import torch
import os
import datetime
import pickle
import dill
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from random import choices
from scipy.stats import entropy, boxcox
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from Networks import Encoder
from Networks import DecoderGRUCover, DecoderSumCover

from Evaluation import EvaluationUtil
from Parameters import Params

params = Params()


class ModelTraining:
    def __init__(self, device, patient_records_file, voc_file, ehr_matrix_file):
        self.device = device
        self.patient_records_file = patient_records_file
        self.voc_file = voc_file
        self.ehr_matrix_file = ehr_matrix_file

        voc = dill.load(open(self.voc_file, 'rb'))
        self.diag_voc = voc['diag_voc']
        self.pro_voc = voc['pro_voc']
        self.med_voc = voc['med_voc']

        self.diagnose_count = len(self.diag_voc.word2idx)
        self.procedure_count = len(self.pro_voc.word2idx)
        self.medication_count = len(self.med_voc.word2idx)

        self.ehr_matrix = dill.load(open(self.ehr_matrix_file, 'rb'))
        self.evaluate_utils = EvaluationUtil()

    def loss_function(self, target_medications, predict_medications, proportion_bce, proportion_multi,
                      coverage_loss=0.0, proportion_coverage=0.0):
        loss_bce_target = np.zeros((1, self.medication_count))
        loss_bce_target[:, target_medications] = 1
        loss_multi_target = np.full((1, self.medication_count), -1)
        for idx, item in enumerate(target_medications):
            loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(predict_medications,
                                                      torch.FloatTensor(loss_bce_target).to(self.device))
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(predict_medications),
                                              torch.LongTensor(loss_multi_target).to(self.device))
        loss = proportion_bce * loss_bce + proportion_multi * loss_multi

        if proportion_coverage != 0:
            loss = loss + proportion_coverage * coverage_loss

        return loss

    def get_performance_on_testset(self, encoder, decoder, patient_records, coverage_type):
        jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg = [], [], [], [], []
        count = 0
        for patient in patient_records:
            for idx, adm in enumerate(patient):
                count += 1
                current_records = patient[:idx + 1]

                query, memory_keys, memory_values = encoder(current_records)

                if coverage_type == 'gru_cover':
                    predict_output = decoder(query, memory_keys, memory_values)
                else:
                    predict_output, _ = decoder(query, memory_keys, memory_values)

                target_medications = adm[params.MEDICATION_INDEX]
                target_multi_hot = np.zeros(self.medication_count)
                target_multi_hot[target_medications] = 1
                predict_prob = torch.sigmoid(predict_output).detach().cpu().numpy()[0]
                predict_multi_hot = predict_prob.copy()

                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)

                predict_multi_hot[predict_multi_hot >= 0.5] = 1
                predict_multi_hot[predict_multi_hot < 0.5] = 0
                predict_medications = list(np.where(predict_multi_hot == 1)[0])

                jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)
                precision = self.evaluate_utils.metric_precision(predict_medications, target_medications)
                recall = self.evaluate_utils.metric_recall(predict_medications, target_medications)
                f1 = self.evaluate_utils.metric_f1(precision, recall)
                prauc = self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

                jaccard_avg.append(jaccard)
                precision_avg.append(precision)
                recall_avg.append(recall)
                f1_avg.append(f1)
                prauc_avg.append(prauc)

        jaccard_avg = np.mean(np.array(jaccard_avg))
        precision_avg = np.mean(np.array(precision_avg))
        recall_avg = np.mean(np.array(recall_avg))
        f1_avg = np.mean(np.array(f1_avg))
        prauc_avg = np.mean(np.array(prauc_avg))

        return jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg

    def trainIters(self, encoder, decoder, encoder_optimizer, decoder_optimizer, coverage_type, patient_records_train,
                   patient_records_test, save_model_path, n_epoch, print_every_iteration=100, save_every_epoch=5,
                   trained_epoch=0, trained_iteration=0):
        start_epoch = trained_epoch + 1
        trained_n_iteration = trained_iteration
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        log_file = open(os.path.join(save_model_path, 'medrec_loss.log'), 'a+')
        encoder_lr_scheduler = ReduceLROnPlateau(encoder_optimizer, mode='max', patience=5, factor=0.1)
        decoder_lr_scheduler = ReduceLROnPlateau(decoder_optimizer, mode='max', patience=5, factor=0.1)

        for epoch in range(start_epoch, start_epoch + n_epoch):
            print_loss = []
            iteration = 0
            for patient in patient_records_train:
                for idx, adm in enumerate(patient):
                    trained_n_iteration += 1
                    iteration += 1
                    current_records = patient[:idx + 1]
                    target_medications = adm[params.MEDICATION_INDEX]
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()

                    query, memory_keys, memory_values = encoder(current_records)
                    if coverage_type == 'gru_cover':
                        predict_output = decoder(query, memory_keys, memory_values)
                        loss = self.loss_function(target_medications, predict_output, 0.8, 0.1)
                        print_loss.append(loss.item())
                    else:  # sum_cover
                        predict_output, coverage_loss = decoder(query, memory_keys, memory_values)
                        loss = self.loss_function(target_medications, predict_output, 0.8, 0.1, coverage_loss, 0.1)
                        print_loss.append(loss.item())
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()

                    if iteration % print_every_iteration == 0:
                        print_loss_avg = np.mean(np.array(print_loss))
                        print_loss = []
                        print(
                            'epoch: {}; time: {}; Iteration: {}; train loss: {:.4f}'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))
                        log_file.write(
                            'epoch: {}; time: {}; Iteration: {};  train loss: {:.4f}\n'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))

            encoder.eval()
            decoder.eval()
            jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg = self.get_performance_on_testset(encoder,
                                                                                                        decoder,
                                                                                                        patient_records_test,
                                                                                                        coverage_type)
            encoder.train()
            decoder.train()

            print(
                'epoch: {}; time: {}; Iteration: {}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; prauc_test: {:.4f}'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    prauc_avg))
            log_file.write(
                'epoch: {}; time: {}; Iteration: {}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; prauc_test: {:.4f}\n'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    prauc_avg))

            encoder_lr_scheduler.step(f1_avg)
            decoder_lr_scheduler.step(f1_avg)

            if epoch % save_every_epoch == 0:
                torch.save(
                    {'medrec_epoch': epoch,
                     'medrec_iteration': trained_n_iteration,
                     'encoder': encoder.state_dict(),
                     'decoder': decoder.state_dict(),
                     'encoder_optimizer': encoder_optimizer.state_dict(),
                     'decoder_optimizer': decoder_optimizer.state_dict()},
                    os.path.join(save_model_path,
                                 'medrec_{}_{}_{:.4f}.checkpoint'.format(epoch, trained_n_iteration, f1_avg)))

        log_file.close()

    def train(self, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate,
              encoder_gru_dropout_rate, encoder_learning_rate, decoder_type, decoder_dropout_rate, decoder_hop_count,
              regular_hop_count, attn_type_kv, attn_type_embedding, least_adm_count, select_adm_count, coverage_dim,
              decoder_learning_rate, save_model_dir='data/model', n_epoch=50, print_every_iteration=100,
              save_every_epoch=1, load_model_name=None):
        print('initializing >>>')
        if load_model_name:
            print('load model from checkpoint file: ', load_model_name)
            checkpoint = torch.load(load_model_name)

        encoder = Encoder(self.device, input_size, hidden_size, self.diagnose_count,
                          self.procedure_count, encoder_n_layers, encoder_embedding_dropout_rate,
                          encoder_gru_dropout_rate)

        if decoder_type == 'gru_cover':
            decoder = DecoderGRUCover(params.device, hidden_size, self.medication_count,
                                      decoder_dropout_rate, least_adm_count, decoder_hop_count,
                                      coverage_dim, attn_type_kv, attn_type_embedding,
                                      regular_hop_count, self.ehr_matrix)
            coverage_type = 'gru_cover'

        elif decoder_type == 'sum_cover':
            decoder = DecoderSumCover(params.device, hidden_size, self.medication_count,
                                      decoder_dropout_rate, decoder_hop_count, attn_type_kv,
                                      attn_type_embedding, least_adm_count, select_adm_count,
                                      regular_hop_count, self.ehr_matrix)
            coverage_type = 'sum_cover'

        else:
            print('wrong decoder type, choose from gru_cover and sum_cover')
            return

        if load_model_name:
            encoder_sd = checkpoint['encoder']
            decoder_sd = checkpoint['decoder']
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.train()
        decoder.train()

        print('build optimizer >>>')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_learning_rate)
        if load_model_name:
            encoder_optimizer_sd = checkpoint['encoder_optimizer']
            decoder_optimizer_sd = checkpoint['decoder_optimizer']
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        print('start training >>>')
        patient_records = pd.read_pickle(self.patient_records_file)
        split_point = int(len(patient_records) * params.TRAIN_RATIO)
        test_count = int(len(patient_records) * params.TEST_RATIO)
        patient_records_train = patient_records[:split_point]
        patient_records_test = patient_records[split_point:split_point + test_count]

        medrec_trained_epoch = 0
        medrec_trained_iteration = 0

        if load_model_name:
            medrec_trained_n_epoch_sd = checkpoint['medrec_epoch']
            medrec_trained_n_iteration_sd = checkpoint['medrec_iteration']
            medrec_trained_epoch = medrec_trained_n_epoch_sd
            medrec_trained_iteration = medrec_trained_n_iteration_sd

        save_model_structure = str(encoder_n_layers) + '_' + str(input_size) + '_' + str(hidden_size)
        save_model_parameters = str(encoder_embedding_dropout_rate) + '_' + str(encoder_gru_dropout_rate) + '_' + str(
            decoder_dropout_rate) + '_' + attn_type_kv + '_' + attn_type_embedding + '_' + str(
            decoder_hop_count) + '_' + str(regular_hop_count)
        save_model_path = os.path.join(save_model_dir, save_model_structure, save_model_parameters)

        self.trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, coverage_type, patient_records_train,
                        patient_records_test, save_model_path, n_epoch, print_every_iteration, save_every_epoch,
                        medrec_trained_epoch, medrec_trained_iteration)

