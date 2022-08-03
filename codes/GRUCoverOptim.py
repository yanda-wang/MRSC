import sys

import dill
import skorch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch import optim
from Optimization import MedRecSeq2SetGRUCoverGateSelCombFixed, MedRecSeq2SetTrainer
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import load as optim_load
from skopt.callbacks import CheckpointSaver

from Parameters import Params

params = Params()

PATIENT_RECORDS_FILE = params.PATIENT_RECORDS_FILE
CONCEPTID_FILE = params.CONCEPTID_FILE
EHR_MATRIX_FILE = params.EHR_MATRIX_FILE 
DEVICE = params.device  # torch.device("cuda" if USE_CUDA else "cpu")
MEDICATION_COUNT = params.MEDICATION_COUNT
DIAGNOSES_COUNT = params.DIAGNOSES_COUNT
PROCEDURES_COUNT = params.PROCEDURES_COUNT

# ENCODER_TYPE = params.ENCODER_TYPE

DIAGNOSE_INDEX = params.DIAGNOSE_INDEX
PROCEDURES_INDEX = params.PROCEDURE_INDEX
MEDICATION_INDEX = params.MEDICATION_INDEX

OPT_SPLIT_TAG_ADMISSION = params.OPT_SPLIT_TAG_ADMISSION  # -1
OPT_SPLIT_TAG_VARIABLE = params.OPT_SPLIT_TAG_VARIABLE  # -2
OPT_MODEL_MAX_EPOCH = params.OPT_MODEL_MAX_EPOCH

TRAIN_RATIO = params.TRAIN_RATIO
TEST_RATIO = params.TEST_RATIO

LOG_FILE = 'data/log/GRUCoverOptim.log'
CHECKPOINT_FILE = 'data/hyper-model/GRUCover_checkpoint.pkl'


def concatenate_single_admission(records):
    x = records[0]
    for item in records[1:]:
        x = x + [OPT_SPLIT_TAG_VARIABLE] + item
    return x


def concatenate_all_admissions(records):
    x = concatenate_single_admission(records[0])
    for admission in records[1:]:
        current_adm = concatenate_single_admission(admission)
        x = x + [OPT_SPLIT_TAG_ADMISSION] + current_adm
    return x


def get_x_y(patient_records):
    x, y = [], []
    for patient in patient_records:
        for idx, adm in enumerate(patient):
            current_records = patient[:idx + 1]
            current_x = concatenate_all_admissions(current_records)
            x.append(np.array(current_x))
            target = adm[MEDICATION_INDEX]
            y.append(np.array(target))
    return np.array(x, dtype=object), np.array(y, dtype=object)


def get_data(patient_records_file):
    patient_records = pd.read_pickle(patient_records_file)
    split_point = int(len(patient_records) * TRAIN_RATIO)
    test_count = int(len(patient_records) * TEST_RATIO)
    train = patient_records[:split_point]
    test = patient_records[split_point:split_point + test_count]

    train_x, train_y = get_x_y(train)
    test_x, test_y = get_x_y(test)
    return train_x, train_y, test_x, test_y


def get_metric(y_predict, y_target):
    f1 = []
    for yp, yt in zip(y_predict, y_target):
        if yp.shape[0] == 0:
            f1.append(0)
        else:
            intersection = list(set(yp.tolist()) & set(yt.tolist()))
            precision = float(len(intersection)) / len(yp.tolist())
            recall = float(len(intersection)) / len(yt.tolist())
            if precision + recall == 0:
                f1.append(0)
            else:
                f1.append(2.0 * precision * recall / (precision + recall))

    avg_f1 = np.mean(np.array(f1))
    return avg_f1


search_space = [Categorical(categories=['64', '128', '200', '256', '300', '400'], name='dimension'),
                Integer(low=1, high=5, name='encoder_n_layers'),
                Real(low=0, high=0.95, name='encoder_embedding_dropout_rate'),
                Real(low=0, high=0.95, name='encoder_gru_dropout_rate'),

                Real(low=0, high=0.95, name='decoder_dropout_rate'),
                Integer(low=5, high=15, name='decoder_least_adm_count'),
                Integer(low=10, high=30, name='decoder_hop_count'),
                Integer(low=1, high=100, name='decoder_coverage_dim'),
                Categorical(categories=['dot', 'general', 'concat'], name='decoder_attn_type_kv'),
                Categorical(categories=['dot', 'general', 'concat'], name='decoder_attn_type_embedding'),
                Integer(low=5, high=15, name='decoder_regular_hop_count'),

                Real(low=1e-5, high=1e-2, prior='log-uniform', name='optimizer_encoder_lr'),
                Real(low=1e-5, high=1e-2, prior='log-uniform', name='optimizer_decoder_lr')
                ]


@use_named_args(dimensions=search_space)
def fitness(dimension, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate, decoder_dropout_rate,
            decoder_least_adm_count, decoder_hop_count, decoder_coverage_dim, decoder_attn_type_kv,
            decoder_attn_type_embedding, decoder_regular_hop_count, optimizer_encoder_lr, optimizer_decoder_lr):
    ehr_matrix = np.load(EHR_MATRIX_FILE, allow_pickle=True)
    input_size = int(dimension)
    hidden_size = int(dimension)

    print('*' * 30)
    print('hyper-parameters')
    print('input size:', input_size)
    print('encoder_n_layers:', encoder_n_layers)
    print('encoder_embedding_dropout_rate:', encoder_embedding_dropout_rate)
    print('encoder_gru_dropout_rate:', encoder_gru_dropout_rate)
    print('encoder_optimizer_lr:{0:.1e}'.format(optimizer_encoder_lr))

    print('decoder_dropout_rate:', decoder_dropout_rate)
    print('decoder_least_adm_count:', decoder_least_adm_count)
    print('decoder_hop_count:', decoder_hop_count)
    print('decoder_coverage_dim:', decoder_coverage_dim)
    print('decoder_attn_type_kv:', decoder_attn_type_kv)
    print('decoder_attn_type_embedding:', decoder_attn_type_embedding)
    print('decoder_regular_hop_count:', decoder_regular_hop_count)
    print('decoder_optimizer_lr:{0:.1e}'.format(optimizer_decoder_lr))

    print()

    model = MedRecSeq2SetTrainer(criterion=nn.BCEWithLogitsLoss, optimizer_encoder=optim.Adam,
                                 optimizer_decoder=optim.Adam, max_epochs=OPT_MODEL_MAX_EPOCH, batch_size=1,
                                 train_split=None, callbacks=[skorch.callbacks.ProgressBar(batches_per_epoch='auto'), ],
                                 device=DEVICE, module=MedRecSeq2SetGRUCoverGateSelCombFixed,
                                 module__device=DEVICE, module__input_size=input_size, module__hidden_size=hidden_size,
                                 module__diagnose_count=DIAGNOSES_COUNT, module__procedures_count=PROCEDURES_COUNT,
                                 module__medication_count=MEDICATION_COUNT,

                                 module__encoder__n_layers=encoder_n_layers.item(),
                                 module__encoder__embedding_dropout_rate=encoder_embedding_dropout_rate,
                                 module__encoder__gru_dropout_rate=encoder_gru_dropout_rate,

                                 module__decoder__dropout_rate=decoder_dropout_rate,
                                 module__decoder__least_adm_count=decoder_least_adm_count,
                                 module__decoder__hop_count=decoder_hop_count,
                                 module__decoder__coverage_dim=decoder_coverage_dim,
                                 module__decoder__attn_type_kv=decoder_attn_type_kv,
                                 module__decoder__attn_type_embedding=decoder_attn_type_embedding,
                                 module__decoder__regular_hop_count=decoder_regular_hop_count,
                                 module__decoder__ehr_adj=ehr_matrix,

                                 optimizer_encoder__lr=optimizer_encoder_lr,
                                 optimizer_decoder__lr=optimizer_decoder_lr
                                 )

    train_x, train_y, test_x, test_y = get_data(PATIENT_RECORDS_FILE)
    model.fit(train_x, train_y)
    predict_y = model.predict(test_x)
    metric = get_metric(predict_y, test_y)

    print('metric:{0:.4f}'.format(metric))

    return -metric


def optimize(n_calls):
    sys.stdout = open(LOG_FILE, 'a')
    checkpoint_saver = CheckpointSaver(CHECKPOINT_FILE, compress=9)

    # optim_result = optim_load(CHECKPOINT_FILE)
    # examined_values = optim_result.x_iters
    # observed_values = optim_result.func_vals
    # result = gp_minimize(fitness, search_space, n_calls=n_calls, verbose=True, callback=[checkpoint_saver],
    #                      x0=examined_values, y0=observed_values, n_initial_points=-len(examined_values))

    result = gp_minimize(fitness, search_space, n_calls=n_calls, verbose=True, callback=[checkpoint_saver])
    print('**********************************')
    print('best result:')
    print('metric:', -result.fun)
    print('optimal hyper-parameters')

    space_dim_name = [item.name for item in search_space]
    for hyper, value in zip(space_dim_name, result.x):
        print(hyper, value)
    sys.stdout.close()


if __name__ == "__main__":
    optimize(25)
