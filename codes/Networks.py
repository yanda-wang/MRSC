import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.parameter import Parameter

from Parameters import Params

params = Params()
DIAGNOSE_INDEX = params.DIAGNOSE_INDEX
PROCEDURE_INDEX = params.PROCEDURE_INDEX
MEDICATION_INDEX = params.MEDICATION_INDEX


class Encoder(nn.Module):
    def __init__(self, device, input_size, hidden_size, diagnoses_count, procedure_count, n_layers=1,
                 embedding_dropout_rate=0, gru_dropout_rate=0, embedding_diagnoses_np=None,
                 embedding_procedures_np=None):
        super(Encoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_diagnoses = nn.Embedding(diagnoses_count, input_size)
        self.embedding_procedures = nn.Embedding(procedure_count, input_size)
        self.n_layers = n_layers
        self.embedding_dropout_rate = embedding_dropout_rate
        self.gru_dropout_rate = gru_dropout_rate

        self.dropout = nn.Dropout(self.embedding_dropout_rate)
        self.gru_diagnoses = nn.GRU(self.input_size, self.hidden_size, self.n_layers,
                                    dropout=(0 if self.n_layers == 1 else self.gru_dropout_rate))

        self.gru_procedures = nn.GRU(self.input_size, self.hidden_size, self.n_layers,
                                     dropout=(0 if self.n_layers == 1 else self.gru_dropout_rate))

        self.linear_embedding = nn.Sequential(nn.ReLU(), nn.Linear(2 * hidden_size, hidden_size))
        self.embedding_diagnoses.weight.data.uniform_(-0.1, 0.1)
        self.embedding_procedures.weight.data.uniform_(-0.1, 0.1)
        if embedding_diagnoses_np is not None:  # use pretrained embedding vectors to initialize the embeddings
            print('use pretrained embedding vectors to initialize diagnoses embeddings')
            self.embedding_diagnoses.weight.data.copy_(torch.from_numpy(embedding_diagnoses_np))
        if embedding_procedures_np is not None:
            print('use pretrained embedding vectors to initialize procedures embeddings')
            self.embedding_procedures.weight.data.copy_(torch.from_numpy(embedding_procedures_np))

    def forward(self, patient_record):
        seq_diagnoses = []
        seq_procedures = []
        memory_values = []
        for admission in patient_record:
            data_diagnoses = self.dropout(
                self.embedding_diagnoses(torch.LongTensor(admission[DIAGNOSE_INDEX]).to(self.device))).mean(
                dim=0, keepdim=True)
            data_procedures = self.dropout(
                self.embedding_procedures(torch.LongTensor(admission[PROCEDURE_INDEX]).to(self.device))).mean(
                dim=0, keepdim=True)
            seq_diagnoses.append(data_diagnoses)
            seq_procedures.append(data_procedures)
            memory_values.append(admission[MEDICATION_INDEX])
        seq_diagnoses = torch.cat(seq_diagnoses).unsqueeze(dim=1)  # dim=(#admission,1,input_size)
        seq_procedures = torch.cat(seq_procedures).unsqueeze(dim=1)  # dim=(#admission,1,input_size)

        # output dim=(#admission,1,hidden_size)
        # hidden dim=(num_layers,1,hidden_size)
        output_diagnoses, hidden_diagnoses = self.gru_diagnoses(seq_diagnoses)
        # output dim=(#admission,1,hidden_size)
        # hidden dim=(num_layers,1,hidden_size)
        output_procedures, hidden_procedures = self.gru_procedures(seq_procedures)

        patient_representations = torch.cat((output_diagnoses, output_procedures), dim=-1).squeeze(
            dim=1)  # dim=(#admission,2*hidden_size)
        queries = self.linear_embedding(patient_representations)  # dim=(#admission,hidden_size)
        query = queries[-1:]  # dim=(1,hidden_size)

        if len(patient_record) > 1:
            memory_keys = queries[:-1]  # dim=(#admission-1,hidden_size)
            memory_values = memory_values[:-1]
        else:
            memory_keys = None
            memory_values = None

        return query, memory_keys, memory_values


class Attn(nn.Module):
    def __init__(self, method, hidden_size, coverage_dim=1):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.coverage_dim = coverage_dim
        if self.method not in ['dot', 'general', 'concat', 'gru_cover', 'sum_cover']:
            raise ValueError(self.method,
                             "is not an appropriate attention method, choose from dot, general, concat, gru_cover, and sum_cover.")

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        elif self.method == 'gru_cover':
            self.attn = nn.Linear(hidden_size * 2 + coverage_dim, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        elif self.method == 'sum_cover':
            self.attn = nn.Linear(hidden_size * 2 + 1, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        self.initialize_weights()

    # score=query.T()*keys
    def dot_score(self, query, keys):
        return torch.sum(query * keys, -1).unsqueeze(0)  # dim=(1,keys.dim(0))

    # score=query.T()*W*keys, W is a matrix
    def general_score(self, query, keys):
        energy = self.attn(keys)
        return torch.sum(query * energy, -1).unsqueeze(0)  # dim=(1, keys.dim(0))

    # score=v.T()*tanh(W*[query;keys])
    def concat_score(self, query, keys):
        energy = self.attn(torch.cat((query.expand(keys.size(0), -1), keys), -1)).tanh()
        return torch.sum(self.v * energy, -1).unsqueeze(0)  # dim=(1, keys.dim(0)

    def gru_cover_score(self, query, keys, last_coverage):
        energy = self.attn(
            torch.cat((query.expand(keys.size(0), -1), keys, last_coverage), -1)).tanh()
        return torch.sum(self.v * energy, -1).unsqueeze(0)

    def sum_cover_score(self, query, keys, last_coverage):
        energy = self.attn(
            torch.cat((query.expand(keys.size(0), -1), keys, last_coverage.reshape(keys.size(0), -1)), -1)).tanh()
        return torch.sum(self.v * energy, -1).unsqueeze(0)

    def initialize_weights(self, init_range=0.1):
        if self.method == 'concat' or self.method == 'gru_cover' or self.method == 'sum_cover':
            self.v.data.uniform_(-init_range, init_range)

        if self.method == 'concat' or self.method == 'general' or self.method == 'gru_cover' or \
                self.method == 'sum_cover':
            nn.init.kaiming_normal_(self.attn.weight)

    def forward(self, query, keys=None, last_coverage=None):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(query, keys)
        elif self.method == 'concat':
            attn_energies = self.concat_score(query, keys)
        elif self.method == 'dot':
            attn_energies = self.dot_score(query, keys)
        elif self.method == 'gru_cover':
            attn_energies = self.gru_cover_score(query, keys, last_coverage)
        elif self.method == 'sum_cover':
            attn_energies = self.sum_cover_score(query, keys, last_coverage)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1)  # dim=(1,keys.dim(0))


class CoverageGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(CoverageGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.coverage2hidden = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)  # parameters to multiply coverage
        self.attention2hidden = nn.Linear(1, 3 * hidden_size, bias=bias)  # parameter to multiply attenttion weight
        self.enHidden2hidden = nn.Linear(input_size, 3 * hidden_size, bias=bias)  # parameter to multiply encoder hidden
        self.deHidden2hidden = nn.Linear(input_size, 3 * hidden_size, bias=bias)  # parameter to multiply decoder hidden

        self.init_parameters()

    def init_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, coverage, attention, enHidden, deHidden):
        gate_coverage = self.coverage2hidden(coverage)  # dim=(1,batch_size,3*hidden)
        gate_attention = self.attention2hidden(attention)  # dim=(1,batch_size,3*hidden)
        gate_enHidden = self.enHidden2hidden(enHidden)  # dim=(1,batch_size,3*hidden)
        gate_deHidden = self.deHidden2hidden(deHidden)  # dim=(1,batch_size,3*hidden)

        coverage_z, coverage_r, coverage_n = gate_coverage.chunk(3, -1)
        attention_z, attention_r, attention_n = gate_attention.chunk(3, -1)
        enHidden_z, enHidden_r, enHidden_n = gate_enHidden.chunk(3, -1)
        deHidden_z, deHidden_r, deHidden_n = gate_deHidden.chunk(3, -1)

        z = torch.sigmoid(coverage_z + attention_z + enHidden_z + deHidden_z)
        r = torch.sigmoid(coverage_r + attention_r + enHidden_r + deHidden_r)
        new_cov = torch.tanh(r * coverage_n + attention_n + enHidden_n + deHidden_n)

        coverage = (1 - z) * new_cov + z * coverage

        return coverage


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, device, item_count, embedding_size, adj_matrix, dropout_rate):
        super(GCN, self).__init__()
        self.device = device
        self.item_count = item_count
        self.embedding_size = embedding_size

        adj_matrix = self.normalize(adj_matrix + np.eye(adj_matrix.shape[0]))
        self.adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)
        self.x = torch.eye(item_count).to(self.device)

        self.gcn1 = GraphConvolution(item_count, embedding_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gcn2 = GraphConvolution(embedding_size, embedding_size)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj_matrix)  # dim=(item_count,embedding*size)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj_matrix)  # dim=(item_count,embedding_size)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class Gate(nn.Module):
    def __init__(self, hidden_size):
        super(Gate, self).__init__()
        self.transform = nn.Linear(hidden_size * 2, hidden_size)
        nn.init.kaiming_normal_(self.transform.weight)

    def forward(self, query, key):
        r = self.transform(torch.cat((query.expand(key.size(0), -1), key), -1))
        gate = torch.sigmoid(r)  # dim=(key.size(0),hidden_size)
        return gate



class DecoderSumCover(nn.Module):
    def __init__(self, device, hidden_size, output_size, dropout_rate=0, hop_count=20, attn_type_kv='dot',
                 attn_type_embedding='dot', least_adm_count=3, select_adm_count=3, regular_hop_count=5, ehr_adj=None):
        super(DecoderSumCover, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.hop_count = hop_count
        self.attn_type_kv = attn_type_kv
        self.attn_type_embedding = attn_type_embedding

        self.least_adm_count = least_adm_count
        self.select_adm_count = select_adm_count
        if self.select_adm_count > self.least_adm_count:
            self.select_adm_count = self.least_adm_count

        self.regular_hop_count = regular_hop_count
        if self.regular_hop_count > self.hop_count:
            self.hop_count = self.regular_hop_count + 5
        self.ehr_adj = ehr_adj

        self.dropout = nn.Dropout(dropout_rate)
        self.attn_coverage = Attn('sum_cover', hidden_size)
        self.attn_kv = Attn(attn_type_kv, hidden_size)
        self.attn_embedding = Attn(attn_type_embedding, hidden_size)
        self.gate = Gate(hidden_size)
        self.ehr_gcn = GCN(device, output_size, hidden_size, ehr_adj, dropout_rate)
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size * 3, hidden_size * 2), nn.ReLU(),
                                    nn.Linear(hidden_size * 2, output_size))

    def forward(self, query, memory_keys, memory_values):
        coverage_loss = 0

        if memory_keys is None:
            embedding_medications = self.ehr_gcn()
            weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(weights_embedding, embedding_medications)
            context_o = context_e

        elif memory_keys.size(0) < self.least_adm_count:  # regular multi-hop reading

            memory_values_multi_hot = np.zeros((len(memory_values), self.output_size))
            for idx, admission in enumerate(memory_values):
                memory_values_multi_hot[idx, admission] = 1
            memory_values_multi_hot = torch.FloatTensor(memory_values_multi_hot).to(self.device)
            embedding_medications = self.ehr_gcn()
            attn_weights_kv = self.attn_kv(query, memory_keys)
            attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
            read_context = torch.mm(attn_values_kv, embedding_medications)
            update_query = torch.add(query, read_context)

            last_query = update_query
            last_context = read_context

            for hop in range(1, self.hop_count):
                embedding_medications = self.ehr_gcn()
                attn_weights_kv = self.attn_kv(last_query, memory_keys)
                attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
                read_context = torch.mm(attn_values_kv, embedding_medications)

                update_query = torch.add(last_query, read_context)
                last_query = update_query
                last_context = read_context

            embedding_medications = self.ehr_gcn()
            attn_weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(attn_weights_embedding, embedding_medications)
            context_o = last_query

        else:  # enough admissions, use coverage and gate
            memory_values_multi_hot = np.zeros((len(memory_values), self.output_size))
            for idx, admission in enumerate(memory_values):
                memory_values_multi_hot[idx, admission] = 1
            memory_values_multi_hot = torch.FloatTensor(memory_values_multi_hot).to(self.device)

            coverage = torch.zeros((1, memory_keys.size(0))).to(self.device)
            embedding_medications = self.ehr_gcn()
            attn_weights_kv = self.attn_kv(query, memory_keys)
            attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
            read_context = torch.mm(attn_values_kv, embedding_medications)

            update_query = torch.add(query, read_context)
            coverage = coverage + attn_weights_kv

            last_query = update_query
            last_context = read_context

            for hop in range(1, self.regular_hop_count):  # regular multi-hop reading
                embedding_medications = self.ehr_gcn()
                attn_weights_kv = self.attn_kv(last_query, memory_keys)
                attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
                read_context = torch.mm(attn_values_kv, embedding_medications)

                update_query = torch.add(last_query, read_context)
                coverage = coverage + attn_weights_kv
                last_query = update_query
                last_context = read_context

            # select admissions with top-N coverage
            select_memory_index = torch.topk(coverage, self.select_adm_count)[1]
            select_memory_keys = memory_keys[select_memory_index].squeeze(0)
            select_memory_values = memory_values_multi_hot[select_memory_index].squeeze(0)
            select_coverage = torch.zeros((1, self.select_adm_count)).to(self.device)

            for hop in range(self.regular_hop_count, self.hop_count):
                embedding_medications = self.ehr_gcn()
                attn_weights_kv = self.attn_coverage(last_query, select_memory_keys, select_coverage)
                attn_values_kv = attn_weights_kv.mm(select_memory_values)
                read_context = torch.mm(attn_values_kv, embedding_medications)

                gate = self.gate(last_query, read_context)
                read_context = gate * read_context

                update_query = torch.add(last_query, read_context)
                coverage_loss += torch.sum(torch.min(select_coverage, attn_weights_kv))
                select_coverage = select_coverage + attn_weights_kv

                last_query = update_query
                last_context = read_context

            embedding_medications = self.ehr_gcn()
            attn_weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(attn_weights_embedding, embedding_medications)
            context_o = last_query

        output = self.output(torch.cat((query, context_o, context_e), -1))
        return output, coverage_loss


class AdmissionSelectGate(nn.Module):
    def __init__(self, coverage_dim, hidden_size):
        super(AdmissionSelectGate, self).__init__()
        self.transform = nn.Linear(coverage_dim, hidden_size)
        nn.init.kaiming_normal_(self.transform.weight)

    def forward(self, coverage):  # dim(coverage)=(#adm, coverage_dim)
        r = self.transform(coverage)  # (#adm, coverage_dim)-> (#adm, hidden_size)
        gate = torch.sigmoid(r)  # element-wise gate, dim=(#adm, hidden_size)
        return gate


class DecoderGRUCover(nn.Module):
    def __init__(self, device, hidden_size, output_size, dropout_rate=0, least_adm_count=3, hop_count=20,
                 coverage_dim=1, attn_type_kv='dot', attn_type_embedding='dot', regular_hop_count=5, ehr_adj=None):
        super(DecoderGRUCover, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.least_adm_count = least_adm_count
        self.hop_count = hop_count
        self.coverage_dim = coverage_dim
        self.attn_type_kv = attn_type_kv
        self.attn_type_embedding = attn_type_embedding
        self.regular_hop_count = regular_hop_count
        if self.regular_hop_count > self.hop_count:
            self.hop_count = self.regular_hop_count + 5
        self.ehr_adj = ehr_adj

        self.dropout = nn.Dropout(dropout_rate)
        self.attn_kv = Attn(attn_type_kv, hidden_size)
        self.attn_coverage = Attn('gru_cover', hidden_size, coverage_dim)
        self.attn_embedding = Attn(attn_type_embedding, hidden_size)
        self.ehr_gcn = GCN(device, output_size, hidden_size, ehr_adj, dropout_rate)
        self.keys_gate = AdmissionSelectGate(self.coverage_dim, self.hidden_size)
        self.read_context_gate = Gate(hidden_size)
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size * 3, hidden_size * 2), nn.ReLU(),
                                    nn.Linear(hidden_size * 2, output_size))
        self.coverage_gru = CoverageGRUCell(hidden_size, coverage_dim)

    def forward(self, query, memory_keys, memory_values):
        if memory_keys is None:
            embedding_medications = self.ehr_gcn()
            weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(weights_embedding, embedding_medications)
            context_o = context_e
        elif memory_keys.size(0) < self.least_adm_count:  # regular multi-hop reading, gated read_context

            memory_values_multi_hot = np.zeros((len(memory_values), self.output_size))
            for idx, admission in enumerate(memory_values):
                memory_values_multi_hot[idx, admission] = 1
            memory_values_multi_hot = torch.FloatTensor(memory_values_multi_hot).to(self.device)
            embedding_medications = self.ehr_gcn()

            attn_weights_kv = self.attn_kv(query, memory_keys)
            attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
            read_context = torch.mm(attn_values_kv, embedding_medications)
            update_query = torch.add(query, read_context)

            last_query = update_query
            last_context = read_context

            for hop in range(1, self.hop_count):
                embedding_medications = self.ehr_gcn()
                attn_weights_kv = self.attn_kv(last_query, memory_keys)
                attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
                read_context = torch.mm(attn_values_kv, embedding_medications)

                update_query = torch.add(last_query, read_context)
                last_query = update_query
                last_context = read_context

            embedding_medications = self.ehr_gcn()
            attn_weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(attn_weights_embedding, embedding_medications)
            context_o = last_query

        else:  # enough admissions, use coverage and gate

            memory_values_multi_hot = np.zeros((len(memory_values), self.output_size))
            for idx, admission in enumerate(memory_values):
                memory_values_multi_hot[idx, admission] = 1
            memory_values_multi_hot = torch.FloatTensor(memory_values_multi_hot).to(self.device)
            embedding_medications = self.ehr_gcn()

            # initial coverage and context
            coverage = torch.zeros((memory_keys.size(0), self.coverage_dim)).to(self.device)
            attn_weights_kv = self.attn_kv(query, memory_keys)  # regular attention, no coverage
            attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
            read_context = torch.mm(attn_values_kv, embedding_medications)

            update_query = torch.add(query, read_context)
            # calculate the coverage
            coverage = self.coverage_gru(coverage.unsqueeze(0), attn_weights_kv.unsqueeze(-1),
                                         memory_keys.unsqueeze(0), query.expand((1, memory_keys.size(0), -1))).squeeze(
                0)

            last_query = update_query
            last_context = read_context

            for hop in range(1, self.regular_hop_count):  # regular multi-hop reading
                embedding_medications = self.ehr_gcn()
                attn_weights_kv = self.attn_kv(last_query, memory_keys)
                attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
                read_context = torch.mm(attn_values_kv, embedding_medications)

                update_query = torch.add(last_query, read_context)
                last_query = update_query
                last_context = read_context
                coverage = self.coverage_gru(coverage.unsqueeze(0), attn_weights_kv.unsqueeze(-1),
                                             memory_keys.unsqueeze(0),
                                             last_query.expand((1, memory_keys.size(0), -1))).squeeze(0)

            gate_keys = self.keys_gate(coverage)  # (#adm, coverage_dim) -> (#adm, hidden_size)
            gated_keys = gate_keys * memory_keys
            coverage_gated = torch.zeros((memory_keys.size(0), self.coverage_dim)).to(self.device)

            for hop in range(self.regular_hop_count, self.hop_count):
                embedding_medications = self.ehr_gcn()
                attn_weights_kv = self.attn_coverage(last_query, gated_keys, coverage_gated)
                attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
                read_context = torch.mm(attn_values_kv, embedding_medications)

                gate = self.read_context_gate(last_query, read_context)
                read_context = gate * read_context

                update_query = torch.add(last_query, read_context)
                coverage_gated = self.coverage_gru(coverage_gated.unsqueeze(0), attn_weights_kv.unsqueeze(-1),
                                                   gated_keys.unsqueeze(0),
                                                   last_query.expand((1, memory_keys.size(0), -1))).squeeze(0)
                last_query = update_query
                last_context = read_context

            embedding_medications = self.ehr_gcn()
            attn_weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(attn_weights_embedding, embedding_medications)
            context_o = last_query

        output = self.output(torch.cat((query, context_o, context_e), -1))
        return output
