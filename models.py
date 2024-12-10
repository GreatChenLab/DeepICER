import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm


class DeepICER(nn.Module):
    def __init__(self, **config):
        super(DeepICER, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        cell_emb_dim = config["CELL"]["EMBEDDING_DIM"]
        num_filters = config["CELL"]["NUM_FILTERS"]
        kernel_size = config["CELL"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp5_hidden_dim = 1024
        mlp_out_dim = 978
        drug_padding = config["DRUG"]["PADDING"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats,
                dim_embedding=drug_embedding,
                padding=drug_padding,
                hidden_feats=drug_hidden_feats)
        self.cell_extractor = CellCNN(cell_emb_dim, num_filters, kernel_size)
        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_decoder = MLPDecoder5(mlp_in_dim, mlp5_hidden_dim, mlp_out_dim)

    def forward(self, bg_d, v_c, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_c = self.cell_extractor(v_c)
        f, att = self.bcn(v_d, v_c)
        x, pred = self.mlp_decoder(f)
        if mode == "train":
            return v_d, v_c, f, pred
        elif mode == "eval":
            return v_d, v_c, pred, att, f, x


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class CellCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size):
        super(CellCNN, self).__init__()
        in_ch = [embedding_dim] + num_filters
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder5(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPDecoder5, self).__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout4 = nn.Dropout(0.25)
        self.fc5 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x))
        x_f = self.dropout4(x)
        x_f = self.fc5(x_f)
        return x, x_f
