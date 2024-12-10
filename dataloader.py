#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

class ICERDataset(data.Dataset):
    def __init__(self, sig_index_value, sig, cp2is, gene_go, cell_tpm, l5_lm_cp_ex, max_drug_nodes=200):
        self.sig_index_value = sig_index_value
        self.sig = sig
        self.cp2is = cp2is
        self.gene_go = gene_go
        self.cell_tpm = cell_tpm
        self.l5_lm_cp_ex = l5_lm_cp_ex
        self.max_drug_nodes = max_drug_nodes
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.sig_index_value)

    def __getitem__(self, index):
        index = self.sig_index_value[index]
        pert_id = self.sig.loc[index,'pert_id']
        sig_id = index
        i_SMILES = self.cp2is.loc[pert_id,'i_SMILES']
        d_graph = self.fc(smiles=i_SMILES, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        pert_time = self.sig.loc[index,'pert_time']
        pert_time_tensor = torch.tensor([float(pert_time)], dtype=torch.float32)
        actual_node_feats = d_graph.ndata.pop('h')
        pert_dose = self.sig.loc[index,'pert_dose']
        pert_dose_tensor = torch.tensor([float(pert_dose)], dtype=torch.float32)
        num_actual_nodes = actual_node_feats.shape[0]
        actual_node_feats = torch.cat((actual_node_feats, pert_dose_tensor.repeat(num_actual_nodes, 1), pert_time_tensor.repeat(num_actual_nodes, 1), torch.zeros(num_actual_nodes, 1)), 1)
        d_graph.ndata['h'] = actual_node_feats
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 76), torch.ones(num_virtual_nodes, 1)), 1)
        d_graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        d_graph = d_graph.add_self_loop()
        cell = self.sig.loc[index,'cell_iname']
        tpm = self.cell_tpm.loc[cell,:]
        go = self.gene_go
        tpm = tpm.sort_index()
        go = go.sort_index()
        cell_feature = go.multiply(tpm, axis=0)
        cell_feature = torch.tensor(cell_feature.values).to(torch.float32)
        y = self.l5_lm_cp_ex.loc[:, index]
        y = y.sort_index()
        y = torch.tensor(y.values, dtype=torch.float32)
        return d_graph, cell_feature, y, sig_id
