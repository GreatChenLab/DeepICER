from models import DeepICER
from time import time
from configs import get_cfg_defaults
from torch.utils.data import DataLoader
import torch
import argparse
import warnings, os
import pandas as pd
import numpy as np
from pathlib import Path
import random
import dgl
import torch.utils.data as data
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

parser = argparse.ArgumentParser(description="DeepICER for identifying ingredient with cellular response")
parser.add_argument('-i', '--infile', help="input file", type=str)
parser.add_argument('-o', '--output', required=True, type=str, help="output dir")
args = parser.parse_args()
device = torch.device('cpu')

def set_seed(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def graph_collate_func(x):
    x = [item for item in x if item[0] is not None]
    if not x:
        return [dgl.graph(()) for _ in range(len([item for item in x if item[2] is not None]))], [], []
    d, c, s = zip(*x)
    d = dgl.batch(d)
    return d, c, s


class ICERDataset(data.Dataset):
    def __init__(self, sig_index_value, sig, gene_go, cell_tpm, max_drug_nodes=200):
        self.sig_index_value = sig_index_value
        self.sig = sig
        self.gene_go = gene_go
        self.cell_tpm = cell_tpm
        self.max_drug_nodes = max_drug_nodes
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.sig_index_value)

    def __getitem__(self, index):
        index = self.sig_index_value[index]
        sig_id = index
        i_SMILES = self.sig.loc[index,'i_SMILES']
        try:
            d_graph = self.fc(smiles=i_SMILES, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        except Exception as e:
            print(f"Error processing SMILES '{i_SMILES}': {e}")
            d_graph = None
        if d_graph is None:
            print(f"No graph generated for SMILES '{i_SMILES}'. Skipping this entry.")
            return None, None, None
        pert_time = self.sig.loc[index,'pert_time']
        pert_time_tensor = torch.tensor([float(pert_time)], dtype=torch.float32)
        actual_node_feats = d_graph.ndata.pop('h')
        pert_dose = self.sig.loc[index,'pert_dose']
        pert_dose_tensor = torch.tensor([float(pert_dose)], dtype=torch.float32)
        num_actual_nodes = actual_node_feats.shape[0]
        if num_actual_nodes >200:
            print(f"The '{i_SMILES}' has more than 200 nodes. Skipping this entry.")
            return None, None, None

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

        return d_graph, cell_feature, sig_id


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file('DeepICER.yaml')
    output_dir = args.output
    output_file = output_dir+'_profile.csv'
    att_file = output_dir+'_att.csv'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print(f"{output_dir} 。")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    p = Path.cwd()
    result_dir = p / output_dir
    data_dir = p / 'data'
    info_path = p / args.infile
    gene_path = data_dir / 'gene_feature_revised.csv'
    cell_path = data_dir / 'ccle_filtered.csv'
    sig_info = pd.read_csv(info_path, index_col = 'sig_id')
    test_sig = sig_info
    gene_go = pd.read_csv(gene_path, header = None, index_col = 0)
    genes = sorted(list(gene_go.index))
    cell_tpm = pd.read_csv(cell_path, index_col = 'cell_line')
    test_sig_info = ICERDataset(test_sig.index.values, test_sig, gene_go, cell_tpm)
    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': False, 'drop_last': False, 'collate_fn': graph_collate_func, 'num_workers': cfg.SOLVER.NUM_WORKERS, 'pin_memory': True}
    params['shuffle'] = False
    params['drop_last'] = False
    test_generator = DataLoader(test_sig_info, **params)
    best_model = DeepICER(**cfg).to(device)
    best_model.load_state_dict(torch.load('data/best_model.pth', map_location=torch.device('cpu')))
    sig_ids, y_preds = test(best_model, test_generator, device)
    df = pd.DataFrame(y_preds, index=sig_ids,columns=genes)
    df.to_csv(result_dir / output_file)
    print(f"Directory for saving result: {output_dir}")

def test(best_model, dataloader, device):
    sig_ids, y_preds, atts = [], [], []
    num_batches = len(dataloader)
    with torch.no_grad():
        for i, (d_graph, cell_feature, sig_id) in enumerate(dataloader):
            if d_graph is None:
                continue
            d_graph, cell_feature = d_graph.to(device, non_blocking=True), torch.stack(cell_feature, dim=0).to(device, non_blocking=True)
            v_d, v_c, y_pred, att, f, x = best_model(d_graph, cell_feature,mode='eval')
            print(att.size())
            sig_ids.extend(sig_id)
            y_pred = y_pred.cpu().numpy().tolist()
            y_preds.extend(y_pred)
    return sig_ids, y_preds

if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
