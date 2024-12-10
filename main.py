from models import DeepICER
from time import time
from utils import set_seed, graph_collate_func
from configs import get_cfg_defaults
from dataloader import ICERDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="DeepICER for identifying ingredient with cellular response") #初始化参数解析器
parser.add_argument('-u', '--gpu', default='0', help="set gpu", type=str, choices=['0', '1', '2'])
parser.add_argument('-cf', '--config', default='DeepICER.yaml', help="path to config file", type=str)
parser.add_argument('-d', '--data', default='data', type=str, help='path to data file')
parser.add_argument('-s', '--split', default='cp_cc05_filter_cell_splited.csv', type=str, help="data splited to train")
parser.add_argument('-is', '--iSMILES', default='cp2is.csv', type=str, help="durg(pert_id) to iSMILES file")
parser.add_argument('-g', '--gene', default='gene_feature_revised.csv', type=str, help="gene vector file")
parser.add_argument('-c', '--cell', default='ccle_filtered.csv', type=str, help="cell tpm file")
parser.add_argument('-l', '--lincs', default='20_Level5_lm_cp_all.feather', type=str, help="LINCS level 5 data file")
parser.add_argument('-o', '--output', required=True, type=str, help="output dir")
args = parser.parse_args() #解析参数

device = torch.device('cuda:'+args.gpu if torch.cuda.is_available() else 'cpu') #设置设备
def main():
    torch.cuda.empty_cache() # 清理显存缓存
    warnings.filterwarnings("ignore", message="invalid value encountered in divide") # 忽略torch分数除0报错
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config) # 加载配置
    set_seed(cfg.SOLVER.SEED) # 设置随机种子
    # suffix = str(int(time() * 1000))[6:]  # 生成结果目录和实验名称
    output_dir = args.output
    # 检查目录是否存在
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print(f"目录 {output_dir} 已存在。")
    print(f"Config yaml: {args.config}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    
    p = Path.cwd() # 当前目录，PosixPath('.')
    result_dir = p / output_dir
    data_dir = p / args.data # 构建数据集路径
    info_path = data_dir / args.split # cp_um_info_splited_cc102.csv
    cp2is_path = data_dir / args.iSMILES # 'cp_pert_info_is.csv'
    gene_path = data_dir / args.gene #'gene_vector.csv'
    cell_path = data_dir / args.cell #'ccle_filtered.csv'
    id2gene_path = data_dir / 'lm_info.csv'
    l5_lm_cp_ex_path = data_dir / args.lincs #'Level5_lm_cp_cc102.feather'
    cp_kpgt_path = data_dir / 'cp_kpgt.csv'

    sig_info = pd.read_csv(info_path, index_col = 'sig_id')
    train_sig = sig_info[sig_info['random_column'] == 'train']
    val_sig = sig_info[sig_info['random_column'] == 'validation']
    test_sig = sig_info[sig_info['random_column'] == 'test']

    #tes_info = pd.read_csv('./data/cp_cc05_filter_cell_splited.csv', index_col = 'sig_id')
    #test_sig = tes_info[tes_info['random_column'] == 'test']

    cp2is = pd.read_csv(cp2is_path, index_col = 'pert_id')
    cp_kpgt = pd.read_csv(cp_kpgt_path)
    gene_go = pd.read_csv(gene_path, header = None, index_col = 0)
    cell_tpm = pd.read_csv(cell_path, index_col = 'cell_line')

    id2gene = pd.read_csv(id2gene_path, index_col = 'pr_gene_id', dtype=str)
    id2gene = id2gene['pr_gene_symbol'].to_dict()
    id2gene = {str(key): value for key, value in id2gene.items()}

    l5_lm_cp_ex = pd.read_feather(l5_lm_cp_ex_path)
    l5_lm_cp_ex.rename(index=id2gene, inplace = True)

    train_sig_info = ICERDataset(train_sig.index.values, train_sig, cp2is, gene_go, cell_tpm, l5_lm_cp_ex)
    val_sig_info = ICERDataset(val_sig.index.values, val_sig, cp2is, gene_go, cell_tpm, l5_lm_cp_ex)
    test_sig_info = ICERDataset(test_sig.index.values, test_sig, cp2is, gene_go, cell_tpm, l5_lm_cp_ex)

    # train_sig_info = ICERDataset(train_sig.index.values, train_sig, cp_kpgt, gene_go, cell_tpm, l5_lm_cp_ex)
    # val_sig_info = ICERDataset(val_sig.index.values, val_sig, cp_kpgt, gene_go, cell_tpm, l5_lm_cp_ex)
    # test_sig_info = ICERDataset(test_sig.index.values, test_sig, cp_kpgt, gene_go, cell_tpm, l5_lm_cp_ex)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'drop_last': True, 'collate_fn': graph_collate_func, 'num_workers': cfg.SOLVER.NUM_WORKERS, 'pin_memory': True} # 定义dataloader参数

    train_generator = DataLoader(train_sig_info, **params) # 构建训练、验证、测试DataLoader
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_sig_info, **params)
    test_generator = DataLoader(test_sig_info, **params)


    model = DeepICER(**cfg).to(device) #初始化模型和优化器

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR) # 构建模型优化器

    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, train_generator, val_generator, test_generator, result_dir, **cfg) # 开始训练
    result = trainer.train()

    with open(result_dir / "model_architecture.txt", "w") as wf: # 保存模型结构
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {output_dir}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
