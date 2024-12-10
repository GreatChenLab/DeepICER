import torch
import torch.nn as nn
import copy
from pathlib import Path
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
from utils import pearson_cc


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, result_dir, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.best_model = None
        self.best_model_state = None
        self.best_epoch = None
        self.best_pcc = 0
        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_pcc_epoch = [], []
        self.test_metrics = {}
        self.output_dir = Path.cwd() / result_dir
        valid_metric_header = ["# Epoch", "PCC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "PCC", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)
        self.load_checkpoint_if_needed()

    def load_checkpoint_if_needed(self):
        checkpoint_path = self.output_dir / f'model_epoch_0.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint)
            self.current_epoch = 0
            print(f'Checkpoint loaded, resuming from epoch {self.current_epoch}')
        else:
            print('No checkpoint found, starting from scratch')


    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs - self.current_epoch):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            pcc, val_loss = self.test(dataloader="val")
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [pcc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_pcc_epoch.append(pcc)
            if self.current_epoch % 10 == 0:
                model_epoch_name = f'model_epoch_{self.current_epoch}.pth'
                torch.save(self.model.state_dict(), self.output_dir / model_epoch_name)
                print(f'Model saved at epoch {self.current_epoch}')
            if pcc >= self.best_pcc:
                self.best_model = copy.deepcopy(self.model)
                self.best_pcc = pcc
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " PCC "
                  + str(pcc))

        pcc, test_loss = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [pcc, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " PCC " + str(pcc))
        self.test_metrics["pcc"] = pcc
        self.test_metrics["test_loss"] = test_loss
        self.save_result()
        return self.test_metrics

    def save_result(self):
        best_model_epoch = f'best_model_epoch_{self.best_epoch}.pth'
        model_epoch = f'model_epoch_{self.current_epoch}.pth'
        torch.save(self.best_model.state_dict(), self.output_dir / best_model_epoch)
        torch.save(self.model.state_dict(), self.output_dir / model_epoch)
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics
        }
        torch.save(state, self.output_dir / "result_metrics.pt")

        val_prettytable_file = self.output_dir / "valid_markdowntable.txt"
        test_prettytable_file = self.output_dir / "test_markdowntable.txt"
        train_prettytable_file = self.output_dir / "train_markdowntable.txt"
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())


    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (d_graph, cell_feature, y_true, sig_id) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            d_graph, cell_feature, y_true = d_graph.to(self.device, non_blocking=True), torch.stack(cell_feature, dim=0).to(self.device, non_blocking=True), torch.stack(y_true, dim=0).to(self.device, non_blocking=True)
            self.optim.zero_grad()
            v_d, v_c, f, y_pred = self.model(d_graph, cell_feature)
            loss = torch.nn.MSELoss()(y_pred, y_true)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch


    def test(self, dataloader="test"):
        test_loss, test_pcc = 0, 0
        y_true, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (d_graph, cell_feature, y_true, sig_id) in enumerate(data_loader):
                d_graph, cell_feature, y_true = d_graph.to(self.device, non_blocking=True), torch.stack(cell_feature, dim=0).to(self.device, non_blocking=True), torch.stack(y_true, dim=0).to(self.device, non_blocking=True)
                if dataloader == "val":
                    v_d, v_c, f, y_pred = self.model(d_graph, cell_feature)
                elif dataloader == "test":
                    v_d, v_c, f, y_pred = self.best_model(d_graph, cell_feature)
                loss = torch.nn.MSELoss()(y_pred, y_true)
                test_loss += loss.item()
                pearson_corr = pearson_cc(y_pred, y_true)
                test_pcc += pearson_corr.item()
        pcc = test_pcc / num_batches
        test_loss = test_loss / num_batches

        return pcc, test_loss
