from Eadro.model import EadroModel
import torch

import os
import time
import copy
import numpy as np
import torch
from torch import nn
from pyprof import profile

class BaseModel(nn.Module):
    def __init__(self, model: EadroModel, lr=1e-3, epoches=50, patience=5, result_dir='./', logger=None, device='cuda'):
        super(BaseModel, self).__init__()
        
        self.epoches = epoches
        self.lr = lr
        self.patience = patience # > 0: use early stop
        self.device = device
        self.logger = logger

        self.model_save_dir = result_dir
        model.set_device(device)
        self.model = model.module
    
    def evaluate(self, test_loader, datatype="Test"):
        self.model.eval()
        hrs = np.zeros(5)
        ranks = []
        TP, FP, FN = 0, 0, 0
        batch_cnt, epoch_loss = 0, 0.0 
        
        with torch.no_grad():
            for features, fault_ids, failure_ids, graphs in test_loader:
                with profile(f"Inference for a batch of failure with batchsize {fault_ids.shape[0]}"):
                    res = self.model.forward(graphs, features, fault_ids)
                    for idx, faulty_nodes in enumerate(res["y_pred"]):
                        culprit = fault_ids[idx].item()
                        if culprit == -1:
                            if faulty_nodes[0] == -1: TP+=1
                            else: FP += 1
                        else:
                            if faulty_nodes[0] == -1: FN+=1
                            else: 
                                TP+=1
                                rank = list(faulty_nodes).index(culprit)
                                for j in range(5):
                                    hrs[j] += int(rank <= j)
                                    ranks.append(rank+1)
                epoch_loss += res["loss"].item()
                batch_cnt += 1
        
        pos = TP+FN
        eval_results = {
                "F1": TP*2.0/(TP+FP+pos) if (TP+FP+pos)>0 else 0,
                "Rec": TP*1.0/pos if pos > 0 else 0,
                "Pre": TP*1.0/(TP+FP) if (TP+FP) > 0 else 0}
        
        for j in [1, 2, 3, 5]:
            eval_results["A@"+str(j)] = hrs[j-1]*1.0/pos
            eval_results["MAR"] = sum(ranks) / len(ranks)
            
        self.logger.info("{} -- {}".format(datatype, ", ".join([k+": "+str(f"{v:.4f}") for k, v in eval_results.items()])))

        return eval_results
    
    def fit(self, train_loader, val_loader=None, test_loader=None, evaluation_epoch=10):
        with profile("training"):
            best_hr1, converge, best_state, eval_res = -1, None, None, None # evaluation
            pre_loss, worse_count = float("inf"), 0 # early break

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)
            
            for epoch in range(1, self.epoches+1):
                self.model.train()
                batch_cnt, epoch_loss = 0, 0.0
                epoch_time_start = time.time()
                for features, fault_ids, failure_ids, graphs in train_loader:
                    optimizer.zero_grad()
                    loss = self.model.forward(graphs, features, fault_ids)['loss']
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_cnt += 1
                epoch_time_elapsed = time.time() - epoch_time_start

                epoch_loss = epoch_loss / batch_cnt
                self.logger.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches, epoch_loss, epoch_time_elapsed))

                ####### early break #######
                if epoch_loss > pre_loss:
                    worse_count += 1
                    if self.patience > 0 and worse_count >= self.patience:
                        self.logger.info("Early stop at epoch: {}".format(epoch))
                        break
                else: worse_count = 0
                pre_loss = epoch_loss

                ####### Evaluate test data during training #######
                if (epoch+1) % evaluation_epoch == 0:
                    val_results = self.evaluate(val_loader, datatype="Validate")
                    if val_results["A@1"] > best_hr1:
                        best_hr1, eval_res, converge  = val_results["A@1"], val_results, epoch
                        best_state = copy.deepcopy(self.model.state_dict())

                    self.save_model(best_state)
            
        if converge > 5:
            self.logger.info("* Best result got at epoch {} with A@1: {:.4f}".format(converge, best_hr1))
        else:
            self.logger.info("Unable to convergence!")

        self.load_model(os.path.join(self.model_save_dir, "model.ckpt"))
        test_results = self.evaluate(test_loader, datatype="Test")
        return test_results, converge
    
    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def save_model(self, state, file=None):
        if file is None: file = os.path.join(self.model_save_dir, "model.ckpt")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, file)
