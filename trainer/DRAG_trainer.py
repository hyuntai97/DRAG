'''
Diversified and Realistic Anomaly Generation based Anomaly Detection (DRAG) Code base

Reference: Hyuntae Kim, Changhee Lee, 
"Enhancing Anomaly Detection via Generating Diversified and Realistic Artificial Anomalies,"
AAAI-24 under review, 2023. 

Code author: Hyuntae Kim (soodaman97@cau.ac.kr)

----------------------------------------------------------

DRAG_trainer.py

(1) Train 
    - Generate artificial anomalies by multiple perturbators
    - Passing the data through the encoder-discriminator pair to produce the logits 
    - Update the entire network parameters using the loss (i.e., cross-entropy loss, perturbations' norm, direction diversity)
        - Encoder
        - Perturbators 
        - Discriminator 

(2) Test 
    - Passing the test data through the encoder-discriminator pair to produce the logits (i.e., anomaly score)
    - Measure the test evaluation metric (e.g, AUC, F1-score)
'''

import os
import copy
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import json 
from itertools import combinations 

#Trainer for PLAD
class Trainer:

    def __init__(self, model, view_model, discriminator_model, 
                 optimizer, clf_optimizer, lamda, device, 
                 k, cos_weight, ratio=20, nl_head=0, metric='AUC'):  
        self.model = model
        self.view_model = view_model
        self.discriminator = discriminator_model
        self.optimizer = optimizer
        self.clf_optimizer = clf_optimizer
        self.lamda = lamda
        self.device = device
        self.k = k
        self.ratio = ratio 
        self.cos_weight = cos_weight
        self.nl_head = nl_head
        self.metric = metric

        self.results = {f'test_{self.metric}':None, 
                        'final_scores':None}
        
    def train(self, train_loader, val_loader, 
              learning_rate, lr_scheduler, total_epochs):
        best_score = -np.inf
        best_model = None

        for epoch in range(total_epochs): 
            self.view_model.train()
            self.model.train()
            lr_scheduler(epoch, learning_rate, self.clf_optimizer)
  
            total_loss = 0
            batch_idx = -1
            for data, target, _ in train_loader:
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)  
                
                # Data Processing
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)
                             
                self.optimizer.zero_grad()
                self.clf_optimizer.zero_grad()    

                #Produce anomalies 
                perturbation_lst = []
                feature_lst = []
                for i in range(len(self.view_model)):
                    perturbation, features = self.view_model[i](data)
                    perturbation_lst.append(perturbation)
                    feature_lst.append(features)

                output_perturbation = torch.stack(perturbation_lst, 0)
                fro_distortion_norm = torch.norm(output_perturbation, p='fro')

                true_feat, _ = self.model(data)

                if self.nl_head == 0:  # NOT USING non-linear head 
                    comb = list(combinations(perturbation_lst, 2))
                elif self.nl_head == 1: # USING non-linear head  
                    comb = list(combinations(feature_lst, 2))

                cosine_sim_loss = 0 
                for i in range(len(comb)):
                    x = comb[i][0]
                    y = comb[i][1]
                    cos_sim = self.cosine_similarity(x, y)
                    cosine_sim_loss += cos_sim

                total_fake_feat = []
                total_robust_feat = []
                for i in range(len(output_perturbation)):
                    y_pixels = output_perturbation[i]
                    fake_feat = true_feat + y_pixels

                    #####-- augmentation perturbation scheme --#####
                    perturb_norm = torch.flatten(y_pixels, start_dim=1)
                    perturb_norm = torch.norm(perturb_norm, p='fro', dim=1)
                    if len(perturb_norm) > self.k:
                        bottom_k = torch.topk(perturb_norm, k=self.k, largest=False)
                    else:
                        bottom_k = torch.topk(perturb_norm, k=0, largest=False)
                    bottom_k = bottom_k.indices.detach().cpu().numpy()

                    mask = torch.ones(true_feat.size(0), dtype=torch.bool)
                    mask[bottom_k] = False

                    bottom_k_feat = fake_feat[~mask]
                    fake_feat = fake_feat[mask]

                    total_fake_feat.append(fake_feat)
                    total_robust_feat.append(bottom_k_feat)
                    #####----------------------------------#####      
                
                total_fake_feat = torch.cat(total_fake_feat, 0)
                total_robust_feat = torch.cat(total_robust_feat, 0)

                true_feat = torch.cat((true_feat, total_robust_feat), 0)
                fake_feat = total_fake_feat

                #Cross-entropy loss of normal samples and anomalies
                logits1 = self.discriminator(true_feat) 
                logits1 = torch.squeeze(logits1, dim = 1)
                ce_loss1 = F.binary_cross_entropy_with_logits(logits1, torch.ones(true_feat.size(0)).to(self.device))
                logits2 = self.discriminator(fake_feat)
                logits2 = torch.squeeze(logits2, dim = 1)
                ce_loss2 = F.binary_cross_entropy_with_logits(logits2, torch.zeros(fake_feat.size(0)).to(self.device))
                loss = ce_loss1 + ce_loss2 + self.lamda * fro_distortion_norm.abs() + cosine_sim_loss * self.cos_weight

                total_loss += loss

                loss.backward()
                self.optimizer.step()
                self.clf_optimizer.step()
                
            #Average Cross-entropy loss 
            total_loss = total_loss/(batch_idx + 1)

            test_score = self.test(val_loader)
            if test_score > best_score:
                best_score = test_score
                best_model = copy.deepcopy(self.model)
                best_clf_model = copy.deepcopy(self.discriminator)
            print('Epoch: {}, Loss: {}, {}: {}'.format(epoch, total_loss.item(), self.metric, test_score))

        self.model = copy.deepcopy(best_model)
        self.discriminator = copy.deepcopy(best_clf_model)
        print('\nBest test {}: {}'.format(
            self.metric, best_score    
        ))
        return best_score

    def test(self, test_loader):  
        self.model.eval()
        label_score = []
        batch_idx = -1
        for data, target, _ in test_loader:
            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)

            target = target.to(torch.float)
            target = torch.squeeze(target)

            feats, _ = self.model(data)
            logits = self.discriminator(feats)
            logits = torch.squeeze(logits, dim = 1)
            sigmoid_logits = torch.sigmoid(logits)
            scores = logits
            label_score += list(zip(target.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
        # Compute test score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores) 

        if self.metric == 'AUC':
            test_metric = roc_auc_score(labels, scores)            
            self.results[f'test_{self.metric}'] = test_metric
            self.results['final_scores'] = scores.tolist()     

        if self.metric == 'F1':
            # f1
            thresh = np.percentile(scores, self.ratio)
            y_pred = np.where(scores >= thresh, 1, 0)

            y_pred2 = np.array([0 if y==1 else 1 for y in y_pred])
            labels2 = np.array([0 if y==1 else 1 for y in labels])
           
            prec, recall, f1, _ = precision_recall_fscore_support(
                labels2, y_pred2, average="binary")
            
            test_metric = f1 
            self.results[f'test_{self.metric}'] = test_metric
            self.results['final_scores'] = scores.tolist()  
        
        return test_metric
        
    def save(self, path):
        torch.save(self.model.state_dict(),os.path.join(path, 'model.pt'))
        torch.save(self.view_model.state_dict(),os.path.join(path, 'view_model.pt'))
        torch.save(self.discriminator.state_dict(),os.path.join(path, 'discriminator_model.pt'))
    
    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        self.view_model.load_state_dict(torch.load(os.path.join(path, 'view_model.pt')))
        self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator_model.pt')))
        
    def save_results(self, export_json):
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def cosine_similarity(self, x, x_):
        x = torch.flatten(x, start_dim=1)
        x_ = torch.flatten(x_, start_dim=1)

        x = F.normalize(x, p=2, dim=-1)
        x_ = F.normalize(x_, p=2, dim=-1)
        dot = torch.matmul(x, x_.T)

        cos_sim = torch.diagonal(dot, 0)

        return cos_sim.mean()