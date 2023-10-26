'''
Diversified and Realistic Anomaly Generation based Anomaly Detection (DRAG) Code base

Reference: Hyuntae Kim, Changhee Lee, 
"Enhancing Anomaly Detection via Generating Diversified and Realistic Artificial Anomalies,"
AAAI-24 under review, 2023. 

Code author: Hyuntae Kim (soodaman97@cau.ac.kr)

----------------------------------------------------------

main_tabular.py

(1) Load data
(2) Load model
(3) Train anomaly detection algorithm (DRAG)
(4) Evaluate the performance of AD algorithm, save models
    - Evaluation metric: AUC or F1-score 
    - Encoder
    - Perturbators 
    - Discriminator 
'''

from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from trainer.DRAG_trainer import Trainer
from perturbator.perturbator_tabular import MultiPerturbator
from models.encoder import MLP_Feature_Extractor
from models.mlp import TabularMLP
import itertools
import json 
from data_preprocess.process_tabular import load_data


def adjust_learning_rate(epoch, learning_rate, optimizer):
    if epoch <= 200:
        lr = learning_rate * 0.001 
    if epoch <= 50:
        lr = learning_rate * 0.01
    if epoch <= 20:
        lr = learning_rate  

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

    return optimizer 

def main():

    # Config setting 
    setting = f'{args.data_name}_ep{args.epochs}_lam{args.lamda}_optim{args.optim}_lr{args.lr}_clflr{args.clf_lr}_K{args.k}_numP{args.num_perturb}_cosweight{args.cos_weight}_nlhead{args.nl_head}'
    path = os.path.join(args.model_dir, f'{args.data_name}_trained_model')
    if not os.path.exists(path):
        os.makedirs(path)
        
    print('>>>>>>>config setting : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    
    # Load data 
    train_dataset, test_dataset, num_features, ratio = load_data(f"{args.data_path}{args.data_name}/")
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True)
    print(f"{args.data_name} number of features: {num_features}")

    # Load model 
    model = MLP_Feature_Extractor(input_dim=num_features, num_hidden_nodes=args.enc_hidden_dim).to(device)

    with open('./perturbator/config/Tabular.json') as f:
        viewmaker_config = json.load(f)

    multi_view_model = MultiPerturbator(config=viewmaker_config, 
                                num_channels=num_features,
                                num_perturb=args.num_perturb)
    view_model = multi_view_model._make_nets().to(device)    
    

    discriminator_model = TabularMLP(mlp_hidden_dim=args.mlp_hidden_dim, input_dim=args.enc_hidden_dim, 
                                     output_dim=1, num_layers=args.mlp_num_layers, 
                                     activation=args.mlp_activation, nl_mlp=args.nl_mlp).to(device)


    if args.optim == 1:
        optimizer = optim.SGD(itertools.chain(model.parameters(),view_model.parameters()),lr=args.lr, momentum=args.mom)
        clf_optimizer = optim.SGD(discriminator_model.parameters(), lr=args.clf_lr, momentum=args.mom)
        print("Optimizer: SGD")
    else:
        optimizer = optim.Adam(itertools.chain(model.parameters(),view_model.parameters()), lr=args.lr, amsgrad = True)
        clf_optimizer = optim.Adam(discriminator_model.parameters(), lr=args.clf_lr, amsgrad = True)
        print("Optimizer: Adam")

    # Train & Test 
    trainer = Trainer(
                    model,
                    view_model, 
                    discriminator_model, 
                    optimizer, 
                    clf_optimizer, 
                    args.lamda, 
                    device, 
                    args.k,
                    cos_weight=args.cos_weight,
                    ratio=ratio,
                    nl_head=args.nl_head, 
                    metric=args.metric
                    )
    if args.eval == 0:
        # Training the model 
        score = trainer.train(train_loader, test_loader, args.clf_lr, 
                              adjust_learning_rate, args.epochs)
        trainer.save(path)

    else:
        if os.path.exists(f'{path}/model.pt'):
            trainer.load(path)
            print("Testing the trained model on Tabular dataset {}".format(args.data_name))      
            print("Saved Model Loaded")
        else:
            print('Saved model not found. Cannot run evaluation.')
            exit()
    score = trainer.test(test_loader)
    trainer.save_results(path + '/results.json')
    print(f'TEST F1: {score}')

if __name__ == '__main__':
    
    torch.set_printoptions(precision=5)
    
    parser = argparse.ArgumentParser(description='DRAG Training')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')                   
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate')   
    parser.add_argument('--clf_lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate of classifier') 
    parser.add_argument('--optim', type=int, default=0, metavar='N',
                        help='0 : Adam 1: SGD')
    parser.add_argument('--mom', type=float, default=0.0, metavar='M',
                        help='momentum')
    parser.add_argument('--model_dir', default='./log_tabular',
                        help='path where to save checkpoint')
    parser.add_argument('--eval', type=int, default=0, metavar='N',
                        help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('-d', '--data_path', type=str, default='./data/')
    parser.add_argument('--metric', type=str, default='F1')
    parser.add_argument('--data_name', type=str, default='thyroid')

    # Model details 
    parser.add_argument('--enc_hidden_dim', type=int, default=64)
    parser.add_argument('--mlp_hidden_dim', type=int, default=32)
    parser.add_argument('--mlp_num_layers', type=int, default=2) 
    parser.add_argument('--mlp_activation', type=str, default='relu') 
    parser.add_argument('--nl_mlp', type=int, default=0) # 0: NOT USING non-linear head / 1: USING non-linear head 

    # DRAG
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--num_perturb', type=int, default=3)
    parser.add_argument('--cos_weight', type=float, default=0.1)
    parser.add_argument('--nl_head', type=int, default=0) # 0: NOT USING non-linear head / 1: USING non-linear head 
    parser.add_argument('--lamda', type=float, default=1, metavar='N',
                        help='Weight of the perturbator loss')
    
    # random seed 
    parser.add_argument('--fix_seed', type=int, default=2023)
    
    args = parser. parse_args()

    torch.manual_seed(args.fix_seed)
    torch.cuda.manual_seed(args.fix_seed)
    torch.cuda.manual_seed_all(args.fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.fix_seed)

    #Model save path
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    main()
