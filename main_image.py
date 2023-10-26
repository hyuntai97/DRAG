'''
Diversified and Realistic Anomaly Generation based Anomaly Detection (DRAG) Code base

Reference: Hyuntae Kim, Changhee Lee, 
"Enhancing Anomaly Detection via Generating Diversified and Realistic Artificial Anomalies,"
AAAI-24 under review, 2023. 

Code author: Hyuntae Kim (soodaman97@cau.ac.kr)

----------------------------------------------------------

main_image.py

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
from data_preprocess.process_fmnist import FMNIST_Dataset
from data_preprocess.process_cifar import CIFAR10_Dataset
from trainer.DRAG_trainer import Trainer
from perturbator.perturbator_image import MultiPerturbator
from models.encoder import LeNet 
from models.mlp import ImageMLP
import itertools
import json 


def adjust_learning_rate(epoch, learning_rate, optimizer):
    if epoch <= 150:
        lr = learning_rate * 0.001 
    if epoch <= 100:
        lr = learning_rate * 0.01
    if epoch <= 20:
        lr = learning_rate  

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

    return optimizer 


def main():

    # Config setting 
    setting = f'{args.data_name}_normal{args.normal_class}_ep{args.epochs}_lam{args.lamda}_optim{args.optim}_lr{args.lr}_clflr{args.clf_lr}_K{args.k}_numP{args.num_perturb}_cosweight{args.cos_weight}_nlhead{args.nl_head}'
    path = os.path.join(args.model_dir, f'{args.data_name}_trained_model_{args.normal_class}')
    if not os.path.exists(path):
        os.makedirs(path)
        
    print('>>>>>>>config setting : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    
    # Load data 
    data_dict = {
        'FMNIST': FMNIST_Dataset, 
        'CIFAR10': CIFAR10_Dataset
                }   
    
    Data = data_dict[args.data_name]
    dataset = Data(args.data_path, args.normal_class)

    train_loader, test_loader = dataset.loaders(batch_size=args.batch_size)
    print(f"{args.data_name} class: ", args.normal_class)

    # Load model 
    model = LeNet(args.num_channels, args.enc_hidden_dim).to(device)
    
    with open(f'./perturbator/config/{args.data_name}.json') as f:
        viewmaker_config = json.load(f)

    multi_view_model = MultiPerturbator(config=viewmaker_config, 
                                num_channels=args.num_channels,
                                num_perturb=args.num_perturb,
                                flat_dim=args.flat_dim)
    view_model = multi_view_model._make_nets().to(device)
    
    discriminator_model = ImageMLP(mlp_hidden_dim=args.mlp_hidden_dim, 
                                   output_dim=1, 
                                   num_layers=args.mlp_num_layers, 
                                   flat_dim=args.flat_dim,
                                   activation=args.mlp_activation).to(device)

    if args.optim == 1:
        optimizer = optim.SGD(itertools.chain(model.parameters(),view_model.parameters()),lr=args.lr, momentum=args.mom)
        clf_optimizer = optim.SGD(discriminator_model.parameters(), lr=args.clf_lr, momentum=args.mom)
        print("Optimizer: SGD")
    else:
        optimizer = optim.Adam(itertools.chain(model.parameters(),view_model.parameters()), lr=args.lr, amsgrad = True)
        if args.data_name == 'FMNIST':
            clf_optimizer = optim.Adam(discriminator_model.parameters(), lr=args.clf_lr, amsgrad=True)
        elif args.data_name == 'CIFAR10':
            clf_optimizer = optim.SGD(discriminator_model.parameters(), lr=args.clf_lr, momentum=args.mom)
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
            print("Testing the trained model on Fashion-MNIST class {}".format(args.normal_class))      
            print("Saved Model Loaded")
        else:
            print('Saved model not found. Cannot run evaluation.')
            exit()
    score = trainer.test(test_loader)
    trainer.save_results(path + '/results.json')
    print(f'TEST AUC: {score}')

if __name__ == '__main__':
    
    torch.set_printoptions(precision=5)
    
    parser = argparse.ArgumentParser(description='DRAG Training')
    parser.add_argument('--normal_class', type=int, default=0, metavar='N',
                    help='Image normal class index')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train')                   
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate')   
    parser.add_argument('--clf_lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate of classifier') 
    parser.add_argument('--optim', type=int, default=0, metavar='N',
                        help='0 : | Adam 1: SGD')
    parser.add_argument('--mom', type=float, default=0.0, metavar='M',
                        help='momentum')
    parser.add_argument('--model_dir', default='./log_fmnist',
                        help='path where to save checkpoint')
    parser.add_argument('--eval', type=int, default=0, metavar='N',
                        help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('-d', '--data_path', type=str, default='./data/')
    parser.add_argument('--metric', type=str, default='AUC')
    parser.add_argument('--data_name', type=str, default='FMNIST')

    # Model details 
    parser.add_argument('--num_channels', type=int, default=1, metavar='N',
                        help='1 : FMNIST | 3: CIFAR10')
    parser.add_argument('--enc_hidden_dim', type=int, default=16,
                        help='16: FMNIST | 32: CIFAR10')
    parser.add_argument('--flat_dim', type=int, default=576,  
                        help='64*3*3: FMNIST | 128*4*4: CIFAR10')
    parser.add_argument('--mlp_hidden_dim', type=int, default=128)
    parser.add_argument('--mlp_num_layers', type=int, default=2,
                        help='2: FMNIST | 3: CIFAR10') 
    parser.add_argument('--mlp_activation', type=str, default='relu',
                        help='relu: FMNIST | leaky_relu: CIFAR10') 

    # DRAG
    parser.add_argument('--k', type=int, default=50)    
    parser.add_argument('--num_perturb', type=int, default=5)
    parser.add_argument('--cos_weight', type=float, default=1e-2)
    parser.add_argument('--nl_head', type=int, default=0) # 0: NOT USING non-linear head / 1: USING non-linear head 
    parser.add_argument('--lamda', type=float, default=1e-4, metavar='N',
                        help='Weight of the perturbator loss')

    # Random Seed 
    parser.add_argument('--fix_seed', type=int, default=2023)

    args = parser.parse_args()

    torch.manual_seed(args.fix_seed)
    torch.cuda.manual_seed(args.fix_seed)
    torch.cuda.manual_seed_all(args.fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.fix_seed)

    # Model save path
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    main()
