# Codebase for "DRAG" 

Paper title: Enhancing Anomaly Detection via Generating Diversified and Realistic Artificial Anomalies

Authors: Hyuntae Kim, Changhee Lee 

Paper Link: TBD 

## Reproduce results --option {default}

### FMNIST 
```
python main.py --data_name {FMNIST} --normal_class {0} --batch_size {512} --epochs {150} --lr {0.005} --clr_lr {1e-5} --optim {0} --mom {0.0} --model_dir {./log_fmnist} --eval {1} --data_path {./data/} --metric {AUC} --k {50} --num_perturb {5} --cos_weight {1e-2} --nl_head {0} --lamda {1e-4} --num_channels {1} --enc_hidden_dim {16} --flat_dim {576} --mlp_hidden_dim {128} --mlp_num_layers {2} --mlp_activation {relu}
```

### CIFAR-10 
```
python main.py --data_name {CIFAR10} --normal_class {0} --batch_size {512} --epochs {150} --lr {0.005} --clr_lr {1e-5} --optim {0} --mom {0.0} --model_dir {./log_cifar} --eval {1} --data_path {./data/} --metric {AUC} --k {30} --num_perturb {3} --cos_weight {1e-3} --nl_head {0} --lamda {1e-4} --num_channels {3} --enc_hidden_dim {32} --flat_dim {2048} --mlp_hidden_dim {128} --mlp_num_layers {3} --mlp_activation {leaky_relu}
```

### Tabular (thyroid) 
```
python main.py --data_name {thyroid} --batch_size {256} --epochs {100} --lr {0.005} --clr_lr {1e-5} --optim {0} --mom {0.0} --model_dir {./log_tabular} --eval {1} --data_path {./data/} --metric {F1} --k {100} --num_perturb {5} --cos_weight {1e-1} --nl_head {0} --nl_mlp {0} --lamda {1} --enc_hidden_dim {64} --mlp_hidden_dim {32} --mlp_num_layers {2} --mlp_activation {relu} 
```

### Base arguments 
`normal_class`: int, Image classification dataset (e.g., FMNIST, CIFAR-10) normal class index 

`batch_size`: int, batch size for training 

`epochs`: int, Number of epochs to train 

`lr`: float, Learning rate of encoder and perturbator networks' optimizer 

`clf_lr`: float, Learning rate of discriminator network's optimizer 

`optim`: int, Option of optimizer Adam or SGD (0/1)

`mom`: float, Learning rate momentum 

`model_dir`: str, Path where to save checkpoint 

`eval`: int, Option whether to load a saved model and evaluate (0/1)

`data_path`: str, Path where the data is stored 

`metric`: str, Evaluation metric option (e.g., AUC, F1)

`k`: int, Number of perturbations used for augmentation from a single perturbator output 

`num_perturb`: int, Number of perturbators 

`cos_weight`: float, Weight of cosine similarity loss 

`nl_head`: int, Option whether to use the non-linear head in perturbator 

`lamda`: float, Weight of perturbation norm loss 

`num_channels`: int, Number of image channels (image only)

`enc_hidden_dim`: int, Encoder hidden dimension

`flat_dim`: int, Length of the flattened image feature map (image only)

`mlp_hidden_dim`: int, MLP (i.e., discriminator) hidden dimension 

`mlp_num_layers`: int, Number of MLP (i.e., discriminator) hidden layers 

`mlp_activation`: int, Activation function of MLP (i.e., discriminator)


## File Directory 
```bash 
├─── data 
│    └── raw dataset 
│
├─── data_preprocess
│    ├── process_cifar.py 
│    ├── process_fmnist.py
│    └── process_tabular.py
│
├─── perturbator 
│    ├── config 
│    │   └── CIFAR10.json
│    │   └── FMNIST.json
│    │   └── Tabular.json
│    ├── perturbator_image.py 
│    └── perturbator_tabular.py
│     
├─── trainer 
│    └── DRAG_trainer.py
│
├─── main_image.py
├─── main_tabular.py
```