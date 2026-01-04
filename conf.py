"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 128
max_len = 256       
d_model = 512       
n_layers = 6      
n_heads = 8        
ffn_hidden = 2048   
drop_prob = 0.1     
label_smoothing = 0.1




init_lr = 1e-4      # 保留作为参考，但实际上 train.py 中使用 lr=1.0
factor = 0.9        
adam_eps = 5e-9     
patience = 10      
warmup = 100        
epoch = 500        
clip = 1.0          
weight_decay = 1e-4  
inf = float('inf')  
