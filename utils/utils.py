import os
import numpy as np
import torch
import random

def mkdir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
def load_model(model, file, use_cuda=True):
    if use_cuda:
        model.load_state_dict(torch.load(file))
    else:
        model.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))
