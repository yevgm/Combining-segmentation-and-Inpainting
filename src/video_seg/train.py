"""
1. Load video - load N frames
2. Create network
3. Define loss
4. Train loop
    a. Select a patch
    b. Add noise to the video on each frame
    c. feed forward
    d. get the result and take loss L(x_hat,x) - where x is original video patch and x_hat is the prediction we want this to reconstruct the original video as most of it should be temporally consistent
    e. backprop the grad (Adam)
5. Save model
"""

import matplotlib.pyplot as plt
import torch
from torch.utils import data
import model
import yaml
import os
import dataloader

# ========================================================================= #
"""
Initialize base parameters
"""

with open(os.path.join(os.getcwd(),'config.yml'), "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# get available device - if GPU will auto detect and use, otherwise will use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your current Device is: ', torch.cuda.get_device_name(0))

# define the dataset parameters for the torch loader
params = {'batch_size': config['Model']['batch_size'],
          'shuffle': True,
          'num_workers': 0}
# ========================================================================= #

# build the network object
net = model.VideoSeg(config, device)

# load the data
dataset = dataloader.DataHandler(config)

# instantiate a Pytorch dataloader object
data_generator = data.DataLoader(dataset, **params)

# call the training scheme
net.train(data_generator)

net.eval(dataset.data)

# empty GPU ram
torch.cuda.empty_cache()
# clean up
torch.cuda.empty_cache()
