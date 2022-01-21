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

import torch
from torch.utils import data
import src.video_seg.model as model
import yaml
import argparse
from config import parse_args_video
import src.video_seg.dataloader as dataloader


def init_config(args):
    """
    Initialize base parameters
    """
    with open(args.video_config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Add to config some additional args
    config['logs_dir'] = args.logs
    config['working_dir'] = args.results
    config['data_path'] = args.output_dir

    return config


def clean_video_train(args):

    # get available device - if GPU will auto detect and use, otherwise will use CPU
    device = torch.device(args.device)
    config = init_config(args)
    # define the dataset parameters for the torch loader
    params = {'batch_size': config['Model']['batch_size'],
              'shuffle': True,
              'num_workers': 0}

    # build the network object
    net = model.VideoSeg(config, device)

    # load the data
    dataset = dataloader.DataHandler(config)

    # instantiate a Pytorch dataloader object
    data_generator = data.DataLoader(dataset, **params)

    # call the training scheme
    net.train(data_generator)

    return net.last_saved_model


def clean_video_eval(args):
    # get available device - if GPU will auto detect and use, otherwise will use CPU
    device = torch.device(args.device)
    config = init_config(args)

    # build the network object
    net = model.VideoSeg(config, device)

    # load the data
    dataset = dataloader.DataHandler(config)

    net.load_model(filename=args.model_path)
    net.eval(dataset.data)

    # empty GPU ram
    torch.cuda.empty_cache()
    # clean up
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_args_video(parser)
    args = parser.parse_args()

    # Train the model on the frames (video) that has been processed by LaMa
    last_saved_model = clean_video_train(args)
    args.model_path = last_saved_model

    # Evaluate the model
    clean_video_eval(args)

