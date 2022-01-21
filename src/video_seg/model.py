import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


class VideoSeg:
    """
    Network class.
    An object to wrap the network with all the methods.
    build, train, predict, save & load models, track performance
    """

    def __init__(self, config, device):
        self.device = device
        self.config = config
        self.channels_in = config['Model']['num_frames'] * 3  # 3 RGB channels

        self.net = self.build_network()
        self.optimizer = self.define_opt()
        self.loss_fn = self.define_loss()
        self.writer = SummaryWriter(os.path.join(config['logs_dir'], 'logs_dir'))
        self.scheduler = self.define_lr_sched()
        self.last_saved_model = []
        print('-net built-')

    def build_network(self):
        """
        This is a fully convolutional network to improve the temporal consistency of the segmentation
        :return: pytorch net object
        """
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        net = nn.Sequential(
            nn.Conv2d(in_channels=self.channels_in, out_channels=32, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.channels_in, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate')
        ).to(self.device)
        net.apply(init_weights)
        return net

    @staticmethod
    def define_loss():
        """
        define the loss function for the network
        :return: loss function handle
        """
        return torch.nn.L1Loss(reduction='sum')

    def define_opt(self):
        """
        define the network optimizer
        :return: optimizer object
        """
        # TODO: read the lr and momentum from config
        learning_rate = float(self.config['Model']['optimizer']['lr'])
        return torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=0.01)

    def define_lr_sched(self):
        """
        take relevant parameters for learning rate scheduler
        :return: lr_scheduler object
        """
        # exponential decay factor
        gamma = self.config['Model']['optimizer']['lr_schedule']['params']['gamma']
        # set decay steps
        milestones = self.config['Model']['optimizer']['lr_schedule']['params']['milestones']
        # when to decay relative to total number of epochs
        step_size = self.config['Model']['optimizer']['lr_schedule']['params']['step_size']
        # total number of epochs
        epochs = self.config['Model']['epochs']

        # check what kind of LR_sched was defined
        if self.config['Model']['optimizer']['lr_schedule']['type'] == 'MultiStepLR':
            return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        elif self.config['Model']['optimizer']['lr_schedule']['type'] == 'StepLR':
            return lr_scheduler.StepLR(self.optimizer, step_size=int(epochs * step_size), gamma=gamma)
        else:
            print('****************** NO LR_SCHED DEFINED SETTING DEFAULT *****************************')
            return lr_scheduler.StepLR(self.optimizer, step_size=epochs // 10, gamma=1 / 1.5)

    def forward(self, input_tensor):
        """
        Apply a forward pass, used for evaluation
        :param input_tensor: MR input tensor to feed to the network
        :return:
        """
        return self.net(input_tensor)

    def calc_loss(self, output, hr_gt_torch):
        """
        Calculate the loss (use cuda if available)
        :param output: network prediction
        :param hr_gt_torch: GT
        :return: loss value
        """
        return self.loss_fn(output, hr_gt_torch).cuda()

    def train(self, data_loader_object):
        """
        Full training scheme. This func
        :param data_loader_object:
        :return:
        """
        print('-starting training-')
        epochs = self.config['Model']['epochs']
        for e in range(epochs):
            t = time.time()
            self.optimizer.zero_grad()
            if e % self.config['Model']['save_every'] == self.config['Model']['save_every'] - 1:
                print(f'saved model at epoch {e}')
                self.save_model(epoch=e, overwrite=False)

            # iterations per epochs
            it = 0
            for (crop, noisy_crop) in data_loader_object:
                x_prediction = self.forward(noisy_crop.to(self.device))
                # loss = self.calc_loss(hr_prediction.to(self.device), hr_gt.to(self.device))
                loss1 = torch.nn.L1Loss(reduction='sum')
                # loss2 = torch.nn.MSELoss(reduction='sum')
                loss = loss1(x_prediction.to(self.device), crop.to(self.device))
                loss.backward()
                it += 1
            print(f'epoch:{e}, loss:{loss.item():.2f}, Time: {(time.time() - t):.2f}, lr={self.optimizer.param_groups[0]["lr"]}')
            # TODO: check about updating after ALL iterations in epoch
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalars('loss', {'loss': loss.item()})
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]["lr"]})

        self.writer.close()
        return

    def eval(self, data):
        num_frames = self.config['Model']['num_frames']
        np_tensor = np.transpose(data, [2, 0, 1]).astype(np.float32)
        os.makedirs(os.path.join(self.config['working_dir'], 'test'), exist_ok=True)
        for i in range(0, np_tensor.shape[0], num_frames * 3):
            result = self.forward(torch.from_numpy(np_tensor[np.newaxis, i:i + num_frames * 3, :, :]).to(self.device))
            result = np.transpose(np.squeeze(result.detach().cpu().numpy()), [1, 2, 0])
            for im in range(0, result.shape[-1], 3):
                image = np.clip(result[:, :, im:im + 3], 0, 1)
                plt.imsave(os.path.join(self.config['working_dir'], 'test', f'{i//3+im//3:05d}.png'), image)
        return

    def save_model(self, epoch=None, scale=None, overwrite=False):
        """
        Saves the model (state-dict, optimizer and lr_sched
        :return:
        """
        if overwrite:
            checkpoint_list = [i for i in os.listdir(os.path.join(self.config['working_dir'])) if i.endswith('.pth.tar')]
            if len(checkpoint_list) != 0:
                os.remove(os.path.join(self.config['working_dir'], checkpoint_list[-1]))

        filename = f'checkpoint{epoch}.pth.tar'
        os.makedirs(os.path.join(self.config['working_dir']), exist_ok=True)
        torch.save({'sd': self.net.state_dict(),
                    'opt': self.optimizer.state_dict()},
                   # 'lr_sched': self.scheduler.state_dict()},
                   os.path.join(self.config['working_dir'], filename))
        self.last_saved_model = os.path.join(self.config['working_dir'], filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['sd'])
        self.optimizer.load_state_dict(checkpoint['opt'])
        # self.scheduler.load_state_dict(checkpoint['lr_sched'])
