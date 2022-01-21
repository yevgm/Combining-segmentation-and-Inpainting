import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn.functional as F

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class PerceptualLoss(nn.Module):
    def __init__(self, normalize_inputs=False):
        super(PerceptualLoss, self).__init__()

        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD

        vgg = torchvision.models.vgg19(pretrained=True).features
        vgg_avg_pooling = []

        for weights in vgg.parameters():
            weights.requires_grad = False

        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)

        self.vgg = nn.Sequential(*vgg_avg_pooling)

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target, mask=None):
        # we expect input and target to be in [0, 1] range
        losses = []

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target

        for layer in self.vgg[:30]:

            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                loss = F.mse_loss(features_input, features_target, reduction='none')

                if mask is not None:
                    cur_mask = F.interpolate(mask, size=features_input.shape[-2:],
                                             mode='bilinear', align_corners=False)
                    loss = loss * (1 - cur_mask)

                loss = loss.mean(dim=tuple(range(1, len(loss.shape))))
                losses.append(loss)

        return losses

    def forward(self, input, target, mask=None):
        losses = self.partial_losses(input, target, mask=mask)
        return torch.stack(losses).sum(dim=0)

    def get_global_features(self, input):

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
        else:
            features_input = input

        features_input = self.vgg(features_input)
        return features_input


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
        self.time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.save_path = os.path.join(self.config['working_dir'], f'Res_{self.time_stamp}')
        self.logs_dir = os.path.join(config['logs_dir'], f'logs_dir_{self.time_stamp}')
        self.writer = SummaryWriter(self.logs_dir)
        self.scheduler = self.define_lr_sched()
        self.last_saved_model = []
        # self.vgg = VGGPerceptualLoss()
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
            # nn.BatchNorm2d(self.channels_in),
            nn.Conv2d(in_channels=self.channels_in, out_channels=64, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=[1, 1], padding_mode='replicate'),
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
        return torch.nn.L1Loss(reduction='mean')

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
        :param input_tensor: input tensor to feed to the network
        :return:
        """
        return self.net(input_tensor)

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
                self.eval(data_loader_object.dataset.data, e)

            # iterations per epochs
            it = 0
            for (crop, noisy_crop) in data_loader_object:
                x_prediction = self.forward(noisy_crop.to(self.device))
                loss1 = torch.nn.L1Loss(reduction='mean')
                # loss2 = PerceptualLoss()
                loss = loss1(x_prediction.to(self.device), crop.to(self.device)) #+ loss2(x_prediction.to(self.device), crop.to(self.device))
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

    def eval(self, data, num_epoch=None):
        num_frames = self.config['Model']['num_frames']
        vid_tensor = np.transpose(data, [2, 0, 1]).astype(np.float32)
        if num_epoch:
            save_path = os.path.join(self.save_path, f'epoch_{num_epoch}')
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = os.path.join(self.save_path, f'EVAL')
            os.makedirs(save_path, exist_ok=True)

        for i in range(0, vid_tensor.shape[0], num_frames * 3):
            # print(f'saving frames {i//3} to {(i + num_frames * 3)//3}, channels {i} to {(i + num_frames * 3)}')
            result = self.forward(torch.from_numpy(vid_tensor[np.newaxis, i:i + num_frames * 3, :, :]).to(self.device))
            result = np.transpose(np.squeeze(result.detach().cpu().numpy()), [1, 2, 0])
            for im in range(0, result.shape[-1], 3):
                plt.imsave(os.path.join(save_path, f'{i//3+im//3:05d}.png'), np.clip(result[:, :, im:im + 3], 0, 1))

        return

    def save_model(self, epoch=None, overwrite=False):
        """
        Saves the model (state-dict, optimizer and lr_sched
        :return:
        """
        if epoch:
            save_path = os.path.join(self.save_path, f'epoch_{epoch}')
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = os.path.join(self.save_path, f'EVAL')
            os.makedirs(save_path, exist_ok=True)

        if overwrite:
            checkpoint_list = [i for i in os.listdir(os.path.join(save_path,f'checkpoints_{self.time_stamp}')) if i.endswith('.pth.tar')]
            if len(checkpoint_list) != 0:
                os.remove(os.path.join(self.config['working_dir'], checkpoint_list[-1]))

        filename = f'checkpoint{epoch}.pth.tar'
        os.makedirs(save_path, exist_ok=True)
        torch.save({'sd': self.net.state_dict(),
                    'opt': self.optimizer.state_dict()},
                   # 'lr_sched': self.scheduler.state_dict()},
                   os.path.join(save_path, filename))
        self.last_saved_model = os.path.join(save_path, filename)


    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['sd'])
        self.optimizer.load_state_dict(checkpoint['opt'])
        # self.scheduler.load_state_dict(checkpoint['lr_sched'])
