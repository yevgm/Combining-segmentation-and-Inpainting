import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data

from src.video_seg.utils import bcolors


class DataHandler(data.Dataset):
    """
    This is a torch class to handle:
    1. Load video
    2. cropping
    3. noising
    3. batching
    """

    def __init__(self, config):
        self.config = config
        self.crop_size = config['Model']['crop_size']
        self.num_frames = config['Model']['num_frames']
        self.work_with_crop = config['Model']['work_with_crop']
        self.video_mean = 0
        self.video_var = 1

        # check if data dir exists
        try:
            assert os.path.isdir(config['data_path']['path'])
            self.data_path = config['data_path']['path']
        except AssertionError as error:
            print(error)
            print(bcolors.FAIL + 'ERROR:\nChosen Data folder does not exist. Please go to config file and update.')
            print(bcolors.FAIL + 'Or enter a command line path python main.py -data /path/to/data/folder')
            exit(1)

        self.data = self.load_video()

    def __len__(self):
        """
        function required to override the built in torch methods
        :return: the length of the dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        return a training example
        :param item: not used, required in the signature by torch
        :return: a training example
        """
        # here we get a crop that is concatenated in the channels dimension
        # the crop will look like [N, crop_w, crop_h, 3]
        vid_crop, noisy_crop = self.get_random_crop()
        return vid_crop, noisy_crop

    def load_video(self):
        """
        Load video
        :return: video tensor [N,W,H,3]
        """
        # if data_path is a dir - load images to tensor
        file_list = sorted([f for f in os.listdir(self.data_path)])
        im_0 = plt.imread(os.path.join(self.data_path, file_list[0]))
        # iterate over the path list and load to tensor
        im_tensor = np.zeros([im_0.shape[0], im_0.shape[1], len(file_list) * 3])
        for ind, im_path in enumerate(file_list):
            im_tensor[:, :, ind * 3:ind * 3 + 3] = plt.imread(os.path.join(self.data_path, im_path))

        # self.video_mean = np.mean(im_tensor)
        # self.video_var = np.var(im_tensor)
        return np.asarray((im_tensor-self.video_mean)/self.video_var)

    def get_random_crop(self):
        """
        Randomly select a crop
        :return: tensor of preset size
        """
        frame = np.random.randint(0, self.data.shape[-1] // 3 - self.num_frames, 1)[0]
        w = np.random.randint(0, self.data.shape[0] - self.crop_size, 1)[0]
        h = np.random.randint(0, self.data.shape[1] - self.crop_size, 1)[0]

        crop = self.data[w:w + self.crop_size, h:h + self.crop_size, 3 * frame:3 * (frame + self.num_frames)]
        noisy_crop = crop + np.random.normal(0, 1, crop.shape) / 20
        return np.transpose(crop, [2, 0, 1]).astype(np.float32), np.transpose(noisy_crop, [2, 0, 1]).astype(np.float32)
