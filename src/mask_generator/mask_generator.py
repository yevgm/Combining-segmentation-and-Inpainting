import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import pickle
from skimage import img_as_ubyte
import PIL.Image as Image

from config import parse_args_mask_generator
try:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog
except:
    print("Detectron v2 is not installed")


class SemanticSegmentationMask:
    def __init__(self, args):
        self.cfg = self.setup_cfg(args)
        self.predictor = DefaultPredictor(self.cfg)
        self.output_dir = args.output_dir
        self.in_files = list(glob.glob(os.path.join(args.input_dir, '**', '*.png'), recursive=True))

    def forward(self):
        for img, fp in self.load_img():
            panoptic_seg, segments_info, img = self.get_segmentation(img)
            image_filename = os.path.split(fp)[-1]
            self.save_mask((panoptic_seg, image_filename))
            break
        return img

    def get_segmentation(self, img):
        im = img_as_ubyte(img)
        im = np.array(im).transpose(1, 2, 0)
        panoptic_seg, segment_info = self.predictor(im)["panoptic_seg"]
        # out = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segment_info)
        # cv2.imshow('ImageWindow', out.get_image()[:, :, ::-1])
        # cv2.waitKey()

        a=1
        return panoptic_seg, segment_info, out.get_image()[:, :, ::-1]

    def load_img(self, mode='RGB'):
        for im_path in self.in_files:
            try:
                img = np.array(Image.open(im_path).convert(mode))
            except OSError:
                print("Can't load file {}".format(im_path))
                sys.exit(1)

            if img.ndim == 3:
                img = np.transpose(img, (2, 0, 1))
            out_img = img.astype('float32') / 255

            yield out_img, im_path

    def save_mask(self, mask):
        '''
        :param mask: (mask_img, mask_filename) tuple
        '''
        fp = os.path.join(self.output_dir, mask[1])
        try:
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
        except OSError:
            print("Can't create dir")

        try:
            cv2.imwrite(fp, mask[0].numpy())
        except OSError:
            print('Saving file {} was unsuccessful'.format(fp))

    @staticmethod
    def setup_cfg(args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
        # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
        # add_panoptic_deeplab_config(cfg)
        # cfg.merge_from_file(args.config_file)
        # cfg.merge_from_list(args.opts)
        confidence_threshold = 0.5  # TODO: add this to config

        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 10
        cfg.MODEL.DEVICE = args.device
        cfg.freeze()
        return cfg


def main(args: argparse.Namespace):
    mp.set_start_method("spawn", force=True)
    # args_dict = vars(args)
    # cfg = setup_cfg(args)

    # Tests TODO: remove this
    args.input_dir = os.path.abspath('../../../LaMa_test_images_seg')

    seg_cls = SemanticSegmentationMask(args)
    seg_cls.forward()
    a=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_args_mask_generator(parser)
    args = parser.parse_args()

    main(args)