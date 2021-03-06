import yaml
yaml.warnings({'YAMLLoadWarning': False})  # disable yaml warnings

import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from PIL import ImageFilter
import PIL.Image as Image

from omegaconf import OmegaConf
from lama.bin import predict
from src.mask_generator.mask_generator import segment
from config import parse_args_lama, print_cls


def main(args, model):
    # define the dir with test images
    test_image_path = args.image_input
    class_to_inpaint = args.remove_class
    # load a list of image names
    image_names = [f for f in sorted(os.listdir(test_image_path)) if f.endswith('.png')
                   or f.endswith('.jpg') or f.endswith('.jpeg')]

    # create inputs and output
    make_dirs(args)

    # iterate over all the images:
    # 1. Segment
    # 2. Save image and mask in input folder
    # 3. Run inpainting (LAMA)
    if not args.skip_seg:
        for image_name in tqdm(image_names, desc="Semantic segmentation in progress"):

            # segment
            filename = os.path.join(test_image_path, f'{image_name}')
            im, mask = segment(filename, model)

            # save
            for i in [class_to_inpaint]:
                # segment returns 21 masks, currently saving only [0] we can/need to save all 21
                # we can also combine the masks if user asks to remove a list of classes - like cars + people
                cur_mask = np.asarray(np.asarray(mask) == i, dtype=np.uint8) * 255

                # save mask and image to inputs only if the mask isn't empty
                if cur_mask.sum() != 0:
                    input_dir = args.input_dir
                    im.save(input_dir + f'/{image_name.split(".")[0]}_{i:03d}.png')
                    cur_mask = Image.fromarray(cur_mask).filter(ImageFilter.MaxFilter(31))
                    # cur_mask = cur_mask.filter(ImageFilter.GaussianBlur(7))
                    cur_mask.save(input_dir + f'/{image_name.split(".")[0]}_{i:03d}_mask.png')


    # run LAMA
    lama_main(args)


def lama_main(args):
    # load config as expected by the 'predict' function
    predict_config = {
        "indir": args.input_dir,
        "outdir": args.output_dir,
        "model": {
            "path": args.lama_model_path,
            "checkpoint": args.lama_model_name
        },
        "dataset": {
            "kind": args.dataset_kind,
            "img_suffix": args.dataset_img_suffix,
            "pad_out_to_modulo": args.dataset_pad_out_to_modulo,
        },
        "out_key": args.out_key,
        "device": args.device
    }
    predict_config_om = OmegaConf.create(predict_config)
    predict.main(predict_config_om)


def choose_seg_model():
    # import segmentation network
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True, force_reload=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True, force_reload=True)
    return model


def make_dirs(args):
    try:
        if not os.path.isdir(args.input_dir):
            os.mkdir(args.input_dir)

        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)

    except OSError:
        print("Can't create directory")
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_args_lama(parser)
    args = parser.parse_args()

    if args.action == 'inpaint':
        model = choose_seg_model()
        main(args, model)
    else:
        print_cls()
