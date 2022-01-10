import os
import numpy as np
import torch
import argparse
from PIL import Image, ImageFilter
from omegaconf import OmegaConf
from lama.bin import predict
from src.mask_generator.mask_generator import segment
from config import parse_args_lama

# import segmentation network
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True, force_reload=True)


def main(args):
    # define the dir with test images
    test_image_path = args.test_image_path

    # load a list of image names
    image_names = [f for f in os.listdir(test_image_path) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

    # iterate over all the images:
    # 1. Segment
    # 2. Save image and mask in input folder
    # 3. Run inpainting (LAMA)
    if not args.skip_seg:
        for image_name in image_names:

            # segment
            filename = os.path.join(test_image_path, f'{image_name}')
            im, mask = segment(filename, model)

            # save
            for i in [15, 3]:
                # segment returns 21 masks, currently saving only [0] we can/need to save all 21
                cur_mask = np.asarray(np.asarray(mask) == i, dtype=np.uint8) * 255

                # save mask and image to inputs only if the mask isn't empty
                if cur_mask.sum() != 0:
                    input_dir = args.input_dir
                    im.save(input_dir + f'/{image_name.split(".")[0]}_{i:03d}.png')
                    cur_mask = Image.fromarray(cur_mask).filter(ImageFilter.MaxFilter(31))
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

# Segmentation labels:
# 0: background
# 1: aeroplane
# 2: bicycle
# 3: bird
# 4: boat
# 5: bottle
# 6: bus
# 7: car
# 8: cat
# 9: chair
# 10: cow
# 11: dining table
# 12: dog
# 13: horse
# 14: motorbike
# 15: person
# 16: potted plant
# 17: sheep
# 18: sofa
# 19: train
# 20: tv monitor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_args_lama(parser)
    args = parser.parse_args()

    main(args)
