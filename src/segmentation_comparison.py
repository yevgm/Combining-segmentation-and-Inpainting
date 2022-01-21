############################################################
# The contents below have been combined using files in the #
# following repository:                                    #
# https://github.com/richzhang/PerceptualSimilarity        #
############################################################

import argparse
import os
import torch
from torchvision import transforms
import PIL.Image as Image
import numpy as np
from src import lpips

classes_flipped = {
    '00': 'background',
    '01': 'aeroplane',
    '02': 'bicycle',
    '03': 'bird',
    '04': 'boat',
    '05': 'bottle',
    '06': 'bus',
    '07': 'car',
    '08': 'cat',
    '09': 'chair',
    '10': 'cow',
    '11': 'dining table',
    '12': 'dog',
    '13': 'horse',
    '14': 'motorbike',
    '15': 'person',
    '16': 'potted plant',
    '17': 'sheep',
    '18': 'sofa',
    '19': 'train',
    '20': 'tv monitor'}
classes = dict((v, k) for k, v in classes_flipped.items())


def parse_func(parent, add_help=False):
    """
        Parse commandline arguments.
        """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)

    repo_root = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.abspath(os.path.join(repo_root, '..'))
    test_folder = os.path.join(repo_root, '..', 'test_data_comparison')
    test_folder = os.path.abspath(test_folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument('--repo-root', default=repo_root, type=str,
                        help='repository main dir')
    parser.add_argument('-t', '--test-data', default=test_folder, type=str,
                        help='provide full dir for the test inpaintings')
    parser.add_argument('--device', default=device, type=str,
                        help='device to use')

    return parser


def compare_results(args):
    dirs = [f for f in os.listdir(args.test_data) if f.endswith('manual')]
    data_dict = {}

    for dir in dirs:
        splitted_dir = dir.split('_')
        manual_res_dir = os.path.join(dir, 'output')
        auto_res_dir = splitted_dir[0] + '_auto/output'
        orig_res_dir = splitted_dir[0] + '_manual/input_no_mask'

        auto_res_dir = os.path.join(args.test_data, auto_res_dir)
        orig_res_dir = os.path.join(args.test_data, orig_res_dir)
        manual_res_dir = os.path.join(args.test_data, manual_res_dir)

        diff_manual = calculate_diff(manual_res_dir, orig_res_dir, 'manual', splitted_dir[0])
        diff_auto = calculate_diff(auto_res_dir, orig_res_dir, 'auto', splitted_dir[0])

        data_dict[splitted_dir[0] + '_auto'] = diff_auto
        data_dict[splitted_dir[0] + '_manual'] = diff_manual

    print_results(data_dict)


def calculate_diff(inpainted_dir, orig_dir, kind, test_data_name):
    lpsis_model_ins = lpips.LPIPS(net='alex')

    orig_images = [f for f in os.listdir(orig_dir) if f.endswith('.png')]
    inpainted_images = [f for f in os.listdir(inpainted_dir) if f.endswith('.png')]
    assert len(orig_images) == len(inpainted_images)

    diff = 0
    with torch.no_grad():
        for orig_image in orig_images:
            orig_image_name = orig_image.split('/')[-1].split('.')[0]
            if kind == 'auto':
                inpainted_image_name = orig_image_name + '_0' + str(classes[test_data_name]) + '_mask.png'
            else:
                inpainted_image_name = orig_image_name + '_mask.png'

            inpainted_image = os.path.join(inpainted_dir, inpainted_image_name)
            orig_image = os.path.join(orig_dir, orig_image)
            # load both images from disk
            im0, im1 = load_images(orig_image, inpainted_image)
            # calculate loss
            d = lpsis_model_ins.forward(im0, im1).squeeze()
            diff += d.numpy()

    return diff / len(orig_images)


def load_images(orig_image, inpainted_image):

    input_image_orig = Image.open(orig_image)
    input_image_inpainted = Image.open(inpainted_image)

    input_image_orig = input_image_orig.convert("RGB")
    input_image_inpainted = input_image_inpainted.convert("RGB")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    orig_im0 = preprocess(input_image_orig)
    inpainted_im1 = preprocess(input_image_inpainted)

    # crop inpainted image to original
    n2, n1 = input_image_orig.size
    inpainted_im1 = inpainted_im1[:, :n1, :n2]

    return orig_im0, inpainted_im1


def print_results(data_dict):
    print('dog manual lpips: {:.5f}, auto lpips: {:.5f}'.format(data_dict['dog_manual'], data_dict['dog_auto']))
    print('bus manual lpips: {:.5f}, auto lpips: {:.5f}'.format(data_dict['bus_manual'], data_dict['bus_auto']))
    print('person manual lpips: {:.5f}, auto lpips: {:.5f}'.format(data_dict['person_manual'], data_dict['person_auto']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_func(parser)
    args = parser.parse_args()

    compare_results(args)
