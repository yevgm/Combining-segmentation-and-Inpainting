import argparse
import os
import torch


def parse_global_args(parent, add_help=False):
    """
        Parse commandline arguments.
        """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)

    repo_root = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(repo_root, 'output')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument('--repo-root', default=repo_root, type=str,
                        help='repository main dir')
    parser.add_argument('-o', '--output-dir', default=output_folder, type=str,
                        help='provide full dir for the output images')

    parser.add_argument('--seed', default=42, type=int,
                        help='random seed for everything')
    parser.add_argument('--device', default=device, type=str,
                        help='device to use')

    return parser


def parse_args_lama(parent, add_help=False):
    """
        Parse commandline arguments.
        """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)
    parser = parse_global_args(parser)

    repo_root = os.path.dirname(os.path.realpath(__file__))
    lama_model_path = os.path.abspath(os.path.join(repo_root, '..', 'lama-fourier'))  # lama-fourier, big-lama
    input_folder = os.path.join(repo_root, 'input')
    test_images_folder = os.path.join(repo_root, 'test_images_comp')

    parser.add_argument('--skip-seg', default=False, type=bool,
                        help='skip segmentation or not')
    parser.add_argument('-i', '--image-input', default=test_images_folder, type=str,
                        help='provide full dir location for images to process')
    parser.add_argument('-c', '--remove-class', default=15, type=int,
                        help="class to remove from images. '-a print_cls' to print all classes")
    parser.add_argument('-a', "--action", default='inpaint',
                        help="choose which action to take: 'inpaint' to process images\n 'print_cls' to print classes",
                        type=str)
    parser.add_argument('--input-dir', default=input_folder, type=str,
                        help='provide full dir location for segmentation output and Lama input')
    parser.add_argument('--lama-model-path', default=lama_model_path, type=str,
                        help='provide lama model path')
    parser.add_argument('--lama-model-name', default='best.ckpt', type=str,
                        help='provide lama specific model name')
    parser.add_argument('--dataset-kind', default='default', type=str,
                        help='')
    parser.add_argument('--dataset-img_suffix', default='.png', type=str,
                        help='')
    parser.add_argument('--dataset-pad_out_to_modulo', default=8, type=int,
                        help='')
    parser.add_argument('--out-key', default='inpainted', type=str,
                        help='')

    return parser


def print_cls():
    print('''
Classes available for segmentation task:
    0: background
    1: aeroplane
    2: bicycle
    3: bird
    4: boat
    5: bottle
    6: bus
    7: car
    8: cat
    9: chair
    10: cow
    11: dining table
    12: dog
    13: horse
    14: motorbike
    15: person
    16: potted plant
    17: sheep
    18: sofa
    19: train
    20: tv monitor 
    
    Please choose an integer.''')