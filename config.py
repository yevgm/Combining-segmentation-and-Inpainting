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
    parser.add_argument('--output-dir', default=output_folder, type=str,
                        help='output dir')

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
    lama_model_path = os.path.abspath(os.path.join(repo_root, '..', 'big-lama'))
    inputs_folder = os.path.join(repo_root, 'inputs')
    test_images_folder = os.path.join(repo_root, 'test_images')

    parser.add_argument('--skip-seg', default=False, type=bool,
                        help='skip segmentation or not')
    parser.add_argument('--test-image-path', default=test_images_folder, type=str,
                        help='test images directory')
    parser.add_argument('--input-dir', default=inputs_folder, type=str,
                        help='input images directory')
    parser.add_argument('--lama-model-path', default=lama_model_path, type=str,
                        help='lama model path')
    parser.add_argument('--lama-model-name', default='best.ckpt', type=str,
                        help='lama specific model name')
    parser.add_argument('--dataset-kind', default='default', type=str,
                        help='')
    parser.add_argument('--dataset-img_suffix', default='.png', type=str,
                        help='')
    parser.add_argument('--dataset-pad_out_to_modulo', default=8, type=int,
                        help='')
    parser.add_argument('--out-key', default='inpainted', type=str,
                        help='')

    return parser
