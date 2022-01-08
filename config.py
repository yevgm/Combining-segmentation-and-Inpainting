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

    parser.add_argument('--save-model', default=False, type=bool,
                        help='save model to output folder or not')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed for everything')
    parser.add_argument('--device', default=device, type=str,
                        help='device to use')

    return parser


def parse_args_mask_generator(parent, add_help=False):
    """
        Parse commandline arguments.
        """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)
    parser = parse_global_args(parser)

    parser.add_argument('--input-dir', default='', type=str,
                        help='input images directory')

    return parser