import argparse
import os


def parse_global_args(parent, add_help=False):
    """
        Parse commandline arguments.
        """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)

    repo_root = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(repo_root, 'output')
    parser.add_argument('--repo-root', default=repo_root, type=str,
                        help='repository main dir')
    parser.add_argument('--output-dir', default=output_folder, type=str,
                        help='output dir')

    parser.add_argument('--save-model', default=False, type=bool,
                        help='save model to output folder or not')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed for everything')

    return parser
