import argparse
from config import parse_global_args


def main(args: argparse.Namespace):

    return NotImplemented


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    args = parser.parse_args()

    main(args)