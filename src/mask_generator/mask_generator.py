import argparse
from config import parse_global_args


class SemanticSegmentationMask:
    def __init__(self, config: dict):
        pass

    def forward(self):
        return NotImplemented


def main(args: argparse.Namespace):
    args_dict = vars(args)
    seg_cls = SemanticSegmentationMask(args_dict)
    seg, seg_metadata = seg_cls.forward()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    args = parser.parse_args()

    main(args)