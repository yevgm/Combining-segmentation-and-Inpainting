import os
from omegaconf import OmegaConf
from lama.bin import predict


def main():
    predict_config = {
        "indir": "/home/george/PycharmProjects/Combining-segmentation-and-Inpainting/lama/LaMa_test_images",
        "outdir": "/home/george/PycharmProjects/Combining-segmentation-and-Inpainting/lama/output",
        "model": {
            "path": "/home/george/PycharmProjects/Combining-segmentation-and-Inpainting/lama/big-lama",
            "checkpoint": "best.ckpt"
        },
        "dataset": {
            "kind": "default",
            "img_suffix": ".png",
            "pad_out_to_modulo": 8,
        },
        "out_key": "inpainted",
        "device": "cuda"
    }

    predict.main(OmegaConf.create(predict_config))

if __name__ == '__main__':
    main()