import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter

# import segmentation network
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True, force_reload=True)


def segment(filename, debug_mode=False):
    """

    :param filename: image path
    :param debug_mode: if you want to show the images to inspect the segmentation
    :return:
    """
    model.eval()
    # this import has to be here otherwise there is a weird import error - did not manage to resolve
    from torchvision import transforms
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    # output = output.byte().cpu().numpy()
    if debug_mode:
        plt.imshow(np.asarray(input_image))
        plt.imshow(r, alpha=0.5)
        plt.show()
    return input_image, r


def main():
    from omegaconf import OmegaConf
    from lama.bin import predict
    # load config as expected by the 'predict' function
    # TODO: remove this to file and put as external config.yml file
    predict_config = {
        "indir": "/home/george/PycharmProjects/Combining-segmentation-and-Inpainting/inputs",
        "outdir": "/home/george/PycharmProjects/Combining-segmentation-and-Inpainting/output",
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
    # define the dir with test images
    test_image_path = './test_images'

    # load a list of image names
    image_names = [f for f in os.listdir(test_image_path) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

    # iterate over all the images:
    # 1. Segment
    # 2. Save image and mask in input folder
    # 3. Run inpainting (LAMA)
    for image_name in image_names:

        # segment
        im, mask = segment(os.path.join(test_image_path,f'{image_name}'))

        # save
        for i in [15]:
            im.save(f'inputs/{image_name.split(".")[0]}_{i:03d}.png')
            # segment returns 21 masks, currently saving only [0] we can/need to save all 21
            cur_mask = np.asarray(np.asarray(mask) == i, dtype=np.uint8) * 255
            cur_mask = Image.fromarray(cur_mask).filter(ImageFilter.MaxFilter(31))
            cur_mask.save(f'inputs/{image_name.split(".")[0]}_{i:03d}_mask.png')

    # run LAMA
    main()
