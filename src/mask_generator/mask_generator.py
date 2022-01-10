import numpy as np
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt


def segment(filename, model, debug_mode=False):
    """

    :param filename: image path
    :param model: torch segmentation model
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

    if debug_mode:
        plt.imshow(np.asarray(input_image))
        plt.imshow(r, alpha=0.5)
        plt.show()
    return input_image, r

