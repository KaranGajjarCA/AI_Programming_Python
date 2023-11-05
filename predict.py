import argparse
import json
import torch
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def arg_parser():
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument('--checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--image_path', dest='path', default='flowers/test/100/image_07896.jpg')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()


def load_checkpoint(device, args):
    checkpoint = torch.load(args.checkpoint)
    model = checkpoint["model"]
    model.classifier = checkpoint["model.classifier"]
    model.class_to_idx = checkpoint["model.class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])
    model.optimizer = checkpoint["optimizer_state_dict"]
    model.to(device)
    return model


def load_cat_names(args):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    image = Image.open(image)
    image_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = np.array(image_transform(image))

    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax


def predict(image_path, model, device, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model. """
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0).float()
    image = image.to(device)

    with torch.no_grad():
        image_output = model.forward(image)

    probability = torch.exp(image_output)
    probs, indexes = probability.topk(topk)
    probs = probs.to('cpu').numpy().tolist()[0]
    indexes = indexes.to('cpu').numpy().tolist()[0]

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [index_to_class[each] for each in indexes]

    return probs, classes


def main():
    args = arg_parser()
    device = torch.device("cuda" if args.gpu else "cpu")
    model = load_checkpoint(device, args)
    cat_to_name = load_cat_names(args)
    probs, classes = predict(args.image_path, model, topk=int(args.top_k))
    labels = [cat_to_name[str(index)] for index in classes]

    # Print final results
    for idx, label in enumerate(labels):
        print("{} with a probability of {}".format(label, probs[idx]))


if __name__ == "__main__":
    main()
