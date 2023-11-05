import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


def arg_parser():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    parser.add_argument('--data_dir', action='store', default="flowers")
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()


def process_data(args):
    # set directory for train, valid and test data
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        "train_transforms": transforms.Compose(
            [transforms.RandomRotation(30), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "valid_transforms": transforms.Compose(
            [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test_transforms": transforms.Compose(
            [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    # Load the datasets with ImageFolder
    image_datasets = {
        "train_datasets": datasets.ImageFolder(train_dir, transform=data_transforms["train_transforms"]),
        "valid_datasets": datasets.ImageFolder(valid_dir, transform=data_transforms["valid_transforms"]),
        "test_datasets": datasets.ImageFolder(test_dir, transform=data_transforms["test_transforms"])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train_loaders": torch.utils.data.DataLoader(image_datasets["train_datasets"], batch_size=64, shuffle=True),
        "valid_loaders": torch.utils.data.DataLoader(image_datasets["valid_datasets"], batch_size=64, shuffle=True),
        "test_loaders": torch.utils.data.DataLoader(image_datasets["test_datasets"], batch_size=64, shuffle=True)
    }
    return data_transforms, image_datasets, dataloaders


def train_model(device, model, optimizer, criterion, data_transforms, image_datasets, dataloaders, args):
    epochs = int(args.epochs)
    print_every = 10
    steps = 0

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders["train_loaders"]):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_accuracy = 0
                validation_loss = 0
                model.eval()
                for valid_inputs, valid_labels in dataloaders["valid_loaders"]:
                    optimizer.zero_grad()
                    valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
                    with torch.no_grad():
                        valid_outputs = model.forward(valid_inputs)
                        loss = criterion(valid_outputs, valid_labels)
                        ps = torch.exp(valid_outputs).data
                        equality = (valid_labels.data == ps.max(1)[1])
                        valid_accuracy += equality.type_as(torch.FloatTensor()).mean()

                validation_loss = loss / len(dataloaders["valid_loaders"])
                valid_accuracy = valid_accuracy / len(dataloaders["valid_loaders"])
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Training Loss: {:.4f}".format(running_loss / print_every),
                      "validation Loss: {:.4f}".format(validation_loss),
                      "Validation Accuracy: {:.4f}".format(valid_accuracy))

                running_loss = 0
    model.train()
    return model


def test_model(model, device, dataloaders):
    test_accuracy = 0
    test_total = 0
    model.eval()
    with torch.no_grad():
        for test_inputs, test_labels in dataloaders["test_loaders"]:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model.forward(test_inputs)
            _, predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_accuracy += (predicted == test_labels).sum().item()
    print(f'Accuracy of test images: {100 * test_accuracy / test_total}%')


def prepare_and_train_model(args):
    data_transforms, image_datasets, dataloaders = process_data(args)
    # initialize model
    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # update model classifier based on arch
    if args.arch and args.arch == "densenet121":
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, int(args.hidden_unit))),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(int(args.hidden_unit), 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, int(args.hidden_unit))),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(int(args.hidden_unit), 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    device = torch.device("cuda" if args.gpu else "cpu")
    model.to(device)

    # Train Model
    model = train_model(device, model, optimizer, criterion, data_transforms, image_datasets, dataloaders, args)

    # Test model
    test_model(model, device, dataloaders)

    # save_model
    model.class_to_idx = image_datasets['train_datasets'].class_to_idx

    torch.save({
        "model.classifier": model.classifier,
        "model.class_to_idx": image_datasets['train_datasets'].class_to_idx,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model": model
    }, args.save_dir)


def main():
    args = arg_parser()
    prepare_and_train_model(args)


if __name__ == "__main__":
    main()