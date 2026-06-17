
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# -------------------------------
# 🚚 Data loading (already implemented)
# -------------------------------
def get_dataloaders(train_size=5000, test_size=1000, batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_subset = torch.utils.data.Subset(full_train, range(train_size))
    test_subset = torch.utils.data.Subset(full_test, range(test_size))

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# -------------------------------
# 🧠 Your task: Define a CNN model
# -------------------------------
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define your CNN architecture here
        raise NotImplementedError("You need to implement this!")

    def forward(self, x, visualize=False):
        # TODO: Implement forward pass
        raise NotImplementedError("You need to implement this!")


# -------------------------------
# 🏋️ Training function
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    # TODO: Implement one epoch of training
    raise NotImplementedError("You need to implement this!")


# -------------------------------
# 🧪 Evaluation function
# -------------------------------
def evaluate(model, dataloader, device):
    # TODO: Implement evaluation logic
    raise NotImplementedError("You need to implement this!")


# -------------------------------
# 🧵 Training loop
# -------------------------------
def train_model(model, trainloader, testloader, device, epochs=10):
    # TODO: Implement the full training loop
    raise NotImplementedError("You need to implement this!")


# -------------------------------
# 🚀 Main function
# -------------------------------
def main():
    trainloader, testloader = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyCNN().to(device)
    train_model(model, trainloader, testloader, device)
    # TODO: add here any necessary code


if __name__ == '__main__':
    main()
