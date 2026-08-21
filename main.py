import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse
from utils.trainer import train, evaluate
from datasets.cifar import get_dataloader
from utils.seed import set_seed

from models.alexnet import AlexNet
from models.googlenet import GoogleNet
from models.resnet import ResNet
from models.vgg import VGG


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #Set Seed
    set_seed(args.seed)

    #Get train_loader and test_loader
    train_loader, test_loader = get_dataloader(args.batch_size) 
    num_classes = 10

    if args.model == "alexnet":
        model = AlexNet(num_classes=num_classes)

    elif args.model == "googlenet":
        model = GoogleNet(n_classes=num_classes)

    elif args.model == "resnet":
        model = ResNet(num_classes=num_classes)

    elif args.model == "vgg":
        model = VGG(num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    #Start Learning
    for epoch in range(args.num_epochs):
        train_acc, train_loss = train(model, optimizer, criterion, train_loader, device)
        test_acc, test_loss = evaluate(model, criterion, test_loader, device)

        if epoch==0 or ((epoch+1)//10) == 0:
            print(f"{epoch}/{args.num_epochs}, test_acc:{test_acc}, test_loss:{test_loss}")








def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=["alexnet", "googlenet", "resnet", "vgg"]
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()