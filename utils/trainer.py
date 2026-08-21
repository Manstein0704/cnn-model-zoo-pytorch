import torch
import torch.optim as optim
from tqdm import tqdm



def trainer(model, optimizer:optim, criterion, train_loader, device):
    model.train()
    n_train, n_acc, total_loss = 0, 0, 0
    
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        n_train += len(labels)
        n_acc += (predictions==labels).sum().item()
        total_loss += (loss.item() * len(labels))

    train_acc = (n_acc / n_train)
    train_loss = (total_loss / n_train)
    return train_acc, train_loss



@torch.no_grad()
def test(model, criterion, test_loader, device):
    model.eval()
    n_test, n_acc, total_loss = 0, 0, 0

    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)
        n_test += len(labels)
        n_acc += (predictions==labels).sum().item()
        total_loss += (loss.item() * len(labels))

    test_acc = (n_acc / n_test)
    test_loss = (total_loss / n_test)

    return test_acc, test_loss




