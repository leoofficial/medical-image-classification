import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import COVID19Net


def estimate_score(data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, label = data
            predicted = model(images)
            predicted = torch.flatten(predicted)
            predicted = torch.where(predicted > 0.5, torch.tensor(1), torch.tensor(0))
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return 100 * correct / total


if __name__ == '__main__':

    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    train_set = torchvision.datasets.ImageFolder(os.path.join('data', 'XRay', 'train'), transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.ImageFolder(os.path.join('data', 'XRay', 'test'), transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2)

    model = COVID19Net()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    file = open('./docs/log', 'a')
    for epoch in range(4):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0], data[1].float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                train_score = estimate_score(train_loader)
                test_score = estimate_score(test_loader)
                log = '[%d, %5d] loss: %.3f, train score: %.3f, test score: %.3f' % \
                      (epoch + 1, i + 1, running_loss / 100, train_score, test_score)
                print(log)
                file.write(log + '\n')
                running_loss = 0.0
    file.close()
    print('Finished Training')

    torch.save(model.state_dict(), './model/covid19_net.pth')
