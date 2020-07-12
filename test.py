import torch
import torchvision
import torchvision.transforms as transforms

from model import COVID19Net
import os

if __name__ == '__main__':

    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    test_set = torchvision.datasets.ImageFolder(os.path.join('data', 'XRay', 'test'), transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2)

    model = COVID19Net()
    model.load_state_dict(torch.load('./model/covid19_net.pth'))

    classes = ['NORMAL', 'PNEUMONIA']
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            predicted = model(images)
            predicted = torch.flatten(predicted)
            predicted = torch.where(predicted > 0.5, torch.tensor(1), torch.tensor(0))
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        print('Accuracy of %5s : %.3f%%' % (classes[i], 100 * class_correct[i] / class_total[i]))
