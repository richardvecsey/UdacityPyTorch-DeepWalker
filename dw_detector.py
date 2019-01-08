# -*- coding: utf-8 -*-
"""
DeepWalker Project
==================

This file is part of the Deep Walker Project which is made for the Udacity
Pytorch Scholarship Project Showcase. The whole project is under the GNU/GPL v2
Lincence. For more information please consult README.md.

***

This file creates red/green detector model based on ResNet-34
To use this model you have prepare dataset with Yolo network

***

@author: Axel Ország-Krisz Dr.
"""

import torch

from torchvision import datasets, models, transforms



device = 'cpu'
epoch_count = 50

def main():

    global device, epoch_count
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    clean_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    ampel_dataset = datasets.ImageFolder('./lamps', transform=clean_transform)
    loader = torch.utils.data.DataLoader(ampel_dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=0)

    model = model = models.resnet34(pretrained=True)
    model.fc = torch.nn.Linear(512, 2, bias=False)
    model.to(device)
    model.load_state_dict(torch.load('./DeepWalkerDetect_state.pth'))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = float(0)
    total_corrects = 0

    for inputs, labels in loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        maxvalues, maxindices = torch.max(outputs, 1)

        total_loss += loss.item() * inputs.size(0)
        total_corrects += torch.sum(maxindices == labels.data)

    total_loss = float(total_loss / len(loader.dataset))
    total_accuracy = float(total_corrects.float() / len(loader.dataset))
    
    print('Total loss: {}; total accuracy {}'.format(total_loss, total_accuracy))

if __name__ == '__main__':
    main()