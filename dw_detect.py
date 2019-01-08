# -*- coding: utf-8 -*-
"""
DeepWalker Project
==================

This file is part of the Deep Walker Project which is made for the Udacity
Pytorch Scholarship Project Showcase. The whole project is under the GNU/GPL v2
Lincence. For more information please consult README.md.

***

This file creates Yolo v3 model.
Spurce inspired by https://github.com/eriklindernoren/PyTorch-YOLOv3.git

***

@author: Axel Ország-Krisz Dr.
"""



import models

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision import datasets, transforms

import loc_utils


device = 'cpu'

def createYoloModel():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    dw_yolo = models.Darknet('./yolov3.cfg')
    dw_yolo.to(device)
    dw_yolo.load_weights('./yolov3.weights')
    
    return dw_yolo