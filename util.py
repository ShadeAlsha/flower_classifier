#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 04:16:30 2020

@author: shaden
"""

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
import numpy as np
from PIL import Image



def transform_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader

def set_model(arch, hidden_units, num_classes, drop_out = 0.3):
    """Inputs:
        - arch (string): the architecture of the training network
        - hidden_units (int): the number of hidden units
    """
    model = getattr(models, arch)(pretrained= True)
    
    #truning gradient off for other parameters
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = model.classifier
    last_units = classifier._modules['0'].in_features
    
    new_classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(last_units, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p= drop_out)),
                          ('fc2', nn.Linear(hidden_units, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = new_classifier
    return model

    
def trained_model(trainloader, validloader, model, epochs, learning_rate, device, print_every =25):
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()
    model.to(device)
    
    steps = 0
    
    
    for epoch in range(epochs):
        running_loss = 0
        
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            
            if steps % print_every == 0: 
                print("Step:", steps)
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            #Turn off dropout
                
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                        
                    valid_loss += batch_loss.item()
                        
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            print("End of Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                   "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
    
                    
            model.train()
    return model

def process_image(pil_image):
    ''' 
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Resizing the image
    width, height = pil_image.size #getting dimensions
    if width > height:
        SIZE = (int(width/height*256), 256)
    else:
        SIZE = (256, int(height/width*256))
    
    pil_image = pil_image.resize(SIZE)
    
    #Cropping the image 
    new_width, new_height = pil_image.size #getting updated dimensions
    left = (new_width - 244)/2
    top = (new_height - 244)/2
    right = (new_width + 244)/2
    bottom = (new_height + 244)/2
    pil_image = pil_image.crop((left, top, right, bottom))

    #Transforming it to Numpy array with Normalization 

    normalized_image = np.array(pil_image)/255

    #normalized_image 

    new_mean = np.array([0.485, 0.456, 0.406])
    new_std = np.array([0.229, 0.224, 0.225])

    new_image = (normalized_image - new_mean)/new_std

    new_image= new_image.transpose((2, 0, 1))
    
    return new_image

def predict_image(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #importing the image
    pil_image = Image.open(image_path)
    image = process_image(pil_image)
    image = image.astype(np.float32)

    images = np.expand_dims(image, axis=0)
    
    #changing it to pytorch tensor
    images = torch.from_numpy(images)

    #make predictions
    images= images.to(device)
    model.eval()
    with torch.no_grad():
        logps = model.forward(images)
        ps = torch.exp(logps)
        probs, classes = ps.topk(topk,largest=True)
        
    model.train()
    return probs, classes 
