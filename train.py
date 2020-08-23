#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:34:48 2020

@author: shaden
"""
import argparse, torch
from util import transform_data, set_model, trained_model

arch_choices = ["vgg" + str(i)for i in [11, 13 ,16, 19]] 
arch_choices += ["vgg" + str(i)+ "_bn"for i in [11, 13 ,16, 19]]


parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')

parser.add_argument('data_directory', type=str, action="store", help = 'Set the learning data directory')
parser.add_argument('--save_directory', type=str, action="store", default = 'checkpoint.pth' ,help='Set the directory of saving the file')
parser.add_argument('--arch', type = str,  action="store", default = 'vgg16', help= 'Set the architecture of the training network', choices= arch_choices)
parser.add_argument('--learning_rate', type= float,  action="store", default = 0.005 ,help= 'Set the value of learning rate')
parser.add_argument('--hidden_units', type= int, action="store", default = 1000, help ='Set the number of hidden units')
parser.add_argument('--epochs', type= int, action="store", default = 5, help ='Set the number of learning epochs')
parser.add_argument('--gpu', action= "store_true", default=False, help = 'Use GPU for training')

args = parser.parse_args()
#Setting the device to gpu or cpu
device = torch.device("cuda" if args.gpu else "cpu")

#The training process
trainloader, validloader, testloader = transform_data(args.data_directory)
model = set_model(args.arch, args.hidden_units, 102)
model = trained_model(trainloader, validloader, model, args.epochs, args.learning_rate, device)

#Saving the model
torch.save(model.state_dict(), args.save_directory)