#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 05:33:35 2020

@author: shaden
"""
import argparse, torch
import numpy as np
import json
from util import predict_image, set_model

num_of_layers = {'36': '19', '24': '13', '30': '16', '20': '11'}
parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')

parser.add_argument('image_path', type=str, action="store", help = 'Single image path')
parser.add_argument('checkpoint', type=str, action="store", help='Choose the model checkpoint')
parser.add_argument('--category_names', type=str, action="store", help='CUse a mapping of categories to real names')
parser.add_argument('--top_k', type= int, action="store", default = 5, help ='Return top K most likely classes')
parser.add_argument('--gpu', action= "store_true", default=False, help = 'Use GPU for testing')

args = parser.parse_args()

#Setting the device to gpu or cpu
device = torch.device("cuda" if args.gpu else "cpu")

#load the model
state_dict = torch.load(args.checkpoint)

num = str(len(state_dict.keys()))
model_arch = "vgg" + num_of_layers[num]

hidden_units = len(state_dict['classifier.fc1.weight'])


model = set_model(model_arch, hidden_units, 102)

model.load_state_dict(state_dict)
model.to(device)


probs, classes = predict_image(args.image_path, model, args.top_k, device)
probs, classes = np.array(probs)[0], np.array(classes)[0]

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[str(class_num+1)] for class_num in classes]
else:
    names = ["category "+ str(i+1) for i in classes]
    
    


i = 1
for name, prob in zip(names, probs):
    print("Flower predicted category with rank "+ str(i)+ " is", name , "with probabilty", prob)
    i +=1
