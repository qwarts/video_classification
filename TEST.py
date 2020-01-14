#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:39:25 2020

@author: vincent
"""


import os
import sys
import json
import subprocess
import numpy as np
import torch

from torch import nn, optim


from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
from train import train_epoch

if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 2#400

    model = generate_model(opt)
    
    print(model)
    opt.model = 'model.pth'
    print("lala " + opt.model)
    # model['arch'] = 'resnet-34'
    
    opt.model = '/data/models/video_classification/resnet-34-kinetics.pth'
    # torch.save(model.state_dict(), opt.model)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)