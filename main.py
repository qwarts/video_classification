import os
import sys
import json
import subprocess
import numpy as np
import torch

from torch import nn, optim

from torchsummary import summary

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
from train import train_epoch, train_main
from target_transforms import ClassLabel

from zipfile import ZipFile

import zipfile

import urllib.request
import shutil

if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 2#400

    model = generate_model(opt)
    
    opt.model = 'model.pth'
    opt.model = '/data/save_1.pth'
    # print("lala " + opt.model)
    # model['arch'] = 'resnet-34'
    # opt.arch = 'resnet-34'
    
    # torch.save(model.state_dict(), opt.model)
    
    
    # print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    
    # print(opt.arch)
    # print(model_data['arch'])
    
    # assert opt.arch == model_data['arch']
    # model.load_state_dict(model_data['state_dict'])
    # model.eval()
    
    # model = model_data
   
    # url = "https://www.kaggle.com/c/16880/datadownload/dfdc_train_part_00.zip"
    # file_name = "/data/databases/deep-fake/data/00.zip"
    # save_dir = "/data/databases/deep-fake/data/"
    # with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
    #     shutil.copyfileobj(response, out_file)
    # with zipfile.ZipFile(file_name) as zf:
    #     zf.extractall(save_dir)
    
    # data_file_path = '/data/databases/deep-fake/train_sample_videos/metadata.json'
    # with open(data_file_path, 'r') as data_file:
    #     labels2 = json.load(data_file)
    #     print(labels2['esxrvsgpvb.mp4']['label'])
    # exit(1)
    
    
    input_root_dir = '/data/databases/deep-fake/data/'
    train_main(model, input_root_dir, opt)
    
   
    
    if opt.verbose:
        print(model)

    print(opt.input)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

# ####
    
#     from dataset import Video
#     from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
#     from temporal_transforms import LoopPadding
    
#     #### LOAD ALL LABELS
    
    import pandas as pd
    labels = pd.read_json('/data/databases/deep-fake/train_sample_videos/metadata.json')
    labels2 = json.load('/data/databases/deep-fake/train_sample_videos/metadata.json')
    print(labels2)
    exit(1)
#     ####
    
    
    
#     for input_file in input_files:
#         video_path = os.path.join(opt.video_root, input_file)
        
        
#         print('INPUT FILE: ' + input_file)
#         filen = input_file.split('/')
#         label = labels[filen[len(filen)-1]]
#         # print(label)
        
        
#         print(video_path)
#         subprocess.call('mkdir tmp', shell=True)
#         subprocess.call('ffmpeg -hide_banner -loglevel panic -i {} tmp/image_%05d.jpg'.format(video_path),
#                         shell=True)

#         criterion = nn.CrossEntropyLoss()
#         if not opt.no_cuda:
#             criterion = criterion.cuda()
#         optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
#         epoch = 1
        
#         spatial_transform = Compose([Scale(opt.sample_size),
#                                      CenterCrop(opt.sample_size),
#                                      ToTensor(),
#                                      Normalize(opt.mean, [1, 1, 1])])
#         temporal_transform = LoopPadding(opt.sample_duration)
        
#         video_dir = '{}tmp/'.format('/data/codebases/video_classification/')
        
#         data = Video(video_dir, spatial_transform=spatial_transform,
#                      temporal_transform=temporal_transform,
#                      sample_duration=opt.sample_duration)
#         data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
#                                                   shuffle=False, num_workers=opt.n_threads, pin_memory=True)
#         import logging
#         from pythonjsonlogger import jsonlogger
        
#         epoch_logger = logging.getLogger('info')
#         batch_logger = logging.getLogger('info')
        
#         elogHandler = logging.StreamHandler()
#         eformatter = jsonlogger.JsonFormatter()
#         elogHandler.setFormatter(eformatter)
#         epoch_logger.addHandler(elogHandler)
        
#         blogHandler = logging.StreamHandler()
#         bformatter = jsonlogger.JsonFormatter()
#         blogHandler.setFormatter(bformatter)
#         batch_logger.addHandler(blogHandler)
    
#         train_epoch(epoch, data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger, label)
        
#         subprocess.call('rm -rf tmp', shell=True)
#     exit(1)
# ####



    import csv
    outputs = []
    outputCSV = []
    i = 0
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        i+=1
        print('INPUT FILE: ' + str(i) + " " + input_file)
        filen = input_file.split('/')
        label = labels[filen[len(filen)-1]]
        
        if os.path.exists(video_path):
            print(video_path)
            subprocess.call('mkdir tmp', shell=True)
            subprocess.call('ffmpeg -hide_banner -loglevel panic -i {} tmp/image_%05d.jpg'.format(video_path),
                            shell=True)

            result = classify_video('tmp', input_file, class_names, model, opt)
            outputs.append(result)
            
            op = []
            op.append(label['label'])
            print(result['clips'])
            for i in range(len(result['clips'])):
                op.append(result['clips'][i]['label'])
            outputCSV.append(op)

            subprocess.call('rm -rf tmp', shell=True)
        else:
            print('{} does not exist'.format(input_file))
            
            
        if i > 50:
            break
    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)

    with open('output.csv', 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(outputCSV)