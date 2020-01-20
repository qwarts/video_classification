import torch
from torch.autograd import Variable
import time
import os
import sys
import torch.nn as nn
import torch.optim as optim
import subprocess
import json

from math import log

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding

import logging
import pandas as pd
from pythonjsonlogger import jsonlogger

from utils import AverageMeter, calculate_accuracy, calculate_accuracy_single_target, calculate_accuracy_mse

# DOCUMENTATION: 
# LOSS FUNCTIONS :  https://pytorch.org/docs/stable/nn.html
#                   https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
#                   https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
#                   https://github.com/pytorch/pytorch/blob/677030b1cb12a2ff32fe85a3c2b9cc547ef47de8/torch/nn/functional.py#L1364


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, label):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
                
        targets = torch.zeros([18, opt.n_classes], dtype=torch.long)
        
        print("Label: " + label['label'])
        
        for j in range(0,18):
            if(label['label'] == 'FAKE'):
                targets[j][0] = 0
            else:
                targets[j][0] = 1


        if not opt.no_cuda:
            targets = targets.cuda(non_blocking=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        
        print(outputs.t())

        loss = criterion(outputs.t(), targets[0])
        acc = calculate_accuracy_single_target(outputs, targets)

        try:
            losses.update(loss.data[0], inputs.size(0))
        except:
            losses.update(loss.data, inputs.size(0))

        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        batch_logger.log(1, {
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']})

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch, i + 1, len(data_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, acc=accuracies))

    epoch_logger.log(1, {
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    opt.checkpoint = 1
    opt.result_path = '/data/'

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(states, save_file_path)


# calculate binary cross entropy
def binary_cross_entropy(actual, predicted):
    sum_score = 0.0
    print(len(actual))
    for i in range(len(actual)):
        print("actual: " + str(actual[i]))
        print("predicted: " + str(predicted[i]))
        b = log(1e-15 + max(0,predicted[i]))
        print(b)
        sum_score += actual[i] * b
        print("sum_score: " + str(sum_score))
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score


def train_main(model, input_root_dir, opt):
    ####

    epoch_logger = logging.getLogger('info')
    batch_logger = logging.getLogger('info')
    
    elogHandler = logging.StreamHandler()
    eformatter = jsonlogger.JsonFormatter()
    elogHandler.setFormatter(eformatter)
    epoch_logger.addHandler(elogHandler)
    
    blogHandler = logging.StreamHandler()
    bformatter = jsonlogger.JsonFormatter()
    blogHandler.setFormatter(bformatter)
    batch_logger.addHandler(blogHandler)

    spatial_transform = Compose([Scale(opt.sample_size),
                                  CenterCrop(opt.sample_size),
                                  ToTensor(),
                                  Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    if not opt.no_cuda:
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    epoch = 1
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    end_time = time.time()

    ii = 0
    
    previous_label = "FAKE"
    pre_previous_label = "FAKE"
    
    for files_dir in os.listdir(input_root_dir):
        sub_path = os.path.join(input_root_dir, files_dir)
        print("Files dir: " + files_dir)
        print("Sub path:" + sub_path)
        
        data_file_path = os.path.join(sub_path,'metadata.json')
        with open(data_file_path, 'r') as data_file:
            labels = json.load(data_file)
            
        batch_size = len(os.listdir(sub_path))
        i = 0
        
        for input_file in os.listdir(sub_path):
            if input_file.endswith(".mp4"):
    
                video_path = os.path.join(sub_path, input_file)

                label = labels[input_file]
                
                if label['label'] != previous_label or label['label'] != pre_previous_label:
                    
                    previous_label = label['label']

                    subprocess.call('mkdir tmp', shell=True)
                    subprocess.call('ffmpeg -hide_banner -loglevel panic -i {}  -vframes 288 tmp/image_%05d.jpg'.format(video_path),
                                    shell=True)
            
                    video_dir = '{}tmp/'.format('/data/codebases/video_classification/')
                    
                    data = Video(video_dir, spatial_transform=spatial_transform,
                                  temporal_transform=temporal_transform,
                                  sample_duration=opt.sample_duration)
                    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)
    
                    for k, (inputs, targets) in enumerate(data_loader):
                        data_time.update(time.time() - end_time)
 
                        print("Label: " + label['label'])
                        
                        # # FOR CROSS ENTROPY LOSS
                        # targets = torch.zeros([18, 1], dtype=torch.long)
                        # for j in range(0,18):
                        #     if(label['label'] == 'FAKE'):
                        #         targets[j][0] = 0
                        #         # targets[j][1] = 1
                        #     else:
                        #         targets[j][0] = 1
                        #         # targets[j][1] = 0
                                
                        # FOR MSE LOSS
                        targets = torch.zeros([18, opt.n_classes], dtype=torch.float)
                        for j in range(0,18):
                            if(label['label'] == 'FAKE'):
                                targets[j][0] = 0.0
                                targets[j][1] = 1.0
                            else:
                                targets[j][0] = 1.0
                                targets[j][1] = 0.0                        
                
                        if not opt.no_cuda:
                            targets = targets.cuda(non_blocking=True)
                        inputs = Variable(inputs)
                        targets = Variable(targets)
                        outputs = model(inputs)

                        print(outputs.t())
                        print(targets.t())
        
                        # FOR CROSS ENTROPY LOSS
                        # loss = criterion(outputs, torch.max(targets, 1)[1])
                        # FOR MSE LOSS
                        loss = criterion(outputs, targets)
                        
                        print(loss)
                        
                        # FOR CROSS ENTROPY LOSS
                        # acc = calculate_accuracy(outputs, targets)
                        # FOR MSE LOSS
                        acc = calculate_accuracy_mse(outputs, targets)
                        
                        print(acc)
                        
                        if(label['label'] == 'FAKE'):
                            try:
                                losses.update(loss.data[0], inputs.size(0))
                            except:
                                losses.update(loss.data, inputs.size(0))
                            accuracies.update(acc, inputs.size(0))
                    
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        else:
                            try:
                                losses.update(loss.data[0], inputs.size(0))
                            except:
                                losses.update(loss.data, inputs.size(0))
                            accuracies.update(acc, inputs.size(0))
                            loss = loss*16
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                
                        batch_time.update(time.time() - end_time)
                        end_time = time.time()
                        
                        batch_logger.log(1, {
                            'epoch': epoch,
                            'batch': i + 1,
                            'iter': (epoch - 1) * batch_size + (i + 1),
                            'loss': losses.val,
                            'acc': accuracies.val,
                            'lr': optimizer.param_groups[0]['lr']})
                
                        print('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                                  epoch, i + 1, batch_size, batch_time=batch_time,
                                  data_time=data_time, loss=losses, acc=accuracies))
                        ii += 1
                    subprocess.call('rm -rf tmp', shell=True)
                i += 1
                
            if ii % 100 == 0:
                save_loc = '/data/codebases/video_classification/model{}.pth'.format(ii)
                torch.save(model.state_dict(), save_loc)
        epoch_logger.log(1, {
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': optimizer.param_groups[0]['lr']})
        print('XXX Epoch: [{0}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                              epoch, i + 1, batch_size, batch_time=batch_time,
                              data_time=data_time, loss=losses, acc=accuracies))
    exit(1)
####
    
    
    
def train_main_multi_batch(model, input_root_dir, opt):
    ####

    epoch_logger = logging.getLogger('info')
    batch_logger = logging.getLogger('info')
    
    elogHandler = logging.StreamHandler()
    eformatter = jsonlogger.JsonFormatter()
    elogHandler.setFormatter(eformatter)
    epoch_logger.addHandler(elogHandler)
    
    blogHandler = logging.StreamHandler()
    bformatter = jsonlogger.JsonFormatter()
    blogHandler.setFormatter(bformatter)
    batch_logger.addHandler(blogHandler)

    spatial_transform = Compose([Scale(opt.sample_size),
                                  CenterCrop(opt.sample_size),
                                  ToTensor(),
                                  Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    if not opt.no_cuda:
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch = 1
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    end_time = time.time()

    ii = 0
    
    previous_label = "FAKE"
    pre_previous_label = "FAKE"
    
    for files_dir in os.listdir(input_root_dir):
        sub_path = os.path.join(input_root_dir, files_dir)
        print("Files dir: " + files_dir)
        print("Sub path:" + sub_path)
        
        data_file_path = os.path.join(sub_path,'metadata.json')
        with open(data_file_path, 'r') as data_file:
            labels = json.load(data_file)
            
        opt.batch_size = 36
        total_batch_size = len(os.listdir(sub_path))
        i = 0
        input_files = os.listdir(sub_path)
        for inp_num in range(1,len(input_files),2):
            print("Lala: " + str(inp_num))
            # print(input_files)
            input_file1 = input_files[inp_num]
            input_file2 = input_files[inp_num-1]
            if input_file1.endswith(".mp4") and input_file2.endswith(".mp4"):
    
                video_path1 = os.path.join(sub_path, input_file1)
                video_path2 = os.path.join(sub_path, input_file2)

                label1 = labels[input_file1]
                label2 = labels[input_file2]
                
                if label1['label'] != previous_label or label1['label'] != pre_previous_label:
                    
                    previous_label = label1['label']

                    subprocess.call('mkdir tmp', shell=True)
                    subprocess.call('ffmpeg -hide_banner -loglevel panic -i {}  -vframes 288 tmp/image_%05d.jpg'.format(video_path1),
                                    shell=True)
                    subprocess.call('ffmpeg -hide_banner -loglevel panic -i {}  -vframes 288 -start_number 289 tmp/image_%05d.jpg'.format(video_path2),
                                    shell=True)

                    video_dir = '{}tmp/'.format('/data/codebases/video_classification/')
                    
                    data = Video(video_dir, spatial_transform=spatial_transform,
                                  temporal_transform=temporal_transform,
                                  sample_duration=opt.sample_duration)

                    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)
     
                    for k, (inputs, targets) in enumerate(data_loader):
                        data_time.update(time.time() - end_time)
 
                        print("Label: " + label1['label'] + ", " + label2['label'])
                        
                        # # FOR CROSS ENTROPY LOSS
                        # targets = torch.zeros([18, 1], dtype=torch.long)
                        # for j in range(0,18):
                        #     if(label['label'] == 'FAKE'):
                        #         targets[j][0] = 0
                        #         # targets[j][1] = 1
                        #     else:
                        #         targets[j][0] = 1
                        #         # targets[j][1] = 0
                                
                        # FOR MSE LOSS
                        targets = torch.zeros([opt.batch_size, opt.n_classes], dtype=torch.float)
                        for j in range(0,int(opt.batch_size/2)):
                            if(label1['label'] == 'FAKE'):
                                targets[j][0] = 0.0
                                targets[j][1] = 1.0
                            else:
                                targets[j][0] = 1.0
                                targets[j][1] = 0.0     
                                
                        for j in range(int(opt.batch_size/2), opt.batch_size):
                            if(label2['label'] == 'FAKE'):
                                targets[j][0] = 0.0
                                targets[j][1] = 1.0
                            else:
                                targets[j][0] = 1.0
                                targets[j][1] = 0.0                        
                
                        if not opt.no_cuda:
                            targets = targets.cuda(non_blocking=True)
                        inputs = Variable(inputs)
                        targets = Variable(targets)
                        outputs = model(inputs)

                        print(outputs.t())
                        print(targets.t())
        
                        # FOR CROSS ENTROPY LOSS
                        # loss = criterion(outputs, torch.max(targets, 1)[1])
                        # FOR MSE LOSS
                        loss = criterion(outputs, targets)
                        
                        print(loss)
                        
                        # FOR CROSS ENTROPY LOSS
                        # acc = calculate_accuracy(outputs, targets)
                        # FOR MSE LOSS
                        acc = calculate_accuracy_mse(outputs, targets)
                        
                        print(acc)
                        
                        try:
                            losses.update(loss.data[0], inputs.size(0))
                        except:
                            losses.update(loss.data, inputs.size(0))
                        accuracies.update(acc, inputs.size(0))
                
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                        batch_time.update(time.time() - end_time)
                        end_time = time.time()
                        
                        batch_logger.log(1, {
                            'epoch': epoch,
                            'batch': i + 1,
                            'iter': (epoch - 1) * opt.batch_size + (i + 1),
                            'loss': losses.val,
                            'acc': accuracies.val,
                            'lr': optimizer.param_groups[0]['lr']})
                
                        print('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                                  epoch, i + 1, opt.batch_size, batch_time=batch_time,
                                  data_time=data_time, loss=losses, acc=accuracies))
                        ii += 1
                    subprocess.call('rm -rf tmp', shell=True)
                i += 1
                
            if ii % 100 == 0:
                save_loc = '/data/codebases/video_classification/model{}.pth'.format(ii)
                torch.save(model.state_dict(), save_loc)
        epoch_logger.log(1, {
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': optimizer.param_groups[0]['lr']})
        print('XXX Epoch: [{0}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                              epoch, i + 1, opt.batch_size, batch_time=batch_time,
                              data_time=data_time, loss=losses, acc=accuracies))
    exit(1)