import torch
from torch.autograd import Variable
import time
import os
import sys
import torch.nn as nn
import torch.optim as optim
import subprocess
import json


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


        # print(targets)
                
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
        # print("Output model: ")
        # print(outputs[0])
        
        # print("Done printing 0")

        # print("Target: ")
        # print(targets)
        
        # print("Done printing 1")
        # print(targets)
        targets1 = torch.max(targets, 1)[1]
        # print(targets1)
        # print("Done printing 2")
        loss = criterion(outputs.t(), targets[0])
        # print("Done printing 3")
        acc = calculate_accuracy_single_target(outputs, targets)
        # print("Done printing 4")

        try:
            losses.update(loss.data[0], inputs.size(0))
            # print("Done printing 5A")
        except:
            losses.update(loss.data, inputs.size(0))
            # print("Done printing 5B")
        accuracies.update(acc, inputs.size(0))
        # print("Done printing 6")

        optimizer.zero_grad()
        # print("Done printing 7")
        loss.backward()
        # print("Done printing 8")
        optimizer.step()
        # print("Done printing 9")

        batch_time.update(time.time() - end_time)
        # print("Done printing 10")
        end_time = time.time()
        # print("Done printing 11")
        
        # print("hoe: " + str(optimizer.param_groups[0]['lr']))

        
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

from math import log
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
    
    from dataset import Video
    from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
    from temporal_transforms import LoopPadding
    
    import logging
    import pandas as pd
    from pythonjsonlogger import jsonlogger
    
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
    
    #### LOAD ALL LABELS
    

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    end_time = time.time()
    
    ####
    ii = 0
    
    previous_label = "FAKE"
    
    for files_dir in os.listdir(input_root_dir):
        sub_path = os.path.join(input_root_dir, files_dir)
        print("Files dir: " + files_dir)
        print("Sub path:" + sub_path)
        
        data_file_path = os.path.join(sub_path,'metadata.json')
        with open(data_file_path, 'r') as data_file:
            labels = json.load(data_file)
            
        # labels = pd.read_json(os.path.join(sub_path,'metadata.json'))
        batch_size = len(os.listdir(sub_path))
        i = 0
        
        for input_file in os.listdir(sub_path):
            if input_file.endswith(".mp4"):
    
                video_path = os.path.join(sub_path, input_file)

                label = labels[input_file]
                
                # print("label : ")
                # print(label['label'])
                # print("previous label: " + previous_label)
                
                if label['label'] != previous_label:
                    
                    # print("Enter")
                    
                    previous_label = label['label']

                    subprocess.call('mkdir tmp', shell=True)
                    subprocess.call('ffmpeg -hide_banner -loglevel panic -i {} tmp/image_%05d.jpg'.format(video_path),
                                    shell=True)
            
                    video_dir = '{}tmp/'.format('/data/codebases/video_classification/')
                    
                    data = Video(video_dir, spatial_transform=spatial_transform,
                                  temporal_transform=temporal_transform,
                                  sample_duration=opt.sample_duration)
                    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)
    
                    # train_epoch(epoch, data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger, label)
                    
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
                        
                        # print(outputs)
                        # # for i in range(len(outputs)):
                        # #     print(i)
                        # #     outputs[i] = max(0,outputs[i])
    
                        # # loss = binary_cross_entropy(targets, outputs)#, targets[0])
                        # print(outputs)
                        # print(targets)
                        # loss = criterion(outputs.t(), targets[0])
                        
                        # loss = criterion(outputs, targets)
                        
                        print(outputs.t())
                        # print(torch.max(targets, 1)[1])
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
                        
                        # acc = calculate_accuracy_single_target(outputs, targets)
                        
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
                # print(i)
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