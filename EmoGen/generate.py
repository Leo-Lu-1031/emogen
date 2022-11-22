import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from math import log
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import torchviz

try:
    import pretty_midi
    import progress
    import torchmetrics
except:
    os.system("pip install pretty_midi progress torchmetrics")
    import pretty_midi
    import progress
    import torchmetrics
    
import time
import sys

print_log = False
if print_log:
    sys.stdout=open("log.txt",'w')
    sys.stdout.close()
    sys.stdout=open("log.txt",'a')

import time
import optparse
import utils as utils
import config as config
from config import device
from dataloader import Dataset
from model import featureGen
from sequence import NoteSeq, EventSeq, ControlSeq
print("Prereqs imported")

import musicclass
print("MusicClass imported")
import musicgen
print("MusicGen imported")

import imgclass
print("ImgClass imported")

batch_size = config.train['batch_size']
# loss_function = nn.CrossEntropyLoss().to(device)
loss_function = nn.MSELoss(reduction='none').to(device)
cosine_loss = nn.CosineEmbeddingLoss().to(device)
accuracy = torchmetrics.Accuracy().to(device)
feature_dim = musicgen.model.feature_dim * musicgen.model.gru_layers

batch_size = config.train['batch_size']

feature = torch.zeros(batch_size, feature_dim, requires_grad=True, device=device)
feature.retain_grad()
optim = optim.Adam([feature], lr = config.train['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

def generate(target, vec=None):
    global feature
    
    if vec is None: vec = 2*torch.randn(batch_size, feature_dim)
    assert vec.shape==torch.Size([batch_size, feature_dim])
    vec = vec.to(device)
    
    target = target.float()
    if len(target.shape)==1: 
        target = target.long().to(device)
        target = F.one_hot(target, 4).float()
    if target.shape[0]==1: target = target.repeat(batch_size, 1)
    assert target.shape==torch.Size([batch_size, 4])
    
    for i in range(1000):
        optim.zero_grad()
        loss = ((feature-vec)**2).sum()
        loss.backward()
        optim.step()
    
    for i in range(10):
        music = musicgen.forward(feature, batch_size, output_type = "softmax")
        # print(music)
        music.retain_grad()
        emotion = musicclass.forward(music)
        emotion.retain_grad()

        optim.zero_grad()
        loss = loss_function(emotion,target).sum(dim=1)
        # print(loss.sum(dim=1))
        loss_ = loss.sum()
        # print(accuracy(emotion, target.argmax(dim=1)))
        
        loss_.backward()
        print(loss_.item())
        # print(emotion.T.max(dim=1).values.tolist())
        # print(emotion.mean(dim=0).tolist())
        # print(loss.item())

        # print(emotion.grad.sum().item())
        # print(music.grad.sum().item())
        # print(feature.grad.sum())
        
        optim.step()
        scheduler.step(loss_)
    
    print(loss.min().item())
    # print(emotion)
    print("Good:",emotion[loss.argmin()].tolist())
    print("Right:",target[0].tolist())
    feature = feature[loss.argmin()].unsqueeze(dim=0).repeat(batch_size,1)
    # print(feature)
    
    music = musicgen.forward(feature, batch_size, output_type = "index")
    # print(music)
    music_hot = F.one_hot(music, config.model['event_dim']).float()
    new_emo = musicclass.forward(music_hot)
    loss = loss_function(new_emo, target).sum(dim=1)
    # print("Final: ", new_emo[loss.argmin()].tolist())
    music = music.T[loss.argmin()].repeat(batch_size, 1).T
    experiment = F.one_hot(music, config.model['event_dim']).float()
    # print(experiment.shape)
    print("Final:", musicclass.forward(experiment)[0].tolist())
    return feature, music, emotion

def validate():
    events, _, label_batch = dataset.paired_batches(batch_size, 200, 10, True)
    label_batch = torch.tensor(label_batch).to(device).long()
    events = F.one_hot(torch.LongTensor(events), config.model['event_dim']).float().to(device)
    fwd = musicclass.forward(events)
    loss = accuracy(fwd, label_batch)
    return loss

def process(img_path):
    if type(img_path) is not list: img_path = [img_path]
    emotion_img = imgclass.forward(img_path, output_type='softmax')
    print("Image emotion:", emotion_img.tolist())
    feature, final, emotion = generate(emotion_img)
    # print(emotion.mean(dim=0).tolist())
    # final = torch.randint(0,config.model['event_dim'],final.shape)
    return feature, final, emotion, emotion_img
    
# for i in range(20): print(validate())
   
if __name__ != '__main__': exit(0)

# filename = input("Filename of your picture (Must be under same dir as this script): ")
filename = 'Pot.jpg'

_,final,_,_ = process(filename)
musicgen.save(final, filename)
    
# generate(3*torch.ones(batch_size))
# generate(torch.randn(batch_size, feature_dim), torch.zeros(batch_size))
# generate(torch.randn(batch_size, feature_dim), torch.zeros(batch_size))