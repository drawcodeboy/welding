import torch

import argparse
import time, sys, os

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np

from utils import evaluate
from models import load_model
from datasets import load_dataset

def main():
    # Device Setting
    device = 'cuda:1'
    print(f"device: {device}")
    
    # Load Dataset
    test_ds = load_dataset(dataset='welding', mode='train')
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=2,
                                          shuffle=True)
    
    # Load Model
    model = load_model(name='DNN').to(device)
    ckpt = torch.load('saved/dnn.welding.886epochs.pth',
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    x, label = next(iter(test_dl))
    x = x.to(device)
    
    output = model(x)
    
    output = output[0].cpu().detach().numpy()
    label = label[0].cpu().detach().numpy()
    
    cs_output = CubicSpline(output[:, 0], output[:, 1])
    x = np.arange(output[:, 0].min(), output[:, 0].max(), 0.001)
    y = cs_output(x)
    plt.plot(x, y, label='Predict', zorder=-1)
    
    cs_label = CubicSpline(label[:, 0], label[:, 1])
    x = np.arange(label[:, 0].min(), label[:, 0].max(), 0.001)
    y = cs_label(x)
    plt.plot(x, y, label='Real', zorder=-2)
    
    color_dict = [
        'red',
        'orange',
        'yellow',
        'green',
        'blue',
        'darkblue',
        'blueviolet',
        'violet',
        'plum'
    ]
    
    for i in range(0, 9):
        x, y = [output[i, 0], label[i, 0]], [output[i, 1], label[i, 1]]
        # plt.plot(x, y, color=color_dict[i], linewidth=1, linestyle='dashed')
        plt.scatter(output[i, 0], output[i, 1], color=color_dict[i])
        plt.scatter(label[i, 0], label[i, 1], color=color_dict[i])
    
    # plt.plot(label[:, 0], label[:, 1], label='Real')
    # plt.plot(output[:, 0], output[:, 1], label='Predict')
    plt.legend()
    
    plt.savefig('test.png', dpi=500)
    
    print(output.shape, label.shape)

if __name__ == '__main__':
    main()