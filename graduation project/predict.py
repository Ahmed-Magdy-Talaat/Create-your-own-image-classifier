import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
import torch
from torch import nn, optim
import futility
import futility
import fmodel

parser = argparse.ArgumentParser(
    description = 'Parser for predict.py'
)

parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
device = args.gpu

path = args.checkpoint

def main():
    model=fmodel.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
        
    prob = fmodel.predict(path_image, model, number_of_outputs, device)
    labels = [cat_to_name[str(index + 1)] for index in np.array(prob[1][0])]
    prob = np.array(probabilities[0][0])
    g=0
    while g < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], prob[i]))
        g += 1
    print("Finished Predicting!")

    
if __name__== "__main__":
    main()