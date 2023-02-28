import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image
import futility
import fmodel

parser = argparse.ArgumentParser(
    description = 'Parser for train.py'
)
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--gpu', action="store", default="gpu")

args = parser.parse_args()
where = args.data_dir
path = args.save_dir
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
lr = args.learning_rate
struct = args.arch
dropout = args.dropout

if torch.cuda.is_available() and power == 'gpu':
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
def main:
    #trainning model
# Train the classifier layers using backpropagation using the pre-trained network to get the features
# Track the loss and accuracy on the validation set to determine the best hyperparameters

def main():
    trainloader, validloader, testloader, train_data = futility.load_data(where)
    model, criterion = fmodel.setup_network(struct,dropout,hidden_units,lr,power)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
    # Train Model
    steps = 0
    running_loss = 0
    print_every = 5
    print("--Training starting--")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            # Move input and label tensors to the default device
            if torch.cuda.is_available() and power =='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            steps += 1

            optimizer.zero_grad()

            #Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')

                        lgps = model.forward(inputs)
                        batch_loss = criterion(lgps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(lgps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    #fmodel.save_checkpoint(traindata,model,path,struct,hidden_units,dropout,lr)
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :struct,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    mmm=1000
    ggg=mmm+100
    print("Saved checkpoint!")
if __name__== "__main__":
    main()