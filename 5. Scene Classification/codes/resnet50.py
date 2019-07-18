from tqdm import tqdm
import numpy as np
import pandas as pd
import glob
import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms import *
import torchvision.models as models
import torch.optim as optim

from utils import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.img_paths = df['img_paths'].values
        self.img_labels = df['img_labels'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        label = self.img_labels[idx]
        return (img, label)

def train(net, train_loader, num_epochs=100):
    # Train
    pbar = tqdm(total=len(train_loader), ascii=True) 

    for epoch in range(1, num_epochs+1):
        pbar.set_description("(%03d/%03d)"%(epoch, num_epochs))
        pbar.n = 0
        pbar.last_print_n = 0 
        pbar.refresh()
        for (imgs, labels) in train_loader:
            imgs = imgs.type(FloatTensor).to(device)
            labels = labels.type(LongTensor).to(device)
            optimizer.zero_grad() 
            outputs = net(imgs)
            loss   = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss":loss.item()})
            pbar.update()
    pbar.close()

def evaluate_model(net, test_loader, plot_cm=True):
    net.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for (imgs, labels) in test_loader:
            imgs = imgs.type(FloatTensor).to(device)
            labels = labels.type(LongTensor).to(device)
            outputs = net(imgs)
            values, indices = outputs.max(1)
            y_pred = indices.data.cpu().numpy()
            y_true = labels.data.cpu().numpy()
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    if plot_cm:
        plot_confusion_matrix(y_pred, y_true, labels= [i for i in range(num_classes)])
    return compute_accuracy(y_pred, y_true)

if __name__ == '__main__':
    # Prepare Data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    transform_train = Compose([transforms.ToPILImage(),
                               transforms.RandomRotation(15),
                               Resize((224, 224)),
                               RandomHorizontalFlip(p=0.5),
                               ToTensor(),
                               Normalize([0.], [1.])])
    train_dataset = Dataset(train_df, transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    transform_test = Compose([transforms.ToPILImage(),
                           Resize((224, 224)),
                           ToTensor(),
                           Normalize([0.], [1.])])
    test_dataset = Dataset(test_df, transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=150)

    # NN Model 
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(2048, num_classes)

    # Hyper-parameters
    num_epochs = 150
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # GPU Usage
    cuda = True if torch.cuda.is_available() else False
    device = "cuda" if cuda else "cpu"
    if cuda:
        net.cuda()
        criterion.cuda()
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # Train Model
    train(net, train_loader, num_epochs=num_epochs)

    train_acc = evaluate_model(net, train_loader, plot_cm=False)
    print("Train Acc", train_acc)

    # Evaluate Model
    test_acc = evaluate_model(net, test_loader, plot_cm=True)
    print("Test Accuracy", test_acc)