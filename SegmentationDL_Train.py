import math

from sklearn.metrics import f1_score, precision_score, recall_score
from Models.DL.Net import ConvolutionNet
from Models.DL.Dataloader import shuffle_split 

import torch
import torch.nn as nn

from Models.DL.Dataloader import SegDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--device', type = str, default="cuda", help='using CUDA for training')

args = parser.parse_args()
if args.device == "cuda" and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True # 启用CuDNN的自动调整功能，以提高深度神经网络的训练性能
    device = torch.device('cuda')
elif args.device == "mps": # Apple Metal Performance Shaders 使用
    device = torch.device('mps')
else: # CPU
    device = torch.device('cpu')

def train():
    os.makedirs("./output", exist_ok=True)
    if True:
        shuffle_split('./Resorce/TrainDataSet/total.txt', './Resorce/TrainDataSet/train.txt', './Resorce/TrainDataSet/test.txt')

    trainData = SegDataset(txt_path="train.txt", transforms=transforms.ToTensor())
    trainDataLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testData = SegDataset(txt_path="test.txt", transforms=transforms.ToTensor())
    testDataLoader = DataLoader(testData, batch_size=args.batch_size, shuffle=True, num_workers=0)

    net = ConvolutionNet()

    print(f'training on {torch.cuda.get_device_name(0)}')
    net.to(device=device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6], 0.1)
    loss_func = nn.BCELoss()

    for epoch in range(args.epochs):
        # training-----------------------------------
        net.train()
        train_loss = 0
        train_acc = 0
        for batch, (batch_x, batch_y) in enumerate(trainDataLoader):
            
            batch_x, batch_y = Variable(batch_x.to(device)), Variable(batch_y.to(device))

            out = net(batch_x)  
            out = torch.squeeze(out, dim=1) 
            loss = loss_func(out, batch_y.float())
            train_loss += loss.item()
            pred = (out > 0.5).float() 
            train_correct = (pred == batch_y).sum() / (128 * 128)
            train_acc += train_correct.item()
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                    % (epoch + 1, args.epochs, batch, math.ceil(len(trainData) / args.batch_size),
                        loss.item(), train_correct.item() / len(batch_x)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        print('Train Loss: %.6f, Acc: %.3f' % (train_loss / (math.ceil(len(trainData)/args.batch_size)),
                                                train_acc / (len(trainData))))
        

        # evaluation--------------------------------
        net.eval()
        eval_loss = 0
        eval_acc = 0
        y_true = []
        y_pred = []
        for batch_x, batch_y in testDataLoader:

            batch_x, batch_y = Variable(batch_x.to(device)), Variable(batch_y.to(device))

            y_true.extend(batch_y.cpu().numpy().flatten())
            out = net(batch_x)
            out = torch.squeeze(out, dim=1)
            loss = loss_func(out, batch_y.float())
            eval_loss += loss.item()
            pred = (out > 0.5).float()
            y_pred.extend(pred.cpu().numpy().flatten())
            num_correct = (pred == batch_y).sum() / (128 * 128)
            eval_acc += num_correct.item()
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(testData)/args.batch_size)),
                                                eval_acc / (len(testData))))
        print('查准率：', precision_score(y_true, y_pred, average='micro'))
        print('查全率：', recall_score(y_true, y_pred, average='macro'))
        print('F1：', f1_score(y_true, y_pred, average='weighted'))

        if (epoch + 1) % 1 == 0:
            torch.save(net.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')

if __name__ == '__main__':
    train()