import math

from sklearn.metrics import f1_score, precision_score, recall_score
from Models.DL.Net import USegNet, SimpleCNN
from Models.DL.Dataloader import shuffle_split 

import torch
import torch.nn as nn

from Models.DL.Dataloader import SegDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

import os
import argparse
import time

showLog = True
learning_rate = 0.1

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--device', type = str, default="cuda", required=True, help='use cuda or other device during training')
parser.add_argument('--model', type = str, default="USegNet", required=True, help='choose model to train')

args = parser.parse_args()
if args.device == "cuda" and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True # 启用CuDNN的自动调整功能，以提高深度神经网络的训练性能
    device = torch.device('cuda')
elif args.device == "mps": # Apple Metal Performance Shaders 
    device = torch.device('mps')
else: # CPU
    device = torch.device('cpu')

def train():
    start = time.time()
    os.makedirs("./output", exist_ok=True)
    shuffle_split('./Resorce/TrainDataSet/total.txt', './Resorce/TrainDataSet/train.txt', './Resorce/TrainDataSet/test.txt')

    trainData = SegDataset(txt_path="train.txt", transforms=transforms.ToTensor())
    trainDataLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testData = SegDataset(txt_path="test.txt", transforms=transforms.ToTensor())
    testDataLoader = DataLoader(testData, batch_size=args.batch_size, shuffle=True, num_workers=0)

    net = SimpleCNN() if args.model == "SCNN" else USegNet()

    print(f'training on {device}, using {net.__class__.__name__}')
    net.to(device=device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40], 0.1)
    loss_func = nn.CrossEntropyLoss()

    mem_use = []
    acc = []
    pre = []
    rec = []
    f1 = []
    for epoch in range(args.epochs):
        # training-----------------------------------
        epochStart = time.time()
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
            if showLog:
                print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                    % (epoch + 1, args.epochs, batch + 1, math.ceil(len(trainData) / args.batch_size),
                        loss.item(), train_correct.item() / len(batch_x)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        if showLog:
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
        acc.append(eval_acc / (len(testData)))
        pre.append(precision_score(y_true, y_pred, average='micro'))
        rec.append(recall_score(y_true, y_pred, average='macro'))
        f1.append(f1_score(y_true, y_pred, average='weighted'))
        if showLog:
            print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(testData)/args.batch_size)),
                                                    acc[-1]))
            print('查准率：', pre[-1])
            print('查全率：', rec[-1])
            print('F1：', f1[-1])

        epochEnd = time.time()
        print(f'epoch {epoch + 1}: ' + 'epoch time: %.2f s' % (epochEnd - epochStart))
        # 显存占用
        mem_use.append(torch.cuda.memory_allocated(device=device))
        print('Memory used: %.2f MB' % (mem_use[-1] / 1024 / 1024))

        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), f'output/dl/{args.model}_params_' + str(epoch + 1) + '.pth')
    
    end = time.time()
    print(f'Finished training, time: {end - start}s')
    print('Max memory used: %.2f MB' % (max(mem_use) / 1024 / 1024))
    print('Average memory used: %.2f MB' % (sum(mem_use) / len(mem_use) / 1024 / 1024))
    torch.save(net.state_dict(), f'output/dl/{args.model}_params_final_{learning_rate}.pth')
    return acc, pre, rec, f1, mem_use, end - start

if __name__ == '__main__':
    acc, pre, rec, f1, mem_use, total_time = train()
    from datetime import datetime

    now = datetime.now()
    # 格式化日期和时间
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(f'dl-{args.model}-train-log.txt', 'a') as f:
        f.write(f'{formatted_now} training on {device}, using {args.model}\n')
        f.write(f'Finished training, time: {total_time}s\n')
        f.write('Max memory used: %.2f MB\n' % (max(mem_use) / 1024 / 1024))
        f.write('Average memory used: %.2f MB\n' % (sum(mem_use) / len(mem_use) / 1024 / 1024))
        f.write('best acc: ' + str(max(acc)) + '\n')
        f.write('best pre: ' + str(max(pre)) + '\n')
        f.write('best rec: ' + str(max(rec)) + '\n')
        f.write('best f1: ' + str(max(f1)) + '\n')

    # 绘制训练过程中的指标变化曲线
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(acc, label='accuracy')
    plt.plot(pre, label='precision')
    plt.plot(rec, label='recall')
    plt.plot(f1, label='f1')
    plt.legend()
    plt.savefig(f'output/dl/{args.model}_metrics.png')
    plt.show()
