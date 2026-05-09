import copy
import time
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import AlexNet
import torch.nn as nn
import pandas as pd

def train_val_data_progress():
    train_data = FashionMNIST(
    root='../data',
    train=True,
    transform=transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor()
    ]),
    download=False)

    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_dataloader=Data.DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    val_dataloader=Data.DataLoader(
        dataset=val_data,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    return train_dataloader,val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epoch):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    criterion=nn.CrossEntropyLoss()
    model=model.to(device)
    best_model_wts=copy.deepcopy(model.state_dict())

    best_acc=0.0
    train_loss_all=[]
    val_loss_all=[]
    train_acc_all=[]
    val_acc_all=[]
    since=time.time()

    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch,num_epoch-1))
        print("--"*10)

        train_loss=0.0
        train_corrects=0
        val_loss=0
        val_corrects=0
        train_num=0
        val_num=0
        rel_loss=0
        #for循环循环了几轮拿什么参考的
        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            model.train()
            output=model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            loss=criterion(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*b_x.size(0)
            train_corrects+=torch.sum(pre_lab==b_y.data)
            train_num+=b_x.size(0)
            print("epoch{}-{} train loss{:.4f}".format(epoch,step,loss))
           
        for step,(b_x,b_y) in enumerate(val_dataloader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            model.eval()
            output=model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            loss=criterion(output,b_y)
            val_loss+=loss.item()*b_x.size(0)
            val_corrects+=torch.sum(pre_lab==b_y.data)
            val_num+=b_x.size(0)

        
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item() /train_num)

        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item() /val_num)

        print("{} train loss{:.4f} train acc{:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} train loss{:.4f} train acc{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc =val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())

    time_use=time.time()-since
    print("训练和验证消耗的时间{:0f}m{:0f}s".format(time_use//60,time_use%60))

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts,"./best_model.pth")
#这句话干什么用的
    train_process =pd.DataFrame(
        data={
            "epoch":range(num_epoch),
            "train_loss_all":train_loss_all,
            "val_loss_all":val_loss_all,
            "train_acc_all":train_acc_all,
            "val_acc_all":val_acc_all,
        }
    )
    return train_process

def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__=='__main__':
    AlexNet=AlexNet()
    train_data,val_data=train_val_data_progress()
    train_process=train_model_process(AlexNet,train_data,val_data,num_epoch=20)
    matplot_acc_loss(train_process)
