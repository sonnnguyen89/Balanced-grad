#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:22:52 2019

@author: son
"""

from __future__ import print_function 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot
import shelve
dtype = torch.cuda.FloatTensor

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        #self.bn1 = torch.nn.BatchNorm2d(32,affine=False)
        self.fc1 = nn.Linear(32*14*14, 10)
        self.dO = torch.zeros(N,H,14,14).type(dtype)


    def forward(self, x,kind):
        #x = self.bn1(self.pool(self.conv1(x)))
        x = self.pool(self.conv1(x))
        if kind ==0:
            x = F.relu(x)
            self.dO.data.zero_()
            self.dO = (x > 0).type(dtype)
        elif kind == 1:
            x = x*self.dO
        else:
            x = F.relu(x)
        x = x.view(-1, 32 * 14 * 14)
        x = self.fc1(x)
        return x


def generate_t(labels):
    result = torch.zeros((len(labels),10))
    for n in range(len(labels)):
        result[n][labels[n]] = 1
    return result


def OR(y,t):
    #y: current output
    #t: correct class output (0 or 1)
    #treat all y as correct class
    y_1_n = (y >= 1).type(dtype)* y
    y_1_p = (y < 1).type(dtype)
    y_1 = (y_1_n + y_1_p) * t  #keep real correct class
    
    #treat all y as mis class
    y_0_n = (y <= 0).type(dtype)* y
    y_0_p = (y > 0).type(dtype) * 0
    y_0 = (y_0_n+y_0_p) * (1-t)
    return y_1 + y_0

def son_OR(y,t):
    #y: current output
    #t: correct class output (0 or 1)
    #treat all y as correct class
    Nit = 3
    z = y - t
    ap = z.mean(1).view([-1,1])
    for it in range(Nit):
        d1 = z-ap
        d2 = (z < ap).type(dtype)
        di = d1*d2*(1-t)
        d2 = (z > ap).type(dtype)
        dii = d1*d2*t
        d = di + dii
        ap = (z-d).mean(1).view([-1,1])
    return t + ap + d

def find_max(M1,M2,M3):
    dec = torch.tensor([-1]*4).type(dtype)
    alpha = torch.tensor(1).type(dtype)
    T1 = (2/len(M1))*(M1*M3).sum() #F1
    T2 = (2/len(M1))*(M2*M3).sum() #F2
    T3 = (2/len(M1))*(M1*M1).sum()  #T4
    T4 = (2/len(M1))*(M1*M2).sum() #T5
    T5 = (2/len(M1))*(M2*M2).sum()  #T6

    a = T1*T1*T4 - T1*T2*T3 
    b = T1*T1*T5 - T2*T2*T3
    #print(a.item())
    c = T1*T2*T5 - T4*T2*T2
    

    dec[0] = T1*T1/T3 #only update input
    dec[1] = T2*T2/T5 #only update output
    delta = b**2 - 4*a*c
    if delta.item() >= 0:
        rt1 = -(b - torch.sqrt(delta))/(2*a)
        rt2  =-(b + torch.sqrt(delta))/(2*a)
        if rt1 >= 0:
            dec[2] = (T2 + rt1*T1).pow(2)/(rt1.pow(2)*T3+2*rt1*T4+T5)
        if rt2>= 0:
            dec[3] =  (T2 + rt2*T1).pow(2)/(rt2.pow(2)*T3+2*rt2*T4+T5)
    max_val = torch.max(dec)
#    print(max_val.item())
    if max_val.item() > 0:
        if torch.equal(max_val,dec[0]):
            f=1
            z = T1/T3
        elif torch.equal(max_val,dec[1]):
            f=2
            z = T2/T5
        else:
            f=3
            if torch.equal(max_val,dec[2]):
                alpha = torch.sqrt(rt1)
            else:
                alpha = torch.sqrt(rt2)
            
            z = (alpha*T1 + T2/alpha)/(T3*alpha.pow(2) + 2*T4 + T5/alpha.pow(2))
    else:
            print('shit happen')
            z,f = torch.tensor(0.001).type(dtype),3     
    return z,alpha,f
 
        
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N,H, M, kernel_size   = 500, 32, 10, 5
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0,), (1,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=N,
                                         shuffle=False)

net = Net()
net.cuda()
torch.manual_seed(13)
#net.double()
net.conv1.weight.data = torch.randn((32,1,5,5)).type(dtype)/200
net.fc1.weight.data = torch.randn((10,32*14*14)).type(dtype)/200
net.conv1.bias.data.zero_().add_(0.1)
net.fc1.bias.data.zero_().add_(0.1)
#optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
both_side = 0
all_time = 0
first_i = 0
MSE = []
Pe_train = []
Pe_test = []
z1 = []
z2 = []

B1 = 0
Xd = 1
first_time = 1
eps = 1e-10
num_backtrack = 0
for epoch in range(1):
    for i, data in enumerate(trainloader):
        images, labels = data
        t = Variable(generate_t(labels),requires_grad = 0).type(dtype)
        x = Variable(images).type(dtype)
        y_pred = net(x,0)
        y = son_OR(y_pred,t)
        #y = t
    # Compute and print loss.
        loss = ((y - y_pred)**2).sum()/y.size()[0]
        dy1 = y - y_pred
        net.zero_grad()
        loss.backward()
        
        #save all weight, bias, and gradient
        gc,gcb = -net.conv1.weight.grad.data.clone(),-net.conv1.bias.grad.data.clone()
        wc,bc = net.conv1.weight.data.clone(),net.conv1.bias.data.clone()
        go,gob = -net.fc1.weight.grad.data.clone(),-net.fc1.bias.grad.data.clone()
        wo,bo = net.fc1.weight.data.clone(),net.fc1.bias.data.clone()
        
        old_wc,old_bc = wc.clone(), bc.clone()
        old_wo,old_bo = wo.clone(), bo.clone()
        
        Xn = (gc*gc).sum() + (gcb*gcb).sum()+(go*go).sum()+(gob*gob).sum()
        B1 = Xn/(Xd+eps)
        Xd = Xn
            
        if first_time == 1:
            Pwc = gc.clone()
            Pbc = gcb.clone()
            Pwo = go.clone()
            Pbo = gob.clone()
            first_time = 0
        else:
            Pwc.mul_(B1).add_(gc)
            Pbc.mul_(B1).add_(gcb)
            Pwo.mul_(B1).add_(go)
            Pbo.mul_(B1).add_(gob)
            
#        
        net.conv1.weight.data,net.conv1.bias.data = Pwc,Pbc
        dydz1 = net(x,1) #m1
#
        net.conv1.weight.data,net.conv1.bias.data = wc,bc
        net.fc1.weight.data,net.fc1.bias.data = Pwo,Pbo
        dydz2 = net(x,2) #m2
        
        net.fc1.weight.data,net.fc1.bias.data = wo,bo

        z,a,f = find_max(dydz1,dydz2,dy1)
        z= z.abs()
#        z,a,f = 0.001,1,3
        #print(z.item(),a.item(),f)
        if f == 1:
            net.conv1.weight.data = wc + z*Pwc
            net.conv1.bias.data = bc + z*Pbc
            z1.append(z.item())
            z2.append(0)
        elif f==2:
            net.fc1.weight.data = wo + z*Pwo
            net.fc1.bias.data = bo + z*Pbo
            z1.append(0)
            z2.append(z.item())
        else:
            net.conv1.weight.data = wc + z*a*Pwc
            net.conv1.bias.data = bc + z*a*Pbc
            net.fc1.weight.data = wo + z/a*Pwo
            net.fc1.bias.data = bo + z/a*Pbo
            both_side +=1
            z1.append((z*a).item())
            z2.append((z/a).item())
        all_time +=1
        #lr = 1e-3
        y_pred = net(x,0)
        y = son_OR(y_pred,t)
        new_loss = ((y - y_pred)**2).sum()/len(x)
        if new_loss.item() > loss.item():
            num_backtrack +=1
            B1=0
            first_time = 1
            Xd=1
            net.conv1.weight.data,net.conv1.bias.data = old_wc.clone(),old_bc.clone()
            net.fc1.weight.data,net.fc1.bias.data = old_wo.clone(),old_bo.clone()
        
#        optimizer.step()
        #print(epoch,i,loss.item())
        MSE.append(loss.item())
            
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images).type(dtype),2)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    Pe_test_current = 1- correct.item() / total
    Pe_test.append(Pe_test_current)
    print(epoch,Pe_test_current)
#    correct = 0
#    total = 0
#    for data in trainloader:
#        images, labels = data
#        outputs = net(Variable(images).type(dtype),2)
#        _,predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted.cpu() == labels).sum()
#    Pe_train_current = 1-  correct.item()/total
#    Pe_train.append(Pe_train_current)
#    print(epoch, Pe_train_current)
#print('both side',both_side/all_time)
#pyplot.plot(Pe_test)
#filename='MNIST_July_3rd.out'
#my_shelf = shelve.open(filename)
#my_shelf['Pe_test_BL'] = globals()['Pe_test']
#my_shelf['Pe_train_BL'] = globals()['Pe_train']
#my_shelf.close()
