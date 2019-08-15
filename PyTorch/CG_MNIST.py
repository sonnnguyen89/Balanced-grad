import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib import pyplot
import shelve
dtype = torch.cuda.FloatTensor
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0,), (1,))])
Nv,Nh, M, kernel_size   = 500, 32, 10, 5
def generate_t(labels):
    result = torch.zeros((len(labels),10)).cuda()
    for n in range(len(labels)):
        result[n][labels[n]] = 1
    return result
def OR(y,t):
    #y: current output
    #t: correct class output (0 or 1)
    #treat all y as correct class
    y_1_n = (y >= 1).type(dtype) * y
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
class Net(nn.Module):
    def __init__(self,Nh,Nv):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,Nh,5,padding=2)
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(Nh)
        self.fc1 = nn.Linear(Nh*14*14,10)
        self.dO = torch.zeros(Nv,Nh,14,14).type(dtype)
    def forward(self, x,kind):
        #0 - regular forward'
    #1 -  dO involve in forward
        #x = self.bn1(self.pool(self.conv1(x)))
        x = self.pool(self.conv1(x))  #32*5*5*28*28
        if kind ==0:
            x = F.relu(x) #500*14*14*32
            #x = F.sigmoid(x)
            self.dO.data.zero_()
            self.dO = (x > 0).type(dtype)
            #self.dO = x*(1-x)
        elif kind == 1:
            x = x*self.dO #500*32*14*14
        else:
            x = F.relu(x)  #500*14*14*32
            #x = F.sigmoid(x)
        x = x.view(-1, 32 * 14 * 14)
        x = self.fc1(x)   #32*14*14*500*10
        return x
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Nv,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=Nv,
                                         shuffle=False)
net = Net(Nh,Nv)
net.cuda()
torch.manual_seed(13)
net.conv1.weight.data = torch.randn((32,1,5,5)).type(dtype)/50
net.fc1.weight.data = torch.randn((10,32*14*14)).type(dtype)/50
net.conv1.bias.data.zero_().add_(0.1)
net.fc1.bias.data.zero_().add_(0.1)
MSE = []
Pe_train = []
Pe_test = []
Nit = 10
B1 = 0
Xd = 1
first_time = 1
eps = 1e-10
num_backtrack = 0
for it in range(1000):
    for i, data in enumerate(trainloader):
        x, labels = data
        x = x.cuda()
        labels = labels.cuda()

        #t_orin = generate_t(labels)
        #t = OR(y,t_orin) #5*500*10
        t = generate_t(labels)
        

        y = net(x,0)
        loss = (t-y).pow(2).sum()/len(x)  #500*10
        old_loss = loss.item()
        print(it,i,old_loss)
        MSE.append(old_loss)
        net.zero_grad()
        loss.backward()
        gwc = -net.conv1.weight.grad.clone()
        gbc = -net.conv1.bias.grad.clone()
        gwo = -net.fc1.weight.grad.clone()
        gbo = -net.fc1.bias.grad.clone()
        Xn = (gwc*gwc).sum() + (gbc*gbc).sum()+(gwo*gwo).sum()+(gbo*gbo).sum()
        B1 = Xn/(Xd+eps)
        Xd = Xn
        if first_time == 1:
            Pwc = gwc.clone()
            Pbc = gbc.clone()
            Pwo = gwo.clone()
            Pbo = gbo.clone()
            first_time = 0
        else:
            Pwc.mul_(B1).add_(gwc)
            Pbc.mul_(B1).add_(gbc)
            Pwo.mul_(B1).add_(gwo)
            Pbo.mul_(B1).add_(gbo)

        dy = t-y
        wc = net.conv1.weight.data.clone()
        bc = net.conv1.bias.data.clone()
        wo = net.fc1.weight.data.clone()
        bo = net.fc1.bias.data.clone()

        net.conv1.weight.data = Pwc.clone()
        net.conv1.bias.data = Pbc.clone()
        net.fc1.weight.data = Pwo.clone()
        net.fc1.bias.data = Pbo.clone()
        dydz = net(x,1)
        H = (dydz*dydz).sum()
        J = (dydz*dy).sum()
        z = J/(H+eps)
        z.abs_()
        n = 0
        while n <=10:
            net.conv1.weight.data = wc + z*Pwc
            net.conv1.bias.data = bc + z*Pbc
            net.fc1.weight.data = wo + z*Pwo
            net.fc1.bias.data = bo + z*Pbo
            y = net(x,2)
            #t = OR(y,t_orin)
            loss = (t-y).pow(2).sum()/Nv
            if loss.item() < old_loss:
                break
            else:
                z = z/2
                first_time = 1
                num_backtrack += 1
            n += 1
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
    print(Pe_test_current)
	#
    correct = 0
    total = 0
    for data in trainloader:
        images, labels = data
        outputs = net(Variable(images).type(dtype),2)
        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    Pe_train_current = 1-  correct.item()/total
    print(it,Pe_train_current)
    Pe_train.append(Pe_train_current)

#filename='MNIST_July_3rd.out'
#my_shelf = shelve.open(filename)
#my_shelf['Pe_test_CG_no_OR'] = globals()['Pe_test']
#my_shelf['Pe_train_CG_no_OR'] = globals()['Pe_train']
#my_shelf.close()

#
#pyplot.plot(Pe_test)