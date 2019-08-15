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
     transforms.Normalize((0, 0,0), (1, 1,1))])
N,H, M, kernel_size   = 500, 32, 10, 5  
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
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        #self.bn1 = torch.nn.BatchNorm2d(32,affine=False)
        self.fc1 = nn.Linear(32*16*16, 10)
        self.dO = Variable(torch.zeros(N,H,16,16).type(dtype),requires_grad = False)

    def forward(self, x,kind):
        x = self.pool(self.conv1(x))
        if kind == 0:
            #involve calculating dO
            #tem = x.clone()
            #x = self.bn1(x)
            x = F.relu(x)
            # x = F.sigmoid(x)
#            tem = tem - (tem.mean(0).mean(1).mean(1)).view(32,1,1)
            #tem_std = (tem.pow(2).mean(0).mean(1).mean(1)+self.bn1.eps).sqrt().view(32,1,1)
            # self.dO.data.zero_()
#            self.dO = (x > 0).type(dtype)/tem_std
            self.dO = (x > 0).type(dtype)
            #self.dO = x*(1-x)
        elif kind == 1:
            x = x*self.dO
        else:
            # x = self.bn1(x)
            x = F.relu(x)
            #x = F.sigmoid(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        return x
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=N,
                                         shuffle=False)
net = Net()
net.cuda()
torch.manual_seed(1)
net.conv1.weight.data = torch.randn((32,3,5,5)).type(dtype)/50
net.fc1.weight.data = torch.randn((10,32*16*16)).type(dtype)/50
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
for it in range(2000):
    for i, data in enumerate(trainloader):
        x, labels = data
        x = x.cuda()
        labels = labels.cuda()

        #t_orin = generate_t(labels)
        #t = son_OR(y,t_orin) #5*500*10
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
#        if B1>2 or i==95:
##        if i%10==0:
#            B1=0
#            first_time = 1
#            Xd=1
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
        dydz = net(x,2)
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
            y = net(x,0)
            #t = son_OR(y,t_orin)
            loss = (t-y).pow(2).sum()/N
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
#    correct = 0
#    total = 0
#    for data in trainloader:
#        images, labels = data
#        outputs = net(Variable(images).type(dtype),2)
#        _,predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted.cpu() == labels).sum()
#    Pe_train_current = 1-  correct.item()/total
#    print(Pe_train_current)
#    Pe_train.append(Pe_train_current)

#pyplot.plot(Pe_test)
#filename='cifar10_July_3rd.out'
#my_shelf = shelve.open(filename)
#my_shelf['Pe_test_CG_no_OR'] = globals()['Pe_test']
#my_shelf.close()
#
#pyplot.plot(Pe_test)