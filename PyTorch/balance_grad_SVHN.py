from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import shelve
dtype = torch.cuda.FloatTensor
N,H, M, kernel_size   = 500, 32, 10, 5   
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
#        self.bn1 = torch.nn.BatchNorm2d(32,affine=False)
        self.fc1 = nn.Linear(32*16*16, 10)
        self.dO = Variable(torch.zeros(N,H,16,16).type(dtype),requires_grad = False)

    def forward(self, x,kind):
        x = self.pool(self.conv1(x))
        if kind == 0:
            #involve calculating dO
#            tem = x.clone()
#            x = self.bn1(x)
            x = F.relu(x)
            # x = F.sigmoid(x)
#            tem = tem - (tem.mean(0).mean(1).mean(1)).view(32,1,1)
#            tem_std = (tem.pow(2).mean(0).mean(1).mean(1)+self.bn1.eps).sqrt().view(32,1,1)
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
def generate_t(labels):
    result = torch.zeros((len(labels),10))
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
def find_max(dydz1,dydz2,dy1):
    dec = torch.tensor([-1]*4).type(dtype)
    alpha = torch.tensor(1).type(dtype)
    F1 = (dydz1*dy1).sum()
    F2 = (dydz2*dy1).sum()
    T1 = F1**2
    T2 = 2*F1*F2
    T3 = F2**2
    T4 = (dydz1*dydz1).sum()
    T5 = 2*(dydz1*dydz2).sum()
    T6 = (dydz2*dydz2).sum()
    a = T1*T5 - T2*T4
    b = T1*T6 - T3*T4  #use even delta
    c = T2*T6 - T3*T5
    dec[0] = T1/T4
    dec[1] = T3/T6
    delta = b**2 - a*c
    if delta.item() >= 0:
        rt1 = -(b - torch.sqrt(delta))/a
        rt2  =-(b + torch.sqrt(delta))/a
        if rt1 >= 0:
            dec[2] = (((rt1*dydz1 + dydz2)*dy1).sum())**2/((rt1*dydz1 + dydz2)*(rt1*dydz1 + dydz2)).sum()
        if rt2>= 0:
            dec[3] =  (((rt2*dydz1 + dydz2)*dy1).sum())**2/((rt2*dydz1 + dydz2)*(rt2*dydz1 + dydz2)).sum()
    max_val = torch.max(dec)
    # print(dec)
    if max_val.item() > 0:
        if torch.equal(max_val,dec[0]):
            f=1
            z = F1/T4
        elif torch.equal(max_val,dec[1]):
            f=2
            z = F2/T6
        else:
            f=3
            if torch.equal(max_val,dec[2]):
                alpha = torch.sqrt(rt1)
            else:
                alpha = torch.sqrt(rt2)
            z = (alpha*F1+F2/alpha)/(alpha**2*T4 + T5 + T6/(alpha**2))
    else:
            print('shit happen')
            z,f = torch.tensor(0.001).type(dtype),3
    return z,alpha,f
class moment():
    def __init__(self,gc,gcb,go,gob):
        self.s_gc, self.r_gc = gc, gc*gc
        self.s_gcb, self.r_gcb = gcb, gcb*gcb
        self.s_go, self.r_go = go, go*go
        self.s_gob, self.r_gob = gob, gob*gob
        self.ro1, self.ro1ex = 0.9,0.9
        self.ro2,self.ro2ex = 0.999,0.999
    def update(self,gc,gcb,go,gob):
        self.s_gc,  self.r_gc  = self.ro1 * self.s_gc +  (1-self.ro1)*gc, self.ro2 *  self.r_gc + (1-self.ro2)*(gc*gc)
        self.s_gcb, self.r_gcb = self.ro1 * self.s_gcb + (1-self.ro1)*gcb, self.ro2 * self.r_gcb + (1-self.ro2)*(gcb*gcb)
        self.s_go,  self.r_go  = self.ro1 * self.s_go +  (1-self.ro1)*go, self.ro2 *  self.r_go + (1-self.ro2)*(go*go)
        self.s_gob, self.r_gob = self.ro1 * self.s_gob + (1-self.ro1)*gob, self.ro2 * self.r_gob + (1-self.ro2)*(gob*gob)
        gc  = (self.s_gc/(1-self.ro1ex))/ (torch.sqrt((self.r_gc/(1-self.ro2ex)))+0.00001)
        gcb = (self.s_gcb/(1-self.ro1ex))/(torch.sqrt((self.r_gcb/(1-self.ro2ex)))+0.00001)
        go  = (self.s_go/(1-self.ro1ex))/ (torch.sqrt((self.r_go/(1-self.ro2ex)))+0.00001)
        gob = (self.s_gob/(1-self.ro1ex))/(torch.sqrt((self.r_gob/(1-self.ro2ex)))+0.00001)
        self.ro1ex *= self.ro1
        self.ro2ex *= self.ro2
        return gc,gcb,go,gob
      
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0, 0), (1, 1))])
torch.manual_seed(1)
trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N,
                                          shuffle=True)

testset = torchvision.datasets.SVHN(root='./data', split = 'test',
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
optimizer = torch.optim.Adam(net.parameters())

both_side = 0
all_time = 0
Pe_test = []

B1 = 0
Xd = 1
first_time = 1
eps = 1e-10
num_backtrack = 0
for epoch in range(1000):
    for i, data in enumerate(trainloader):

        images, labels = data
    #for t in range (300):
        t = Variable(generate_t(labels),requires_grad = 0).type(dtype)
        x = Variable(images).type(dtype)
        y_pred = net(x,0)
        y = son_OR(y_pred,t)
       # y = t

        loss = ((y - y_pred)**2).sum()/len(x)
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
        #####experiment momentum#############################
#        if first_i ==0:
#            m_g = moment(gc,gcb,go,gob)
#            first_i = 1
#        else:
#            gc,gcb,go,gob =m_g.update(gc,gcb,go,gob)


        ###########################################################
#        
        Xn = (gc*gc).sum() + (gcb*gcb).sum()+(go*go).sum()+(gob*gob).sum()
        B1 = Xn/(Xd+eps)
        Xd = Xn
#        if B1>2 or i==140:
##        if i%10==0:
#            B1=0
#            first_time = 1
#            Xd=1
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
        elif f==2:
            net.fc1.weight.data = wo + z*Pwo
            net.fc1.bias.data = bo + z*Pbo
        else:
            net.conv1.weight.data = wc + z*a*Pwc
            net.conv1.bias.data = bc + z*a*Pbc
            net.fc1.weight.data = wo + z/a*Pwo
            net.fc1.bias.data = bo + z/a*Pbo   
            both_side += 1
        all_time +=1
        
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
        print(epoch,i,loss.item())

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images).type(dtype),0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    current_Pe_test = 1-correct.item()/total
    print('Pe test', current_Pe_test)
    Pe_test.append(current_Pe_test)
#
#filename='SVHN_July_3rd.out'
#my_shelf = shelve.open(filename)
#my_shelf['Pe_test_BL'] = globals()['Pe_test']
#my_shelf.close()

print('both side: ',both_side/all_time)