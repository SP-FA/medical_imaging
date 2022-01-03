import torch
import torchvision
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from Model.resnet18_torch import ResNet18
from Model.KNN_torch import KNN
from LoadData.LoadData_torch import InputImg
from torch.autograd import Variable

batchsz = 64
taskNumber = 3
size = 0.2
width = 128
epoch = 100

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([width,width]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
dataSets = InputImg('./Data', width, transform)
train_size = int(len(dataSets) * (1 - size))
test_size  = int(len(dataSets) - train_size)
train, test = torch.utils.data.random_split(dataSets, [train_size, test_size])

def create_class_weight(labels_dict, total):
    class_weight = []
    for i in labels_dict:
        class_weight.append(total / i)
    return torch.tensor(class_weight)

labels_dict = [0, 0, 0]
for x, y in train:
    labels_dict[y-1] += 1
cls_wght = create_class_weight(labels_dict, train_size)

train = DataLoader(train, batch_size = batchsz, shuffle = True)
test  = DataLoader(test , batch_size = batchsz, shuffle = True)
#以上是加载数据

def one_hot(y):
    lst = []
    for i in y:
        if (i == 0):
            lst.append([1, 0, 0])
        elif (i == 1):
            lst.append([0, 1, 0])
        else:
            lst.append([0, 0, 1])
    lst = torch.Tensor(lst)
    return lst

device = torch.device('cuda')
model = ResNet18(taskNumber).to(device)

criteon = nn.CrossEntropyLoss(weight = cls_wght).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)

for i in range(epoch):
    model.train()
    for x, y in train:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        logits = logits.to(device)
        # loss: tensor scalar
        # loss = criteon(logits, y)
        res = KNN(logits, y, logits, y, 7).get_res()
        res = one_hot(res)
        res = torch.Tensor(res)
        res = res.to(device)
        res = Variable(res, requires_grad=True)
        loss = criteon(res, y)
        print(loss)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print(i, 'loss:', loss.item())

    model.eval()
    with torch.no_grad():
        # test
        total_correct = 0
        total_num = 0
        for x, label in train:
            x = x.to(device)
            label = label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
        acc = total_correct / total_num
        print(i, 'train acc:', acc)

    with torch.no_grad():
        # test
        total_correct = 0
        total_num = 0
        for x, label in test:
            x = x.to(device)
            label = label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
        acc = total_correct / total_num
        print(i, 'test acc:', acc)