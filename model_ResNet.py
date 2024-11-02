import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import time

tic = time.time()
BATCH_SIZE = 64
LR = 0.0001
EPOCH = 200
# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda')


transforms_train = transforms.Compose([
    transforms.Grayscale(),#使用ImageFolder默认扩展为三通道，重新变回去就行
    transforms.RandomHorizontalFlip(),#随机翻转
    transforms.ColorJitter(brightness=0.5, contrast=0.5),#随机调整亮度和对比度
    transforms.ToTensor()
])
transforms_valid = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])


# fer2013Plus数据集
# path_train = 'face_images/fer2013Plus/train'
# path_valid = 'face_images/fer2013Plus/test'

# FaceEmotion 数据集
path_train = 'face_images/FaceEmotion/train'
path_valid = 'face_images/FaceEmotion/test'


data_train = torchvision.datasets.ImageFolder(root=path_train,transform=transforms_train)
data_valid = torchvision.datasets.ImageFolder(root=path_valid,transform=transforms_valid)

train_set = torch.utils.data.DataLoader(dataset=data_train,batch_size=BATCH_SIZE,shuffle=True)
valid_set = torch.utils.data.DataLoader(dataset=data_valid,batch_size=BATCH_SIZE,shuffle=False)


model = torchvision.models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=7, bias=True)

model.to(DEVICE)
optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
            #optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


train_loss = []
train_ac = []
valid_loss = []
valid_ac = []
y_pred = []


def train(model,device,dataset,optimizer,epoch):
    model.train()
    correct = 0
    total_loss = 0
    total_samples = 0
    for i, (x, y) in tqdm(enumerate(dataset)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = criterion(output, y)
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        loss.backward()
        optimizer.step()

    train_ac.append(correct / total_samples)
    train_loss.append(total_loss / total_samples)

    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(epoch, total_loss / total_samples, correct,
                                                                 total_samples, 100 * correct / total_samples))

def valid(model,device,dataset):
    model.eval()
    correct = 0
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataset)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    valid_ac.append(correct / total_samples)
    valid_loss.append(total_loss / total_samples)
    print("Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(total_loss / total_samples, correct, total_samples,
                                                             100. * correct / total_samples))
    val_loss = total_loss / total_samples
    return val_loss


def RUN():
    # 早停参数
    patience = 10  # 容忍的epoch数量
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(1,EPOCH+1):
        '''if epoch==15 :
            LR = 0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        if(epoch>30 and epoch%15==0):
            LR*=0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        '''
        #尝试动态学习率
        train(model,device=DEVICE,dataset=train_set,optimizer=optimizer,epoch=epoch)
        val_loss = valid(model,device=DEVICE,dataset=valid_set)
        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # 可以在这里保存最佳模型
            torch.save(model,'model/model_res18_FE_epo_200_lr_00001_batch_64_loss_1.pkl')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break



def print_plot(train_plot, valid_plot, train_text, valid_text, ac, name):
    x = [i for i in range(1, len(train_plot) + 1)]
    plt.plot(x, train_plot, label=train_text)
    plt.plot(x[-1], train_plot[-1], marker='o')
    plt.annotate("%.2f%%" % (train_plot[-1] * 100) if ac else "%.4f" % (train_plot[-1]), xy=(x[-1], train_plot[-1]))
    plt.plot(x, valid_plot, label=valid_text)
    plt.plot(x[-1], valid_plot[-1], marker='o')
    plt.annotate("%.2f%%" % (valid_plot[-1] * 100) if ac else "%.4f" % (valid_plot[-1]), xy=(x[-1], valid_plot[-1]))
    plt.title("model_res18_FE_epo_200_lr_00001_batch_64_loss_1")
    plt.legend()


if __name__ == '__main__':
    RUN()
    toc = time.time()
    print("Times:", (toc - tic))
    print_plot(train_loss,valid_loss,"train_loss","valid_loss",False,"loss.jpg")
    # print_plot(train_ac,valid_ac,"train_ac","valid_ac",True,"ac.jpg")
    plt.savefig('result_img/resn18_result_FE_epo_200_lr_00001_batch_64_loss_1.jpg')
