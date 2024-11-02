import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from model_VGG import *
from model_ResNet import *

DEVICE = torch.device('cuda')
# DEVICE = torch.device('cpu')

transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# path_test = 'face_images/FER2013/PrivateTest'
path_test = 'face_images/FaceEmotion/test'
data_test = torchvision.datasets.ImageFolder(root=path_test, transform=transforms)
test_set = DataLoader(dataset=data_test, batch_size=32, shuffle=False)


# 加载模型
model = torch.load("model/model_vgg_FE_epo_1000_lr_00001_batch_64_acc_1.pkl", map_location=torch.device('cpu'))
# model = torch.load("model/model_cnn.pkl")
model.to(device=DEVICE)
model.eval()

# 初始化每个类别的统计变量
class_correct = list(0. for _ in range(len(data_test.classes)))
class_total = list(0. for _ in range(len(data_test.classes)))

# 测试模型
with torch.no_grad():
    for images, labels in test_set:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 打印每个类别的准确率
for i in range(len(data_test.classes)):
    print('类别 {} 的准确率: {:.2f}%'.format(
        data_test.classes[i], 100 * class_correct[i] / class_total[i]))


correct = 0
total_samples = 0
with torch.no_grad():
    for i, (x, y) in tqdm(enumerate(test_set)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        total_samples += x.size(0)


print('模型的总正确率为{:.2f}%'.format(100. * correct / total_samples))