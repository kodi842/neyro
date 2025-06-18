import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Подготовка данных
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.ImageFolder(root='D:/study/neyro/5/5/data/test', transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# Загрузка модели
net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
for param in net.parameters():
    param.requires_grad = False
net.classifier[6] = nn.Linear(4096, len(dataset.classes))
net = net.to(device)

# Обучение с выводом ошибки
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

print("Начало обучения...")
for epoch in range(5):
    running_loss = 0.0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Эпоха {epoch+1}, Ошибка: {running_loss/10:.4f}')
            running_loss = 0.0

print("Обучение завершено")
torch.save(net.state_dict(), 'model.pth')