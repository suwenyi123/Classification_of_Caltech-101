import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import CaltechDataset

# 设置路径
train_path = '/home/wenyisu/NNandDL/second/1/dataset/images/train'
test_path = '/home/wenyisu/NNandDL/second/1/dataset/images/test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 是否使用预训练模型
use_pretrained = True

# 数据增强标准化参数
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# 加载数据集
train_dataset = CaltechDataset(train_path, transform=transform_train)
test_dataset = CaltechDataset(test_path, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 模型初始化
model = models.resnet01(pretrained=use_pretrained)

# 冻结参数逻辑优化
if use_pretrained:
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻最后两个残差块（layer3和layer4）
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "layer2" in name:  # 修改处：解冻深层
            param.requires_grad = True
    
    # 重新定义全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, 101)
    )
    # 解冻全连接层
    for param in model.fc.parameters():
        param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    [
        {'params': model.layer3.parameters(), 'lr': 1e-5},  # 深层使用更小学习率
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.layer2.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ],
    weight_decay=1e-6
)

epochs = 50

# 学习率调度器（分层调度）
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# TensorBoard
log_dir = '/home/wenyisu/NNandDL/second/optimized/runs/caltech101'
writer = SummaryWriter(log_dir=log_dir)

# 训练参数
best_val_acc = 0.0
patience = 20  # 缩短早停耐心值（因为参数更多）
no_improve_count = 0

# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    model.train()  # 修改处：直接使用全局train模式
    
    train_loss, train_correct = 0.0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # 梯度裁剪（应用到所有可训练参数）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
    
    scheduler.step()
    
    avg_train_loss = train_loss / len(train_dataset)
    avg_train_acc = train_correct / len(train_dataset)
    
    # 验证
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(test_dataset)
    avg_val_acc = val_correct / len(test_dataset)
    
    if avg_val_acc > best_val_acc + 1e-4:
        best_val_acc = avg_val_acc
        no_improve_count = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_val_acc
        }, f"{log_dir}/best_model.pth")
    else:
        no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 记录指标
    writer.add_scalars('Loss', {'Train': avg_train_loss, 'Val': avg_val_loss}, epoch)
    writer.add_scalars('Accuracy', {'Train': avg_train_acc, 'Val': avg_val_acc}, epoch)
    
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.2e}")

writer.close()