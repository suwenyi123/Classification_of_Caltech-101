import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from dataset import CaltechDataset
from utils import accuracy
from torchvision.models import ResNet50_Weights

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 设置路径
train_path = '/home/wenyisu/NNandDL/second/1/dataset/images/train'
test_path = '/home/wenyisu/NNandDL/second/1/dataset/images/test'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型保存目录
model_dir = '/home/wenyisu/NNandDL/second/1/models'
os.makedirs(model_dir, exist_ok=True)

# 数据预处理
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# 加载数据
train_dataset = CaltechDataset(train_path, transform=transform_train)
test_dataset = CaltechDataset(test_path, transform=transform_test)

# 训练和验证函数
def train_model(model, optimizer, criterion, scheduler, train_loader, test_loader, epochs, model_name):
    """完整训练流程，返回最佳验证准确率"""
    best_val_acc = 0.0
    best_epoch = 0
    writer = SummaryWriter(log_dir=f'runs/{model_name}')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            acc1, _ = accuracy(outputs, labels, topk=(1, 5))
            train_correct += acc1.item() * labels.size(0)
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                acc1, _ = accuracy(outputs, labels, topk=(1, 5))
                val_correct += acc1.item() * labels.size(0)
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total
        
        # 学习率调整
        if scheduler:
            scheduler.step(avg_val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', avg_train_acc, epoch)
        writer.add_scalar('Accuracy/Val', avg_val_acc, epoch)
        
        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(), 
                os.path.join(model_dir, f'{model_name}_best.pth')
            )
        
        # 打印训练进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | "
              f"训练损失: {avg_train_loss:.4f}, 训练准确率: {avg_train_acc:.2f}% | "
              f"验证损失: {avg_val_loss:.4f}, 验证准确率: {avg_val_acc:.2f}% | "
              f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch}) | "
              f"学习率: {current_lr:.6f}")
    
    writer.close()
    return best_val_acc

def main():
    # 设置模型（使用预训练ResNet-50，微调最后几层）
    print("初始化模型...")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # 冻结大部分预训练层，仅微调最后几层
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻最后一个残差块和全连接层
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # 修改输出层为101类
    model.fc = nn.Linear(model.fc.in_features, 101)
    model = model.to(device)
    
    # 设置优化器（分层学习率）
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 5e-4},  # 微调层
        {'params': model.fc.parameters(), 'lr': 1e-3}      # 新输出层
    ], weight_decay=1e-4)
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 设置数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 训练模型
    print("开始训练...")
    model_name = "resnet50_caltech101"
    best_accuracy = train_model(
        model, optimizer, nn.CrossEntropyLoss(), scheduler,
        train_loader, test_loader, epochs=25, model_name=model_name
    )
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}_final.pth'))
    
    # 打印结果
    print("\n=== 训练完成 ===")
    print(f"最佳验证准确率: {best_accuracy:.2f}%")
    print(f"模型保存路径: {model_dir}")
    print(f"TensorBoard日志: runs/{model_name}")
    
    # 保存实验结果
    with open(os.path.join(model_dir, '实验结果.txt'), 'w') as f:
        f.write(f"ResNet-50 最佳验证准确率: {best_accuracy:.2f}%\n")
        f.write(f"训练配置:\n")
        f.write(f"- 预训练: ImageNet\n")
        f.write(f"- 微调层: layer4 + fc\n")
        f.write(f"- 学习率: layer4=5e-4, fc=1e-3\n")
        f.write(f"- 批次大小: {batch_size}\n")
        f.write(f"- 权重衰减: 1e-4\n")
        f.write(f"- 训练轮次: 25\n")
    
    print(f"实验配置已保存至: {os.path.join(model_dir, '实验结果.txt')}")

if __name__ == "__main__":
    main()