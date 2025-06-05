import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import json
import time
from PIL import Image
from sklearn.model_selection import ParameterGrid

# 设置随机种子和设备
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"可用GPU内存: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# 数据路径
train_path = '/home/wenyisu/NNandDL/second/1/dataset/images/train'
test_path = '/home/wenyisu/NNandDL/second/1/dataset/images/test'
model_dir = '/home/wenyisu/NNandDL/second/1/models_efficientnet_b4_new'
os.makedirs(model_dir, exist_ok=True)

# 数据预处理
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.Resize(380),
    transforms.RandomResizedCrop(380, scale=(0.6, 1.4)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(380),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    normalize
])

# 自定义数据集类
class CaltechDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # 获取所有类别
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        # 加载所有图像路径和标签
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path) and img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((img_path, self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 准确率计算函数
def accuracy(output, target, topk=(1,)):
    """计算指定topk的准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# 训练函数
def train_model(config, pretrained=True, unfrozen_layers=-6):
    """完整训练流程，返回最佳验证准确率"""
    # 创建TensorBoard写入器
    model_type = "pretrained" if pretrained else "random_init"
    model_name = f"{model_type}_{config['trial_number']}"
    writer = SummaryWriter(log_dir=f'runs/{model_name}')
    
    # 初始化模型
    if pretrained:
        print(f"使用预训练的EfficientNet-B4模型 - Trial {config['trial_number']}")
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    else:
        print(f"使用随机初始化的EfficientNet-B4模型 - Trial {config['trial_number']}")
        model = models.efficientnet_b4(weights=None)
    
    # 冻结/解冻策略
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻指定层数
    if unfrozen_layers is not None:
        for param in model.features[unfrozen_layers:].parameters():
            param.requires_grad = True
    
    # 修改输出层为101类
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 101)
    model = model.to(device)  # 移至GPU
    
    # 设置优化器
    if pretrained:
        # 分层学习率（预训练模型）
        optimizer = optim.Adam([
            {'params': model.features[unfrozen_layers:].parameters(), 'lr': config['lr_features']},
            {'params': model.classifier.parameters(), 'lr': config['lr_classifier']}
        ], weight_decay=config['weight_decay'])
    else:
        # 统一学习率（随机初始化模型）
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    train_dataset = CaltechDataset(train_path, transform=transform_train)
    test_dataset = CaltechDataset(test_path, transform=transform_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True  
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True  
    )
    
    # 训练过程
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_correct_top5 = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 移至GPU
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            train_correct += acc1.item() * labels.size(0)
            train_correct_top5 += acc5.item() * labels.size(0)
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        avg_train_acc_top5 = train_correct_top5 / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_correct_top5 = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 移至GPU
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                val_correct += acc1.item() * labels.size(0)
                val_correct_top5 += acc5.item() * labels.size(0)
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total
        avg_val_acc_top5 = val_correct_top5 / val_total
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train/Top1', avg_train_acc, epoch)
        writer.add_scalar('Accuracy/Train/Top5', avg_train_acc_top5, epoch)
        writer.add_scalar('Accuracy/Val/Top1', avg_val_acc, epoch)
        writer.add_scalar('Accuracy/Val/Top5', avg_val_acc_top5, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(), 
                os.path.join(model_dir, f'{model_name}_best.pth')
            )
        
        if device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**2
        else:
            gpu_memory = 0
        
        print(f"Trial {config['trial_number']} | Epoch {epoch+1}/{config['epochs']} | "
              f"训练损失: {avg_train_loss:.4f}, 训练准确率: {avg_train_acc:.2f}% | "
              f"验证损失: {avg_val_loss:.4f}, 验证准确率: {avg_val_acc:.2f}% | "
              f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch}) | "
              f"学习率: {current_lr:.6f} | GPU内存: {gpu_memory:.2f} MB")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}_final.pth'))
    
    # 保存实验结果
    result = {
        'model_type': model_type,
        'config': config,
        'unfrozen_layers': unfrozen_layers,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_accuracy': avg_val_acc,
        'final_val_loss': avg_val_loss,
        'training_time': training_time
    }
    
    result_path = os.path.join(model_dir, f'result_{model_name}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    writer.close()
    return best_val_acc

# 网格搜索超参数优化
def grid_search_optimization():
    print("开始GridSearchCV超参数优化...")
    
    param_grid = {
        'batch_size': [32,64],
        'lr_features': [1e-5,3e-4],
        'lr_classifier': [8e-4,1e-3],
        'weight_decay': [1e-4,1e-3],
        'unfrozen_layers': [-6, -4] 
    }
    
    # 记录所有结果
    all_results = []
    trial_number = 0
    
    for params in ParameterGrid(param_grid):
        trial_number += 1
        config = {
            'trial_number': trial_number,
            'batch_size': params['batch_size'],
            'epochs': 30,
            'lr_features': params['lr_features'],
            'lr_classifier': params['lr_classifier'],
            'weight_decay': params['weight_decay']
        }
        
        print(f"\n=== 开始实验: Trial {trial_number} ===")
        print(f"参数: {config}")
        print(f"解冻层数: features[{params['unfrozen_layers']}:]")
        
        # 训练模型
        best_accuracy = train_model(
            config, 
            pretrained=True,
            unfrozen_layers=params['unfrozen_layers']
        )
        
        # 记录结果
        result = {
            'trial_number': trial_number,
            'config': config,
            'unfrozen_layers': params['unfrozen_layers'],
            'best_accuracy': best_accuracy
        }
        
        all_results.append(result)
        print(f"实验完成! 最佳准确率: {best_accuracy:.2f}%")
    
    # 保存所有实验结果
    with open(os.path.join(model_dir, 'grid_search_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 找出最佳参数组合
    best_result = max(all_results, key=lambda x: x['best_accuracy'])
    print("\n=== 网格搜索结果汇总 ===")
    print(f"最佳准确率: {best_result['best_accuracy']:.2f}%")
    print(f"最佳参数: {best_result['config']}")
    print(f"解冻层数: features[{best_result['unfrozen_layers']}:]")
    
    return best_result['config'], best_result['unfrozen_layers']

# 对比预训练与随机初始化
def comparison_experiment(best_config, unfrozen_layers):
    print("\n=== 开始预训练 vs 随机初始化对比实验 ===")
    
    # 记录结果
    results = {}
    
    # 测试预训练模型
    print("\n--- 测试预训练模型 ---")
    config_pretrained = best_config.copy()
    config_pretrained['trial_number'] = 9999
    pretrained_acc = train_model(
        config_pretrained, 
        pretrained=True,
        unfrozen_layers=unfrozen_layers
    )
    results['pretrained'] = pretrained_acc
    
    # 测试随机初始化模型
    print("\n--- 测试随机初始化模型 ---")
    config_random = {
        'trial_number': 9998,
        'batch_size': best_config['batch_size'],
        'epochs': 50,
        'lr': best_config['lr_classifier'],
        'weight_decay': best_config['weight_decay']
    }
    random_acc = train_model(
        config_random, 
        pretrained=False,
        unfrozen_layers=None  # 随机初始化时不冻结任何层
    )
    results['random'] = random_acc
    
    # 保存对比结果
    with open(os.path.join(model_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 输出对比结果
    print("\n=== 对比实验结果 ===")
    print(f"预训练模型准确率: {pretrained_acc:.2f}%")
    print(f"随机初始化模型准确率: {random_acc:.2f}%")
    print(f"预训练带来的准确率提升: {pretrained_acc - random_acc:.2f}%")
    
    return results

# 主函数
def main():
    # 运行网格搜索优化
    best_config, unfrozen_layers = grid_search_optimization()
    
    # 运行对比实验
    comparison_experiment(best_config, unfrozen_layers)
    
    print("\n=== 所有实验完成 ===")
    print(f"模型保存路径: {model_dir}")
    print(f"TensorBoard日志: runs/")
    print("请使用 `tensorboard --logdir=runs` 查看训练可视化结果")
    if torch.cuda.is_available():
        print(f"训练过程中最大GPU内存使用: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()