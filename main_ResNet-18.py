import os
import itertools
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset import CaltechDataset

# 超参数空间
HYPERPARAM_SPACE = {
    'train_mode': ['pretrained', 'scratch'],  # 新增训练模式参数
    'unfreeze_layers': [['layer4'], ['layer3', 'layer4']],
    'lr': [3e-4, 5e-5],
    'batch_size': [32, 64],
    'epochs': [30,50],
    'dropout_rate': [0.5],
    'hidden_dim': [512],
    'weight_decay': [1e-4, 1e-3]
}

BASE_OUTPUT_DIR = "/home/wenyisu/NNandDL/second/optimized/experiments"
DATA_PATHS = { 
    'train': '/home/wenyisu/NNandDL/second/optimized/dataset/images/train',
    'test': '/home/wenyisu/NNandDL/second/optimized/dataset/images/test'
}

def create_experiment_dir(config):
    """创建实验目录结构"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode_prefix = "pretrained" if config['train_mode'] == 'pretrained' else "scratch"
    folder_name = (
        f"{mode_prefix}_"
        f"bs{config['batch_size']}_"
        f"lr{config['lr']:.0e}_"
        f"layers-{'-'.join(config['unfreeze_layers'])}_"
        f"ep{config['epochs']}_"
        f"do{config['dropout_rate']}_"
        f"hd{config['hidden_dim']}_"
        f"wd{config['weight_decay']:.0e}"
    )
    experiment_dir = os.path.join(BASE_OUTPUT_DIR, f"{timestamp}_{folder_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    return experiment_dir

def save_config(config, experiment_dir):
    """保存配置到文件"""
    with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

def calculate_class_weights(dataset):
    """计算类别权重"""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    sorted_classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in sorted_classes]
    class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    return class_weights / class_weights.sum()

def train_model(config, device, trial_number):
    experiment_dir = create_experiment_dir(config)
    save_config(config, experiment_dir)
    
    print("\n" + "="*80)
    mode = "预训练" if config['train_mode'] == 'pretrained' else "随机初始化"
    print(f"实验 {trial_number} - {mode}训练:")
    for k, v in config.items():
        if k != 'train_mode':
            print(f"  {k:15}: {v}")
    print("="*80 + "\n")

    writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "logs"))

    # 数据增强
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CaltechDataset(DATA_PATHS['train'], transform=transform_train)
    test_dataset = CaltechDataset(DATA_PATHS['test'], transform=transform_test)

    class_weights = calculate_class_weights(train_dataset).to(device)

    train_loader = DataLoader(train_dataset, 
                            batch_size=config['batch_size'], 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True,
                            drop_last=True)
    test_loader = DataLoader(test_dataset, 
                           batch_size=config['batch_size'],
                           shuffle=False, 
                           num_workers=4)

    # 模型初始化
    if config['train_mode'] == 'pretrained':
        model = models.resnet18(pretrained=True)
    else:
        model = models.resnet18(pretrained=False)  # 随机初始化
    
    # 冻结/解冻层（仅预训练模式有效）
    if config['train_mode'] == 'pretrained':
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(layer in name for layer in config['unfreeze_layers']):
                param.requires_grad = True

    # 增强分类头
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(config['dropout_rate']),
        nn.Linear(num_ftrs, config['hidden_dim']*2),
        nn.BatchNorm1d(config['hidden_dim']*2),
        nn.ReLU(),
        nn.Dropout(config['dropout_rate']),
        nn.Linear(config['hidden_dim']*2, config['hidden_dim']),
        nn.ReLU(),
        nn.Linear(config['hidden_dim'], 101)
    )
    model = model.to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, 
        label_smoothing=0.1
    )

    # 优化器和调度器
    if config['train_mode'] == 'pretrained':
        # 预训练模式：对backbone和分类头使用不同学习率
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        head_params = model.fc.parameters()
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': config['lr']/10},
            {'params': head_params, 'lr': config['lr']}
        ], weight_decay=config['weight_decay'])
    else:
        # 随机初始化模式：统一学习率
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                             T_0=20, 
                                                             T_mult=2, 
                                                             eta_min=1e-7)

    start_time = datetime.now()
    best_acc = 0.0
    best_epoch = 0
    final_val_acc = 0.0
    final_val_loss = 0.0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = correct / len(train_dataset)
        
        model.eval()
        val_loss, val_correct = 0, 0
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
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', avg_val_acc, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, os.path.join(experiment_dir, "checkpoints", "best_model.pth"))
        
        final_val_acc = avg_val_acc
        final_val_loss = avg_val_loss
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    writer.close()
    
    # 构建JSON结果
    result = {
        "model_type": "pretrained" if config['train_mode'] == 'pretrained' else 'scratch',
        "config": {
            "trial_number": trial_number,
            "batch_size": config['batch_size'],
            "epochs": config['epochs'],
            "lr": config['lr'],
            "weight_decay": config['weight_decay'],
            "dropout_rate": config['dropout_rate'],
            "hidden_dim": config['hidden_dim'],
            "unfreeze_layers": config['unfreeze_layers'],
            "train_mode": config['train_mode']
        },
        "unfrozen_layers": len(config['unfreeze_layers']) if config['train_mode'] == 'pretrained' else 0,
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "final_val_accuracy": final_val_acc,
        "final_val_loss": final_val_loss,
        "training_time": training_time
    }
    
    # 保存单个实验结果
    with open(os.path.join(experiment_dir, "experiment_result.json"), 'w') as f:
        json.dump(result, f, indent=4)
    
    return result

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_results = []

    param_names = list(HYPERPARAM_SPACE.keys())
    param_values = [HYPERPARAM_SPACE[name] for name in param_names]
    
    total_combinations = 1
    for v in param_values:
        total_combinations *= len(v)
        
    print(f"Total hyperparameter combinations: {total_combinations}")

    trial_number = 1
    for combination in itertools.product(*param_values):
        config = dict(zip(param_names, combination))
        config['hidden_dim'] = 512  # 固定hidden_dim
        
        try:
            result = train_model(config, device, trial_number)
            all_results.append(result)
            trial_number += 1
            
        except Exception as e:
            print(f"实验失败: {str(e)}")
            error_result = {
                "model_type": "pretrained" if config['train_mode'] == 'pretrained' else 'scratch',
                "config": {
                    "trial_number": trial_number,
                    "batch_size": config['batch_size'],
                    "epochs": config['epochs'],
                    "lr": config['lr'],
                    "weight_decay": config['weight_decay'],
                    "dropout_rate": config['dropout_rate'],
                    "hidden_dim": config['hidden_dim'],
                    "unfreeze_layers": config['unfreeze_layers'],
                    "train_mode": config['train_mode']
                },
                "unfrozen_layers": len(config['unfreeze_layers']) if config['train_mode'] == 'pretrained' else 0,
                "best_val_accuracy": 0.0,
                "best_epoch": 0,
                "final_val_accuracy": 0.0,
                "final_val_loss": float('inf'),
                "training_time": 0,
                "error": str(e)[:100]
            }
            all_results.append(error_result)
            trial_number += 1
    
    # 保存所有实验结果
    results_path = os.path.join(BASE_OUTPUT_DIR, "all_experiments_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # 生成对比结果
    comparison = {}
    for result in all_results:
        if 'error' in result:
            continue
            
        config = result['config']
        key = f"bs{config['batch_size']}_lr{config['lr']}_layers-{'-'.join(config['unfreeze_layers'])}_do{config['dropout_rate']}_wd{config['weight_decay']}"
        
        if key not in comparison:
            comparison[key] = {'pretrained': None, 'scratch': None}
        
        if config['train_mode'] == 'pretrained':
            comparison[key]['pretrained'] = result
        else:
            comparison[key]['scratch'] = result
    
    # 计算提升并保存对比结果
    comparison_results = []
    for key, pair in comparison.items():
        if pair['pretrained'] and pair['scratch']:
            pretrained_acc = pair['pretrained']['best_val_accuracy']
            scratch_acc = pair['scratch']['best_val_accuracy']
            improvement = (pretrained_acc - scratch_acc) / scratch_acc * 100
            
            comparison_results.append({
                "config": pair['pretrained']['config'],
                "pretrained_accuracy": pretrained_acc,
                "scratch_accuracy": scratch_acc,
                "improvement_percentage": improvement,
                "absolute_difference": pretrained_acc - scratch_acc
            })
    
    # 保存对比结果
    comparison_path = os.path.join(BASE_OUTPUT_DIR, "pretrained_vs_scratch_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    # 打印最佳结果
    if comparison_results:
        best_comparison = max(comparison_results, key=lambda x: x['improvement_percentage'])
        print("\n=== 预训练效果最佳对比 ===")
        print(f"配置: {best_comparison['config']}")
        print(f"预训练准确率: {best_comparison['pretrained_accuracy']:.4f}")
        print(f"随机初始化准确率: {best_comparison['scratch_accuracy']:.4f}")
        print(f"提升: {best_comparison['improvement_percentage']:.2f}%")    