import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from timm import create_model
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.vision_transformer import VisionTransformer, Block
from kaf_act import RFFActivation
from functools import partial
import os
import yaml
from timm.scheduler import CosineLRScheduler
import numpy as np

# \
#accelerate launch --num_processes 3 --gpu_ids 0,1,2 vit_KAf.py

class KAFMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.activation = RFFActivation(use_layernorm=True)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  
        x = self.drop(x)  
        x = self.KAF(x)  
        x = self.drop(x)  
        return x


class KAFBlock(Block):
    """ 修改后的Transformer Block，使用KAF替代MLP """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__(dim=dim, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        proj_drop=drop,
                        attn_drop=attn_drop, 
                        drop_path=drop_path, 
                        norm_layer=norm_layer)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = KAFMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop=drop
        )


class KAFViT(VisionTransformer):
    """ 使用KAF的ViT """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        

        for i in range(len(self.blocks)):
            block = self.blocks[i]  
            self.blocks[i] = KAFBlock(
                dim=block.norm1.normalized_shape[0],  
                num_heads=block.attn.num_heads,
                mlp_ratio=4., 
                qkv_bias=block.attn.qkv.bias is not None,
                drop=0.1, 
                attn_drop=0.0, 
                drop_path=block.drop_path.drop_prob if hasattr(block, 'drop_path') else 0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6) 
            )

class ImageNetDataModule:
    def __init__(self, config):
        self.config = config
        self.train_dir = config['data']['train_dir']
        self.val_dir = config['data']['val_dir']
        self.batch_size = config['data']['batch_size']
        self.num_workers = config['data']['num_workers']
        
        self.setup_transforms()
        self.setup_mixup()
        
    def setup_transforms(self):
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform(
                config_str='rand-m9-mstd0.5-inc1',
                hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25)
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def setup_mixup(self):
        self.mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=0.1,
            num_classes=1000
        )
        
    def setup(self):
        self.train_dataset = datasets.ImageFolder(self.train_dir, self.transform_train)
        self.val_dataset = datasets.ImageFolder(self.val_dir, self.transform_val)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True  # 丢弃最后一个不完整的batch
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False  # 验证集不需要丢弃
        )
        
class KAFViTTrainer:
    """训练器类"""
    def __init__(self, config):
        self.config = config
        self.setup_accelerator()
        self.setup_model()
        self.setup_criterion()
        self.setup_optimizer()
        self.best_acc = 0
        self.best_test_acc = 0  # 添加最佳测试集准确率记录
        
    def setup_accelerator(self):
        self.accelerator = Accelerator()
        self.logger = get_logger(__name__)
        set_seed(42)
        
    def setup_model(self):
        # 首先加载预训练的ViT-Ti模型
        original_model = create_model(
            'vit_tiny_patch16_224',
            pretrained=True,
            num_classes=1000
        )
        
        # 创建我们的KAFViT模型，传入所有必要的参数
        self.model = KAFViT(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        )
        
        # 加载预训练权重
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in original_model.state_dict().items() 
                         if k in model_dict and 'mlp' not in k}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        
        if self.accelerator.is_main_process:
            self.logger.info("Successfully loaded pretrained ViT-Ti weights")
            
        # 如果配置文件中指定了额外的预训练权重路径，则加载它
        if self.config['training'].get('extra_pretrained_path'):
            self.load_extra_weights()
            
    def load_extra_weights(self):
        weights_path = self.config['training']['extra_pretrained_path']
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            if self.accelerator.is_main_process:
                self.logger.info(f"Loaded extra weights from {weights_path}")
        
    def setup_criterion(self):
        self.criterion_train = SoftTargetCrossEntropy()
        self.criterion_val = LabelSmoothingCrossEntropy(smoothing=0.1)
        
    def setup_optimizer(self):
        # 使用AdamW优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay']),
            betas=(0.9, 0.999)
        )
        
        # 使用cosine调度器
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config['training']['epochs'],
            lr_min=float(self.config['training']['min_lr']),
            warmup_lr_init=float(self.config['training']['warmup_lr_init']),
            warmup_t=self.config['training']['warmup_epochs'],
            cycle_limit=1,
            t_in_epochs=True,
        )
        
    def prepare(self, datamodule):
        self.model, self.optimizer, datamodule.train_loader, \
        datamodule.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, datamodule.train_loader,
            datamodule.val_loader, self.scheduler
        )
        self.train_loader = datamodule.train_loader
        self.val_loader = datamodule.val_loader
        self.mixup_fn = datamodule.mixup_fn
        
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, 
                        disable=not self.accelerator.is_local_main_process)
        
        for images, labels in progress_bar:
            if self.mixup_fn is not None:
                images, labels = self.mixup_fn(images, labels)
                
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion_train(outputs, labels)
            
            self.accelerator.backward(loss)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(dim=-1)).sum().item()
        
        # 确保统计量在正确设备上
        total_loss_tensor = torch.tensor(total_loss, device=self.accelerator.device)
        total_tensor = torch.tensor(total, device=self.accelerator.device)
        correct_tensor = torch.tensor(correct, device=self.accelerator.device)
        
        gathered_loss = self.accelerator.gather(total_loss_tensor).sum().item()
        gathered_total = self.accelerator.gather(total_tensor).sum().item()
        gathered_correct = self.accelerator.gather(correct_tensor).sum().item()
        
        return gathered_loss / len(self.train_loader), 100. * gathered_correct / gathered_total

    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.val_loader, 
                        disable=not self.accelerator.is_local_main_process)
        
        for images, labels in progress_bar:
            outputs = self.model(images)
            loss = self.criterion_val(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 确保统计量在正确设备上
        total_loss_tensor = torch.tensor(total_loss, device=self.accelerator.device)
        total_tensor = torch.tensor(total, device=self.accelerator.device)
        correct_tensor = torch.tensor(correct, device=self.accelerator.device)
        
        gathered_loss = self.accelerator.gather(total_loss_tensor).sum().item()
        gathered_total = self.accelerator.gather(total_tensor).sum().item()
        gathered_correct = self.accelerator.gather(correct_tensor).sum().item()
        
        return gathered_loss / len(self.val_loader), 100. * gathered_correct / gathered_total
    
    @torch.no_grad()
    def test(self):
        """测试方法，类似于validate但使用测试集"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.val_loader,
                          disable=not self.accelerator.is_local_main_process)
        
        for images, labels in progress_bar:
            outputs = self.model(images)
            loss = self.criterion_val(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 修改这里：确保 tensor 在正确的设备上
        total_loss = self.accelerator.gather(
            torch.tensor(total_loss, device=self.accelerator.device)
        ).mean().item()
        total = self.accelerator.gather(
            torch.tensor(total, device=self.accelerator.device)
        ).sum().item()
        correct = self.accelerator.gather(
            torch.tensor(correct, device=self.accelerator.device)
        ).sum().item()
        
        return 100. * correct / total
    
    def save_checkpoint(self, epoch, val_acc):
        if self.accelerator.is_main_process and val_acc > self.best_acc:
            self.best_acc = val_acc
            self.accelerator.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_acc': self.best_acc,
                    'best_test_acc': self.best_test_acc,  # 保存最佳测试集准确率
                },
                self.config['training']['checkpoint_path']
            )
    
    def train(self):
        epochs = self.config['training']['epochs']
        test_interval = 1
        
        # 创建或清空日志文件
        if self.accelerator.is_main_process:
            with open('training_log.txt', 'w') as f:
                f.write('Epoch,Train_Acc,Val_Acc,Test_Acc,LR\n')
        
        for epoch in range(epochs):
            if self.accelerator.is_main_process:
                self.logger.info(f'\nEpoch: {epoch+1}/{epochs}')
            
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step(epoch + 1)
            
            # 每10个epoch进行一次测试
            if (epoch + 1) % test_interval == 0:
                test_acc = self.test()
                if test_acc > self.best_test_acc:
                    self.best_test_acc = test_acc
            else:
                test_acc = 0  # 非测试epoch标记为0
            
            # 记录每个epoch的结果
            if self.accelerator.is_main_process:
                self.logger.info(
                    f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
                self.logger.info(
                    f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
                if test_acc > 0:
                    self.logger.info(f'Test Acc: {test_acc:.2f}% | Best Test Acc: {self.best_test_acc:.2f}%')
                self.logger.info(
                    f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                
                # 保存到CSV格式的日志文件
                with open('training_log.txt', 'a') as f:
                    f.write(f'{epoch+1},{train_acc:.2f},{val_acc:.2f},{test_acc:.2f},{self.optimizer.param_groups[0]["lr"]:.6f}\n')
            
            self.save_checkpoint(epoch, val_acc)
            
        if self.accelerator.is_main_process:
            self.logger.info(f'Training completed.')
            self.logger.info(f'Best validation accuracy: {self.best_acc:.2f}%')
            self.logger.info(f'Best test accuracy: {self.best_test_acc:.2f}%')
            
            # 保存最终结果
            with open('final_results.txt', 'w') as f:
                f.write('Final Results:\n')
                f.write(f'Best Validation Accuracy: {self.best_acc:.2f}%\n')
                f.write(f'Best Test Accuracy: {self.best_test_acc:.2f}%\n')
                f.write(f'Final Test Accuracy: {self.test():.2f}%\n')

def main():
    # 使用绝对路径加载配置文件
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 正确初始化accelerator（移除手动CUDA设置）
    accelerator = Accelerator(
        mixed_precision='fp16',
        device_placement=True
    )
    
    # 初始化数据模块
    datamodule = ImageNetDataModule(config)
    datamodule.setup()  # 不再需要手动传递accelerator
    
    # 初始化训练器
    trainer = KAFViTTrainer(config)
    trainer.prepare(datamodule)
    
    # 开始训练
    trainer.train()
if __name__ == '__main__':
    # 使用accelerate启动脚本
    main()