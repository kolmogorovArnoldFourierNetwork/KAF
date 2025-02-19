import torch
import torch.nn as nn
import torch.nn.functional as F
from fastkan.fastkan import FastKAN
from .FANLayer import FANLayer
import torch
import random
import sys
from kaf_act import RFFActivation
from kat_rational.kat_1dgroup import KAT_Group

#根据fastkan的超参数，rank=0.125
FLOPs_MAP = {
    "zero": 0,
    "identity": 0,
    "relu": 1,
    'square_relu': 2,
    "sigmoid":4,
    "silu":5,
    "tanh":6,
    "gelu": 14,
    "polynomial2": 1+2+3-1,
    "polynomial3": 1+2+3+4-1,
    "polynomial5": 1+2+3+4+5-1,
}

class KAF(nn.Module):
    def __init__(self, args, input_size=32 * 32 * 3, num_classes=3, dropout_prob=0.1, num_grids=8, 
                 spline_dropout=0,act_exp=1.64):
        super(KAF, self).__init__()
        self.input_size = getattr(args, 'input_size', input_size)
        self.num_classes = getattr(args, 'num_classes', num_classes)
        self.dropout_rate = getattr(args, 'dropout_prob', dropout_prob)
        self.num_grids = getattr(args, 'num_grids', num_grids)
        self.spline_dropout = getattr(args, 'spline_dropout', spline_dropout)
        self.act_exp = getattr(args, 'act_exp', act_exp)
        
        layers_width = getattr(args, 'layers_width', [])
        layer_sizes = [self.input_size] + layers_width + [self.num_classes]
        self.layer_sizes = layer_sizes
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if args.batch_norm:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            self.layers.append(RFFActivation(
                num_grids=self.num_grids,
                dropout=self.spline_dropout,
                activation_expectation=self.act_exp,
                use_layernorm=False,
                base_activation=F.gelu
            ))
            self.layers.append(nn.Dropout(p=self.dropout_rate))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return x
    
    def total_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total
    
    def total_flops(self):
        total_flops = 0
        for i in range(len(self.layer_sizes) - 1):
            total_flops += self.layer_sizes[i] * self.layer_sizes[i+1]
            total_flops += self.layer_sizes[i+1] * self.num_grids  
            total_flops += 2 * self.num_grids * self.layer_sizes[i+1]  
        
        return total_flops
    

FLOPs_MAP = {
    'sigmoid': 1,
    'tanh': 1,
}

class KAF_Text(nn.Module):
    def __init__(self, args):
        super(KAF_Text, self).__init__()
        self.input_size = getattr(args, 'input_size', 32 * 32 * 3)
        self.num_classes = getattr(args, 'num_classes', 3)
        self.dropout_prob = getattr(args, 'dropout_prob', 0.1)
        self.num_grids = getattr(args, 'num_grids', 8)
        self.spline_dropout = getattr(args, 'spline_dropout', 0)
        self.act_exp = getattr(args, 'act_exp', 1.64)
        self.batch_norm = getattr(args, 'batch_norm', False)
        self.activation = getattr(args, 'activation', nn.ReLU)
        self.activation_name = getattr(args, 'activation_name', 'relu')

        # 嵌入层
        self.embedding = nn.EmbeddingBag(args.input_size, args.layers_width[0], sparse=False)
        
        # 定义层宽度
        layers_width = [args.layers_width[0]] + args.layers_width
        self.layers_width = layers_width + [self.num_classes]

        self.layers = nn.ModuleList()
        for i in range(len(layers_width) - 1):
            self.layers.append(nn.Linear(layers_width[i], layers_width[i+1]))
            if args.batch_norm:
                self.layers.append(nn.BatchNorm1d(layers_width[i+1]))
            self.layers.append(RFFActivation(
                num_grids=self.num_grids,
                dropout=self.spline_dropout,
                activation_expectation=self.act_exp,
                use_layernorm=False,
                base_activation=F.gelu
            ))
            self.layers.append(nn.Dropout(p=self.dropout_rate))
        
        # 最后一层输出
        self.layers.append(nn.Linear(self.layers_width[-2], self.num_classes))
    
    def forward(self, inputs):

        text, offsets = inputs
        x = self.embedding(text, offsets)
        for layer in self.layers:
            x = layer(x)
        return x

    def layer_flops(self, din, dout, nonlinearity):

        return 2 * din * dout + FLOPs_MAP.get(nonlinearity, 1) * dout

    def layer_parameters(self, din, dout):

        return dout * (din + 1)

    def batchnorm_flops(self, dout):

        return dout * 4

    def batchnorm_parameters(self, dout):

        return 2 * dout

    def total_flops(self):

        total_flops = 0
        for i in range(len(self.layers_width) - 1):
            total_flops += self.layer_flops(self.layers_width[i], self.layers_width[i + 1], self.activation_name)
            if self.batch_norm:
                total_flops += self.batchnorm_flops(self.layers_width[i + 1])
        return total_flops

    def total_parameters(self):

        total_parameters = 0
        for i in range(len(self.layers_width) - 1):
            total_parameters += self.layer_parameters(self.layers_width[i], self.layers_width[i + 1])
            if self.batch_norm:
                total_parameters += self.batchnorm_parameters(self.layers_width[i + 1])
        return total_parameters

class GPKAN(nn.Module):
    def __init__(self, args, input_size=32 * 32 * 3, num_classes=3, dropout_prob=0.1, num_grids=8, 
                 spline_dropout=0,act_exp=1.7890,act_cfg=dict(type="KAT", act_init=["identity", "gelu"])):
        super(GPKAN, self).__init__()
        self.input_size = getattr(args, 'input_size', input_size)
        self.num_classes = getattr(args, 'num_classes', num_classes)
        self.dropout_prob = getattr(args, 'dropout_prob', dropout_prob)
        self.num_grids = getattr(args, 'num_grids', num_grids)
        self.spline_dropout = getattr(args, 'spline_dropout', spline_dropout)
        self.act_exp = getattr(args, 'act_exp', act_exp)
        
        self.layers_width = getattr(args, 'layers_width', [])
        #目前只支持两层
        self.fc1 = nn.Linear(self.input_size, self.layers_width[0], bias=True)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][0])
        self.drop1 = nn.Dropout(self.dropout_prob)
        self.act2 = KAT_Group(mode = act_cfg['act_init'][1])
        self.fc2 = nn.Linear(self.layers_width[0], self.num_classes, bias=True)
        self.drop2 = nn.Dropout(self.dropout_prob)

    
    def forward(self, x):
        # 添加一个维度，将 2D 输入 (batch_size, features) 转换为 3D (batch_size, 1, features)
        x = x.unsqueeze(1)  # 在第1维添加维度
        x = self.act2(x)
        x = self.drop1(x)
        # 在进入 fc1 前压缩回 2D
        x = x.squeeze(1)
        
        x = self.fc1(x)
        # 再次添加维度用于 act2
        x = x.unsqueeze(1)
        x = self.act2(x)
        x = self.drop2(x)
        # 最后压缩回 2D
        x = x.squeeze(1)
        
        x = self.fc2(x)
        return x
    
    def total_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total
    
    def total_flops(self):
        total_flops = 0
        
        return total_flops
    
class FAN(nn.Module):
    def __init__(self, args, input_size=32 * 32 * 3, num_classes=3, dropout_prob=0.1, 
                 p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FAN, self).__init__()
        self.input_size = getattr(args, 'input_size', input_size)
        self.num_classes = getattr(args, 'num_classes', num_classes)
        self.dropout_prob = getattr(args, 'dropout_prob', dropout_prob)
        self.layers_width = getattr(args, 'layers_width', [])
        
        # 构建网络层
        self.layers = nn.ModuleList()
        
        # 第一层：输入层到隐藏层
        self.layers.append(nn.Sequential(
            FANLayer(self.input_size, self.layers_width[0], 
                    p_ratio=p_ratio, activation=activation, use_p_bias=use_p_bias),
            nn.Dropout(p=self.dropout_prob)
        ))
        
        # 中间层（如果有的话）
        for i in range(len(self.layers_width)-1):
            self.layers.append(nn.Sequential(
                FANLayer(self.layers_width[i], self.layers_width[i+1],
                        p_ratio=p_ratio, activation=activation, use_p_bias=use_p_bias),
                nn.Dropout(p=self.dropout_prob)
            ))
        
        # 输出层
        self.layers.append(FANLayer(self.layers_width[-1], self.num_classes,
                                  p_ratio=p_ratio, activation=activation, use_p_bias=use_p_bias))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def total_flops(self):
        total_flops = 0
        # 计算每一层的FLOPs
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Sequential):
                fan_layer = layer[0]  # 获取FANLayer实例
            else:
                fan_layer = layer
                
            # 计算线性变换的FLOPs
            if i == 0:
                input_size = self.input_size
            else:
                input_size = self.layers_width[i-1]
                
            if i == len(self.layers) - 1:
                output_size = self.num_classes
            else:
                output_size = self.layers_width[i]
                
            # 为p分量的线性变换计算FLOPs
            p_output_dim = int(output_size * fan_layer.p_ratio)
            total_flops += input_size * p_output_dim * 2  # 乘2是因为有乘法和加法操作
            
            # 为g分量的线性变换计算FLOPs
            g_output_dim = output_size - p_output_dim * 2
            total_flops += input_size * g_output_dim * 2
            
            # 为三角函数计算添加FLOPs
            total_flops += p_output_dim * 2  # cos和sin操作
            
        return total_flops

class FAN_Text(nn.Module):
    def __init__(self, args, input_size=32 * 32 * 3, num_classes=3, dropout_prob=0.1, 
                 p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FAN_Text, self).__init__()
        # 处理参数
        self.input_size = getattr(args, 'input_size', input_size)
        self.num_classes = getattr(args, 'num_classes', num_classes)
        self.dropout_prob = getattr(args, 'dropout_prob', dropout_prob)
        self.layers_width = getattr(args, 'layers_width', [])
        
        # 嵌入层
        self.embedding = nn.EmbeddingBag(args.input_size, args.layers_width[0], sparse=False)
        
        # 修改这里：定义层宽度
        self.hidden_dim = args.layers_width[0]  # 使用嵌入维度作为隐藏层维度
        
        # 构建网络层
        self.layers = nn.ModuleList()
        
        # 第一层：从嵌入维度到隐藏层
        self.layers.append(nn.Sequential(
            FANLayer(self.hidden_dim, self.hidden_dim,  # 修改输入输出维度
                    p_ratio=p_ratio, activation=activation, use_p_bias=use_p_bias),
            nn.Dropout(p=self.dropout_prob)
        ))
        
        # 中间层（如果有的话）
        for i in range(len(self.layers_width)-1):
            self.layers.append(nn.Sequential(
                FANLayer(self.hidden_dim, self.hidden_dim,  # 修改输入输出维度
                        p_ratio=p_ratio, activation=activation, use_p_bias=use_p_bias),
                nn.Dropout(p=self.dropout_prob)
            ))
        
        # 输出层
        self.layers.append(FANLayer(self.hidden_dim, self.num_classes,  # 最后一层到输出类别数
                                  p_ratio=p_ratio, activation=activation, use_p_bias=use_p_bias))
    
    def forward(self, inputs):
        text, offsets = inputs
        x = self.embedding(text, offsets)  # 得到文本嵌入
        
        for layer in self.layers:
            x = layer(x)
        return x
    
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def total_flops(self):
        total_flops = 0
        # 计算嵌入层的FLOPs
        total_flops += self.embedding.num_embeddings * self.embedding.embedding_dim
        
        # 计算FAN层的FLOPs
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Sequential):
                fan_layer = self.layers[i][0]  # 获取FANLayer实例
            else:
                fan_layer = self.layers[i]
            
            # 获取输入和输出维度
            if i == 0:
                input_size = self.embedding.embedding_dim
            else:
                input_size = fan_layer.input_linear_p.in_features
                
            # 计算p分量和g分量的FLOPs
            p_flops = input_size * fan_layer.input_linear_p.out_features * 2  # 乘加各算一次
            g_flops = input_size * fan_layer.input_linear_g.out_features * 2
            
            # 加上三角函数的FLOPs（每个p输出需要一个三角函数计算）
            trig_flops = fan_layer.input_linear_p.out_features * 2  # cos和sin各算一次
            
            total_flops += p_flops + g_flops + trig_flops
            
        return total_flops

class GPKAN_Text(nn.Module):
    def __init__(self, args, act_cfg=dict(type="KAT", act_init=["identity", "gelu"])):
        super(GPKAN_Text, self).__init__()
        # 处理参数
        self.input_size = getattr(args, 'input_size', 32 * 32 * 3)
        self.num_classes = getattr(args, 'num_classes', 3)
        self.dropout_prob = getattr(args, 'dropout_prob', 0.1)
        self.act_cfg = act_cfg
        
        # 嵌入层
        self.embedding = nn.EmbeddingBag(args.input_size, args.layers_width[0], sparse=False)
        
        # 定义层
        self.fc1 = nn.Linear(args.layers_width[0], args.layers_width[0], bias=True)
        self.act1 = KAT_Group(mode=act_cfg['act_init'][0])
        self.drop1 = nn.Dropout(self.dropout_prob)
        self.act2 = KAT_Group(mode=act_cfg['act_init'][1])
        self.fc2 = nn.Linear(args.layers_width[0], self.num_classes, bias=True)
        self.drop2 = nn.Dropout(self.dropout_prob)
    
    def forward(self, inputs):
        text, offsets = inputs
        x = self.embedding(text, offsets)
        
        x = x.unsqueeze(1)
        x = self.act1(x)
        x = self.drop1(x)
        x = x.squeeze(1)
        
        x = self.fc1(x)
        x = x.unsqueeze(1)
        x = self.act2(x)
        x = self.drop2(x)
        x = x.squeeze(1)
        
        x = self.fc2(x)
        return x
    
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def total_flops(self):
        total_flops = 0
        # 计算嵌入层的FLOPs
        total_flops += self.embedding.num_embeddings * self.embedding.embedding_dim
        # 计算线性层的FLOPs
        total_flops += self.fc1.in_features * self.fc1.out_features * 2
        total_flops += self.fc2.in_features * self.fc2.out_features * 2
        return total_flops

    