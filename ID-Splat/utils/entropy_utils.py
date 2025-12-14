"""
熵计算工具模块
提供各种熵计算函数，用于分析语言特征的信息量
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_shannon_entropy(features, eps=1e-8):
    """
    计算Shannon熵
    
    Args:
        features: 特征张量 [N, D]
        eps: 避免log(0)的小值
        
    Returns:
        entropy: Shannon熵值
    """
    # 将特征转换为概率分布
    probs = F.softmax(features, dim=-1)
    probs = torch.clamp(probs, min=eps)
    
    # 计算熵: H = -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    
    return torch.mean(entropy)

def compute_cross_entropy(features, target_probs=None, eps=1e-8):
    """
    计算交叉熵
    
    Args:
        features: 特征张量 [N, D]
        target_probs: 目标概率分布，如果为None则使用均匀分布
        eps: 避免log(0)的小值
        
    Returns:
        cross_entropy: 交叉熵值
    """
    if target_probs is None:
        # 使用均匀分布作为目标
        target_probs = torch.ones_like(features) / features.shape[-1]
    
    probs = F.softmax(features, dim=-1)
    probs = torch.clamp(probs, min=eps)
    
    cross_entropy = -torch.sum(target_probs * torch.log(probs), dim=-1)
    return torch.mean(cross_entropy)

def compute_feature_diversity_entropy(features, eps=1e-8):
    """
    计算特征多样性熵
    
    Args:
        features: 特征张量 [N, D]
        eps: 避免log(0)的小值
        
    Returns:
        diversity_entropy: 特征多样性熵
    """
    # 归一化特征
    normalized_features = F.normalize(features, dim=-1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.mm(normalized_features, normalized_features.t())
    
    # 将对角线元素设为0（避免自相似性）
    mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
    similarity_matrix = similarity_matrix * (1 - mask)
    
    # 计算每个特征的平均相似度
    avg_similarity = torch.sum(similarity_matrix, dim=1) / (similarity_matrix.shape[1] - 1)
    
    # 将相似度转换为多样性
    diversity = 1 - avg_similarity
    
    # 计算多样性的熵
    diversity_probs = F.softmax(diversity, dim=0)
    diversity_probs = torch.clamp(diversity_probs, min=eps)
    
    diversity_entropy = -torch.sum(diversity_probs * torch.log(diversity_probs))
    
    return diversity_entropy

def compute_weighted_entropy(features, weights=None, eps=1e-8):
    """
    计算加权熵
    
    Args:
        features: 特征张量 [N, D]
        weights: 权重张量 [N, 1]，如果为None则使用均匀权重
        eps: 避免log(0)的小值
        
    Returns:
        weighted_entropy: 加权熵值
    """
    if weights is None:
        weights = torch.ones(features.shape[0], 1, device=features.device)
    
    # 归一化权重
    weights = weights / torch.sum(weights)
    
    # 计算每个特征的熵
    probs = F.softmax(features, dim=-1)
    probs = torch.clamp(probs, min=eps)
    
    individual_entropy = -torch.sum(probs * torch.log(probs), dim=-1, keepdim=True)
    
    # 计算加权熵
    weighted_entropy = torch.sum(weights * individual_entropy)
    
    return weighted_entropy

def compute_conditional_entropy(features, condition_features, eps=1e-8):
    """
    计算条件熵
    
    Args:
        features: 特征张量 [N, D]
        condition_features: 条件特征张量 [N, C]
        eps: 避免log(0)的小值
        
    Returns:
        conditional_entropy: 条件熵值
    """
    # 计算联合概率分布
    joint_features = torch.cat([features, condition_features], dim=-1)
    joint_probs = F.softmax(joint_features, dim=-1)
    
    # 计算条件概率分布
    condition_probs = F.softmax(condition_features, dim=-1)
    
    # 计算条件熵
    conditional_entropy = -torch.sum(joint_probs * torch.log(joint_probs + eps), dim=-1)
    conditional_entropy += torch.sum(condition_probs * torch.log(condition_probs + eps), dim=-1)
    
    return torch.mean(conditional_entropy)

def compute_mutual_information(features1, features2, eps=1e-8):
    """
    计算互信息
    
    Args:
        features1: 第一个特征张量 [N, D1]
        features2: 第二个特征张量 [N, D2]
        eps: 避免log(0)的小值
        
    Returns:
        mutual_info: 互信息值
    """
    # 计算各个熵
    entropy1 = compute_shannon_entropy(features1, eps)
    entropy2 = compute_shannon_entropy(features2, eps)
    
    # 计算联合熵
    joint_features = torch.cat([features1, features2], dim=-1)
    joint_entropy = compute_shannon_entropy(joint_features, eps)
    
    # 互信息 = H(X) + H(Y) - H(X,Y)
    mutual_info = entropy1 + entropy2 - joint_entropy
    
    return mutual_info

def compute_entropy_metrics(features, weights=None, return_dict=True):
    """
    计算多种熵指标
    
    Args:
        features: 特征张量 [N, D]
        weights: 权重张量 [N, 1]
        return_dict: 是否返回字典格式
        
    Returns:
        entropy_metrics: 熵指标字典或元组
    """
    metrics = {}
    
    # Shannon熵
    metrics['shannon_entropy'] = compute_shannon_entropy(features)
    
    # 交叉熵
    metrics['cross_entropy'] = compute_cross_entropy(features)
    
    # 特征多样性熵
    metrics['diversity_entropy'] = compute_feature_diversity_entropy(features)
    
    # 加权熵
    metrics['weighted_entropy'] = compute_weighted_entropy(features, weights)
    
    # 最大可能熵（均匀分布）
    max_entropy = torch.log(torch.tensor(features.shape[-1], dtype=torch.float, device=features.device))
    metrics['max_entropy'] = max_entropy
    
    # 归一化熵（相对于最大熵）
    metrics['normalized_entropy'] = metrics['shannon_entropy'] / max_entropy
    
    if return_dict:
        return metrics
    else:
        return tuple(metrics.values())

def analyze_entropy_trends(entropy_history, window_size=10):
    """
    分析熵的变化趋势
    
    Args:
        entropy_history: 熵值历史列表
        window_size: 滑动窗口大小
        
    Returns:
        trend_analysis: 趋势分析结果
    """
    if len(entropy_history) < window_size:
        return {"error": "历史数据不足"}
    
    entropy_tensor = torch.tensor(entropy_history)
    
    # 计算滑动平均
    moving_avg = torch.conv1d(
        entropy_tensor.unsqueeze(0).unsqueeze(0),
        torch.ones(1, 1, window_size) / window_size,
        padding=window_size // 2
    ).squeeze()
    
    # 计算趋势（线性回归斜率）
    x = torch.arange(len(entropy_tensor), dtype=torch.float)
    slope = torch.polyfit(x, entropy_tensor, 1)[0]
    
    # 计算稳定性（标准差）
    stability = torch.std(entropy_tensor)
    
    # 计算变化率
    change_rate = (entropy_tensor[-1] - entropy_tensor[0]) / len(entropy_tensor)
    
    return {
        "moving_average": moving_avg.tolist(),
        "trend_slope": slope.item(),
        "stability": stability.item(),
        "change_rate": change_rate.item(),
        "is_increasing": slope > 0,
        "is_stable": stability < 0.1  # 阈值可调整
    } 