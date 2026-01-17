#!/usr/bin/env python3
"""
Deep Learning-based Loop Closure Detection
使用卷积神经网络提取特征，计算两个图像的相似度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import cv2


class SiameseNetwork(nn.Module):
    """
    孪生网络用于计算两个图像特征的相似度
    """
    def __init__(self, feature_dim: int = 256, input_channels: int = 3):
        super(SiameseNetwork, self).__init__()
        self.feature_dim = feature_dim
        
        # 共享的卷积编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResBlock 1
            self._make_res_block(64, 64, stride=1),
            self._make_res_block(64, 128, stride=2),
            
            # ResBlock 2
            self._make_res_block(128, 128, stride=1),
            self._make_res_block(128, 256, stride=2),
            
            # ResBlock 3
            self._make_res_block(256, 256, stride=1),
            self._make_res_block(256, 512, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def _make_res_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码图像为特征向量
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            特征向量 [B, feature_dim]
        """
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        features = self.feature_projection(features)
        return F.normalize(features, p=2, dim=1)
    
    def compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        计算两个特征向量的相似度（余弦相似度）
        
        Args:
            feat1: 特征向量1 [B, feature_dim]
            feat2: 特征向量2 [B, feature_dim]
            
        Returns:
            相似度 [B]
        """
        return F.cosine_similarity(feat1, feat2, dim=1)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x1: 图像1 [B, C, H, W]
            x2: 图像2 [B, C, H, W]
            
        Returns:
            相似度 [B]
        """
        feat1 = self.encode(x1)
        feat2 = self.encode(x2)
        return self.compute_similarity(feat1, feat2)


class DeepLoopDetector:
    """
    基于深度学习的回环检测
    """
    def __init__(
        self,
        model_path: str = None,
        feature_dim: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        similarity_threshold: float = 0.5,
        top_k: int = 10
    ):
        """
        初始化回环检测器
        
        Args:
            model_path: 预训练模型路径
            feature_dim: 特征维度
            device: 计算设备 (cuda/cpu)
            similarity_threshold: 相似度阈值
            top_k: 返回最相似的前K个候选
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        
        # 初始化网络
        self.siamese = SiameseNetwork(feature_dim=feature_dim).to(device)
        
        if model_path:
            self.siamese.load_state_dict(torch.load(model_path, map_location=device))
        
        self.siamese.eval()
        
        # 特征数据库
        self.feature_database: List[torch.Tensor] = []
        self.frame_ids: List[int] = []
        
    def add_frame(self, image: np.ndarray, frame_id: int) -> None:
        """
        添加帧到特征数据库
        
        Args:
            image: 输入图像 [H, W, 3] (uint8)
            frame_id: 帧ID
        """
        # 预处理图像
        image_tensor = self._preprocess_image(image)
        
        # 提取特征
        with torch.no_grad():
            feature = self.siamese.encode(image_tensor)
        
        self.feature_database.append(feature)
        self.frame_ids.append(frame_id)
    
    def detect_loop_closure(
        self,
        query_image: np.ndarray,
        query_id: int,
        min_distance: int = 50
    ) -> List[Dict]:
        """
        检测回环闭合
        
        Args:
            query_image: 查询图像 [H, W, 3]
            query_id: 查询帧ID
            min_distance: 最小时间距离（避免检测相邻帧）
            
        Returns:
            检测到的回环候选列表，每个候选包含:
            {
                'query_id': int,
                'reference_id': int,
                'similarity': float,
                'rank': int
            }
        """
        if len(self.feature_database) == 0:
            return []
        
        # 提取查询图像特征
        query_tensor = self._preprocess_image(query_image)
        with torch.no_grad():
            query_feature = self.siamese.encode(query_tensor)
        
        # 计算与数据库中所有特征的相似度
        similarities = []
        for ref_feature in self.feature_database:
            sim = F.cosine_similarity(query_feature, ref_feature, dim=1).item()
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # 过滤：只考虑满足时间距离的候选
        valid_indices = []
        for idx, ref_id in enumerate(self.frame_ids):
            if abs(query_id - ref_id) >= min_distance:
                valid_indices.append(idx)
        
        if len(valid_indices) == 0:
            return []
        
        # 获取最相似的候选
        valid_similarities = similarities[valid_indices]
        top_indices = np.argsort(-valid_similarities)[:self.top_k]
        
        loop_candidates = []
        for rank, idx in enumerate(top_indices):
            actual_idx = valid_indices[idx]
            similarity = similarities[actual_idx]
            
            if similarity >= self.similarity_threshold:
                loop_candidates.append({
                    'query_id': query_id,
                    'reference_id': self.frame_ids[actual_idx],
                    'similarity': float(similarity),
                    'rank': rank + 1
                })
        
        return loop_candidates
    
    def query_database(self, query_image: np.ndarray, top_k: int = None) -> List[Dict]:
        """
        查询相似的图像
        
        Args:
            query_image: 查询图像
            top_k: 返回最相似的K个（默认使用构造函数中设置的值）
            
        Returns:
            最相似的图像列表
        """
        if top_k is None:
            top_k = self.top_k
        
        query_tensor = self._preprocess_image(query_image)
        with torch.no_grad():
            query_feature = self.siamese.encode(query_tensor)
        
        similarities = []
        for ref_feature in self.feature_database:
            sim = F.cosine_similarity(query_feature, ref_feature, dim=1).item()
            similarities.append(sim)
        
        similarities = np.array(similarities)
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'database_id': self.frame_ids[idx],
                'similarity': float(similarities[idx]),
                'rank': rank + 1
            })
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像为模型输入格式
        
        Args:
            image: 输入图像 [H, W, 3] (uint8)
            
        Returns:
            预处理后的张量 [1, 3, H, W]
        """
        # 确保是BGR格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 调整大小为224x224
        image = cv2.resize(image, (224, 224))
        
        # 标准化
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # 转换为张量 [3, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        # 添加批次维度 [1, 3, H, W]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def clear_database(self) -> None:
        """清空特征数据库"""
        self.feature_database = []
        self.frame_ids = []
    
    def save_database(self, save_path: str) -> None:
        """保存特征数据库"""
        data = {
            'features': [f.cpu().numpy() for f in self.feature_database],
            'frame_ids': self.frame_ids
        }
        np.save(save_path, data, allow_pickle=True)
    
    def load_database(self, load_path: str) -> None:
        """加载特征数据库"""
        data = np.load(load_path, allow_pickle=True).item()
        self.feature_database = [
            torch.from_numpy(f).to(self.device) for f in data['features']
        ]
        self.frame_ids = data['frame_ids']


# 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = DeepLoopDetector(
        feature_dim=256,
        similarity_threshold=0.5
    )
    
    # 模拟添加图像帧
    for i in range(10):
        dummy_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detector.add_frame(dummy_image, frame_id=i)
    
    # 查询相似图像
    query_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    results = detector.query_database(query_image, top_k=5)
    
    print("查询结果:")
    for result in results:
        print(f"  帧 {result['database_id']}: 相似度 {result['similarity']:.4f}")
