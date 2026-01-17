#!/usr/bin/env python3
"""
视觉里程计改进模块
集成增强特征跟踪器到LVI-SAM
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Feature:
    """特征点数据结构"""
    pt: Tuple[float, float]       # 2D坐标
    track_id: int                 # 跟踪ID
    depth: float = -1             # 深度值
    track_count: int = 0          # 跟踪帧数
    descriptor_type: int = 0      # 0=KLT, 1=ORB
    response: float = 0           # 特征响应
    velocity: Tuple[float, float] = (0, 0)  # 运动速度


class EnhancedVisualOdometry:
    """
    增强视觉里程计模块
    改进项:
    1. 多描述符特征提取 (KLT + ORB)
    2. 自适应特征分布 (网格均匀)
    3. 深度融合
    4. 鲁棒性特征筛选
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化视觉里程计
        
        Args:
            config: 配置字典，包含参数设置
        """
        self.config = config or self._default_config()
        
        # 特征检测参数
        self.n_features = self.config.get('n_features', 150)
        self.max_track_cnt = self.config.get('max_track_cnt', 30)
        self.min_feature_distance = self.config.get('min_feature_distance', 50)
        
        # 网格参数
        self.grid_rows = self.config.get('grid_rows', 4)
        self.grid_cols = self.config.get('grid_cols', 4)
        
        # 状态
        self.next_track_id = 0
        self.prev_gray = None
        self.prev_features: List[Feature] = []
        self.cur_features: List[Feature] = []
        
        # 初始化特征检测器
        self.orb_detector = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # 相机参数
        self.camera_K = None
        self.camera_D = None
        
        logger.info("✓ 增强视觉里程计初始化完成")
    
    @staticmethod
    def _default_config() -> Dict:
        """默认配置"""
        return {
            'n_features': 150,
            'max_track_cnt': 30,
            'min_feature_distance': 50,
            'grid_rows': 4,
            'grid_cols': 4,
            'klt_window_size': 21,
            'klt_max_level': 3,
            'klt_criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            'depth_threshold': (0.1, 10.0),  # 最小和最大深度
        }
    
    def set_camera_model(self, K: np.ndarray, D: np.ndarray):
        """
        设置相机内参
        
        Args:
            K: 内参矩阵 (3x3)
            D: 畸变系数
        """
        self.camera_K = K.copy()
        self.camera_D = D.copy()
    
    def track_image(self, img: np.ndarray, timestamp: float) -> List[Feature]:
        """
        处理新图像，进行特征跟踪
        
        Args:
            img: 输入图像 (BGR或灰度)
            timestamp: 时间戳
            
        Returns:
            当前帧的特征列表
        """
        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # 第一帧处理
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self._detect_new_features(gray)
            return self.cur_features
        
        # 跟踪前一帧的特征
        if self.prev_features:
            self._track_features(self.prev_gray, gray)
        
        # 检测新特征
        new_features = self._detect_new_features(gray)
        
        # 合并新旧特征
        self._merge_features(new_features)
        
        # 保存当前帧
        self.prev_gray = gray.copy()
        self.prev_features = [f for f in self.cur_features]
        
        return self.cur_features
    
    def _track_features(self, prev_gray: np.ndarray, cur_gray: np.ndarray):
        """
        使用KLT跟踪特征点
        
        Args:
            prev_gray: 前一帧灰度图
            cur_gray: 当前帧灰度图
        """
        if not self.prev_features:
            return
        
        # 准备前一帧特征坐标
        prev_pts = np.array([f.pt for f in self.prev_features], dtype=np.float32)
        
        # KLT跟踪
        cur_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray,
            prev_pts, None,
            winSize=(self.config['klt_window_size'], self.config['klt_window_size']),
            maxLevel=self.config['klt_max_level'],
            criteria=self.config['klt_criteria']
        )
        
        # 更新特征
        self.cur_features = []
        for i, (prev_f, st) in enumerate(zip(self.prev_features, status)):
            if st[0]:  # 跟踪成功
                f = Feature(
                    pt=tuple(cur_pts[i]),
                    track_id=prev_f.track_id,
                    depth=prev_f.depth,
                    track_count=prev_f.track_count + 1,
                    descriptor_type=0,  # KLT
                    response=prev_f.response,
                    velocity=(
                        cur_pts[i][0] - prev_pts[i][0],
                        cur_pts[i][1] - prev_pts[i][1]
                    )
                )
                
                # 检查边界
                if self._in_border(f.pt):
                    self.cur_features.append(f)
    
    def _detect_new_features(self, gray: np.ndarray) -> List[Feature]:
        """
        使用ORB检测新特征点（网格均匀分布）
        
        Args:
            gray: 灰度图
            
        Returns:
            新检测到的特征列表
        """
        # ORB特征检测
        kpts = self.orb_detector.detect(gray, None)
        
        if not kpts:
            return []
        
        # 转换为Point2f格式
        pts = np.array([kpt.pt for kpt in kpts], dtype=np.float32)
        responses = np.array([kpt.response for kpt in kpts])
        
        # 网格分配
        h, w = gray.shape
        grid_h = h // self.grid_rows
        grid_w = w // self.grid_cols
        
        grid_features = {}
        for pt, resp in zip(pts, responses):
            grid_x = min(int(pt[0] // grid_w), self.grid_cols - 1)
            grid_y = min(int(pt[1] // grid_h), self.grid_rows - 1)
            grid_id = grid_y * self.grid_cols + grid_x
            
            if grid_id not in grid_features:
                grid_features[grid_id] = []
            grid_features[grid_id].append((pt, resp))
        
        # 从每个网格选择最佳特征
        new_features = []
        features_per_cell = max(1, self.n_features // (self.grid_rows * self.grid_cols))
        
        for cell_pts in grid_features.values():
            # 按响应值排序
            cell_pts.sort(key=lambda x: x[1], reverse=True)
            
            for pt, resp in cell_pts[:features_per_cell]:
                # 检查与现有特征的距离
                valid = True
                for f in self.cur_features:
                    dist = np.linalg.norm(pt - np.array(f.pt))
                    if dist < self.min_feature_distance:
                        valid = False
                        break
                
                if valid and self._in_border(tuple(pt)):
                    new_feat = Feature(
                        pt=tuple(pt),
                        track_id=self.next_track_id,
                        track_count=1,
                        descriptor_type=1,  # ORB
                        response=float(resp)
                    )
                    new_features.append(new_feat)
                    self.next_track_id += 1
        
        return new_features
    
    def _merge_features(self, new_features: List[Feature]):
        """
        合并新检测的特征到当前特征列表
        
        Args:
            new_features: 新检测的特征
        """
        # 添加新特征
        for f in new_features:
            if len(self.cur_features) >= self.n_features:
                break
            self.cur_features.append(f)
        
        # 删除跟踪失败的特征（低track_count的新特征）
        self.cur_features = [
            f for f in self.cur_features
            if not (f.track_count < 2 and f.descriptor_type == 1)
        ]
        
        # 删除长期失踪的特征
        self.cur_features = [
            f for f in self.cur_features
            if f.track_count <= self.max_track_cnt
        ]
    
    def fuse_depth(self, depth_map: np.ndarray):
        """
        融合深度图信息
        
        Args:
            depth_map: 深度图 (float)
        """
        depth_min, depth_max = self.config['depth_threshold']
        
        for f in self.cur_features:
            x, y = int(f.pt[0]), int(f.pt[1])
            
            if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                d = depth_map[y, x]
                if depth_min <= d <= depth_max:
                    f.depth = float(d)
    
    def get_features(self) -> List[Feature]:
        """获取当前特征"""
        return self.cur_features
    
    def get_features_as_array(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取特征为数组格式（兼容LVI-SAM）
        
        Returns:
            (点坐标, ID, 深度)
        """
        if not self.cur_features:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        
        pts = np.array([f.pt for f in self.cur_features], dtype=np.float32)
        ids = np.array([f.track_id for f in self.cur_features], dtype=np.int32)
        depths = np.array([f.depth for f in self.cur_features], dtype=np.float32)
        
        return pts, ids, depths
    
    def _in_border(self, pt: Tuple[float, float], border: int = 1) -> bool:
        """检查点是否在图像边界内"""
        if self.prev_gray is None:
            return True
        h, w = self.prev_gray.shape
        return border <= pt[0] < w - border and border <= pt[1] < h - border
    
    def get_statistics(self) -> Dict:
        """获取特征统计信息"""
        if not self.cur_features:
            return {}
        
        track_counts = [f.track_count for f in self.cur_features]
        has_depth = sum(1 for f in self.cur_features if f.depth > 0)
        
        return {
            'n_features': len(self.cur_features),
            'avg_track_count': np.mean(track_counts),
            'max_track_count': np.max(track_counts),
            'features_with_depth': has_depth,
            'descriptor_types': {
                'klt': sum(1 for f in self.cur_features if f.descriptor_type == 0),
                'orb': sum(1 for f in self.cur_features if f.descriptor_type == 1),
            }
        }


def compare_trackers(img: np.ndarray, 
                     enhanced_vo: EnhancedVisualOdometry,
                     baseline_params: Dict = None) -> Dict:
    """
    对比增强版本和基础版本
    
    Args:
        img: 输入图像
        enhanced_vo: 增强视觉里程计
        baseline_params: 基础版本参数
        
    Returns:
        对比结果字典
    """
    # 增强版本
    enhanced_features = enhanced_vo.track_image(img, 0)
    enhanced_stats = enhanced_vo.get_statistics()
    
    # 基础版本（标准ORB）
    orb_basic = cv2.ORB_create(nfeatures=150)
    kpts_basic = orb_basic.detect(img, None)
    
    basic_stats = {
        'n_features': len(kpts_basic),
        'avg_response': np.mean([kpt.response for kpt in kpts_basic]) if kpts_basic else 0,
    }
    
    return {
        'enhanced': enhanced_stats,
        'baseline': basic_stats,
        'improvement': {
            'n_features_ratio': enhanced_stats['n_features'] / max(basic_stats['n_features'], 1),
            'stability': enhanced_stats['avg_track_count'],
        }
    }


if __name__ == '__main__':
    # 示例使用
    print("增强视觉里程计模块")
    
    # 创建实例
    vo = EnhancedVisualOdometry()
    
    # 模拟图像处理
    test_img = cv2.imread('/home/cx/lvi-sam/test_image.jpg')
    if test_img is not None:
        features = vo.track_image(test_img, 0)
        stats = vo.get_statistics()
        
        print(f"\n特征统计:")
        print(f"  特征数: {stats['n_features']}")
        print(f"  平均跟踪计数: {stats['avg_track_count']:.2f}")
        print(f"  有深度信息: {stats['features_with_depth']}")
