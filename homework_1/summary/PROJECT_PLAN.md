# LVI-SAM 校园场景SLAM优化项目

## 项目概述

基于LVI-SAM框架，针对校园场景数据进行多模块优化研究。通过改进视觉里程计、激光里程计、回环检测和因子图优化等模块，提升SLAM系统的定位精度和鲁棒性。

### 硬件配置
- **RGB-D相机**: RealSense D435i (RGB + 深度)
- **3D LiDAR**: Velodyne VLP-16 (32线激光雷达)
- **IMU**: MTi-680G 惯性测量单元
- **地面真值**: RTK GPS (cm级精度)

### 评估基准
- **APE (Absolute Pose Error)**: 绝对位姿误差
- **ATE (Absolute Trajectory Error)**: 绝对轨迹误差
- **ARE (Absolute Rotation Error)**: 绝对旋转误差
- **评估工具**: EVO (轨迹评估框架)
- **指标**: RMSE值

---

## 项目结构

```
lvi-sam/
├── src/LVI-SAM-Easyused/
│   ├── src/
│   │   ├── lidar_odometry/          # 激光里程计（改进）
│   │   └── visual_odometry/         # 视觉里程计（改进）
│   │       ├── visual_feature/      # 特征提取改进
│   │       ├── visual_estimator/    # 深度估计改进
│   │       └── visual_loop/         # 回环检测（Deep Learning）
│   ├── config/                      # 传感器配置
│   └── launch/                      # 启动文件
│
├── improvements/                    # 改进模块目录（新增）
│   ├── visual_feature_enhanced/     # 增强特征提取
│   ├── depth_estimation/            # 深度估计方法
│   ├── dynamic_removal/             # 动态物体去除
│   ├── point_cloud_matching/        # 点云匹配改进
│   ├── loop_closure_dl/             # 深度学习回环检测
│   └── factor_graph_opt/            # 因子图优化
│
├── experiments/                     # 实验管理
│   ├── baseline/                    # 基线结果（原LVI-SAM）
│   ├── improved_v1/                 # 改进版本1
│   ├── improved_v2/                 # 改进版本2
│   └── evaluation/                  # 评估结果
│       ├── trajectories/            # 生成的轨迹
│       ├── metrics/                 # 性能指标
│       └── plots/                   # 可视化图表
│
└── reports/                         # 实验报告
    └── final_report.md              # 最终报告
```

---

## 核心改进方案

### 1. 视觉里程计改进 (Visual Odometry Enhancement)

#### 1.1 特征点提取改进
**当前方案**: KLT特征跟踪 + Harris角点检测

**改进方向**:
- [ ] **SIFT/SURF特征**: 更具鲁棒性的描述符
- [ ] **ORB特征**: 实时性与鲁棒性平衡
- [ ] **SuperPoint**: 深度学习特征点检测和描述
- [ ] **特征点均匀分布**: 改进特征选择算法

**实现步骤**:
```cpp
// 新建 visual_feature_enhanced/enhanced_tracker.h/cpp
class EnhancedFeatureTracker {
  - supportMultipleDescriptors()  // 支持多种特征描述符
  - improveFeatureDistribution()  // 均匀分布策略
  - adaptiveTracking()            // 自适应跟踪
};
```

#### 1.2 深度估计改进
**当前方案**: LiDAR投影得到的深度

**改进方向**:
- [ ] **单目深度估计**: CNN基础深度预测
- [ ] **立体深度增强**: 基于RGB-D的深度补全
- [ ] **融合策略**: LiDAR+单目深度融合
- [ ] **无序点优先**: 处理LiDAR稀疏区域

**实现步骤**:
```cpp
// 新建 depth_estimation/depth_predictor.h/cpp
class DepthEstimator {
  - monocularDepthPrediction()     // 单目深度预测
  - stereoDepthRefinement()        // 立体深度精化
  - liddarDepthFusion()            // LiDAR融合
};
```

---

### 2. 激光里程计改进 (LiDAR Odometry Enhancement)

#### 2.1 动态物体去除
**背景**: 校园场景包含行人、车辆等动态物体

**改进方向**:
- [ ] **运动一致性检测**: 光流判断动态点
- [ ] **点云分割**: 欧式聚类识别动态物体
- [ ] **残差分析**: 基于配准残差检测动态点
- [ ] **时间一致性**: 多帧信息检测变化

**实现步骤**:
```cpp
// 新建 dynamic_removal/dynamic_filter.h/cpp
class DynamicObjectRemover {
  - motionConsistencyCheck()      // 运动一致性
  - euclideanClustering()         // 聚类分割
  - residualAnalysis()            // 残差分析
  - temporalConsistency()         // 时间一致性
};
```

#### 2.2 点云匹配改进
**当前方案**: 基于特征的ICP/GICP

**改进方向**:
- [ ] **NDT (Normal Distribution Transform)**: 概率匹配
- [ ] **Generalized-ICP**: 改进的变体
- [ ] **Point-to-Plane ICP**: 更精确的配准
- [ ] **多层级配准**: 粗到细的配准策略
- [ ] **深度学习配准**: 神经网络加速ICP

**实现步骤**:
```cpp
// 新建 point_cloud_matching/advanced_matcher.h/cpp
class AdvancedPointCloudMatcher {
  - ndtMatching()                 // NDT配准
  - point2planeICP()              // P2L-ICP
  - multiScaleRegistration()      // 多层级配准
  - learningBasedMatching()       // 深度学习配准
};
```

---

### 3. 回环检测改进 (Loop Closure Detection)

#### 3.1 深度学习回环检测
**当前方案**: 基于DBoW2词袋模型

**改进方向**:
- [ ] **深度神经网络**: CNN特征提取 (ResNet, VGG)
- [ ] **Siamese网络**: 两个图像相似度计算
- [ ] **3D CNN**: 利用深度图像信息
- [ ] **光流结合**: 增强时间一致性检测

**实现步骤**:
```python
# 新建 loop_closure_dl/deep_loop_detector.py
class DeepLoopDetector:
    def __init__(self, model_path):
        self.encoder = ResNet18(pretrained=True)  # 特征编码
        self.siamese = SiameseNetwork()           # 相似度计算
        
    def extract_features(self, image):
        return self.encoder(image)
    
    def compute_similarity(self, feat1, feat2):
        return self.siamese(feat1, feat2)
    
    def detect_loop(self, current_frame, reference_frames):
        # 检测回环
        pass
```

---

### 4. 因子图优化改进 (Factor Graph Optimization)

#### 4.1 新增约束因子
- [ ] **视觉回环因子**: 从深度学习检测的回环
- [ ] **IMU预积分因子**: 改进误差模型
- [ ] **点面因子**: 改进的LiDAR点云因子
- [ ] **光度度量因子**: 直接法优化

**实现步骤**:
```cpp
// 新建 factor_graph_opt/custom_factors.h
class DeepLoopFactor : public gtsam::NoiseModelFactor2 {
    // 深度学习回环约束因子
};

class ImprovedLidarFactor : public gtsam::NoiseModelFactor1 {
    // 改进的LiDAR点面因子
};

class PhotometricFactor : public gtsam::NoiseModelFactor1 {
    // 光度度量因子
};
```

#### 4.2 优化算法改进
- [ ] **更新排序**: 优化因子处理顺序
- [ ] **增量式求解**: 改进边缘化策略
- [ ] **鲁棒性提升**: 动态噪声模型调整

---

## 实验方案

### Phase 1: 基线建立 (Week 1-2)
```bash
# 1. 运行原始LVI-SAM获取基线
roslaunch lvi_sam Husky.launch
rosbag play campus_data.bag

# 2. 提取轨迹并评估
python eval.py --trajectory results/trajectory.txt \
               --ground_truth ground_truth.txt \
               --output baseline_metrics.json

# 3. 保存基线结果
mkdir -p experiments/baseline
cp results/* experiments/baseline/
```

### Phase 2: 单模块改进测试 (Week 3-6)
```bash
# 2.1 测试改进的视觉里程计
roslaunch lvi_sam_enhanced visual_only.launch
rosbag play campus_data.bag
# 评估...

# 2.2 测试改进的激光里程计
roslaunch lvi_sam_enhanced lidar_only.launch
rosbag play campus_data.bag
# 评估...

# 2.3 测试改进的回环检测
roslaunch lvi_sam_enhanced loop_detection.launch
rosbag play campus_data.bag
# 评估...
```

### Phase 3: 综合改进测试 (Week 7-8)
```bash
# 3.1 全模块改进版本
roslaunch lvi_sam_enhanced full_improved.launch
rosbag play campus_data.bag

# 3.2 超参数调优
python hyperparameter_tuning.py

# 3.3 最终评估
python final_evaluation.py
```

---

## 性能评估方法

### 使用EVO工具

#### 安装
```bash
pip install evo
```

#### 评估脚本
```python
# evaluation/evaluate_trajectory.py
#!/usr/bin/env python3
import evo
from evo.tools import file_interface
from evo.tools import plot
from evo.main_ape import ape
from evo.main_ate import ate

def evaluate(estimated_traj, reference_traj, output_dir):
    # 加载轨迹
    traj_ref = file_interface.read_tum_trajectory_file(reference_traj)
    traj_est = file_interface.read_tum_trajectory_file(estimated_traj)
    
    # 计算APE (绝对位姿误差)
    ape_result = ape(traj_ref, traj_est)
    print(f"APE RMSE: {ape_result.stats['rmse']:.4f} m")
    print(f"APE Mean: {ape_result.stats['mean']:.4f} m")
    print(f"APE Median: {ape_result.stats['median']:.4f} m")
    
    # 计算ATE (绝对轨迹误差)
    ate_result = ate(traj_ref, traj_est)
    print(f"ATE RMSE: {ate_result.stats['rmse']:.4f} m")
    
    # 计算ARE (绝对旋转误差)
    # ... 实现旋转误差计算
    
    # 绘制结果
    plot.plot_mode = "xy"
    fig = plot.plot_trajectory(traj_ref, traj_est)
    fig.savefig(f"{output_dir}/trajectory_comparison.png")
    
    # 保存指标
    metrics = {
        "APE_RMSE": ape_result.stats['rmse'],
        "APE_Mean": ape_result.stats['mean'],
        "ATE_RMSE": ate_result.stats['rmse'],
    }
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    evaluate("results/trajectory.txt", "ground_truth.txt", "experiments/eval_v1")
```

### 评估指标汇总表

| 方法 | APE RMSE (m) | ATE RMSE (m) | ARE RMSE (deg) | 说明 |
|------|-------------|-------------|----------------|------|
| 原始LVI-SAM | - | - | - | 基线方案 |
| +特征增强 | - | - | - | 改进视觉特征 |
| +深度估计 | - | - | - | 改进深度 |
| +动态去除 | - | - | - | 去除动态物体 |
| +点云匹配 | - | - | - | 改进配准 |
| +回环检测 | - | - | - | 深度学习回环 |
| +全模块 | - | - | - | 所有改进 |

---

## 实验报告框架

### 1. 摘要 (Abstract)
- 研究背景与意义
- 主要贡献
- 关键结果

### 2. 引言 (Introduction)
- SLAM技术概述
- 现有方法分析
- 本工作的创新点

### 3. 方法 (Methodology)
- 系统架构图
- 各模块改进详解
- 理论基础

### 4. 实验 (Experiments)
- 数据集描述
- 实验设置
- 结果分析
- 对比分析

### 5. 结论与展望 (Conclusion & Future Work)
- 主要成果总结
- 改进方向
- 实际应用价值

---

## 关键文件清单

### 需要创建的新文件
```
improvements/
├── visual_feature_enhanced/
│   ├── enhanced_tracker.h
│   ├── enhanced_tracker.cpp
│   ├── descriptor_factory.h
│   └── feature_distribution.h
│
├── depth_estimation/
│   ├── depth_predictor.h
│   ├── depth_predictor.cpp
│   ├── monocular_depth.py
│   └── fusion_strategy.h
│
├── dynamic_removal/
│   ├── dynamic_filter.h
│   ├── dynamic_filter.cpp
│   └── motion_consistency.h
│
├── point_cloud_matching/
│   ├── advanced_matcher.h
│   ├── advanced_matcher.cpp
│   ├── ndt_matcher.h
│   └── p2l_icp.h
│
├── loop_closure_dl/
│   ├── deep_loop_detector.py
│   ├── siamese_network.py
│   ├── feature_extractor.py
│   └── loop_db.py
│
└── factor_graph_opt/
    ├── custom_factors.h
    ├── deep_loop_factor.h
    ├── improved_lidar_factor.h
    └── photometric_factor.h

scripts/
├── train_loop_detector.py
├── evaluate_trajectory.py
├── benchmark_all.py
└── generate_report.py
```

### 需要修改的现有文件
```
src/LVI-SAM-Easyused/CMakeLists.txt       # 添加新模块编译
src/LVI-SAM-Easyused/package.xml          # 添加新依赖
launch/improved_full.launch               # 新启动文件
config/campus_params.yaml                 # 校园场景参数
```

---

## 技术栈

### 编程语言
- **C++17**: 核心算法实现
- **Python 3.8+**: 数据处理与评估

### 深度学习框架
- **PyTorch**: 神经网络训练
- **TorchVision**: 计算机视觉模型

### 其他关键库
- **OpenCV**: 图像处理
- **PCL**: 点云处理
- **GTSAM**: 因子图优化
- **EVO**: 轨迹评估
- **TensorFlow**: 可选的模型部署

---

## 注意事项

1. **校园场景特点**
   - 动态物体较多（行人、车辆）
   - 重复纹理区域多
   - GPS信号可能不稳定
   - 需要针对性改进

2. **实验严谨性**
   - 所有对比实验使用相同的数据
   - 固定随机种子以保证可重复性
   - 至少进行3次独立实验取平均值
   - 记录所有超参数配置

3. **计算资源**
   - GPU支持深度学习模型加速
   - 实时性与精度的权衡
   - 内存占用监控

4. **代码规范**
   - 遵循ROS和C++编码规范
   - 完整的注释和文档
   - 单元测试覆盖
   - 版本控制管理

---

## 参考资源

### 相关论文
- LVI-SAM (Shan et al., 2021)
- VINS-Mono (Qin et al., 2018)
- LIO-SAM (Shan et al., 2020)
- SuperPoint (DeTone et al., 2018)
- ORB-SLAM2 (Mur-Artal & Tardós, 2017)

### 工具与框架
- EVO: https://github.com/MichaelGrupp/evo
- PyTorch: https://pytorch.org/
- OpenCV: https://opencv.org/
- PCL: https://pointclouds.org/

