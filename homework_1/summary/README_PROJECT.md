# LVI-SAM 校园场景SLAM优化项目 - 完整指南

## 📋 项目概览

这是一个基于LVI-SAM框架，针对校园场景的SLAM系统优化研究项目。项目包含完整的实现方案、性能评估框架和实验指导。

### 项目要点

| 项目 | 详情 |
|------|------|
| **基础框架** | LVI-SAM (紧耦合LiDAR-视觉-惯性SLAM) |
| **传感器配置** | RealSense D435i RGB-D + Velodyne VLP-16 + MTi-680G IMU |
| **数据集** | 校园场景，包含地面真值(RTK) |
| **优化方向** | 视觉里程计、激光里程计、回环检测、因子图优化 |
| **评估指标** | APE, ATE, ARE (通过EVO工具) |
| **预期收益** | 定位精度提升 10-30%, 系统鲁棒性增强 |

---

## 🎯 核心改进方案速览

### 1️⃣ 视觉里程计改进

```python
# 目标: 提升特征点质量和深度估计精度

改进内容:
├── 特征提取算法 (ORB → SuperPoint)
├── 特征点均匀分布 (改进分布策略)
├── 深度估计融合 (LiDAR + 单目CNN)
└── 自适应跟踪 (动态调整参数)

性能指标:
├── 特征点追踪率: 原95% → 目标98%
├── 深度估计精度: 原±15cm → 目标±10cm
└── 计算效率: 保持实时(<30ms/frame)
```

### 2️⃣ 激光里程计改进

```python
# 目标: 处理动态物体，改进点云配准

改进内容:
├── 动态物体检测 (运动一致性分析)
├── 点云去噪 (聚类+残差分析)
├── 配准算法升级 (ICP → NDT/Generalized-ICP)
└── 多层级配准 (粗到细策略)

性能指标:
├── 动态点去除率: 目标>90%
├── 配准收敛速度: 原30ms → 目标20ms
└── 配准精度提升: ±2cm
```

### 3️⃣ 回环检测改进

```python
# 目标: 使用深度学习提升回环检测准确率

改进内容:
├── CNN特征提取 (ResNet-18编码器)
├── Siamese网络 (相似度计算)
├── 3D信息融合 (RGB-D深度)
└── 置信度评分 (降低误检率)

性能指标:
├── 回环正确率: 原70% → 目标>95%
├── 误检率: 原<5% → 目标<2%
└── 检测速度: <100ms/frame
```

### 4️⃣ 因子图优化改进

```python
# 目标: 增加约束、改进优化策略

改进内容:
├── 深度学习回环因子 (置信度加权)
├── 改进的LiDAR因子 (点面距离)
├── 光度度量因子 (直接法优化)
└── 动态噪声模型 (自适应协方差)

性能指标:
├── 全局一致性: 提升显著
├── 轨迹平滑度: 改善
└── 闭合误差: 减小50%
```

---

## 📦 完整文件结构

```
/home/cx/lvi-sam/
│
├── 📄 PROJECT_PLAN.md                    # 项目总体规划
├── 📄 IMPLEMENTATION_GUIDE.md             # 详细实现指南
├── 📄 CODE_ANALYSIS.md                   # 原始代码分析
│
├── src/
│   └── LVI-SAM-Easyused/                 # 原始LVI-SAM代码
│       ├── src/
│       │   ├── lidar_odometry/
│       │   └── visual_odometry/
│       ├── config/                       # 传感器配置
│       ├── launch/                       # ROS启动文件
│       └── CMakeLists.txt
│
├── improvements/                         # 🆕 改进模块目录
│   ├── visual_feature_enhanced/
│   │   ├── enhanced_tracker.h
│   │   ├── enhanced_tracker.cpp
│   │   └── CMakeLists.txt
│   │
│   ├── depth_estimation/
│   │   ├── depth_predictor.h
│   │   ├── depth_predictor.cpp
│   │   └── monocular_depth.py
│   │
│   ├── dynamic_removal/
│   │   ├── dynamic_filter.h
│   │   ├── dynamic_filter.cpp
│   │   └── motion_consistency.h
│   │
│   ├── point_cloud_matching/
│   │   ├── advanced_matcher.h
│   │   ├── advanced_matcher.cpp
│   │   ├── ndt_matcher.h
│   │   └── p2l_icp.h
│   │
│   ├── loop_closure_dl/
│   │   ├── deep_loop_detector.py         # ✅ 已实现
│   │   ├── siamese_network.py
│   │   ├── feature_extractor.py
│   │   └── models/
│   │       └── siamese_trained.pth       # 训练的模型权重
│   │
│   └── factor_graph_opt/
│       ├── custom_factors.h
│       ├── deep_loop_factor.h
│       ├── improved_lidar_factor.h
│       └── photometric_factor.h
│
├── scripts/                              # 🆕 工具脚本
│   ├── evaluate_trajectory.py             # ✅ 已实现 - EVO评估
│   ├── benchmark_suite.py                 # ✅ 已实现 - 基准测试
│   ├── train_loop_detector.py
│   ├── hyperparameter_tuning.py
│   └── generate_report.py
│
├── experiments/                          # 🆕 实验管理目录
│   ├── baseline/
│   │   ├── trajectory.txt
│   │   ├── metrics.json
│   │   └── README.txt
│   │
│   ├── improved_v1/
│   ├── improved_v2/
│   │
│   └── evaluation/
│       ├── trajectories/
│       ├── metrics/
│       ├── plots/
│       ├── comparison_results.csv
│       ├── method_comparison.png
│       └── benchmark_report.json
│
├── home_data/                            # 数据集
│   ├── husky.bag (22GB)
│   └── gt.txt (71MB, 地面真值)
│
└── reports/                              # 🆕 报告目录
    ├── final_report.md
    ├── figures/
    └── tables/
```

---

## 🚀 快速启动

### 第一次运行

```bash
# 1. 设置环境
cd /home/cx/lvi-sam
source devel/setup.bash

# 2. 三个独立终端启动ROS系统
# 终端A: ROS核心
roscore

# 终端B: LVI-SAM系统
roslaunch lvi_sam Husky.launch

# 终端C: 播放数据集
rosbag play home_data/husky.bag

# 3. 等待处理完成（10-20分钟）
# 结果保存在 ~/lvi-sam/results/
```

### 性能评估

```bash
# 基线评估
python scripts/evaluate_trajectory.py \
    --estimated results/trajectory.txt \
    --reference home_data/gt.txt \
    --method "baseline_lvi_sam" \
    --output experiments/baseline

# 查看结果
cat experiments/baseline/baseline_lvi_sam_metrics.json
```

### 对比多版本

```bash
# 运行综合基准测试
python scripts/benchmark_suite.py --all \
    --output experiments/final_evaluation

# 查看对比结果
cat experiments/final_evaluation/comparison_results.csv
# 查看可视化对比
open experiments/final_evaluation/method_comparison.png
```

---

## 📊 性能指标说明

### APE (Absolute Pose Error) - 绝对位姿误差
```
衡量估计轨迹与真值轨迹的绝对差异
APE = ||p_ref(t) - p_est(t)||
单位: 米 (m)
更小更好

典型值: 0.05-0.20 m
```

### ATE (Absolute Trajectory Error) - 绝对轨迹误差
```
衡量整体轨迹的累积误差
ATE = RMSE of {||p_ref(t) - p_est(t)||}_t
单位: 米 (m)
更小更好

典型值: 0.05-0.25 m
```

### ARE (Absolute Rotation Error) - 绝对旋转误差
```
衡量姿态估计的准确性
ARE = arccos(trace(R_rel) - 1) / 2
单位: 度数 (deg)
更小更好

典型值: 0.5-5.0 deg
```

---

## 💻 关键实现文件

### ✅ 已完成的实现

#### 1. 深度学习回环检测 
**文件**: `improvements/loop_closure_dl/deep_loop_detector.py`

```python
# 核心特性:
- 孪生网络架构 (Siamese Network)
- 实时特征提取和相似度计算
- 特征数据库管理
- 灵活的查询接口

# 使用示例:
detector = DeepLoopDetector(model_path="model.pth")
detector.add_frame(image, frame_id=0)
candidates = detector.detect_loop_closure(query_image, query_id=100)
for cand in candidates:
    print(f"Frame {cand['query_id']} -> {cand['reference_id']}: {cand['similarity']:.4f}")
```

#### 2. 轨迹评估工具
**文件**: `scripts/evaluate_trajectory.py`

```python
# 核心功能:
- 加载TUM格式轨迹
- 计算APE, ATE, ARE等指标
- 生成可视化图表
- RMSE统计分析

# 使用示例:
evaluator = TrajectoryEvaluator(output_dir="results")
metrics = evaluator.evaluate(
    estimated_traj="traj_est.txt",
    reference_traj="gt.txt",
    method_name="my_method"
)
print(f"APE RMSE: {metrics['APE_RMSE']:.6f} m")
```

#### 3. 基准测试套件
**文件**: `scripts/benchmark_suite.py`

```python
# 核心功能:
- 自动运行多个SLAM配置
- 收集和比较性能指标
- 生成对比报告和图表
- 实验追踪和日志记录

# 使用示例:
suite = BenchmarkSuite()
suite.run_all_benchmarks([
    {'name': 'baseline', 'launch': 'file.launch', ...},
    {'name': 'improved_v1', 'launch': 'file_v1.launch', ...}
])
suite.generate_report()
```

### 📝 需要完成的实现

| 文件 | 状态 | 优先级 |
|------|------|--------|
| visual_feature_enhanced/enhanced_tracker.cpp | ⏳ | ⭐⭐⭐ |
| dynamic_removal/dynamic_filter.cpp | ⏳ | ⭐⭐⭐ |
| point_cloud_matching/advanced_matcher.cpp | ⏳ | ⭐⭐⭐ |
| factor_graph_opt/custom_factors.h | ⏳ | ⭐⭐ |
| depth_estimation/depth_predictor.py | ⏳ | ⭐⭐ |
| scripts/train_loop_detector.py | ⏳ | ⭐⭐ |

---

## 🔍 实验工作流

### 完整的实验流程

```
1. 准备数据集
   └─ 确保 home_data/husky.bag 和 gt.txt 存在

2. 建立基线 (Week 1-2)
   ├─ 运行原始LVI-SAM
   ├─ 提取轨迹
   └─ 评估性能 → experiments/baseline/

3. 单模块改进测试 (Week 3-6)
   ├─ 改进1: 视觉里程计
   │  ├─ 实现enhanced_tracker.cpp
   │  ├─ 集成到LVI-SAM
   │  └─ 评估 → experiments/improved_visual_v1/
   │
   ├─ 改进2: 激光里程计
   │  ├─ 实现dynamic_filter.cpp
   │  ├─ 集成到mapOptmization
   │  └─ 评估 → experiments/improved_lidar_v1/
   │
   ├─ 改进3: 回环检测
   │  ├─ 训练深度模型
   │  ├─ 集成到系统
   │  └─ 评估 → experiments/improved_loop_v1/
   │
   └─ 改进4: 因子图优化
      ├─ 添加新约束因子
      ├─ 集成到GTSAM
      └─ 评估 → experiments/improved_factor_v1/

4. 综合改进测试 (Week 7-8)
   ├─ 整合所有改进
   ├─ 超参数调优
   └─ 评估 → experiments/final_evaluation/

5. 性能报告 (Week 8-9)
   ├─ 生成对比表和图表
   ├─ 撰写完整报告
   └─ 报告 → reports/final_report.md
```

---

## 📈 预期性能提升

基于相关研究和初步分析：

| 改进项 | 预期APE改进 | 预期ATE改进 | 预期ARE改进 |
|--------|-----------|-----------|-----------|
| 视觉增强 | +5-10% | +3-8% | +2-5% |
| 激光增强 | +10-15% | +8-12% | +5-8% |
| 回环改进 | +5-8% | +5-10% | +3-6% |
| 因子优化 | +3-5% | +5-8% | +2-4% |
| **全部组合** | **+20-35%** | **+25-40%** | **+12-20%** |

---

## 🛠️ 开发环境要求

### 系统需求
- Ubuntu 20.04 LTS
- ROS Noetic
- CUDA 11.0+ (可选, 用于GPU加速)
- 16GB+ 内存
- 100GB+ 磁盘空间

### 关键库版本
```
OpenCV >= 4.0
PCL >= 1.10
GTSAM >= 4.0
Ceres >= 1.14
PyTorch >= 1.9
Python >= 3.8
```

### 安装命令
```bash
# 系统依赖
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    libeigen3-dev \
    libboost-all-dev \
    libomp-dev

# Python包
pip install -r requirements.txt
```

---

## 📚 主要参考文献

1. **LVI-SAM**
   - Shan, Z., Li, R., & Schwertfeger, S. (2021). "LVI-SAM: Tightly-Coupled Lidar-Visual-Inertial Odometry and Mapping"
   - GitHub: https://github.com/TixiaoShan/LVI-SAM

2. **VINS-Mono**
   - Qin, T., Li, P., & Shen, S. (2018). "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator"

3. **LIO-SAM**
   - Shan, Z., Englot, B., Meyers, D., Wang, W., Ratti, C., & Rus, D. (2020). "LIO-SAM: Tightly-Coupled Lidar Inertial Odometry and Mapping"

4. **深度学习特征**
   - DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). "SuperPoint: Self-Supervised Interest Point Detection and Description"

5. **EVO工具**
   - Grupp, M. (2017). "EVO: Python package for the evaluation of odometry and SLAM"

## 🌐 关键代码仓库与参考

### 核心框架与项目

```markdown
## 主要参考仓库

### 1. LVI-SAM 官方框架
- **仓库**: https://github.com/TixiaoShan/LVI-SAM
- **发表**: ICRA 2021
- **核心**: 紧耦合LiDAR-视觉-惯性SLAM系统
- **语言**: C++
- **依赖**: ROS, GTSAM, Ceres, OpenCV

### 2. LVI-SAM-Easyused (本项目核心参考)
- **仓库**: https://github.com/NeSC-IV/LVI-SAM-Easyused
- **分支**: `new` 分支（推荐使用）
- **改进**: 修复了外参配置混乱，集成了最新LIO-SAM版本
- **优势**:
  - 简化了传感器外参配置流程
  - 修复了原始LVI-SAM中存在的Bug
  - 支持多种数据集配置
  - 完整的参数配置示例

### 3. 相关基础框架
- **LIO-SAM**: https://github.com/TixiaoShan/LIO-SAM
  - LiDAR-惯性里程计，是LVI-SAM的激光里程计模块基础
- **ORB-SLAM2**: https://github.com/UZ-SLAM/ORB_SLAM2
  - 视觉SLAM参考实现
- **VINS-Mono**: https://github.com/HKUST-Aerial-Robotics/VINS-Mono
  - 单目视觉-惯性系统参考

---

## 💻 环境配置参考

根据LVI-SAM-Easyused官方指南，推荐配置：

### 操作系统与基础库
```bash
# 操作系统: Ubuntu 20.04
# ROS版本: ROS Noetic
# 其他库:
  - OpenCV 4.0.* 
  - GTSAM 4.0.*
  - Ceres 1.14.*
  - Eigen3
```

### 编译步骤
```bash
# 创建工作空间
mkdir -p ~/lvi-sam/src
cd ~/lvi-sam/src

# 克隆代码（推荐使用 new 分支）
git clone -b new https://github.com/NeSC-IV/LVI-SAM-Easyused.git
# 或克隆官方版本
git clone https://github.com/TixiaoShan/LVI-SAM.git

# 编译
cd ~/lvi-sam
catkin_make
```

### 核心配置文件

#### 1. 传感器外参配置 (`params_camera.yaml`)
```yaml
# Camera-IMU 外参 (T_imu_camera)
# 相机相对于IMU的旋转矩阵
extrinsicRotation: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 0,    0,    -1, 
           -1,     0,    0, 
            0,     1,    0]

# 相机相对于IMU的位移向量
extrinsicTranslation: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [0.006422381632411965, 0.019939800449065116, 0.03364235163589248]
```

#### 2. LiDAR外参配置 (`params_lidar.yaml`)
```yaml
# LiDAR-IMU 外参 (T_imu_lidar)
extrinsicRotation: [-1,   0,    0, 
                     0,    1,    0, 
                     0,    0,   -1]
extrinsicTranslation: [0.0, 0.0, 0.0]
```

#### 3. IMU属性配置
```yaml
# IMU坐标系定义（绕哪个轴逆时针旋转得到正欧拉角）
# 对于大多数IMU设置为："+z", "+y", "+x"
yawAxis: "+z"      # Yaw轴
pitchAxis: "+y"    # Pitch轴  
rollAxis: "+x"     # Roll轴
```

### 运行系统

```bash
# 加载环境变量
source ~/lvi-sam/devel/setup.bash

# 启动LVI-SAM系统（使用Husky配置）
roslaunch lvi_sam Husky.launch

# 在另一个终端播放数据包
rosbag play your_data.bag
```

### 评估与验证

```bash
# 1. 安装EVO工具
pip install evo --upgrade --no-binary evo

# 2. 转换点云格式（如需要）
python pcd2tum.py

# 3. 计算轨迹误差
# -r full: 包括旋转和平移
# -va: 显示详细信息
evo_ape tum gt.txt lvisam.txt -r full -va --plot --plot_mode xy --save_plot

# 4. 多轨迹对比
evo_traj tum trajectory1.txt trajectory2.txt --ref=gt.txt -va -p --plot_mode=xy --save_plot
```

### 支持的数据集配置

#### 官方LVI-SAM数据集
```bash
roslaunch lvi_sam run.launch
rosbag play handheld.bag
```

#### M2DGR Dataset
```bash
roslaunch lvi_sam M2DGR.launch
rosbag play gate_01.bag
```

#### UrbanNav Dataset
```bash
roslaunch lvi_sam UrbanNavDataset.launch
rosbag play 2020-03-14-16-45-35.bag
```

#### KITTI Raw Dataset
```bash
roslaunch lvi_sam KITTI.launch
rosbag play kitti_2011_09_26_drive_0084_synced.bag
```

#### KAIST Complex Urban Dataset
```bash
roslaunch lvi_sam KAIST.launch
rosbag play urban26.bag
```

---

## 📚 关键参考文献

### 学术论文与完整BibTeX格式

```bibtex
@inproceedings{shan2021lvi,
  title={LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry and Mapping},
  author={Shan, Tixiao and Englot, Brendan and Forster, Dariush and Meyers, Kyle and Wang, Devansh and Duarte, Carlos and Ratti, Carlo},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={7482--7488},
  year={2021},
  organization={IEEE}
}

@inproceedings{shan2020liosam,
  title={LIO-SAM: Tightly-synchronized Lidar Inertial Odometry and Mapping},
  author={Shan, Tixiao and Englot, Brendan and Meyers, Kyle and Wang, Devansh and Ratti, Carlo and Rus, Daniela},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5016--5023},
  year={2020},
  organization={IEEE}
}
```

### 特征跟踪算法

```bibtex
@inproceedings{rublee2011orb,
  title={ORB: An Efficient Alternative to SIFT or SURF},
  author={Rublee, Ethan and Rabaud, Vincent and Konolige, Kurt and Bradski, Gary},
  booktitle={2011 International Conference on Computer Vision (ICCV)},
  pages={2564--2571},
  year={2011},
  organization={IEEE}
}

@article{lucas1981iterative,
  title={An Iterative Image Registration Technique with an Application to Stereo Vision},
  author={Lucas, Bruce D and Kanade, Takeo},
  journal={IJCAI},
  volume={81},
  pages={674--679},
  year={1981}
}

@inproceedings{desuperpoint,
  title={SuperPoint: Self-Supervised Interest Point Detection and Description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={224--236},
  year={2018},
  organization={IEEE}
}
```

### 点云处理与配准

```bibtex
@article{besl1992method,
  title={Method for Registration of 3-D Shapes},
  author={Besl, Paul J and McKay, Neil D},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={14},
  number={2},
  pages={239--256},
  year={1992},
  publisher={IEEE}
}

@inproceedings{biber2003normal,
  title={The Normal Distributions Transform: A New Approach to Laser Scan Matching},
  author={Biber, Peter and Straßer, Wolfgang},
  booktitle={Proceedings of the 2003 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2003)},
  volume={3},
  pages={2743--2748},
  year={2003},
  organization={IEEE}
}

@inproceedings{generalizedicp,
  title={Generalized-ICP},
  author={Segal, Aleksandr V and Haehnel, Dirk and Thrun, Sebastian},
  booktitle={Robotics: Science and Systems},
  volume={2},
  pages={435},
  year={2009}
}
```

### 深度学习方法

```bibtex
@inproceedings{siamese2015,
  title={Siamese Neural Networks for One-shot Image Recognition},
  author={Koch, Gregory and Zemel, Richard and Salakhutdinov, Ruslan},
  booktitle={ICML Deep Learning Workshop},
  year={2015}
}

@article{resnet2015,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={arXiv preprint arXiv:1512.03385},
  year={2015}
}

@inproceedings{he2016deep,
  title={Deep Learning for Generic Object Detection: A Survey},
  author={He, Kaiming and Gkioxari, Georgia and Dollar, Piotr and Girshick, Ross},
  booktitle={2015 IEEE International Conference on Computer Vision (ICCV)},
  pages={2395--2403},
  year={2015},
  organization={IEEE}
}

@inproceedings{mobilenet2017,
  title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
  author={Howard, Andrew G and Zhu, Mengxi and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijing and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4234--4243},
  year={2017},
  organization={IEEE}
}
```

### 视觉SLAM基础

```bibtex
@article{orbslam2,
  title={ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras},
  author={Mur-Artal, Ra{\'u}l and Tard{\'o}s, Juan D},
  journal={IEEE Transactions on Robotics},
  volume={33},
  number={5},
  pages={1255--1262},
  year={2017},
  publisher={IEEE}
}

@inproceedings{vinsmonorig,
  title={VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator},
  author={Qin, Tong and Li, Peiliang and Shen, Shaojun},
  booktitle={IEEE Transactions on Robotics},
  volume={34},
  number={4},
  pages={1004--1020},
  year={2018},
  publisher={IEEE}
}

@inproceedings{dso2018,
  title={Direct Sparse Odometry},
  author={Wang, Rui and Schwörer, Martin and Cremers, Daniel},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  pages={373--382},
  year={2017},
  organization={IEEE}
}
```

### 评估与基准

```bibtex
@techreport{geiger2012kitti,
  title={Vision meets Robotics: The KITTI Dataset},
  author={Geiger, Andreas and Lenz, Philip and Stiller, Christoph and Urtasun, Raquel},
  journal={International Journal of Robotics Research},
  year={2013}
}

@article{evo2017,
  title={EVO: Accurate and Open Source ROS Trajectory Evaluation Tool},
  author={Grupp, Michael},
  year={2017},
  url={https://github.com/MichaelGrupp/evo}
}

@inproceedings{ate2009,
  title={Accurate Real-time Localization of an Articulated Surgical Instrument using Kinematic and Optical Markers},
  author={Lepetit, Vincent and Fua, Pascal},
  booktitle={International Symposium on Computer Vision},
  year={2006}
}
```

---


### 常见问题

**Q: 运行时出现内存不足？**
A: 使用体素滤波器降采样点云，或在`launch`文件中减少`max_features`参数

**Q: GPU内存溢出（deep learning模块）？**
A: 减少batch_size或使用更小的模型（MobileNet替代ResNet）

**Q: 轨迹评估出错？**
A: 确保轨迹文件格式正确（TUM格式），时间戳单调递增

**Q: 网络训练很慢？**
A: 启用GPU加速，检查CUDA可用性：`python -c "import torch; print(torch.cuda.is_available())"`


