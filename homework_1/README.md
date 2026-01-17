# 🤖 LVI-SAM 校园场景SLAM优化研究项目

> 基于LVI-SAM框架的多模块优化研究，涵盖视觉里程计、激光里程计、回环检测和因子图优化

## 📋 项目信息

- **项目名称**: LVI-SAM 校园场景SLAM系统优化
- **基础框架**: [LVI-SAM](https://github.com/TixiaoShan/LVI-SAM) - 紧耦合LiDAR-视觉-惯性里程计
- **传感器配置**: RealSense D435i (RGB-D) + Velodyne VLP-16 (LiDAR) + MTi-680G (IMU)
- **数据集**: 校园场景，地面真值由RTK GPS提供
- **评估工具**: [EVO](https://github.com/MichaelGrupp/evo) - 轨迹评估工具
- **性能指标**: APE, ATE, ARE (RMSE值)
- **预期目标**: 相对基线改进 **20-35%** 的定位精度

## 🎯 核心创新点

### ✨ 四大改进方向

| # | 改进方向 | 关键技术 | 目标 |
|---|---------|---------|------|
| 1 | **视觉里程计** | SuperPoint特征 + 深度估计融合 | 特征追踪率 95%→98% |
| 2 | **激光里程计** | 动态物体去除 + 改进的点云配准 | 配准精度±2cm, 动态去除>90% |
| 3 | **回环检测** | CNN特征 + Siamese相似度网络 | 正确率>95%, 误检率<2% |
| 4 | **因子图优化** | 新约束因子 + 动态噪声模型 | 闭合误差-50%, 全局一致性↑ |

## 📁 项目结构

```
📦 /home/cx/lvi-sam/
├── 📖 【从这里开始】
│   ├── README_PROJECT.md          ⭐ 完整项目指南（推荐首先阅读）
│   ├── QUICK_START.md              ⭐ 快速参考（5分钟快速入门）
│   └── IMPLEMENTATION_GUIDE.md      ⭐ 详细实现指南（逐步教程）
│
├── 📚 【参考文档】
│   ├── CODE_ANALYSIS.md            原始LVI-SAM代码分析
│   └── PROJECT_PLAN.md             项目总体规划
│
├── 💻 【源代码】
│   └── src/LVI-SAM-Easyused/       原始LVI-SAM框架
│       ├── src/
│       │   ├── lidar_odometry/     激光里程计模块
│       │   └── visual_odometry/    视觉里程计模块
│       ├── config/                 传感器参数配置
│       └── launch/                 ROS启动文件
│
├── 🆕 【改进模块】- improvements/
│   ├── visual_feature_enhanced/    增强视觉特征提取
│   ├── depth_estimation/           改进深度估计
│   ├── dynamic_removal/            动态物体检测去除
│   ├── point_cloud_matching/       改进点云配准算法
│   ├── loop_closure_dl/            深度学习回环检测 ✅
│   └── factor_graph_opt/           改进因子图优化
│
├── 🔧 【工具脚本】- scripts/
│   ├── evaluate_trajectory.py      ✅ 轨迹评估工具
│   ├── benchmark_suite.py          ✅ 基准测试套件
│   ├── train_loop_detector.py      深度学习模型训练
│   └── generate_report.py          报告生成脚本
│
├── 🧪 【实验结果】- experiments/
│   ├── baseline/                   基线结果
│   ├── improved_v1/                改进版本1
│   ├── final_evaluation/           最终对比评估
│   └── evo_analysis/               EVO详细分析
│
├── 📊 【报告】- reports/
│   ├── final_report.md             完整研究报告
│   ├── figures/                    论文图表
│   └── tables/                     性能对比表
│
└── 📦 【数据集】- home_data/
    ├── husky.bag (22GB)            Husky UGV传感器数据
    └── gt.txt (71MB)               RTK地面真值轨迹
```

## 🚀 快速开始（5分钟）

### 1. 环境配置

```bash
cd /home/cx/lvi-sam
source devel/setup.bash
```

### 2. 三个独立终端启动系统

```bash
# 终端A: ROS核心
roscore

# 终端B: LVI-SAM系统
roslaunch lvi_sam Husky.launch

# 终端C: 播放数据集
rosbag play home_data/husky.bag
```

### 3. 性能评估

```bash
python scripts/evaluate_trajectory.py \
    --estimated results/trajectory.txt \
    --reference home_data/gt.txt \
    --method "baseline_lvi_sam" \
    --output experiments/baseline
```

### 4. 查看结果

```bash
cat experiments/baseline/baseline_lvi_sam_metrics.json
```

## 📊 核心评估指标

```
┌─────────────────────────────────────────────────────────┐
│ APE RMSE (Absolute Pose Error)                         │
│ 衡量估计轨迹与真值轨迹的绝对误差                         │
│ 原始LVI-SAM: ~0.10 m                                    │
│ 目标改进后: ~0.065 m (改进35%)                          │
├─────────────────────────────────────────────────────────┤
│ ATE RMSE (Absolute Trajectory Error)                   │
│ 衡量整体轨迹的累积误差                                  │
│ 原始LVI-SAM: ~0.15 m                                    │
│ 目标改进后: ~0.09 m (改进40%)                           │
├─────────────────────────────────────────────────────────┤
│ ARE RMSE (Absolute Rotation Error)                     │
│ 衡量姿态估计的准确性                                    │
│ 原始LVI-SAM: ~2.5 deg                                   │
│ 目标改进后: ~1.8 deg (改进28%)                          │
└─────────────────────────────────────────────────────────┘
```

## 📚 阅读路线图

### 新手入门（强烈推荐按顺序阅读）

```
1️⃣ QUICK_START.md (本文件)
   └─ 了解项目基本概况、快速命令

2️⃣ README_PROJECT.md
   └─ 深入理解项目全景、工作流、文件结构

3️⃣ CODE_ANALYSIS.md
   └─ 理解原始LVI-SAM代码架构

4️⃣ IMPLEMENTATION_GUIDE.md
   └─ 逐步实现各个改进模块
   └─ 包含代码示例和集成方法

5️⃣ PROJECT_PLAN.md
   └─ 查看详细的项目规划和时间表
```

### 快速查阅

- 🔍 **需要理解代码?** → `CODE_ANALYSIS.md`
- 🚀 **需要快速启动?** → `QUICK_START.md`
- 📖 **需要完整指南?** → `README_PROJECT.md`
- 💻 **需要实现指导?** → `IMPLEMENTATION_GUIDE.md`
- 📋 **需要项目规划?** → `PROJECT_PLAN.md`

## 🛠️ 已完成的实现

### ✅ 深度学习回环检测
**文件**: `improvements/loop_closure_dl/deep_loop_detector.py`

完整的孪生网络实现，包括：
- 特征提取编码器 (CNN)
- 相似度计算模块
- 特征数据库管理
- 回环候选查询

```python
# 使用示例
from improvements.loop_closure_dl.deep_loop_detector import DeepLoopDetector

detector = DeepLoopDetector(model_path="model.pth", similarity_threshold=0.5)
detector.add_frame(image, frame_id=0)
candidates = detector.detect_loop_closure(query_image, query_id=100)
```

### ✅ 轨迹评估工具
**文件**: `scripts/evaluate_trajectory.py`

完整的性能评估框架，包括：
- TUM格式轨迹加载
- APE, ATE, ARE计算
- 轨迹可视化
- 性能对比分析

```bash
# 使用示例
python scripts/evaluate_trajectory.py \
    --estimated traj_est.txt \
    --reference gt.txt \
    --method "my_method" \
    --output results/
```

### ✅ 基准测试套件
**文件**: `scripts/benchmark_suite.py`

自动化测试框架，包括：
- 多方法自动对比
- 性能指标收集
- 结果报告生成
- 实验追踪记录

```bash
# 使用示例
python scripts/benchmark_suite.py --all --output experiments/
```

## 🔄 建议的研究流程

### Phase 1: 基线建立 (Week 1-2)

```
目标: 了解系统，建立性能基线

步骤:
1. 阅读CODE_ANALYSIS.md了解代码
2. 运行原始LVI-SAM系统
3. 评估基线性能 → experiments/baseline/
4. 熟悉评估工具和工作流
```

### Phase 2: 单模块改进 (Week 3-6)

```
目标: 逐个实现并测试各改进模块

建议顺序:
1. 视觉里程计改进 (visual_feature_enhanced/)
   - 实现enhanced_tracker.cpp
   - 集成到LVI-SAM
   - 评估性能

2. 激光里程计改进 (dynamic_removal/)
   - 实现dynamic_filter.cpp
   - 集成到mapOptmization
   - 评估性能

3. 回环检测改进 (loop_closure_dl/)
   - 训练深度学习模型 ⭐ 已有框架
   - 集成到系统
   - 评估性能

4. 因子图优化改进 (factor_graph_opt/)
   - 实现新约束因子
   - 集成到GTSAM
   - 评估性能
```

### Phase 3: 综合测试 (Week 7-8)

```
目标: 整合所有改进，进行性能对比

步骤:
1. 创建综合改进版启动文件
2. 运行完整系统
3. 进行超参数调优
4. 生成最终性能报告 → experiments/final_evaluation/
```

### Phase 4: 报告撰写 (Week 8-9)

```
目标: 撰写完整的研究报告

内容:
1. 摘要与引言
2. 相关工作分析
3. 方法论详解
4. 实验结果与分析
5. 性能对比表和图表
6. 结论与展望
```

## 💻 技术栈

### 编程语言
- **C++14/17**: 核心算法和ROS节点
- **Python 3.8+**: 数据处理、深度学习、评估

### 关键框架库
- **ROS Noetic**: 机器人操作系统
- **OpenCV 4.0+**: 计算机视觉
- **PCL 1.10+**: 点云处理
- **GTSAM 4.0+**: 因子图优化
- **Ceres 1.14+**: 非线性优化
- **PyTorch 1.9+**: 深度学习

### 工具与平台
- **Ubuntu 20.04 LTS**: 操作系统
- **CUDA 11.0+**: GPU计算 (可选)
- **EVO**: 轨迹评估工具
- **Git**: 版本控制

## 📈 性能预期

基于相关研究和初步分析，各模块改进预期：

| 改进模块 | APE改进 | ATE改进 | ARE改进 | 计算开销 |
|---------|--------|--------|--------|---------|
| 基线 | - | - | - | 1x |
| +视觉增强 | +5-10% | +3-8% | +2-5% | 1.1x |
| +激光增强 | +10-15% | +8-12% | +5-8% | 1.2x |
| +回环改进 | +5-8% | +5-10% | +3-6% | 1.3x |
| +因子优化 | +3-5% | +5-8% | +2-4% | 1.35x |
| **全部改进** | **+20-35%** | **+25-40%** | **+12-20%** | ~1.4x |

## 🐛 常见问题

### Q1: 如何快速了解项目？
**A**: 按阅读路线图依次阅读：QUICK_START → README_PROJECT → CODE_ANALYSIS

### Q2: 第一个改进模块应该选择哪个？
**A**: 推荐从回环检测开始 (已有完整框架)，或视觉里程计 (影响最大)

### Q3: 如何验证改进效果？
**A**: 使用 `scripts/evaluate_trajectory.py` 工具对比基线和改进版本的APE/ATE/ARE

### Q4: 系统要求是什么？
**A**: Ubuntu 20.04, ROS Noetic, 16GB内存, GPU可选 (用于深度学习加速)

### Q5: 是否必须实现所有改进？
**A**: 不必须。可根据时间和兴趣选择特定模块，但建议至少完成3个模块获得显著效果

## 📞 获取帮助

| 问题类型 | 查看文件 |
|---------|---------|
| 代码理解 | CODE_ANALYSIS.md |
| 快速上手 | QUICK_START.md |
| 详细指南 | README_PROJECT.md, IMPLEMENTATION_GUIDE.md |
| 项目规划 | PROJECT_PLAN.md |
| 工具使用 | 脚本头部注释和docstring |
| ROS问题 | http://wiki.ros.org/ |
| 深度学习 | PyTorch官方文档 |

## 🎓 学习资源

### 推荐论文

1. **LVI-SAM**: Shan et al., 2021 - "LVI-SAM: Tightly-Coupled Lidar-Visual-Inertial Odometry and Mapping"
2. **VINS-Mono**: Qin et al., 2018 - "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator"
3. **LIO-SAM**: Shan et al., 2020 - "LIO-SAM: Tightly-Coupled Lidar Inertial Odometry and Mapping"
4. **SuperPoint**: DeTone et al., 2018 - "SuperPoint: Self-Supervised Interest Point Detection and Description"
5. **ORB-SLAM2**: Mur-Artal & Tardós, 2017 - "ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras"

### 在线资源

- EVO工具: https://github.com/MichaelGrupp/evo
- LVI-SAM官方: https://github.com/TixiaoShan/LVI-SAM
- ROS Wiki: http://wiki.ros.org/
- PCL教程: https://pcl.readthedocs.io/
- GTSAM文档: https://gtsam.org/

## ⭐ 项目亮点

✨ **完整的项目框架** - 从基线到改进的系统化方案
✨ **详细的实现指导** - 包含代码示例和集成方法
✨ **自动化评估工具** - 简化性能比较过程
✨ **多个改进方向** - 可根据兴趣灵活选择
✨ **实战应用价值** - 改进后的系统可用于实际场景

## 📝 许可证

项目基于 [LVI-SAM](https://github.com/TixiaoShan/LVI-SAM) 的GPL-3.0许可证

## 🙏 致谢

感谢以下项目和工具的支持：
- LVI-SAM原始团队
- EVO轨迹评估工具
- ROS社区
- PyTorch和深度学习生态

---

## 📋 核心文档导航

### 📖 【快速参考】
- **这是你现在正在读的文件** - 项目概览和快速入门
- [QUICK_START.md](QUICK_START.md) - 5分钟快速参考和常用命令

### 📚 【详细指南】
- [README_PROJECT.md](README_PROJECT.md) - ⭐ 完整项目指南（推荐首先阅读）
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - 逐步实现教程
- [CODE_ANALYSIS.md](CODE_ANALYSIS.md) - 原始代码结构分析

### 📋 【规划文档】
- [PROJECT_PLAN.md](PROJECT_PLAN.md) - 项目总体规划和时间表

---
@inproceedings{lvisam2021shan,
  title={LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry via Smoothing and Mapping},
  author={Shan, Tixiao and Englot, Brendan and Ratti, Carlo and Rus Daniela},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5692-5698},
  year={2021},
  organization={IEEE}
}
