# LVI-SAM 改进项目实施指南

## 快速开始

### 环境设置

#### 1. 安装必要的依赖库

```bash
# ROS依赖
sudo apt-get install ros-noetic-desktop-full
sudo apt-get install ros-noetic-pointcloud-to-laserscan
sudo apt-get install ros-noetic-pcl-ros

# Python依赖
pip install torch torchvision
pip install opencv-python
pip install numpy scipy
pip install evo  # EVO轨迹评估工具
pip install matplotlib seaborn

# 编译依赖
sudo apt-get install libeigen3-dev
sudo apt-get install libgtsam-dev
sudo apt-get install libceres-dev
```

#### 2. 编译项目

```bash
cd /home/cx/lvi-sam
catkin_make -j4
source devel/setup.bash
```

---

## Phase 1: 基线建立（Week 1-2）

### 目标
- 获取原始LVI-SAM的基线性能
- 建立评估框架
- 熟悉系统运行流程

### 任务清单

#### Task 1.1: 收集基线轨迹

```bash
# 终端1: 启动ROS核心
roscore

# 终端2: 启动LVI-SAM系统
cd /home/cx/lvi-sam
source devel/setup.bash
roslaunch lvi_sam Husky.launch

# 终端3: 播放数据集
cd /home/cx/lvi-sam
source devel/setup.bash
rosbag play home_data/husky.bag

# 等待处理完成，结果保存在 ~/lvi-sam/results/
```

#### Task 1.2: 提取轨迹格式

```bash
# 确保results目录中有轨迹文件
cd /home/cx/lvi-sam/results
ls -la

# 如果需要转换格式，使用项目提供的脚本
python ../src/LVI-SAM-Easyused/pcd2tum.py
```

#### Task 1.3: 性能评估

```bash
# 使用评估脚本
python scripts/evaluate_trajectory.py \
    --estimated results/trajectory.txt \
    --reference home_data/gt.txt \
    --method "baseline_lvi_sam" \
    --output experiments/baseline

# 查看结果
cat experiments/baseline/baseline_lvi_sam_metrics.json
```

#### Task 1.4: 保存基线结果

```bash
# 保存结果到基线目录
mkdir -p experiments/baseline
cp results/* experiments/baseline/
cp experiments/baseline_lvi_sam_metrics.json experiments/baseline/

# 记录配置
echo "LVI-SAM原始版本，Husky数据集" > experiments/baseline/README.txt
```

**预期结果**:
- APE RMSE: ~0.05-0.15 m
- 基线评估指标保存在 `experiments/baseline/metrics.json`

---

## Phase 2: 单模块改进测试（Week 3-6）

### 2.1 视觉里程计改进（Week 3-4）

#### 子任务2.1.1: 实现增强特征跟踪器

**文件**: `/home/cx/lvi-sam/improvements/visual_feature_enhanced/enhanced_tracker.cpp`

```cpp
#include "enhanced_tracker.h"

EnhancedFeatureTracker::EnhancedFeatureTracker(
    DescriptorType descriptor_type,
    int max_features,
    bool uniform_distribution
) : descriptor_type_(descriptor_type),
    max_features_(max_features),
    use_uniform_distribution_(uniform_distribution),
    feature_id_counter_(0) {
    
    initializeDetector();
}

void EnhancedFeatureTracker::initializeDetector() {
    switch(descriptor_type_) {
        case DescriptorType::ORB:
            detector_ = cv::ORB::create(max_features_);
            descriptor_extractor_ = cv::ORB::create(max_features_);
            break;
        case DescriptorType::SIFT:
            detector_ = cv::SIFT::create();
            descriptor_extractor_ = cv::SIFT::create();
            break;
        case DescriptorType::SURF:
            detector_ = cv::xfeatures2d::SURF::create();
            descriptor_extractor_ = cv::xfeatures2d::SURF::create();
            break;
        case DescriptorType::KLT:
        default:
            // KLT跟踪使用opencv的角点检测
            break;
    }
}

std::vector<EnhancedFeatureTracker::TrackedFeature>
EnhancedFeatureTracker::trackFeatures(
    const cv::Mat& image,
    const cv::Mat& depth_image) {
    
    current_image_ = image.clone();
    
    std::vector<TrackedFeature> tracked;
    
    if(descriptor_type_ == DescriptorType::KLT) {
        tracked = kltTracking(image);
    } else if(descriptor_type_ == DescriptorType::ORB) {
        tracked = orbTracking(image);
    } else if(descriptor_type_ == DescriptorType::SIFT) {
        tracked = siftTracking(image);
    }
    
    // 应用均匀分布策略
    if(use_uniform_distribution_) {
        tracked = enforceUniformDistribution(tracked, image.size());
    }
    
    // 融合深度信息
    if(!depth_image.empty()) {
        for(auto& feature : tracked) {
            int x = static_cast<int>(feature.pt.x);
            int y = static_cast<int>(feature.pt.y);
            
            if(x >= 0 && x < depth_image.cols && 
               y >= 0 && y < depth_image.rows) {
                float depth = depth_image.at<float>(y, x);
                if(depth > 0) {
                    feature.depth = depth;
                }
            }
        }
    }
    
    current_features_ = tracked;
    return tracked;
}

// ... 其他方法实现
```

#### 子任务2.1.2: 集成到LVI-SAM

修改 `src/LVI-SAM-Easyused/src/visual_odometry/visual_feature/feature_tracker.cpp`

```cpp
// 在feature_tracker.cpp中添加选项来使用增强跟踪器
#include "../../improvements/visual_feature_enhanced/enhanced_tracker.h"

// 全局变量
EnhancedFeatureTracker* enhanced_tracker[NUM_OF_CAM];
bool use_enhanced_tracker = true;

// 初始化
if(use_enhanced_tracker) {
    for(int i = 0; i < NUM_OF_CAM; i++) {
        enhanced_tracker[i] = new EnhancedFeatureTracker(
            EnhancedFeatureTracker::DescriptorType::ORB,
            300,
            true  // 使用均匀分布
        );
    }
}
```

#### 子任务2.1.3: 性能测试

```bash
# 创建改进版启动文件
cat > src/LVI-SAM-Easyused/launch/visual_improved_v1.launch << 'EOF'
<launch>
    <param name="/use_enhanced_tracker" value="true"/>
    <param name="/tracker_type" value="ORB"/>  <!-- ORB, SIFT, SURF -->
    
    <include file="$(find lvi_sam)/launch/Husky.launch">
    </include>
</launch>
EOF

# 运行测试
python scripts/evaluate_trajectory.py \
    --estimated results/trajectory.txt \
    --reference home_data/gt.txt \
    --method "visual_enhanced_v1" \
    --output experiments/improved_v1
```

### 2.2 激光里程计改进（Week 4-5）

#### 子任务2.2.1: 实现动态物体过滤

**文件**: `/home/cx/lvi-sam/improvements/dynamic_removal/dynamic_filter.cpp`

```cpp
#include "dynamic_filter.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>

DynamicObjectRemover::DynamicObjectRemover(const FilterConfig& config)
    : config_(config) {}

DynamicObjectRemover::FilterConfig DynamicObjectRemover::defaultConfig() {
    FilterConfig config;
    config.motion_threshold = 0.5f;
    config.flow_consistency_threshold = 0.8f;
    config.clustering_tolerance = 0.5f;
    config.min_cluster_size = 10;
    config.max_cluster_size = 10000;
    config.residual_threshold = 0.1f;
    config.confidence_threshold = 0.7f;
    config.temporal_window_size = 5;
    config.temporal_consistency_ratio = 0.6f;
    return config;
}

std::vector<DynamicObjectRemover::PointLabel>
DynamicObjectRemover::detectByEuclideanClustering(
    const std::vector<Eigen::Vector3f>& points) {
    
    std::vector<PointLabel> labels(points.size());
    
    // 构建KD-树
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(size_t i = 0; i < points.size(); ++i) {
        pcl::PointXYZ p;
        p.x = points[i](0);
        p.y = points[i](1);
        p.z = points[i](2);
        cloud->push_back(p);
    }
    
    // 欧式聚类
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>
    );
    tree->setInputCloud(cloud);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(config_.clustering_tolerance);
    ec.setMinClusterSize(config_.min_cluster_size);
    ec.setMaxClusterSize(config_.max_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    
    // 分析每个聚类
    for(size_t i = 0; i < cluster_indices.size(); ++i) {
        const auto& indices = cluster_indices[i].indices;
        
        // 判断聚类是否为动态物体
        float motion_variance = 0.0f;
        
        // 简单启发式：大面积、运动一致性强的聚类可能是动态物体
        if(indices.size() > config_.min_cluster_size * 2) {
            for(const auto& idx : indices) {
                labels[idx].is_dynamic = true;
                labels[idx].cluster_id = i;
                labels[idx].confidence = 0.8f;
            }
        }
    }
    
    return labels;
}

std::vector<Eigen::Vector3f>
DynamicObjectRemover::filterPointCloud(
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<PointLabel>& labels,
    float confidence_threshold) {
    
    std::vector<Eigen::Vector3f> filtered;
    
    for(size_t i = 0; i < points.size(); ++i) {
        if(!labels[i].is_dynamic || 
           labels[i].confidence < confidence_threshold) {
            filtered.push_back(points[i]);
        }
    }
    
    total_points_last_ = points.size();
    dynamic_points_last_ = points.size() - filtered.size();
    
    return filtered;
}
```

#### 子任务2.2.2: 集成到地图优化

修改 `src/LVI-SAM-Easyused/src/lidar_odometry/mapOptmization.cpp`

```cpp
// 添加头文件
#include "../../improvements/dynamic_removal/dynamic_filter.h"

// 全局变量
DynamicObjectRemover* dynamic_remover;

// 初始化
void initializeDynamicRemover() {
    DynamicObjectRemover::FilterConfig config = 
        DynamicObjectRemover::defaultConfig();
    dynamic_remover = new DynamicObjectRemover(config);
}

// 在点云处理中使用
void filterDynamicPoints(
    pcl::PointCloud<PointType>::Ptr cloudIn,
    pcl::PointCloud<PointType>::Ptr cloudOut) {
    
    // 转换为Eigen格式
    std::vector<Eigen::Vector3f> points;
    for(const auto& p : cloudIn->points) {
        points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
    }
    
    // 检测动态点
    auto labels = dynamic_remover->detectByEuclideanClustering(points);
    
    // 过滤动态点
    auto filtered = dynamic_remover->filterPointCloud(
        points, labels, 0.7f
    );
    
    // 转换回PCL格式
    cloudOut->clear();
    for(const auto& p : filtered) {
        PointType pt;
        pt.x = p(0);
        pt.y = p(1);
        pt.z = p(2);
        cloudOut->push_back(pt);
    }
}
```

#### 子任务2.2.3: 性能测试

```bash
# 创建改进版启动文件
cat > src/LVI-SAM-Easyused/launch/lidar_improved_v1.launch << 'EOF'
<launch>
    <param name="/enable_dynamic_removal" value="true"/>
    <param name="/dynamic_removal_method" value="euclidean"/>
    
    <include file="$(find lvi_sam)/launch/Husky.launch">
    </include>
</launch>
EOF

# 运行测试
python scripts/evaluate_trajectory.py \
    --estimated results/trajectory.txt \
    --reference home_data/gt.txt \
    --method "lidar_enhanced_v1" \
    --output experiments/improved_v1
```

### 2.3 回环检测改进（Week 5-6）

#### 子任务2.3.1: 训练深度学习模型

**文件**: `/home/cx/lvi-sam/scripts/train_loop_detector.py`

```python
#!/usr/bin/env python3
import torch
import torch.optim as optim
from improvements.loop_closure_dl.deep_loop_detector import SiameseNetwork

def train_siamese_network(
    train_loader,
    val_loader,
    epochs=50,
    learning_rate=0.001,
    output_path="improvements/loop_closure_dl/models/siamese_trained.pth"
):
    """训练孪生网络"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseNetwork(feature_dim=256).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CosineSimilarity(dim=1)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            similarity = model(img1, img2)
            
            loss = torch.nn.functional.mse_loss(similarity, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 验证
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for img1, img2, labels in val_loader:
                    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                    similarity = model(img1, img2)
                    loss = torch.nn.functional.mse_loss(similarity, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"  Validation Loss: {avg_val_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), output_path)
    print(f"✓ 模型已保存: {output_path}")

if __name__ == "__main__":
    # TODO: 实现数据加载器和训练循环
    pass
```

#### 子任务2.3.2: 集成到系统

创建ROS节点: `/home/cx/lvi-sam/src/LVI-SAM-Easyused/src/visual_odometry/visual_loop/deep_loop_node.cpp`

```cpp
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
// ... 其他头文件

// 调用Python深度学习模块进行回环检测

class DeepLoopClosureNode {
private:
    ros::NodeHandle nh_;
    // ... 其他成员
    
public:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        // 调用深度学习模块检测回环
        // 发布回环约束
    }
};
```

---

## Phase 3: 综合改进测试（Week 7-8）

### 任务3.1: 整合所有改进

```bash
# 创建综合改进版启动文件
cat > src/LVI-SAM-Easyused/launch/improved_full.launch << 'EOF'
<launch>
    <!-- 视觉里程计改进 -->
    <param name="/use_enhanced_tracker" value="true"/>
    <param name="/tracker_type" value="ORB"/>
    
    <!-- 激光里程计改进 -->
    <param name="/enable_dynamic_removal" value="true"/>
    
    <!-- 回环检测改进 -->
    <param name="/use_deep_loop_detector" value="true"/>
    
    <!-- 因子图优化改进 -->
    <param name="/use_improved_factors" value="true"/>
    
    <include file="$(find lvi_sam)/launch/Husky.launch">
    </include>
</launch>
EOF
```

### 任务3.2: 运行性能对比

```bash
# 运行综合测试
python scripts/benchmark_suite.py --all \
    --output experiments/final_evaluation

# 查看对比结果
cat experiments/final_evaluation/comparison_results.csv
```

### 任务3.3: 超参数调优

```bash
# 创建调优脚本
cat > scripts/hyperparameter_tuning.py << 'EOF'
#!/usr/bin/env python3
import optuna
from evaluate_trajectory import TrajectoryEvaluator

def objective(trial):
    """Optuna目标函数"""
    # 定义超参数搜索空间
    clustering_tol = trial.suggest_float('clustering_tol', 0.1, 1.0)
    confidence_thresh = trial.suggest_float('confidence_thresh', 0.5, 0.9)
    
    # ... 设置参数并运行评估
    
    # 返回APE RMSE（最小化）
    return ape_rmse

# 启动超参数优化
study = optuna.create_study()
study.optimize(objective, n_trials=50)

print(f"最优参数: {study.best_params}")
print(f"最优APE RMSE: {study.best_value}")
EOF

python scripts/hyperparameter_tuning.py
```

---

## Phase 4: 性能评估与报告（Week 8-9）

### 任务4.1: 完整性能评估

```bash
# 使用EVO工具进行深层评估
mkdir -p experiments/evo_analysis

# 计算多种误差指标
evo_ape tum home_data/gt.txt results/trajectory.txt --save_results experiments/evo_analysis/ape.zip

evo_rpe tum home_data/gt.txt results/trajectory.txt --delta 1 --plot experiments/evo_analysis/rpe.pdf

# 轨迹对齐与缩放
evo_traj tum home_data/gt.txt results/trajectory.txt --plot --plot_mode xy

# 生成详细报告
evo_res experiments/evo_analysis/*.zip -p
```

### 任务4.2: 撰写实验报告

**报告框架** (`reports/final_report.md`):

```markdown
# LVI-SAM 校园场景SLAM优化研究报告

## 1. 摘要
- 研究背景与意义
- 主要贡献与创新点
- 关键性能指标

## 2. 引言
- SLAM技术概述
- 存在的问题
- 本研究的解决方案

## 3. 相关工作
- LVI-SAM综述
- 视觉里程计相关研究
- 激光里程计相关研究
- 回环检测相关研究

## 4. 方法论
### 4.1 视觉里程计改进
- 特征提取算法对比
- 深度估计方法
- 性能分析

### 4.2 激光里程计改进
- 动态物体去除方法
- 点云匹配算法
- 性能分析

### 4.3 回环检测改进
- 深度学习方案
- 网络架构
- 训练策略

### 4.4 因子图优化
- 新增约束因子
- 优化策略
- 性能提升

## 5. 实验与评估
### 5.1 实验设置
- 数据集描述
- 硬件配置
- 参数设定

### 5.2 单模块评估
- 基线性能
- 各改进模块性能
- 性能对比表

### 5.3 综合改进评估
- 全模块改进性能
- 轨迹可视化
- 误差分析

### 5.4 计算复杂度分析
- 实时性评估
- 资源占用分析

## 6. 结论与展望
- 主要成果
- 发现与洞察
- 未来改进方向

## 7. 附录
- 详细性能指标表
- 配置文件
- 代码片段
```

### 任务4.3: 生成性能对比表

```bash
# 创建汇总脚本
python << 'EOF'
import json
import pandas as pd
from pathlib import Path

results = {}

# 读取各版本的结果
for method in ['baseline', 'visual_v1', 'lidar_v1', 'loop_v1', 'full_improved']:
    metrics_file = f'experiments/{method}/{method}_metrics.json'
    if Path(metrics_file).exists():
        with open(metrics_file) as f:
            results[method] = json.load(f)

# 创建DataFrame
df = pd.DataFrame(results).T
df = df[['APE_RMSE', 'ATE_RMSE', 'ARE_RMSE']]

print("\n性能指标对比表")
print("="*60)
print(df.to_string())

# 计算改进百分比
baseline_ape = df.loc['baseline', 'APE_RMSE']
print("\n\n相对于基线的改进百分比")
print("="*60)
for method in df.index[1:]:
    improvement = (baseline_ape - df.loc[method, 'APE_RMSE']) / baseline_ape * 100
    print(f"{method:20s}: {improvement:+.2f}%")

# 保存为CSV
df.to_csv('experiments/final_metrics_summary.csv')
print("\n✓ 结果已保存到: experiments/final_metrics_summary.csv")
EOF
```

---

## 文件清单与检查点

### 必须完成的文件

- [ ] `/home/cx/lvi-sam/improvements/visual_feature_enhanced/enhanced_tracker.h/cpp`
- [ ] `/home/cx/lvi-sam/improvements/depth_estimation/depth_predictor.h/cpp`
- [ ] `/home/cx/lvi-sam/improvements/dynamic_removal/dynamic_filter.h/cpp`
- [ ] `/home/cx/lvi-sam/improvements/point_cloud_matching/advanced_matcher.h/cpp`
- [ ] `/home/cx/lvi-sam/improvements/loop_closure_dl/deep_loop_detector.py`
- [ ] `/home/cx/lvi-sam/improvements/factor_graph_opt/custom_factors.h`
- [ ] `/home/cx/lvi-sam/scripts/evaluate_trajectory.py`
- [ ] `/home/cx/lvi-sam/scripts/benchmark_suite.py`
- [ ] `/home/cx/lvi-sam/reports/final_report.md`

### 实验输出目录

```
experiments/
├── baseline/               # 基线结果
│   ├── trajectory.txt
│   ├── metrics.json
│   └── plots/
├── improved_v1/           # 改进版本1
│   ├── trajectory.txt
│   ├── metrics.json
│   └── plots/
├── final_evaluation/      # 最终评估
│   ├── comparison_results.csv
│   ├── method_comparison.png
│   └── benchmark_report.json
└── evo_analysis/          # EVO详细分析
    ├── ape.zip
    ├── rpe.pdf
    └── plots/
```

---

## 常见问题解决

### Q: ROS通信错误
**A**: 确保已启动roscore，且所有终端都执行了`source devel/setup.bash`

### Q: 内存不足
**A**: 减少点云分辨率，或使用体素滤波器下采样

### Q: GPU/CUDA相关错误
**A**: 检查PyTorch和CUDA版本兼容性，或使用CPU模式运行

### Q: EVO安装失败
**A**: `pip install --upgrade evo` 或从源代码安装

---

## 参考资源

- LVI-SAM论文与代码: https://github.com/TixiaoShan/LVI-SAM
- EVO工具: https://github.com/MichaelGrupp/evo
- ROS入门: http://wiki.ros.org/
- PyTorch文档: https://pytorch.org/docs/

