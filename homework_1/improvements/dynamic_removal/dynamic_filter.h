#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <memory>

/**
 * @brief 动态物体过滤器
 * 用于检测和移除点云中的动态物体
 */
class DynamicObjectRemover {
public:
    struct FilterConfig {
        // 运动一致性检测参数
        float motion_threshold;           // 运动速度阈值
        float flow_consistency_threshold; // 光流一致性阈值
        
        // 欧式聚类参数
        float clustering_tolerance;       // 聚类距离阈值
        int min_cluster_size;            // 最小聚类尺寸
        int max_cluster_size;            // 最大聚类尺寸
        
        // 残差分析参数
        float residual_threshold;         // 残差阈值
        float confidence_threshold;       // 置信度阈值
        
        // 时间一致性参数
        int temporal_window_size;         // 时间窗口大小
        float temporal_consistency_ratio; // 时间一致性比率
    };

    struct PointLabel {
        int point_id;
        bool is_static;              // 是否为静态点
        bool is_dynamic;             // 是否为动态点
        float confidence;            // 置信度 [0, 1]
        int cluster_id;              // 所属聚类ID
        float motion_speed;          // 运动速度
        std::string removal_reason;  // 移除原因
    };

    /**
     * @brief 构造函数
     * @param config 过滤配置
     */
    explicit DynamicObjectRemover(const FilterConfig& config = defaultConfig());

    /**
     * @brief 设置配置
     */
    void setConfig(const FilterConfig& config) { config_ = config; }

    /**
     * @brief 获取默认配置
     */
    static FilterConfig defaultConfig();

    /**
     * @brief 使用运动一致性检测动态点
     * @param current_points 当前帧点云
     * @param current_image 当前图像
     * @param previous_points 上一帧点云
     * @param previous_image 上一帧图像
     * @return 点标签
     */
    std::vector<PointLabel> detectByMotionConsistency(
        const std::vector<Eigen::Vector3f>& current_points,
        const cv::Mat& current_image,
        const std::vector<Eigen::Vector3f>& previous_points,
        const cv::Mat& previous_image
    );

    /**
     * @brief 使用欧式聚类检测动态物体
     * @param points 点云
     * @return 点标签
     */
    std::vector<PointLabel> detectByEuclideanClustering(
        const std::vector<Eigen::Vector3f>& points
    );

    /**
     * @brief 使用残差分析检测动态点
     * @param points 点云
     * @param registration_residuals 配准残差
     * @return 点标签
     */
    std::vector<PointLabel> detectByResidualAnalysis(
        const std::vector<Eigen::Vector3f>& points,
        const std::vector<float>& registration_residuals
    );

    /**
     * @brief 使用时间一致性检测动态点
     * @param point_histories 点的历史位置 [时间 x 点索引]
     * @return 点标签
     */
    std::vector<PointLabel> detectByTemporalConsistency(
        const std::vector<std::vector<Eigen::Vector3f>>& point_histories
    );

    /**
     * @brief 融合多种检测方法
     * @param detections 多种检测方法的结果
     * @return 融合后的点标签
     */
    std::vector<PointLabel> fusionDetections(
        const std::vector<std::vector<PointLabel>>& detections
    );

    /**
     * @brief 过滤点云，移除动态点
     * @param points 输入点云
     * @param labels 点标签
     * @param confidence_threshold 置信度阈值
     * @return 过滤后的点云
     */
    std::vector<Eigen::Vector3f> filterPointCloud(
        const std::vector<Eigen::Vector3f>& points,
        const std::vector<PointLabel>& labels,
        float confidence_threshold = 0.7
    );

    /**
     * @brief 获取最后一次检测的统计信息
     */
    void getStatistics(int& total_points, int& dynamic_points, float& dynamic_ratio) const;

private:
    FilterConfig config_;
    int total_points_last_ = 0;
    int dynamic_points_last_ = 0;

    /**
     * @brief 计算光流
     */
    std::vector<cv::Point2f> computeOpticalFlow(
        const cv::Mat& prev_img,
        const cv::Mat& curr_img,
        const std::vector<cv::Point2f>& prev_features
    );

    /**
     * @brief 欧式聚类实现
     */
    std::vector<std::vector<int>> euclideanCluster(
        const std::vector<Eigen::Vector3f>& points
    );

    /**
     * @brief 计算点之间的距离
     */
    float pointDistance(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) const;

    /**
     * @brief 统计聚类中的动态点
     */
    void analyzeCluster(
        const std::vector<int>& cluster,
        const std::vector<float>& residuals,
        std::vector<PointLabel>& labels
    );
};

#endif // DYNAMIC_OBJECT_REMOVER_H
