#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

/**
 * @brief 增强型特征跟踪器
 * 支持多种特征描述符和改进的特征分布策略
 */
class EnhancedFeatureTracker {
public:
    enum DescriptorType {
        KLT = 0,           // Kanade-Lucas-Tomasi (原始)
        SIFT = 1,          // Scale-Invariant Feature Transform
        SURF = 2,          // Speeded Up Robust Features
        ORB = 3,           // Oriented FAST and Rotated BRIEF
        SUPERPOINT = 4     // Deep learning-based SuperPoint
    };

    struct TrackedFeature {
        int id;
        cv::Point2f pt;           // 2D图像坐标
        cv::Vec3f pts_3d;         // 3D点云坐标（若有）
        cv::Mat descriptor;       // 特征描述符
        float response;           // 特征响应强度
        float depth;              // 深度值
        bool is_valid;            // 是否有效
    };

    /**
     * @brief 构造函数
     * @param descriptor_type 特征描述符类型
     * @param max_features 最大特征点数
     * @param uniform_distribution 是否使用均匀分布策略
     */
    EnhancedFeatureTracker(
        DescriptorType descriptor_type = ORB,
        int max_features = 300,
        bool uniform_distribution = true
    );

    /**
     * @brief 初始化跟踪器
     * @param image 初始图像
     */
    void initialize(const cv::Mat& image);

    /**
     * @brief 追踪特征点
     * @param image 当前帧图像
     * @param depth_image 深度图像（可选）
     * @return 追踪的特征点集合
     */
    std::vector<TrackedFeature> trackFeatures(
        const cv::Mat& image,
        const cv::Mat& depth_image = cv::Mat()
    );

    /**
     * @brief 提取新特征点
     * @param image 当前帧图像
     * @param mask 感兴趣区域掩码
     * @return 新提取的特征点
     */
    std::vector<TrackedFeature> detectNewFeatures(
        const cv::Mat& image,
        const cv::Mat& mask = cv::Mat()
    );

    /**
     * @brief 获取当前追踪的特征点
     */
    const std::vector<TrackedFeature>& getCurrentFeatures() const {
        return current_features_;
    }

    /**
     * @brief 移除无效特征点
     */
    void removeInvalidFeatures();

    /**
     * @brief 获取特征分布情况（用于可视化）
     */
    cv::Mat visualizeFeatureDistribution(const cv::Size& img_size) const;

private:
    DescriptorType descriptor_type_;
    int max_features_;
    bool use_uniform_distribution_;
    
    std::vector<TrackedFeature> current_features_;
    std::vector<TrackedFeature> previous_features_;
    
    cv::Mat previous_image_;
    cv::Mat current_image_;
    
    int feature_id_counter_;

    // 特征提取器
    std::shared_ptr<cv::Feature2D> detector_;
    std::shared_ptr<cv::Feature2D> descriptor_extractor_;

    /**
     * @brief 初始化特征检测器
     */
    void initializeDetector();

    /**
     * @brief 应用均匀分布策略
     */
    std::vector<TrackedFeature> enforceUniformDistribution(
        const std::vector<TrackedFeature>& features,
        const cv::Size& img_size
    );

    /**
     * @brief 自适应跟踪
     */
    std::vector<TrackedFeature> adaptiveTracking(const cv::Mat& image);

    /**
     * @brief KLT跟踪实现
     */
    std::vector<TrackedFeature> kltTracking(const cv::Mat& image);

    /**
     * @brief SIFT跟踪实现
     */
    std::vector<TrackedFeature> siftTracking(const cv::Mat& image);

    /**
     * @brief ORB跟踪实现
     */
    std::vector<TrackedFeature> orbTracking(const cv::Mat& image);

    /**
     * @brief 特征匹配
     */
    std::vector<cv::DMatch> matchFeatures(
        const cv::Mat& desc1,
        const cv::Mat& desc2
    );
};

#endif // ENHANCED_TRACKER_H
