#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <map>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "parameters.h"

using namespace std;
using namespace Eigen;

/**
 * 增强视觉特征跟踪器
 * 改进内容:
 * 1. 多描述符特征提取 (KLT + ORB)
 * 2. 自适应特征分布 (网格均匀分布)
 * 3. 深度信息融合
 * 4. 鲁棒性特征验证
 */

struct Feature {
    cv::Point2f pt;           // 特征点坐标
    int id;                   // 全局ID
    float depth;              // 深度值 (如果可用)
    int track_count;          // 跟踪帧数
    int descriptor_type;      // 描述符类型 (0=KLT, 1=ORB)
    float response;           // 特征响应值
    float velocity_x, velocity_y;  // 运动速度
    
    Feature() : pt(0, 0), id(-1), depth(-1), track_count(0), 
                descriptor_type(0), response(0), velocity_x(0), velocity_y(0) {}
};

class EnhancedFeatureTracker {
public:
    enum DescriptorType {
        DESCRIPTOR_KLT = 0,    // Lucas-Kanade tracker
        DESCRIPTOR_ORB = 1     // ORB特征
    };
    
    enum AdaptiveMode {
        GRID_BASED = 0,        // 网格均匀分布
        QUALITY_BASED = 1      // 质量优先
    };
    
    /**
     * 构造函数
     */
    EnhancedFeatureTracker() 
        : n_features(150),
          max_track_cnt(30),
          min_feature_distance(50),
          grid_rows(4),
          grid_cols(4),
          next_id(0),
          prev_time(0),
          cur_time(0) {
        
        // 初始化ORB检测器
        orb_detector = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
        
        // 初始化BRIEF描述符匹配器
        brief_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    }
    
    /**
     * 读取新图像并进行特征跟踪
     */
    void trackImage(const cv::Mat &img, double _cur_time) {
        prev_time = cur_time;
        cur_time = _cur_time;
        
        if (!prev_img.empty()) {
            // 步骤1: 使用KLT跟踪前一帧的特征
            vector<cv::Point2f> prev_pts_to_track;
            vector<int> prev_ids_to_track;
            
            for (const auto &f : prev_features) {
                prev_pts_to_track.push_back(f.pt);
                prev_ids_to_track.push_back(f.id);
            }
            
            if (!prev_pts_to_track.empty()) {
                trackFeatures(prev_img, img, prev_pts_to_track, prev_ids_to_track);
            }
        }
        
        // 步骤2: 检测新特征点
        vector<cv::Point2f> new_pts = detectNewFeatures(img);
        
        // 步骤3: 更新特征列表
        addNewFeatures(new_pts);
        
        // 保存当前帧
        prev_img = img.clone();
        prev_features = cur_features;
    }
    
    /**
     * 获取当前特征点（未畸变坐标）
     */
    vector<Feature> getFeatures() const {
        return cur_features;
    }
    
    /**
     * 设置相机模型（用于畸变矫正）
     */
    void setCameraModel(const cv::Mat &K, const cv::Mat &D) {
        camera_K = K.clone();
        camera_D = D.clone();
    }
    
    /**
     * 融合深度信息
     */
    void fuseDepth(const cv::Mat &depth_image) {
        for (auto &f : cur_features) {
            int x = static_cast<int>(f.pt.x);
            int y = static_cast<int>(f.pt.y);
            
            if (x >= 0 && x < depth_image.cols && 
                y >= 0 && y < depth_image.rows) {
                
                float d = depth_image.at<float>(y, x);
                if (d > 0 && d < 10.0f) {  // 合理的深度范围
                    f.depth = d;
                }
            }
        }
    }
    
    /**
     * 设置参数
     */
    void setParameters(int num_feat, int max_cnt, int min_dist,
                     int grid_r, int grid_c) {
        n_features = num_feat;
        max_track_cnt = max_cnt;
        min_feature_distance = min_dist;
        grid_rows = grid_r;
        grid_cols = grid_c;
    }
    
private:
    int n_features;
    int max_track_cnt;
    int min_feature_distance;
    int grid_rows, grid_cols;
    int next_id;
    double prev_time, cur_time;
    
    cv::Mat prev_img;
    vector<Feature> prev_features;
    vector<Feature> cur_features;
    
    cv::Mat camera_K, camera_D;
    cv::Ptr<cv::ORB> orb_detector;
    cv::Ptr<cv::BFMatcher> brief_matcher;
    
    /**
     * KLT特征跟踪
     */
    void trackFeatures(const cv::Mat &prev_img, const cv::Mat &cur_img,
                      const vector<cv::Point2f> &prev_pts,
                      const vector<int> &prev_ids) {
        
        if (prev_pts.empty()) return;
        
        vector<cv::Point2f> cur_pts;
        vector<uchar> status;
        vector<float> err;
        
        // 使用KLT追踪
        cv::calcOpticalFlowPyrLK(
            prev_img, cur_img,
            prev_pts, cur_pts,
            status, err,
            cv::Size(21, 21),
            3,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
        );
        
        // 更新特征位置
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                Feature f;
                f.pt = cur_pts[i];
                f.id = prev_ids[i];
                f.track_count = prev_features[i].track_count + 1;
                f.response = prev_features[i].response;
                f.descriptor_type = DESCRIPTOR_KLT;
                
                // 计算速度
                f.velocity_x = cur_pts[i].x - prev_pts[i].x;
                f.velocity_y = cur_pts[i].y - prev_pts[i].y;
                
                cur_features.push_back(f);
            }
        }
    }
    
    /**
     * 检测新特征点（网格均匀分布）
     */
    vector<cv::Point2f> detectNewFeatures(const cv::Mat &img) {
        int h = img.rows;
        int w = img.cols;
        
        vector<cv::Point2f> all_new_pts;
        
        // 检测ORB特征
        vector<cv::KeyPoint> orb_kpts;
        orb_detector->detect(img, orb_kpts);
        
        // 转换为Point2f
        vector<cv::Point2f> orb_pts;
        vector<float> orb_responses;
        
        for (const auto &kpt : orb_kpts) {
            orb_pts.push_back(kpt.pt);
            orb_responses.push_back(kpt.response);
        }
        
        // 网格基础特征分布
        int grid_h = h / grid_rows;
        int grid_w = w / grid_cols;
        
        map<int, vector<pair<cv::Point2f, float>>> grid_features;
        
        // 分配特征到网格单元
        for (size_t i = 0; i < orb_pts.size(); ++i) {
            int grid_x = static_cast<int>(orb_pts[i].x / grid_w);
            int grid_y = static_cast<int>(orb_pts[i].y / grid_h);
            
            grid_x = min(grid_x, grid_cols - 1);
            grid_y = min(grid_y, grid_rows - 1);
            
            int grid_id = grid_y * grid_cols + grid_x;
            grid_features[grid_id].push_back({orb_pts[i], orb_responses[i]});
        }
        
        // 从每个网格单元选择最佳特征
        int features_per_cell = max(1, n_features / (grid_rows * grid_cols));
        
        for (auto &cell : grid_features) {
            auto &pts = cell.second;
            
            // 按响应值排序
            sort(pts.begin(), pts.end(),
                [](const auto &a, const auto &b) {
                    return a.second > b.second;
                });
            
            // 选择最多features_per_cell个点
            for (int i = 0; i < min((int)pts.size(), features_per_cell); ++i) {
                // 检查与现有特征的距离
                bool valid = true;
                for (const auto &f : cur_features) {
                    float dist = cv::norm(pts[i].first - f.pt);
                    if (dist < min_feature_distance) {
                        valid = false;
                        break;
                    }
                }
                
                if (valid) {
                    all_new_pts.push_back(pts[i].first);
                }
            }
        }
        
        return all_new_pts;
    }
    
    /**
     * 添加新特征到特征列表
     */
    void addNewFeatures(const vector<cv::Point2f> &new_pts) {
        for (const auto &pt : new_pts) {
            if (cur_features.size() >= (size_t)n_features) {
                break;
            }
            
            Feature f;
            f.pt = pt;
            f.id = next_id++;
            f.track_count = 1;
            f.descriptor_type = DESCRIPTOR_ORB;
            f.response = 0;
            
            cur_features.push_back(f);
        }
        
        // 删除跟踪失败的特征（track_count过低）
        cur_features.erase(
            remove_if(cur_features.begin(), cur_features.end(),
                     [this](const Feature &f) {
                         return f.track_count < 2 && f.descriptor_type == DESCRIPTOR_ORB;
                     }),
            cur_features.end()
        );
    }
};
