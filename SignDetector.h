#ifndef SIGN_DETECTOR_H
#define SIGN_DETECTOR_H

#include <opencv2\opencv.hpp>
#include <string>
#include <vector>

class SignDetector {
public:
    struct DetectedSign {
        cv::Rect boundingBox;
        std::string label;
    };

    std::vector<DetectedSign> detect(const cv::Mat& frame);
    void drawDetections(cv::Mat& frame, const std::vector<DetectedSign>& detections);

private:
    bool isStopSign(const std::vector<cv::Point>& contour);
    bool isYieldSign(const std::vector<cv::Point>& contour);
    bool isSpeedLimitSign(const cv::Rect& rect);
    bool isRed(const cv::Mat& mask, const cv::Rect& rect);
};

#endif
#pragma once
