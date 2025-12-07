#include <opencv2\opencv.hpp>
#include "SignDetector.h"

using namespace cv;
using namespace std;

vector<SignDetector::DetectedSign> SignDetector::detect(const Mat& inputFrame) {
    vector<DetectedSign> results;
    Mat frame, hsv, maskRed1, maskRed2, maskRed, maskGray;

    // 1️- Image Resizing
    resize(inputFrame, frame, Size(640, 480));

    // 2️- Convert to HSV
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // 3️- Histogram Equalization on V channel
    vector<Mat> channels;
    split(hsv, channels);
    equalizeHist(channels[2], channels[2]);
    merge(channels, hsv);

    // 4️- Gaussian Denoising
    GaussianBlur(hsv, hsv, Size(5, 5), 0);

    // 5️- Color Threshold for Red Channel
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), maskRed1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), maskRed2);
    maskRed = maskRed1 | maskRed2;
    GaussianBlur(maskRed, maskRed, Size(5, 5), 0);

    vector<vector<Point>> contours;
    findContours(maskRed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area < 500) continue;

        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.04 * arcLength(cnt, true), true);
        Rect box = boundingRect(approx);

        if (isStopSign(approx))
            results.push_back({ box, "STOP" });
        else if (isYieldSign(approx))
            results.push_back({ box, "YIELD" });
    }

    // Speed Limit detection - white regions
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    threshold(gray, maskGray, 200, 255, THRESH_BINARY);
    morphologyEx(maskGray, maskGray, MORPH_CLOSE, Mat(), Point(-1, -1), 3);

    findContours(maskGray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area < 800) continue;

        Rect box = boundingRect(cnt);
        if (isSpeedLimitSign(box))
            results.push_back({ box, "SPEED LIMIT" });
    }

    return results;
}

bool SignDetector::isStopSign(const vector<Point>& contour) {
    return contour.size() == 8;
}

bool SignDetector::isYieldSign(const vector<Point>& contour) {
    return contour.size() == 3;
}

bool SignDetector::isSpeedLimitSign(const Rect& rect) {
    float aspect = (float)rect.width / rect.height;
    return aspect > 0.6 && aspect < 1.2; // Rough square (sign is typically near-square)
}

bool SignDetector::isRed(const Mat& mask, const Rect& rect) {
    Mat roi = mask(rect);
    double nonZero = countNonZero(roi);
    double area = rect.width * rect.height;
    return (nonZero / area) > 0.5; // More than 50% red pixels
}

void SignDetector::drawDetections(Mat& frame, const vector<DetectedSign>& detections) {
    for (auto& d : detections) {
        rectangle(frame, d.boundingBox, Scalar(0, 255, 0), 2);
        putText(frame, d.label,
            Point(d.boundingBox.x, d.boundingBox.y - 5),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    }
}