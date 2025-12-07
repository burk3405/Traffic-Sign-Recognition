#include <opencv2\opencv.hpp>
#include "SignDetector.h"
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

namespace {
    static double pointDist(const Point& a, const Point& b) {
        return std::hypot(double(a.x - b.x), double(a.y - b.y));
    }

    // Relative standard deviation of polygon edge lengths (stddev / mean)
    static double relativeEdgeLengthStdDev(const std::vector<Point>& poly) {
        int n = (int)poly.size();
        if (n < 3) return 1.0;
        std::vector<double> lengths;
        lengths.reserve(n);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            double l = pointDist(poly[i], poly[(i + 1) % n]);
            lengths.push_back(l);
            sum += l;
        }
        double mean = sum / n;
        double var = 0.0;
        for (double l : lengths) var += (l - mean) * (l - mean);
        var /= n;
        double sd = std::sqrt(var);
        return mean > 0.0 ? sd / mean : 1.0;
    }

    // Compute circularity for a contour (value close to 1 => circle)
    static double contourCircularity(const std::vector<Point>& contour) {
        double area = contourArea(contour);
        double perim = arcLength(contour, true);
        if (perim <= 1.0) return 0.0;
        return (4.0 * CV_PI * area) / (perim * perim);
    }

    // Arrow template (right-pointing); Hu-moment-based match is rotation/scale-invariant
    static const std::vector<Point>& getArrowTemplate() {
        static std::vector<Point> tpl;
        if (tpl.empty()) {
            tpl = {
                Point(0,20), Point(40,20), Point(40,10),
                Point(60,30), Point(40,50), Point(40,40),
                Point(0,40)
            };
        }
        return tpl;
    }
}

vector<SignDetector::DetectedSign> SignDetector::detect(const Mat& inputFrame) {
    vector<DetectedSign> results;

    // Work on a copy of the original image (preserve original dimensions)
    Mat frame = inputFrame.clone();
    if (frame.empty()) return results;

    Mat hsv, maskRed1, maskRed2, maskRed, maskGray;

    // Convert to HSV and equalize V channel to improve robustness to lighting
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    vector<Mat> channels;
    split(hsv, channels);
    equalizeHist(channels[2], channels[2]);
    merge(channels, hsv);

    // Denoise
    GaussianBlur(hsv, hsv, Size(5, 5), 0);

    // Color threshold for red (wider range to accommodate lighting)
    inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), maskRed1);
    inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), maskRed2);
    maskRed = maskRed1 | maskRed2;

    // Clean up mask using morphology
    Mat kernelR = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(maskRed, maskRed, MORPH_OPEN, kernelR, Point(-1, -1), 1);
    morphologyEx(maskRed, maskRed, MORPH_CLOSE, kernelR, Point(-1, -1), 1);

    vector<vector<Point>> contours;
    findContours(maskRed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    const double imgArea = double(frame.cols) * double(frame.rows);
    const double minRedArea = max(500.0, 0.0005 * imgArea); // scale with image size

    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area < minRedArea) continue;

        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.04 * arcLength(cnt, true), true);
        Rect box = boundingRect(approx);
        if (box.area() <= 0) continue;

        // STOP / YIELD / WRONG WAY checks (stop/yield have stronger geometry-based heuristics)
        if (isStopSign(approx)) {
            results.push_back({ box, "STOP" });
            continue;
        } else if (isYieldSign(approx)) {
            results.push_back({ box, "YIELD" });
            continue;
        } else if (isWrongWaySign(frame, box, hsv)) {
            results.push_back({ box, "WRONG WAY" });
            continue;
        }
    }

    // SPEED LIMIT and ONE WAY detection - white regions (harsher rules)
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    Mat blurredGray;
    GaussianBlur(gray, blurredGray, Size(5, 5), 0);
    threshold(blurredGray, maskGray, 200, 255, THRESH_BINARY);

    Mat k2 = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(maskGray, maskGray, MORPH_CLOSE, k2, Point(-1, -1), 2);
    morphologyEx(maskGray, maskGray, MORPH_OPEN, k2, Point(-1, -1), 1);

    findContours(maskGray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Increase min white area to reduce small false positives
    const double minWhiteArea = max(1000.0, 0.0012 * imgArea);

    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area < minWhiteArea) continue;

        Rect box = boundingRect(cnt);
        if (box.area() <= 0) continue;

        // Prefer ONE WAY (arrow) detections first
        if (isOneWaySign(frame, box)) {
            results.push_back({ box, "ONE WAY" });
            continue;
        }

        // Quick shape filter: speed limit signs are usually near-square / circular in bounding box
        float aspect = (float)box.width / float(max(1, box.height));
        if (aspect < 0.8f || aspect > 1.25f) continue;

        // Additional aspect constraint from helper
        if (!isSpeedLimitSign(box)) continue;

        // Validate ROI content: require bright background + dark, letter-like components
        Rect clipped = box & Rect(0, 0, gray.cols, gray.rows);
        if (clipped.area() <= 0) continue;
        Mat roiGray = gray(clipped);

        // Must be bright enough to be a white panel
        Scalar meanVal = mean(roiGray);
        if (meanVal[0] < 180.0) continue;

        // Detect dark components (letters/digits) using inverted Otsu
        Mat lettersMask;
        threshold(roiGray, lettersMask, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

        // Clean small noise and close strokes (kernel scales with ROI)
        int letterK = max(1, min(clipped.width, clipped.height) / 50);
        Mat kLetter = getStructuringElement(MORPH_RECT, Size(letterK, letterK));
        morphologyEx(lettersMask, lettersMask, MORPH_OPEN, kLetter, Point(-1, -1), 1);
        morphologyEx(lettersMask, lettersMask, MORPH_CLOSE, kLetter, Point(-1, -1), 1);

        // Find dark contours (candidate letters)
        vector<vector<Point>> letterContours;
        findContours(lettersMask, letterContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double lettersAreaSum = 0.0;
        int letterCount = 0;
        double largestLetterArea = 0.0;
        double minLetterArea = max(40.0, 0.001 * clipped.area()); // stricter min letter size

        for (const auto& lc : letterContours) {
            double la = contourArea(lc);
            if (la < minLetterArea) continue;
            if (la > 0.45 * clipped.area()) continue; // too large -> not a digit
            Rect lr = boundingRect(lc);
            double lrAspect = double(lr.width) / double(max(1, lr.height));
            if (lrAspect < 0.08 || lrAspect > 12.0) continue;
            lettersAreaSum += la;
            ++letterCount;
            largestLetterArea = max(largestLetterArea, la);
        }

        double darkRatio = lettersAreaSum / double(clipped.area());
        double circ = contourCircularity(cnt);

        // Check for a red border (common on circular speed-limit signs)
        Mat roiHSV = hsv(clipped);
        Mat r1, r2, redMaskROI;
        inRange(roiHSV, Scalar(0, 70, 50), Scalar(10, 255, 255), r1);
        inRange(roiHSV, Scalar(170, 70, 50), Scalar(180, 255, 255), r2);
        redMaskROI = r1 | r2;
        double redRatio = double(countNonZero(redMaskROI)) / double(max(1, clipped.area()));

        bool accept = false;

        // Rule 1: circular sign with red border and some dark strokes
        if (circ > 0.60 && redRatio > 0.04 && darkRatio > 0.02 && letterCount >= 1) {
            accept = true;
        }

        // Rule 2: near-square panel with multiple digit-like blobs
        if (!accept && aspect > 0.88f && aspect < 1.12f && letterCount >= 2 && darkRatio > 0.02) {
            accept = true;
        }

        // Rule 3: fallback — at least two decent letter blobs + stronger dark coverage
        if (!accept && letterCount >= 2 && darkRatio > 0.035) {
            accept = true;
        }

        if (accept) {
            results.push_back({ box, "SPEED LIMIT" });
        }
    }

    // Basic non-max suppression to reduce duplicates
    vector<DetectedSign> filtered;
    for (const auto& d : results) {
        bool keep = true;
        for (auto it = filtered.begin(); it != filtered.end();) {
            Rect inter = (d.boundingBox & it->boundingBox);
            double iArea = double(inter.area());
            double uArea = double(d.boundingBox.area() + it->boundingBox.area() - inter.area());
            double iou = (uArea > 0.0) ? (iArea / uArea) : 0.0;
            if (iou > 0.5) {
                if (d.boundingBox.area() > it->boundingBox.area()) {
                    it = filtered.erase(it);
                    continue;
                } else {
                    keep = false;
                    break;
                }
            }
            ++it;
        }
        if (keep) filtered.push_back(d);
    }

    return filtered;
}

bool SignDetector::isStopSign(const vector<Point>& contour) {
    // Fallback heuristic: roughly octagonal and convex
    return (contour.size() >= 7 && contour.size() <= 9 && isContourConvex(contour));
}

bool SignDetector::isYieldSign(const vector<Point>& contour) {
    // Fallback heuristic: triangle and convex
    return (contour.size() == 3 && isContourConvex(contour));
}

bool SignDetector::isSpeedLimitSign(const Rect& rect) {
    float aspect = (float)rect.width / rect.height;
    return aspect > 0.6f && aspect < 1.2f; // Rough square (sign is typically near-square)
}

bool SignDetector::isRed(const Mat& mask, const Rect& rect) {
    Rect r = rect & Rect(0, 0, mask.cols, mask.rows);
    if (r.area() <= 0) return false;
    Mat roi = mask(r);
    double nonZero = double(countNonZero(roi));
    double area = double(r.area());
    return (nonZero / area) > 0.5; // More than 50% red pixels
}

// Detect WRONG WAY (red rectangular sign with white text)
bool SignDetector::isWrongWaySign(const Mat& frame, const Rect& rect, const Mat& hsv) {
    Rect r = rect & Rect(0, 0, frame.cols, frame.rows);
    if (r.area() <= 0) return false;

    // Quick aspect check: WRONG WAY is usually horizontal rectangle
    double aspect = double(r.width) / double(max(1, r.height));
    if (aspect < 1.2 || aspect > 4.0) return false;

    Mat roiHSV = hsv(r);

    // White/near-white mask (letters)
    Mat whiteMask;
    inRange(roiHSV, Scalar(0, 0, 200), Scalar(180, 100, 255), whiteMask);
    int k = max(1, min(r.width, r.height) / 40);
    Mat ksm = getStructuringElement(MORPH_RECT, Size(k, k));
    morphologyEx(whiteMask, whiteMask, MORPH_OPEN, ksm, Point(-1, -1), 1);
    morphologyEx(whiteMask, whiteMask, MORPH_CLOSE, ksm, Point(-1, -1), 1);

    // Count white blobs (letter-like)
    vector<vector<Point>> wcontours;
    findContours(whiteMask, wcontours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double minLetterArea = max(12.0, 0.0004 * r.area());
    int letterCount = 0;
    double lettersAreaSum = 0.0;
    for (const auto& wc : wcontours) {
        double la = contourArea(wc);
        if (la < minLetterArea) continue;
        if (la > 0.6 * r.area()) continue;
        Rect lr = boundingRect(wc);
        double ar = double(lr.width) / double(max(1, lr.height));
        if (ar < 0.05 || ar > 12.0) continue;
        ++letterCount;
        lettersAreaSum += la;
    }

    double whiteRatio = double(countNonZero(whiteMask)) / double(r.area());

    // Check red coverage inside candidate (must be predominantly red)
    Mat red1, red2, rmask;
    inRange(roiHSV, Scalar(0, 70, 50), Scalar(10, 255, 255), red1);
    inRange(roiHSV, Scalar(170, 70, 50), Scalar(180, 255, 255), red2);
    rmask = red1 | red2;
    double redRatio = double(countNonZero(rmask)) / double(r.area());

    // Heuristic thresholds: require red background, some white letters, and aspect
    if (redRatio > 0.35 && letterCount >= 2 && whiteRatio > 0.015 && aspect >= 1.2) {
        return true;
    }

    return false;
}

// Detect ONE WAY (white panel containing an arrow-shaped dark blob OR dark panel with white arrow)
bool SignDetector::isOneWaySign(const Mat& frame, const Rect& rect) {
    Rect r = rect & Rect(0, 0, frame.cols, frame.rows);
    if (r.area() <= 0) return false;

    Mat roi = frame(r);
    Mat roiGray;
    cvtColor(roi, roiGray, COLOR_BGR2GRAY);

    // First attempt: black arrow on white background
    Mat darkMask;
    threshold(roiGray, darkMask, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    int letterK = max(1, min(r.width, r.height) / 50);
    Mat km = getStructuringElement(MORPH_RECT, Size(letterK, letterK));
    morphologyEx(darkMask, darkMask, MORPH_OPEN, km, Point(-1, -1), 1);
    morphologyEx(darkMask, darkMask, MORPH_CLOSE, km, Point(-1, -1), 1);

    vector<vector<Point>> dcontours;
    findContours(darkMask, dcontours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double largestArea = 0.0;
    int largestIdx = -1;
    for (size_t i = 0; i < dcontours.size(); ++i) {
        double a = contourArea(dcontours[i]);
        if (a > largestArea) {
            largestArea = a;
            largestIdx = int(i);
        }
    }

    double areaRatio = largestArea / double(max(1, r.area()));
    const vector<Point>& arrowTpl = getArrowTemplate();

    if (largestIdx >= 0 && areaRatio > 0.02 && areaRatio < 0.7) {
        double match = matchShapes(dcontours[largestIdx], arrowTpl, 1, 0.0);
        if (match < 0.35) {
            return true;
        }
    }

    // Second attempt: white arrow on dark background (inverse)
    Scalar meanVal = mean(roiGray);
    if (meanVal[0] < 110.0) {
        Mat lightMask;
        threshold(roiGray, lightMask, 0, 255, THRESH_BINARY | THRESH_OTSU);
        morphologyEx(lightMask, lightMask, MORPH_OPEN, km, Point(-1, -1), 1);
        morphologyEx(lightMask, lightMask, MORPH_CLOSE, km, Point(-1, -1), 1);

        vector<vector<Point>> lcontours;
        findContours(lightMask, lcontours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double lLargest = 0.0;
        int lIdx = -1;
        for (size_t i = 0; i < lcontours.size(); ++i) {
            double a = contourArea(lcontours[i]);
            if (a > lLargest) {
                lLargest = a;
                lIdx = int(i);
            }
        }

        double lAreaRatio = lLargest / double(max(1, r.area()));
        if (lIdx >= 0 && lAreaRatio > 0.02 && lAreaRatio < 0.7) {
            double match = matchShapes(lcontours[lIdx], arrowTpl, 1, 0.0);
            if (match < 0.35) {
                return true;
            }
        }
    }

    return false;
}

void SignDetector::drawDetections(Mat& frame, const vector<DetectedSign>& detections) {
    for (const auto& d : detections) {
        rectangle(frame, d.boundingBox, Scalar(0, 255, 0), 2);
        putText(frame, d.label,
            Point(d.boundingBox.x, max(12, d.boundingBox.y - 5)),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    }
}
