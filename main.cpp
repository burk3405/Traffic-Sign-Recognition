#include <opencv2\opencv.hpp>
#include "SignDetector.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {

    string imgName;
    cout << "Enter image filename: ";
    cin >> imgName;

    string filePath = imgName;
    Mat frame = imread(filePath);

    if (frame.empty()) {
        cout << "Error: Cannot load image: " << filePath << endl;
        return -1;
    }

    SignDetector detector;
    auto detections = detector.detect(frame);

    detector.drawDetections(frame, detections);

    // Ensure results directory exists
    system("mkdir -p results");
    string savePath = "results/output_" + imgName;
    imwrite(savePath, frame);

    cout << "Detections: " << detections.size() << endl;

    // Ensure logs folder exists
    system("mkdir -p logs");
    ofstream log("logs/detection_log.csv", ios::app);
    log << imgName << "," << detections.size() << "\n";
    log.close();

    cout << "Processed image saved to: " << savePath << endl;

    return 0;
}