#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "common.h"

void Time::start() {
    _start_time = std::chrono::steady_clock::now();
}

void Time::stop() {
    _end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = _end_time - _start_time;
    _used_time += diff.count() * 1000;
}

void Time::clear() {
    _used_time = 0.0;
}

double Time::used_time() {
    return _used_time;
}

cv::Mat read_image(const std::string& img_path) {
  cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  return img;
}

void hwc_2_chw_data(const cv::Mat& hwc_img, float* data) {
  int rows = hwc_img.rows;
  int cols = hwc_img.cols;
  int chs = hwc_img.channels();
  for (int i = 0; i < chs; ++i) {
    cv::extractChannel(hwc_img, cv::Mat(rows, cols, CV_32FC1, data + i * rows * cols), i);
  }
}

bool file_exists(const std::string& path) {
#ifdef _WIN32
    struct _stat buffer;
    return (_stat(path.c_str(), &buffer) == 0);
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
#endif  // !_WIN32
}
