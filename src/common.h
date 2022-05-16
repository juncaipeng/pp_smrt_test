#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <stdarg.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#pragma once

class Time {
private:
  std::chrono::time_point<std::chrono::steady_clock> _start_time;
  std::chrono::time_point<std::chrono::steady_clock> _end_time;
  double _used_time;

public:
  Time() {
      _used_time = 0.0;
  }

  void start();
  void stop();
  void clear();
  double used_time();   // return the used time (ms)
};

cv::Mat read_image(const std::string& img_path);
void hwc_2_chw_data(const cv::Mat& hwc_img, float* data);

template<typename T>
std::string vector_2_str(std::vector<T> input) {
  std::stringstream ss;
  for (auto i : input) {
    ss << i << " ";
  }
  return ss.str();
}

bool file_exists(const std::string& path);
