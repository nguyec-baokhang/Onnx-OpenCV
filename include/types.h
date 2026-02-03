#include <opencv2/opencv.hpp>
#pragma once

struct YoloBoundingBox{
  cv::Rect bounding_box;
  double confidence;
  int class_id;
};