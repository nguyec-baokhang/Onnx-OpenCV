#ifndef KALMAN
#define KALMAN 

#include <opencv2/opencv.hpp>

cv::Mat convert_bbox_to_z(cv::Rect bbox){
  float *box_data = reinterpret_cast<float *>(bbox.data);
}

class KalmanBoxTracker{
  private: 
    kf = cv::KalmanFilter(7,4);
    kf.transitionMatrix = (cv::Mat_<float>(7,7) << 
      1,0,0,0,1,0,0,
      0,1,0,0,0,1,0,
      0,0,1,0,0,0,1,
      0,0,0,1,0,0,0,
      0,0,0,0,1,0,0,
      0,0,0,0,0,1,0,
      0,0,0,0,0,0,1
    );
    kf.measurementMatrix = (cv::Mat_<float>(4,7) << 
      1,0,0,0,0,0,0,
      0,1,0,0,0,0,0,
      0,0,1,0,0,0,0,
      0,0,0,1,0,0,0
    )
    kf.measurementNoiseCov = cv::Mat::eye(4,4,CV_32F);
    kf.measurementNoiseCov(cv::Rect(2,2,2,2) * 10);
    kf.errorCovPost = cv::Mat::eye(7,7, CV_32F);
    kf.errorCovPost(cv::Rect(4,4,4,4) * 1000);
    kf.processNoiseCov = cv::Mat::eye(7,7,CV_32F);
    kf.processNoiseCov.at<float>(-1,-1) *= 0.01;
    kf.processNoiseCov(cv::Rect(4,4,4,4) *= 0.01);


}

#endif // !KALMAN