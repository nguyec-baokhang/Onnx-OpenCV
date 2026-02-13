#include "kalman_filter.h"
#include <opencv2/opencv.hpp>

void Kalman_Filter_2D::predict(cv::Mat state_est, cv::Mat cov_mat, cv::Mat state_transition_mat, cv::Mat control_input){
  x = state_est;
  P = cov_mat;
  F = state_transition_mat;
  u = control_input;
  cv::Mat Ft;
  cv::transpose(F,Ft);
  x = F * x + u;
  P = F * P * Ft;
}

void Kalman_Filter_2D::measurement_update(cv::Mat act_meas, cv::Mat measurement_func, cv::Mat meas_noise_cov){
  cv::Mat Ht;
  H = measurement_func;
  R = meas_noise_cov;
  int n = x.rows;
  cv::Mat I = cv::Mat::eye(n,n,CV_32F);
  cv::transpose (H, Ht);
  cv::Mat y = act_meas - H * x; // innovation
  cv::Mat S = H * P * Ht + R; // innovation covariance 
  cv::Mat K = P * Ht * (S.inv()); 
  x = x + K * y;
  P = (I - K * H) * P;
  
}

cv::Mat Kalman_Filter_2D::state_estimate_getter(){
  std::cout << "State estimate: " << x << "\n";
  return x;
}

cv::Mat Kalman_Filter_2D::covariance_getter(){
  std::cout << "Covariance: " << P << "\n";
  return P;
}

void Kalman_Filter_2D::matrix_check(cv::Mat &matrix){
  if(matrix.empty()){
      throw std::invalid_argument("The matrix is empty");
    }
  if(matrix.rows != matrix.cols){
    int n = std::max(matrix.rows,matrix.cols);
    cv::Mat squareMat = cv::Mat::zeros(n,n,matrix.type());
    matrix.copyTo(squareMat(cv::Rect(0, 0, matrix.cols, matrix.rows)));
    matrix = squareMat;
  }
}