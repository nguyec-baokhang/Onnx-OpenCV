#ifndef KALMAN
#define KALMAN 

#include <opencv2/opencv.hpp>

class Kalman_Filter_2D{
  private: 
    cv::Mat x,P,F,H,u,z,R;
  public: 
    void predict(cv::Mat state_est, cv::Mat cov_mat, cv::Mat state_transition_mat, cv::Mat control_input);
    void measurement_update(cv::Mat act_meas, cv::Mat measurement_func, cv::Mat meas_noise_cov);
    void matrix_check(cv::Mat &matrix);
    cv::Mat state_estimate_getter();
    cv::Mat covariance_getter();
};

#endif // !KALMAN