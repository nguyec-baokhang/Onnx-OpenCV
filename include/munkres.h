#ifndef MUNKRES
#define MUNKRES

#include <opencv2/opencv.hpp>

class Munkres{
  private:
    bool check;
    cv::Mat mask;
    std::vector<std::vector<bool>> marked;

  public: 
    void rowReduce(cv::Mat &costMatrix);
    void colReduce(cv::Mat &costMatrix);
    cv::Mat mask_assignment(cv::Mat &matrix);
    bool check_requirement(cv::Mat mask);
    std::vector<std::vector<bool>> minimal_line_marking(cv::Mat &matrix, cv::Mat &mask);
    void min_subtraction(cv::Mat &matrix, const std::vector<std::vector<bool>> &marked);
    void matrix_check(cv::Mat &matrix);
    cv::Mat main_loop(cv::Mat &matrix);
};

#endif // !MUNKRES