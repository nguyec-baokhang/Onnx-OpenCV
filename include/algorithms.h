#pragma once 
#include <opencv2/opencv.hpp>

// Munkres helper functions 
void rowReduce(cv::Mat &costMatrix){
  for(int i = 0; i < costMatrix.rows; i++){
    double minVal;
    cv::minMaxLoc(costMatrix.row(i), &minVal, nullptr);
    costMatrix.row(i) -= minVal;
  }
}

void colReduce(cv::Mat &costMatrix){
  for(int i = 0; i < costMatrix.cols; i++){
    double minVal;
    cv::minMaxLoc(costMatrix.col(i), &minVal, nullptr);
    costMatrix.col(i) -= minVal;
  }
}

cv::Mat mask_assignment(cv::Mat &matrix){
  cv::Mat mask = cv::Mat::zeros(matrix.size(), CV_8U);
  std::vector<bool> row_assigned(matrix.rows, false);
  std::vector<bool> col_assigned(matrix.cols, false);

  for(int i = 0; i < matrix.rows; i++){
    for(int j = 0; j < matrix.cols; j++){
      if(matrix.at<float>(i,j) == 0 && !row_assigned[i] && !col_assigned[j]){
        mask.at<uchar>(i,j) = 1;
        row_assigned[i] = true;
        col_assigned[j] = true; 
      }
    }
  }
  std::cout << mask << "\n";

  return mask;
}

bool check_requirement(cv::Mat mask){
  int n = mask.rows; 
  int assignment_counter{0};

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      if(mask.at<uchar>(i,j) == 1){
        assignment_counter++;
      }
    }
  }

  if(n == assignment_counter){
    return true;
  }

  return false;
}

std::vector<std::vector<bool>> minimal_line_marking(cv::Mat &matrix, cv::Mat &mask){
  int n = matrix.rows;
  std::vector<std::vector<bool>> marked;
  std::vector<bool> row_marked(n,false);
  std::vector<bool> col_marked(n,false);
  bool change;

  // Mark all rows with no assignment
  for(int i = 0; i < n; i++){
    bool has_assignment = false;
    for(int j = 0; j < n; j++){
      if(mask.at<uchar>(i,j) == 1){
        has_assignment = true;
      }
    }
    if(!has_assignment) row_marked[i] = true;
  }

  do{
    change = false;
    // Iterate through marked rows' column with 0
    for(int i = 0; i < n; i++){
      if(row_marked[i]){
        for(int j = 0; j < n; j++){
          if(matrix.at<float>(i,j) == 0 && !col_marked[j]){
            col_marked[j] = true;
            change = true;
          }
        }
      }
    }

    // Iterate through marked columns, mark row with assignment 
    for(int j = 0; j < n; j++){
      if(col_marked[j]){
        for(int i = 0; i < n; i++){
          if(mask.at<uchar>(i,j) == 1 && !row_marked[i]){
            row_marked[i] = true;
            change = true;
          }
        }
      }
    }
  }while(change);

  return {row_marked, col_marked};
}

void min_subtraction(cv::Mat &matrix, const std::vector<std::vector<bool>> &marked){
  float min_uncovered = std::numeric_limits<float>::max();
  for(int i = 0; i < matrix.rows; i++){
    if ((marked[0][i])){
      for(int j = 0; j < matrix.cols; j++){
        if (!(marked[1][j])){
          if(matrix.at<float>(i,j) < min_uncovered){
            min_uncovered = matrix.at<float>(i,j);
          }
        }
      }
    }
  }
  std::cout << min_uncovered << "\n";

  for(int i = 0; i < matrix.rows; i++){
    for(int j = 0; j < matrix.cols; j++){
      if((marked[0][i]) && !(marked[1][j])){
        matrix.at<float>(i,j) -= min_uncovered;
      }
      else if(!(marked[0][i]) && (marked[1][j])){
        matrix.at<float>(i,j) += min_uncovered;
      }
    }
  }

  std::cout << matrix << "\n";
}

namespace Algorithms
{
  cv::Mat Munkres(cv::Mat &matrix){
    bool check;
    cv::Mat mask;
    std::vector<std::vector<bool>> marked;

    if(matrix.empty()){
      throw std::invalid_argument("The matrix is empty");
    }
    if(matrix.rows != matrix.cols){
      int n = std::max(matrix.rows,matrix.cols);
      cv::Mat squareMat = cv::Mat::zeros(n,n,matrix.type());
      matrix.copyTo(squareMat(cv::Rect(0, 0, matrix.cols, matrix.rows)));
      matrix = squareMat;
    }

    rowReduce(matrix);
    colReduce(matrix);

    do{
      mask = mask_assignment(matrix);
      check = check_requirement(mask);
      marked = minimal_line_marking(matrix,mask);
      min_subtraction(matrix,marked);
    }while(check);

    return matrix;
  }
}
