#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

Ort::Session LoadYoloModel(const Ort::Env &env, const std::string &modelPath){
  Ort::SessionOptions yoloSessionOption;
  Ort::Session yoloModelSession(env,modelPath.c_str(),yoloSessionOption);
  return yoloModelSession;
}

cv::Mat LetterBox(cv::Mat img, cv::Size new_size){
  int fill_value = 114;
  float aspect_ratio = std::min(
    static_cast<float>(new_size.width) / img.cols,
    static_cast<float>(new_size.height) / img.rows
  );

  int resized_w = static_cast<int>(img.cols * aspect_ratio);
  int resized_h = static_cast<int>(img.rows * aspect_ratio);

  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(resized_w, resized_h));

  cv::Mat padded_img(new_size, CV_8UC3, cv::Scalar(fill_value, fill_value, fill_value));

  int x_range_start = (new_size.width - resized_w) / 2;
  int y_range_start = (new_size.height - resized_h) / 2;

  cv::Rect roi(x_range_start, y_range_start, resized_w, resized_h);
  resized_img.copyTo(padded_img(roi));

  // cv::imshow("Padded image",padded_img);
  // cv::waitKey(0);

  return padded_img;
}

cv::Mat LoadImage(const std::string &imagePath){
  cv::Mat image = cv::imread(imagePath);
  cv::Mat padded_image;
  padded_image = LetterBox(image,cv::Size(640,640));
  return cv::dnn::blobFromImage(
    padded_image,
    1.0 / 255, // Scale 0 -> 1 
    cv::Size(640,640),
    cv::Scalar::all(0),
    true,
    false,
    CV_32F
  );
}

int main(){
  const Ort::Env env(ORT_LOGGING_LEVEL_ERROR,"YOLO");
  const std::string modelPath = "../models/yolov8n.onnx";
  const std::string imagePath = "../images/image1.jpg";

  Ort::Session yolo_model_session = LoadYoloModel(env,modelPath);
  cv::Mat blob = LoadImage(imagePath);
  // cv::imshow("Image Window",blob);
  return 0;
}