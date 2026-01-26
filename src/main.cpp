#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <fstream> 

#include "utilities.h"


int main(){
  const Ort::Env env(ORT_LOGGING_LEVEL_ERROR,"YOLO");
  const std::string modelPath = "../models/yolo11n.onnx";
  const std::string imagePath = "../images/image1.jpg";
  const std::string labelsPath = "../labels/coco.txt";
  std::string videoPath = "../videos/tennis.mp4";

  cv::Mat frame;
  
  cv::VideoCapture video(videoPath); 
  if (!video.isOpened()) {
    std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
    return -1;
  } 

  Ort::Session yolo_model_session = Utilities::LoadYoloModel(env,modelPath);

  // Inferencing 
  Ort::AllocatorWithDefaultOptions allocator;
  const std::string input_name = yolo_model_session.GetInputNameAllocated(0,allocator).get();
  const std::string output_name = yolo_model_session.GetOutputNameAllocated(0,allocator).get();
  const char* input_names[] = {input_name.c_str()};
  const char* output_names[] = {output_name.c_str()};

  std::vector<std::string> labels = Utilities::LoadLabels(labelsPath);

  cv::namedWindow("Computer Vision");
  while(true){
    video >> frame; 
    if(frame.empty()){
      break;
    }
    
    cv::Mat blob = Utilities::ImageToBlob(frame);
    Ort::Value input_tensor = Utilities::BlobToOnnxTensor(blob);
    std::vector<Ort::Value> output = yolo_model_session.Run(
      Ort::RunOptions{nullptr},
      input_names,
      &input_tensor,
      1,
      output_names,
      1
    );

    cv::Mat raw_boxes = Utilities::getYoloBox(output);
    std::vector<YoloBoundingBox> filtered_boxes = Utilities::ProcessYoloOutputs(raw_boxes, frame.size());
    // Highlight the YOLO box that most overlaps with the CamShift region
    Utilities::drawing(filtered_boxes, frame, labels);
    cv::imshow("Computer Vision", frame);
    int key = cv::waitKey(1000 / 120);
    if(key==27){
      break;
    }
  }

  cv::destroyAllWindows();
  video.release();
  return 0;
}