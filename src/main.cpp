#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <fstream> 

bool selectObj = false; 
int trackedObj = 0; 
cv::Rect selection; 
cv::Mat image;

struct YoloBoundingBox{
  cv::Rect bounding_box;
  double confidence;
  int class_id;
};

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

  return padded_img;
}

void ClipBox(cv::Rect &box, const cv::Size &shape){
  box.x = std::max(0,std::min(box.x,shape.width));
  box.y = std::max(0,std::min(box.y,shape.height));
  box.width = std::max(0, std::min(box.width, shape.width - box.x));
  box.height = std::max(0, std::min(box.height, shape.height - box.y));
}

cv::Mat ImageToBlob(const cv::Mat &image){
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

Ort::Value BlobToOnnxTensor(const cv::Mat &blob){
  // Create tensor 
  std::vector<int64_t> tensor_shape = {1,3,640,640};
  int64_t input_size = 1 * 3 * 640 * 640;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value tensor = Ort::Value::CreateTensor(
    memory_info,
    (float*) (blob.data), // C-style casting
    input_size,
    tensor_shape.data(),
    tensor_shape.size()
  );
  return tensor;
}

cv::Mat getYoloBox(std::vector<Ort::Value> &output){
  std::vector<int64_t> output_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
  return cv::Mat(
    cv::Size(static_cast<int>(output_shape[2]),
            static_cast<int>(output_shape[1])),
    CV_32F,
    output[0].GetTensorMutableData<float>()
  ).t();
}

void ScaleYoloBox(YoloBoundingBox &box, const cv::Size &original_shape){
  cv::Size yolo_shape(640,640);
  float scale_ratio = std::min(
    static_cast<float>(yolo_shape.height) / original_shape.height,
    static_cast<float>(yolo_shape.width) / original_shape.width
  );
  float new_unpad_w = original_shape.width * scale_ratio;
  float new_unpad_h = original_shape.height * scale_ratio;
  float letterbox_padding_x = (yolo_shape.width - new_unpad_w) / 2;
  float letterbox_padding_y = (yolo_shape.height - new_unpad_h) / 2;
  box.bounding_box.x -= letterbox_padding_x; 
  box.bounding_box.y -= letterbox_padding_y; 
  box.bounding_box.x /= scale_ratio;
  box.bounding_box.y /= scale_ratio;
  box.bounding_box.width /= scale_ratio;
  box.bounding_box.height /= scale_ratio;
  ClipBox(box.bounding_box, original_shape);
}

std::vector<YoloBoundingBox> ProcessYoloOutputs(const cv::Mat &raw_boxes, const cv::Size &original_shape){
  int num_classes{80};
  int data_width = num_classes + 4; 
  float *box_data = reinterpret_cast<float *>(raw_boxes.data);
  std::vector<YoloBoundingBox> processed_boxes;

  for(int row = 0; row < raw_boxes.rows; row++){
    // Step 1: Extract coords
    float center_x = box_data[0];
    float center_y = box_data[1];
    float width = box_data[2];
    float height = box_data[3];
    float x = std::max(center_x - 0.5 * width, 0.0);
    float y = std::max(center_y - 0.5 * height, 0.0);
    cv::Rect bounding_box(static_cast<int>(x), static_cast<int>(y), static_cast<int>(width), static_cast<int>(height));
    cv::Mat class_scores = cv::Mat(1, num_classes, CV_32FC1, box_data + 4);
    cv::Point class_id_tmp;
    double confidence;
    cv::minMaxLoc(class_scores, nullptr, &confidence, nullptr, &class_id_tmp);
    int class_id = class_id_tmp.x;

    // Step 2: Filter by confidence
    if(confidence > 0.25){
      YoloBoundingBox parsed_box{bounding_box,confidence,class_id};
      processed_boxes.push_back(parsed_box);
    }
    box_data += data_width;
  }
  // Step 3: Scale boxes
  float conf_thres{0.25};
  float iou{0.7};
  std::vector<cv::Rect>boxes;
  std::vector<float>confidences;
  std::vector<int>filtered_indices;
  for(size_t i = 0; i < processed_boxes.size(); i++){
    auto &box = processed_boxes[i];
    ScaleYoloBox(box,original_shape);
    // Step 4: Non-max suppression 
    boxes.push_back(box.bounding_box);
    confidences.push_back(static_cast<float>(box.confidence));
  }
  cv::dnn::NMSBoxes(boxes, confidences, conf_thres, iou, filtered_indices);
  std::vector<YoloBoundingBox> filtered_boxes;
  for(size_t i = 0; i < filtered_indices.size(); i++){
    filtered_boxes.push_back(processed_boxes[filtered_indices[i]]);
  }

  return filtered_boxes;
}

void onMouse(int event ,int x, int y, int flag, void*){
  static cv::Point origin;
  if(selectObj){
    selection.x = std::min(x,origin.x);
    selection.y = std::min(y,origin.y);
    selection.width = std::abs(x-origin.x);
    selection.height = std::abs(y-origin.y);

    selection &= cv::Rect(0, 0, image.cols, image.rows);
  }

  switch(event){
    case cv::EVENT_LBUTTONDOWN: 
      origin = cv::Point(x,y);
      selection = cv::Rect(0, 0, x, y);
      selectObj = true;
      break;
    case cv::EVENT_LBUTTONUP: 
      selectObj = false;
      if(selection.width > 0 && selection.height > 0){
        trackedObj = -1; // tracked object has not calculated Camshift
      }
      break;

  }
}

double computeIoU(const cv::Rect &box, const cv::Rect &focusedBox){
  int intersectionArea = (focusedBox & box).area();
  int unionArea = focusedBox.area() + box.area() - intersectionArea;
  return unionArea > 0 ?  static_cast<double>(intersectionArea) / unionArea : 0.0;
}

std::vector<std::string> LoadLabels(const std::string labelsPath){
  std::vector<std::string> labels;
  std::string label;
  std::ifstream coco(labelsPath);
  while (std::getline(coco,label)){
    labels.push_back(label);
  }
  return labels;
}

void drawing(const std::vector<YoloBoundingBox> &boxes, cv::Mat &frame, const std::vector<std::string> &labels){
  for(const auto &box: boxes) {
    std::string label = labels[box.class_id];
    // Draw bounding box
    cv::rectangle(frame, box.bounding_box, cv::Scalar(0, 255, 0), 2);
    // Draw label background
    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = std::max(box.bounding_box.y, labelSize.height);
    cv::rectangle(frame, cv::Point(box.bounding_box.x, top - labelSize.height - 5),
                cv::Point(box.bounding_box.x + labelSize.width, top + baseLine - 5),
                cv::Scalar(0, 255, 0), cv::FILLED);
    // Draw label text with confidence
    char label_text[128];
    snprintf(label_text, sizeof(label_text), "%s: %.2f", label.c_str(), box.confidence);
    cv::putText(frame, label_text, cv::Point(box.bounding_box.x, top - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
  }
}


int main(){
  const Ort::Env env(ORT_LOGGING_LEVEL_ERROR,"YOLO");
  const std::string modelPath = "../models/yolo11n.onnx";
  const std::string imagePath = "../images/image1.jpg";
  const std::string labelsPath = "../labels/coco.txt";
  std::string videoPath = "../videos/tennis.mp4";

  const float overlapedThreshold{0.3};
  cv::Mat frame;
  cv::Mat hsv, hue, mask, hist, backproj;
  cv::Rect trackedWindow;             // tracking window
  int hsize = 16;                   // for histogram
  float hranges[] = {0,180};        // for histogram
  const float* phranges = hranges;  // for histogram
  cv::RotatedRect trackedBox;
  
  cv::VideoCapture video(videoPath); 
  if (!video.isOpened()) {
    std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
    return -1;
  } 

  Ort::Session yolo_model_session = LoadYoloModel(env,modelPath);
  // Single image processing
  // cv::Mat image = cv::imread(imagePath);  
  // cv::Mat blob = ImageToBlob(image);
  // Ort::Value input_tensor = BlobToOnnxTensor(blob);

  // Inferencing 
  Ort::AllocatorWithDefaultOptions allocator;
  const std::string input_name = yolo_model_session.GetInputNameAllocated(0,allocator).get();
  const std::string output_name = yolo_model_session.GetOutputNameAllocated(0,allocator).get();
  const char* input_names[] = {input_name.c_str()};
  const char* output_names[] = {output_name.c_str()};

  std::vector<std::string> labels = LoadLabels(labelsPath);

  cv::namedWindow("Computer Vision");
  while(true){
    video >> frame; 
    if(frame.empty()){
      break;
    }
    frame.copyTo(image);
    cv::setMouseCallback("Computer Vision",onMouse, NULL);
    cv::cvtColor(image,hsv, cv::COLOR_BGR2HSV);

    if(trackedObj){
      cv::inRange(hsv,cv::Scalar(0,30,10),cv::Scalar(180,256,10),mask);
      int ch[] = {0,0};
      hue.create(hsv.size(), hsv.depth());
      cv::mixChannels(&hsv,1,&hue,1,ch,1);
      if(trackedObj < 0){
        // Setup chanel h and mask ROI
        cv::Mat roi(hue,selection),maskroi(mask,selection);
        // Calculate ROI histogram
        calcHist(&roi,1,0,maskroi,hist,1,&hsize,&phranges);
        // Normalization of histogram 
        normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

        trackedWindow = selection; 
        trackedObj = 1;
      }
      // Backproject histogram
      calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
      backproj &= mask;
      trackedBox = CamShift(backproj, trackedWindow, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER , 10, 1 ));
    }

    cv::Mat blob = ImageToBlob(frame);
    Ort::Value input_tensor = BlobToOnnxTensor(blob);
    std::vector<Ort::Value> output = yolo_model_session.Run(
      Ort::RunOptions{nullptr},
      input_names,
      &input_tensor,
      1,
      output_names,
      1
    );

    cv::Mat raw_boxes = getYoloBox(output);
    std::vector<YoloBoundingBox> filtered_boxes = ProcessYoloOutputs(raw_boxes, frame.size());

    if(trackedObj > 0){
      std::vector<YoloBoundingBox> focused_boxes;
      cv::Rect focusedBox = trackedBox.boundingRect();
      for(const auto &box : filtered_boxes){
        if(computeIoU(box.bounding_box, focusedBox) > overlapedThreshold){
          focused_boxes.push_back(box);
        }
      }
      drawing(focused_boxes, frame, labels);
    } else {
      drawing(filtered_boxes, frame, labels);
    }
    cv::imshow("Computer Vision", frame);
    retangle(frame,trackedBox)
    int key = cv::waitKey(1000 / 120);
    if(key==27){
      break;
    }
  }

  cv::destroyAllWindows();
  video.release();
  return 0;
}