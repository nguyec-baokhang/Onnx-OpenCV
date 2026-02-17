#ifndef SORT
#define SORT 

class SORT{
  private: 
    int max_age, min_hit, frame_count;
    float iou_threshold{0.3};
    std::vector<int> trackers;

  public:
    cv::Mat update(cv::Mat dets);
}

#endif // !SORT