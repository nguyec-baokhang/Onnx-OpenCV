#ifndef SORT_H
#define SORT_H

///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <vector>

#include "types.h"
#include "KalmanTracker.h"

class SORT {
  private:
    int max_age_{};
    int min_hits_{};
    int frame_count_{};
    float iou_threshold_{};

    std::vector<KalmanTracker> trackers_;

  public:
    SORT(int max_age = 1, int min_hits = 3, float iou_threshold = 0.3f);

    std::vector<TrackingBox> update(const std::vector<YoloBoundingBox> &detections);
};

#endif // !SORT_H