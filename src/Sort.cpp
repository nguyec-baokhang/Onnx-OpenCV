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

#include <algorithm>
#include <cfloat>
#include <limits>
#include <set>

#include "Sort.h"
#include "Hungarian.h"
#include "types.h"

namespace {
double GetIOU(const cv::Rect_<float> &bb_test, const cv::Rect_<float> &bb_gt) {
	const float in = (bb_test & bb_gt).area();
	const float un = bb_test.area() + bb_gt.area() - in;
	if (un < static_cast<float>(DBL_EPSILON)) {
		return 0.0;
	}
	return static_cast<double>(in / un);
}

cv::Rect_<float> ToRectFloat(const cv::Rect &box) {
	return cv::Rect_<float>(static_cast<float>(box.x),
													static_cast<float>(box.y),
													static_cast<float>(box.width),
													static_cast<float>(box.height));
}
} // namespace

SORT::SORT(int max_age, int min_hits, float iou_threshold)
		: max_age_(max_age), min_hits_(min_hits), frame_count_(0),
			iou_threshold_(iou_threshold) {}

std::vector<TrackingBox> SORT::update(const std::vector<YoloBoundingBox> &detections) {
	frame_count_++;

	std::vector<cv::Rect_<float>> predicted_boxes;
	predicted_boxes.reserve(trackers_.size());

	for (auto it = trackers_.begin(); it != trackers_.end();) {
		const cv::Rect_<float> pred_box = (*it).predict();
		if (pred_box.x >= 0 && pred_box.y >= 0) {
			predicted_boxes.push_back(pred_box);
			++it;
		} else {
			it = trackers_.erase(it);
		}
	}

	const size_t trk_num = predicted_boxes.size();
	const size_t det_num = detections.size();

	std::vector<int> assignment;
	std::set<int> unmatched_detections;
	std::set<int> unmatched_tracks;
	std::set<int> all_items;
	std::set<int> matched_items;
	std::vector<cv::Point> matched_pairs;

	if (trk_num > 0 && det_num > 0) {
		std::vector<std::vector<double>> iou_matrix(trk_num, std::vector<double>(det_num, 0.0));
		for (size_t i = 0; i < trk_num; ++i) {
			for (size_t j = 0; j < det_num; ++j) {
				iou_matrix[i][j] = 1.0 - GetIOU(predicted_boxes[i], ToRectFloat(detections[j].bounding_box));
			}
		}

		HungarianAlgorithm hungarian;
		hungarian.Solve(iou_matrix, assignment);

		unmatched_detections.clear();
		unmatched_tracks.clear();
		all_items.clear();
		matched_items.clear();

		if (det_num > trk_num) {
			for (size_t n = 0; n < det_num; ++n) {
				all_items.insert(static_cast<int>(n));
			}
			for (size_t i = 0; i < trk_num; ++i) {
				if (assignment[i] >= 0) {
					matched_items.insert(assignment[i]);
				}
			}
			std::set_difference(all_items.begin(), all_items.end(),
													matched_items.begin(), matched_items.end(),
													std::inserter(unmatched_detections, unmatched_detections.begin()));
		} else if (det_num < trk_num) {
			for (size_t i = 0; i < trk_num; ++i) {
				if (assignment[i] == -1) {
					unmatched_tracks.insert(static_cast<int>(i));
				}
			}
		}

		matched_pairs.clear();
		for (size_t i = 0; i < trk_num; ++i) {
			if (assignment[i] == -1) {
				continue;
			}
			if (1.0 - iou_matrix[i][assignment[i]] < iou_threshold_) {
				unmatched_tracks.insert(static_cast<int>(i));
				unmatched_detections.insert(assignment[i]);
			} else {
				matched_pairs.emplace_back(static_cast<int>(i), assignment[i]);
			}
		}

		for (const auto &pair : matched_pairs) {
			trackers_[pair.x].update(ToRectFloat(detections[pair.y].bounding_box));
		}
	} else {
		for (size_t i = 0; i < trk_num; ++i) {
			unmatched_tracks.insert(static_cast<int>(i));
		}
		for (size_t i = 0; i < det_num; ++i) {
			unmatched_detections.insert(static_cast<int>(i));
		}
	}

	for (const auto det_idx : unmatched_detections) {
		trackers_.emplace_back(ToRectFloat(detections[det_idx].bounding_box));
	}

	std::vector<TrackingBox> results;
	results.reserve(trackers_.size());
	for (auto it = trackers_.begin(); it != trackers_.end();) {
		if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits_ || frame_count_ <= min_hits_)) {
			TrackingBox res;
			res.box = (*it).get_state();
			res.id = (*it).m_id + 1;
			res.frame = frame_count_;
			results.push_back(res);
			++it;
		} else {
			++it;
		}

		if (it != trackers_.end() && (*it).m_time_since_update > max_age_) {
			it = trackers_.erase(it);
		}
	}

	return results;
}