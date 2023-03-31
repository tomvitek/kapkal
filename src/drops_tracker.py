from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from drop_detector import find_drops

DROP_PX_MARGIN = 10
DROP_MAX_FRAME_MOVEMENT_PX = np.array((5, 20))
MAX_ACTIVE_DROPS = 50
TRACKER_MIN_STD = 5
TRACKER_MIN_MAXIMUM = 50

@dataclass
class DropTracker:
    bbox: Tuple[int, int, int, int]
    init_frame: int
    active: bool

    def track(self, frame: cv2.UMat):
        # Find center of brightness in currenty bbox
        bbox_img = self.get_img(frame)
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2GRAY)
        
        # Find center of brightness
        x_sum = np.sum(bbox_img, axis=0)
        y_sum = np.sum(bbox_img, axis=1)
        x_avg = np.average(np.arange(len(x_sum)), weights=x_sum)
        y_avg = np.average(np.arange(len(y_sum)), weights=y_sum)
        
        # Update bbox
        self.bbox[0] = int(self.bbox[0] + x_avg - self.bbox[2]/2)
        self.bbox[1] = int(self.bbox[1] + y_avg - self.bbox[3]/2)
        self.bbox = limit_bbox_to_frame(self.bbox, frame.shape[:2])

        # Check that drop is still visible
        bbox_img = self.get_img(frame)
        drop_std = cv2.meanStdDev(bbox_img)[1][0][0]
        drop_max = bbox_img.max()
        if drop_std < TRACKER_MIN_STD or drop_max < TRACKER_MIN_MAXIMUM:
            self.active = False

    def get_img(self, frame: cv2.UMat):
        return frame[self.bbox[1]:self.bbox[1]+self.bbox[3], self.bbox[0]:self.bbox[0]+self.bbox[2]]

class DropsTracker:
    def __init__(self, init_frame, drops, thresh_min, thresh_max, vid_resolution: Tuple[int, int]) -> None:
        self.frame_i = 0
        self.trackers: List[DropTracker] = []

        for drop in drops:
            drop_bbox = self.detected_drop_to_bbox(drop, vid_resolution)
            drop_tracker = DropTracker(drop_bbox, 0, True)
            self.trackers.append(drop_tracker)

        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        

    def track(self, frame: cv2.UMat):
        self.frame_i += 1   

        # Find new drops
        detected_drops = find_drops(frame, self.thresh_min, self.thresh_max)
        if len(detected_drops) > MAX_ACTIVE_DROPS:
            # Bad frame, disable all trackers and skip
            for tracker in self.trackers:
                tracker.active = False
            return
        
        # Update trackers. This also removes trackers that no longer see their drops
        for tracker in self.get_active_trackers():
            tracker.track(frame)

        for tracker in self.get_active_trackers():
            closest_drop, closest_pos_delta = self._find_nearest_drop(tracker, detected_drops)

            if np.all(np.abs(closest_pos_delta) < DROP_MAX_FRAME_MOVEMENT_PX):
                # If the closest drop is close enough, update tracker's bbox
                tracker.bbox = self.detected_drop_to_bbox(closest_drop, frame.shape)
                detected_drops.pop(detected_drops.index(closest_drop))
    
        
        # Add new drops
        for detected_drop in detected_drops:
            drop_bbox = self.detected_drop_to_bbox(detected_drop, frame.shape)
            drop_tracker = DropTracker(drop_bbox, self.frame_i, True)
            self.trackers.append(drop_tracker)


                    
    def get_drop_tracker(self, drop: Tuple[int, int, int, int]):
        drop_center = (drop[0] + drop[2] / 2, drop[1] + drop[3] / 2)
        for drop_tracker in self.trackers:
            tracked_bbox = drop_tracker.bbox
            tracked_drop_center = (tracked_bbox[0] + tracked_bbox[2] / 2, tracked_bbox[1] + tracked_bbox[3] / 2)
            drop_diff = (tracked_drop_center[0] - drop_center[0], tracked_drop_center[1] - drop_center[1])
            drop_r = np.sqrt(drop_diff[0] ** 2 + drop_diff[1] ** 2)
            if (drop_r < 20):
                return drop_tracker
        return None

    def detected_drop_to_bbox(self, drop: Tuple[int, int, int, int], frame_size: Tuple[int, int]):
        bbox = [drop[0] - DROP_PX_MARGIN, drop[1] - DROP_PX_MARGIN, drop[2] + DROP_PX_MARGIN * 2, drop[3] + DROP_PX_MARGIN * 2]
        bbox = limit_bbox_to_frame(bbox, frame_size)
        return bbox

    def get_active_trackers(self):
        return [drop_tracker for drop_tracker in self.trackers if drop_tracker.active]
    
    def _distance(self, drop1_bbox: Tuple[int, int, int, int], drop2_bbox: Tuple[int, int, int, int]):
        drop1_center = np.array((drop1_bbox[0] + drop1_bbox[2] / 2, drop1_bbox[1] + drop1_bbox[3] / 2))
        drop2_center = np.array((drop2_bbox[0] + drop2_bbox[2] / 2, drop2_bbox[1] + drop2_bbox[3] / 2))
        drop_delta = drop1_center - drop2_center
        drop_r = np.sqrt(drop_delta[0] ** 2 + drop_delta[1] ** 2)
        return drop_r, drop_delta
    
    def _find_nearest_drop(self, tracker: DropTracker, detected_drops: List[Tuple[int, int, int, int]]):
        closest_drop = None
        closest_dist_score = 999999999
        closest_pos_delta = (999999999, 999999999)
        for detected_drop in detected_drops:
            dist, pos_delta = self._distance(detected_drop, tracker.bbox)
            dist_score = dist #np.sqrt((pos_delta[0] * 2)**2 + pos_delta[1]**2)
            if dist_score < closest_dist_score:
                closest_drop = detected_drop
                closest_dist_score = dist_score
                closest_pos_delta = pos_delta
        
        return closest_drop, closest_pos_delta

def limit_bbox_to_frame(bbox: List[int], frame_size: Tuple[int, int]):
        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[0] + bbox[2] > frame_size[1]:
            bbox[2] = frame_size[1] - bbox[0]
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[1] + bbox[3] > frame_size[0]:
            bbox[3] = frame_size[0] - bbox[1]
        return bbox

