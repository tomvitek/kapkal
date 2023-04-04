
from ast import List
from dataclasses import dataclass
import numpy as np
from drops_tracker import DropTracker


@dataclass
class DropLog:
    frame_i: int
    x: np.ndarray
    y: np.ndarray


class DropsManager:
    def __init__(self, video_file: str):
        self.drop_logs: List[DropLog] = []
        self.video_file = video_file
    
    def log_drops(self, frame_i, drop_trackers: List(DropTracker)):
        x = np.array([(drop_tracker.bbox[0] if drop_tracker.active else np.nan) for drop_tracker in drop_trackers])
        y = np.array([(drop_tracker.bbox[1] if drop_tracker.active else np.nan) for drop_tracker in drop_trackers])
        self.drop_logs.append(DropLog(frame_i, x, y))
    
    def save(self, min_drop_frames: int = 10):
        total_drops = len(self.drop_logs[-1].x)

        drop_table_x = np.zeros((len(self.drop_logs), total_drops + 1))
        drop_table_x[:, 0] = np.arange(len(self.drop_logs))
        drop_table_x[:, 1:] = np.array([_ndarray_to_fixed_len(drop_log.x, total_drops) for drop_log in self.drop_logs])

        drop_table_y = np.zeros((len(self.drop_logs), total_drops + 1))
        drop_table_y[:, 0] = np.arange(len(self.drop_logs))
        drop_table_y[:, 1:] = np.array([_ndarray_to_fixed_len(drop_log.y, total_drops) for drop_log in self.drop_logs])
        
        x_not_nan_mask = np.logical_not(np.isnan(drop_table_x[:, 1:]))
        x_not_nan_sum = np.sum(x_not_nan_mask, axis=0)
        
        drops_to_delete = []
        for i in range(1, total_drops + 1):
            if x_not_nan_sum[i - 1] < min_drop_frames:
                # Schedule drop for deletion
                drops_to_delete.append(i)

        drop_table_x = np.delete(drop_table_x, drops_to_delete, axis=1)
        drop_table_y = np.delete(drop_table_y, drops_to_delete, axis=1)
        print(f"Total drops: {drop_table_x.shape[1] - 1}, filtered drops: {total_drops - (drop_table_x.shape[1] - 1)}")

        np.savetxt(self.video_file + "_x.csv", drop_table_x, delimiter=",")
        np.savetxt(self.video_file + "_y.csv", drop_table_y, delimiter=",")
    
        
def _ndarray_to_fixed_len(arr: np.ndarray, length: int, fill_value: int = np.nan):
    if len(arr) >= length:
        return arr[:length]
    else:
        return np.concatenate((arr, np.full(length - len(arr), fill_value)))
    