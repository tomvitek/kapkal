from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import lmfit


@dataclass
class DropsData:
    frames: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vy: np.ndarray

    def copy(self):
        return DropsData(
            np.copy(self.frames),
            np.copy(self.x),
            np.copy(self.y),
            np.copy(self.vy)
        )


@dataclass
class DropResult:
    vy1: float
    vy1_std: float
    intercept1: float
    vy2: float
    vy2_std: float
    intercept2: float


class DropsFilter:
    def __init__(self, 
                 vy_std_threshold: float, 
                 x_std_threshold: float,
                 min_fit_frames: int = 5,
                 frames_from_inversion_offset: int = 2
                 ):
        self.vy_std_threshold = vy_std_threshold
        self.x_std_threshold = x_std_threshold
        self.min_fit_frames = min_fit_frames
        self.frames_from_inversion_offset = frames_from_inversion_offset
        self.reset_counters()
    
    def reset_counters(self):
        self.filtered_x_std_count = 0
        self.filtered_min_fit_frames_count = 0


    def apply_x_std_filter(self, drops_data: DropsData, std_threshold: float = 15) -> DropsData:
        """Filter drops with big x deviation"""
        for i in range(1, drops_data.x.shape[1]):
            not_nan_mask = np.logical_not(np.isnan(drops_data.x[:, i]))
            if not_nan_mask.max() == 0:
                continue
            std = np.std(drops_data.x[not_nan_mask, i])
            if std > std_threshold:
                drops_data.x[:, i] = np.nan
                drops_data.y[:, i] = np.nan
                drops_data.vy[:, i] = np.nan
        return drops_data

    def smooth_y(self, drops_data: DropsData, smooth_factor: int = 7) -> DropsData:
        """Smooth drops y"""
        smoothed_data = drops_data.copy()
        for i in range(1, drops_data.y.shape[1]):
            smoothed_data.y[:, i] = np.convolve(drops_data.y[:, i], np.ones(
                smooth_factor) / smooth_factor, mode="same")
        smoothed_data.vy = np.diff(smoothed_data.y, axis=0)
        return smoothed_data

    def find_drops_inversion_frames(self, drops_data: DropsData, diff_frame_count=5, min_frame=20) -> List[Tuple[int, DropsData]]:
        """Find voltage inversion frame and suitable drops"""
        MIN_SPEED_DELTA = 20
        MIN_INVERTED_COUNT = 2
        # find voltage inversion frame and suitable drops
        min_frame = diff_frame_count * 2 if min_frame < diff_frame_count * 2 else min_frame

        inversion_frames_list = []

        for i in range(min_frame, drops_data.vy.shape[0] - diff_frame_count):
            not_nan_mask = np.logical_not(np.isnan(drops_data.vy[i, :]))
            not_nan_mask_prev = np.logical_not(
                np.isnan(drops_data.vy[i - diff_frame_count, :]))
            not_nan_mask = np.logical_and(not_nan_mask, not_nan_mask_prev)

            if np.sum(not_nan_mask) == 0:
                continue

            speed_diffs = np.abs(
                drops_data.vy[i, not_nan_mask] - drops_data.vy[i - diff_frame_count, not_nan_mask])
            speed_diffs_limit_mask = speed_diffs > 3
            inverted_mask = np.any(
                [
                    (np.all([
                        (drops_data.vy[i, not_nan_mask] < 0),
                        (drops_data.vy[i - diff_frame_count,
                         not_nan_mask] > 0),
                        speed_diffs_limit_mask
                    ], axis=0)),
                    (np.all([
                        (drops_data.vy[i, not_nan_mask] > 0),
                        (drops_data.vy[i - diff_frame_count,
                         not_nan_mask] < 0),
                        speed_diffs_limit_mask
                    ], axis=0)),
                ], axis=0
            )
            inverted_count = np.sum(inverted_mask)
            speed_delta = np.abs(
                drops_data.vy[i, not_nan_mask][inverted_mask] - drops_data.vy[i - diff_frame_count, not_nan_mask][inverted_mask])
            
            if np.sum(speed_delta) >= MIN_SPEED_DELTA and inverted_count >= MIN_INVERTED_COUNT:
                inversion_frame = i - int(diff_frame_count / 2)
                filtered_drops_data = DropsData(
                    frames=drops_data.frames,
                    x=drops_data.x[:, not_nan_mask][:, inverted_mask],
                    y=drops_data.y[:, not_nan_mask][:, inverted_mask],
                    vy=drops_data.vy[:, not_nan_mask][:, inverted_mask]
                )
                inversion_frames_list.append((inversion_frame, filtered_drops_data))

            # Remove inversion frames next to each other
            if len(inversion_frames_list) > 1:
                for j in range(len(inversion_frames_list) - 1):
                    if inversion_frames_list[j + 1][0] - inversion_frames_list[j][0] < 5:
                        if len(inversion_frames_list[j][1].vy) > len(inversion_frames_list[j + 1][1].vy):
                            inversion_frames_list.pop(j + 1)
                        else:
                            inversion_frames_list.pop(j)
                        
                        j -= 1

        return inversion_frames_list

    def analyze_drops(self,
                      filtered_drops_data: DropsData, 
                      inversion_frame: int
                      ) -> List[DropResult]:
        """Analyze drops"""
        drops_results = []
        for i in range(filtered_drops_data.y.shape[1]):
            drop_result = self.analyze_drop(
                filtered_drops_data.frames, 
                filtered_drops_data.x[:, i],
                filtered_drops_data.y[:, i], 
                inversion_frame)
            
            if drop_result is not None:
                drops_results.append(drop_result)
        return drops_results

    def analyze_drop(self, 
                     frames: np.ndarray, 
                     x: np.ndarray,
                     y: np.ndarray,
                     inversion_frame: int
                     ) -> DropResult:
        """Fits drop to two lines and returns its velocity before and after voltage inversion"""

        interval1 = np.arange(inversion_frame - self.frames_from_inversion_offset, -1, -1)
        interval2 = np.arange(inversion_frame + self.frames_from_inversion_offset, frames.shape[0], 1)
        y1 = y[interval1]
        y2 = y[interval2]
        x1 = x[interval1]
        x2 = x[interval2]

        not_nan_mask = np.logical_not(np.isnan(y1))
        x1 = x1[not_nan_mask]
        y1 = y1[not_nan_mask]
        frames1 = frames[interval1][not_nan_mask]

        not_nan_mask = np.logical_not(np.isnan(y2))
        x2 = x2[not_nan_mask]
        y2 = y2[not_nan_mask]
        frames2 = frames[interval2][not_nan_mask]
        # Fit first part of y, before inversion.

        y1_fit_params = self.fit_y_near_inversion(frames1, x1, y1)
        y2_fit_params = self.fit_y_near_inversion(frames2, x2, y2)

        if y1_fit_params is None or y2_fit_params is None:
            return None

        # If slopes have the same sign, then drop didn't invert its velocity
        if y1_fit_params['slope'].value * y2_fit_params['slope'].value > 0:
            return None

        return DropResult(
            vy1=y1_fit_params['slope'].value,
            vy2=y2_fit_params['slope'].value,
            vy1_std=y1_fit_params['slope'].stderr,
            vy2_std=y2_fit_params['slope'].stderr,
            intercept1=y1_fit_params['intercept'].value,
            intercept2=y2_fit_params['intercept'].value
        )

    def fit_y_near_inversion(self,
                             frames: np.ndarray,
                             x: np.ndarray,
                             y: np.ndarray
                             ) -> lmfit.Parameters:
        # Find the part of y that is straight (fit can be performed within std limits) using binary search
        if len(frames) < self.min_fit_frames:
            self.filtered_min_fit_frames_count += 1
            return None
        
        left = 0
        right = len(frames) - 1
        fit_params = None
        while left + 3 < right:
            if right < self.min_fit_frames:
                self.filtered_min_fit_frames_count += 1
                return None
            middle = (left + right) // 2
            fit_params = self.fit_y(frames[:middle], y[:middle])

            # Sometimes fit fails for unknown reason, it is rare so we just ignore it
            if fit_params is None or fit_params['slope'].stderr is None:
                return None
            if fit_params['slope'].stderr > self.vy_std_threshold:
                right = middle
            else:
                left = middle + 1

        if np.std(x[:right]) > self.x_std_threshold:
            self.filtered_x_std_count += 1
            return None
        
        shortening = 0.8

        if int(right * shortening) < len(frames):
            fit_params = self.fit_y(frames[:int(right * shortening)], y[:int(right * shortening)])
        else:
            return None

        return fit_params

    def fit_y(self, frames: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
        # Returns linear fit params
        model = lmfit.models.LinearModel()
        params = model.guess(y, x=frames)
        fit = model.fit(y, params, x=frames)
        return fit.params
