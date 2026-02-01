"""
ByteTrack Implementation for Face Tracking

Simplified ByteTrack for tracking faces across video frames.
More robust than simple IoU tracking.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes.
    State: [x, y, w, h, dx, dy, dw, dh]
    """
    count = 0
    
    def __init__(self, bbox):
        """Initialize with [x1, y1, x2, y2]"""
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ])
        
        # Measurement uncertainty
        self.kf.R[2:,2:] *= 10.
        
        # Process uncertainty
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # Convert bbox to [x, y, w, h]
        self.kf.x[:4] = self._convert_bbox(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """Update tracker with new detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox(bbox))
        
    def predict(self):
        """Predict next state."""
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] *= 0.0
        if self.kf.x[3] + self.kf.x[7] <= 0:
            self.kf.x[7] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._convert_x_to_bbox(self.kf.x)
        
    def get_state(self):
        """Return current bbox estimate."""
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox(bbox):
        """Convert [x1,y1,x2,y2] to [x,y,w,h]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        return np.array([x, y, w, h]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x):
        """Convert [x,y,w,h] to [x1,y1,x2,y2]"""
        w = x[2]
        h = x[3]
        return np.array([
            x[0] - w/2.,
            x[1] - h/2.,
            x[0] + w/2.,
            x[1] + h/2.
        ]).flatten()


def iou_batch(bboxes1, bboxes2):
    """
    Compute IoU between two sets of bounding boxes (vectorized for speed).
    
    Args:
        bboxes1: Nx4 array [x1, y1, x2, y2]
        bboxes2: Mx4 array [x1, y1, x2, y2]
    Returns:
        NxM IoU matrix
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)
    
    bboxes1 = np.expand_dims(bboxes1, 1)
    bboxes2 = np.expand_dims(bboxes2, 0)
    
    # Vectorized IoU computation
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    
    intersection = w * h
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - intersection
    
    iou = intersection / np.maximum(union, 1e-6)
    
    return iou


class ByteTrack:
    """
    ByteTrack for multi-object tracking (optimized for single face tracking).
    Uses Kalman filter + Hungarian matching for robust temporal tracking.
    
    Lightweight CPU implementation - tracking overhead is minimal compared to GPU inference.
    """
    
    def __init__(self, track_thresh=0.5, match_thresh=0.8, max_age=30):
        """
        Args:
            track_thresh: Detection confidence threshold (default: 0.5)
            match_thresh: IoU threshold for matching tracks (default: 0.8, high for faces)
            max_age: Maximum frames to keep lost tracks (default: 30 frames)
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections):
        """
        Update trackers with new detections.
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, score]
            
        Returns:
            List of active tracks: [(x1, y1, x2, y2, track_id), ...]
        """
        self.frame_count += 1
        
        # Get predictions from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Separate high and low confidence detections
        if len(detections) > 0:
            high_dets = detections[detections[:, 4] >= self.track_thresh]
            low_dets = detections[detections[:, 4] < self.track_thresh]
        else:
            high_dets = np.empty((0, 5))
            low_dets = np.empty((0, 5))
        
        # Match high confidence detections
        matched, unmatched_dets, unmatched_trks = self._associate(
            high_dets, trks, self.match_thresh
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(high_dets[m[0], :4])
        
        # Create new trackers for unmatched high confidence detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(high_dets[i, :4])
            self.trackers.append(trk)
        
        # Remove dead tracklets
        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()
            if trk.time_since_update < self.max_age and trk.hits >= 1:
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def _associate(self, detections, trackers, iou_threshold):
        """
        Associate detections to tracked objects using IoU.
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
        
        if min(iou_matrix.shape) > 0:
            # Hungarian algorithm for optimal assignment
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
