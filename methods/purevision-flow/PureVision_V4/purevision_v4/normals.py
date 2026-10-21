"""Causal per-sequence normal state and explicit source-frame routing."""
import cv2
import numpy as np

from ._reference.normal_policy import constrained
from ._reference.reliable_normal import ReliableNormal as ReferenceNormal
from ._reference.visual_support import road_mask as reference_road_mask, supported_flow as reference_supported_flow


def image_texture_std(rgb):
    """Spatial luminance variation in 8-bit units, not variation across color channels."""
    return float(np.std(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY).astype(float)))


def road_mask(rgb, k, nominal, config):
    mask = reference_road_mask(rgb, k, nominal, config)
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    luma = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    if np.count_nonzero(interior) < 2 or np.std(luma[interior].astype(float)) < config.get("min_texture_std", 1.):
        return np.zeros_like(mask)
    return mask


def supported_flow(flow, confidence, source, target, config):
    try:
        return reference_supported_flow(flow, confidence, source, target, config)
    except (cv2.error, np.linalg.LinAlgError, FloatingPointError):
        return np.zeros_like(confidence), dict(region_status="numerical_failure", weighted_coverage=0.)


class NormalTracker:
    def __init__(self, k, nominal, config):
        self.k = np.asarray(k, float)
        self.nominal = np.asarray(nominal, float)
        self.config = dict(config)
        self.reset()

    def reset(self):
        self.state = ReferenceNormal(self.k, self.nominal, self.config)
        self.last_target = None
        self.last_update = None

    def step(self, flow, confidence, source_id, target_id, source_time, target_time):
        if target_time <= source_time or target_id <= source_id:
            raise ValueError("Normal input pair must be ordered")
        if self.last_target is not None and (source_id, source_time) != self.last_target:
            raise ValueError("Normal pairs must be contiguous within one sequence")
        dt = target_time - source_time
        gap = dt > self.config["max_pair_dt_seconds"]
        expired = self.last_update is not None and source_time - self.last_update > self.config["max_hold_seconds"] + 1e-9
        if gap or expired:
            self.state = ReferenceNormal(self.k, self.nominal, self.config)
            self.last_update = None
        try:
            record = self.state.step(flow, np.zeros_like(confidence) if gap else confidence)
        except (cv2.error, np.linalg.LinAlgError, FloatingPointError, ValueError):
            # Reset any partially modified candidate state; preserve only last accepted normal.
            previous, ever = self.state.previous.copy(), self.state.ever_valid
            stale = self.state.stale
            self.state = ReferenceNormal(self.k, self.nominal, self.config)
            self.state.previous, self.state.ever_valid, self.state.stale = previous, ever, stale
            record = self.state.step(np.zeros_like(flow), np.zeros_like(confidence))
            record["reason"] = "normal_numerical_failure"
        if gap:
            record["reason"] = "time_gap"
        if record["status"] == "update":
            self.last_update = target_time
        record.update(source_id=source_id, target_id=target_id, source_time_s=source_time,
                      target_time_s=target_time, normal_frame="source_camera")
        self.last_target = (target_id, target_time)
        return record


def select_normal(method, nominal, record, source_id):
    if record["source_id"] != source_id or record["normal_frame"] != "source_camera":
        raise ValueError("Source-frame normal mismatch")
    if method == "F_R":
        return np.asarray(nominal, float)
    if method == "F_RC":
        # Preserve the validated optical-frame roll constraint; pitch/height stay fixed.
        return constrained(np.asarray(record["normal"], float), np.asarray(nominal, float))
    if method == "F_B":
        return np.asarray(record["normal"], float)
    raise ValueError(f"Unknown geometry method: {method}")
