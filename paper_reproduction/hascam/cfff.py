"""Two-stage cosine selection and completion memory (paper Algorithm 1).

The algorithm is for sequential visual enhancement. Its appearance memory is
not an oracle instance identity; visually identical cars can collapse together.
"""
from collections import deque
import numpy as np
from .detection import Detection


def unit(feature):
    f = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(f))
    if not np.isfinite(f).all() or norm <= 1e-12:
        raise ValueError("Appearance extractor returned a non-finite/zero vector")
    return f / norm


def cosine(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Feature dimension changed: {a.shape} versus {b.shape}")
    return float(np.clip(np.dot(unit(a), unit(b)), -1, 1))


class CFFF:
    """extractor(rgb, detection) must return an actually computed feature vector.

    .525 and .85 are author-source constants, L=100 is a recorded assumption
    because neither paper nor bundled source provides a usable memory limit.
    """
    def __init__(self, extractor, threshold=.525, current_weight=.85, max_memory=100,
                 confidence_threshold=.9, consecutive_steps=10):
        if not -1 <= threshold <= 1 or not 0 < current_weight <= 1:
            raise ValueError("Invalid CFFF cosine threshold/fusion weight")
        if max_memory < 1 or consecutive_steps < 1:
            raise ValueError("CFFF memory and confidence streak must be positive")
        self.extractor = extractor
        self.threshold = float(threshold)
        self.current_weight = float(current_weight)
        self.max_memory = int(max_memory)
        self.confidence_threshold = float(confidence_threshold)
        self.consecutive_steps = int(consecutive_steps)
        self.reset()

    def reset(self):
        self.monitored = deque(maxlen=self.max_memory)
        self.current_feature = None
        self.current_features = []
        self.confidence_streak = 0
        self.completed_count = 0
        self.feature_dim = None
        self.last_event = {"completed": False, "completion_count": 0,
                           "completed_detection": None, "selected": False,
                           "similarity": None, "new_target": False, "acquired": False}

    def choose(self, rgb, detections: list[Detection], count_step=True):
        # Sorting by descending detector confidence is Algorithm 1 line 3.
        candidates = sorted(detections, key=lambda d: d.confidence, reverse=True)
        chosen = None
        similarity = None
        chosen_feature = None
        self.last_event = {"completed": False, "completion_count": self.completed_count,
                           "completed_detection": None, "selected": False,
                           "similarity": None, "new_target": False, "acquired": False}
        for det in candidates:
            f = unit(self.extractor(rgb, det))
            if self.feature_dim is None:
                self.feature_dim = len(f)
            if len(f) != self.feature_dim:
                raise ValueError("Extractor changed appearance feature dimension within episode")
            similarity = None if self.current_feature is None else cosine(f, self.current_feature)
            if similarity is not None and similarity < self.threshold:
                continue
            # Second stage excludes appearances whose enhancement completed.
            if any(cosine(f, monitored) >= self.threshold for monitored in self.monitored):
                self.current_feature = None
                self.current_features = []
                self.confidence_streak = 0
                continue
            chosen, chosen_feature = det, f
            break
        if chosen is None:
            # Author environment resets a lost active reference. Completed
            # memory persists, allowing subsequent unobserved-target selection.
            self.current_feature = None
            self.current_features = []
            self.confidence_streak = 0
            return None
        acquired = self.current_feature is None
        if acquired:
            self.current_feature = chosen_feature.copy()
        else:
            w = self.current_weight
            self.current_feature = w * chosen_feature + (1-w) * self.current_feature
        self.current_features.append(chosen_feature.copy())
        if count_step:
            self.confidence_streak = self.confidence_streak + 1 if chosen.confidence >= self.confidence_threshold else 0
        self.last_event.update(selected=True, similarity=similarity, new_target=acquired, acquired=acquired)
        if self.confidence_streak >= self.consecutive_steps:
            # Equal-weight temporal fusion is the actual author implementation.
            # Paper calls it weighted but gives no coefficients.
            self.monitored.append(unit(np.mean(self.current_features, axis=0)))
            self.completed_count += 1
            self.last_event.update(completed=True, completion_count=self.completed_count,
                                   completed_detection=chosen)
            self.current_feature = None
            self.current_features = []
            self.confidence_streak = 0
        return chosen

    def metadata(self):
        return {"task": "sequential_visual_enhancement", "threshold": self.threshold,
                "current_weight": self.current_weight, "completion_fusion": "arithmetic_mean",
                "max_memory": self.max_memory, "confidence_threshold": self.confidence_threshold,
                "consecutive_steps": self.consecutive_steps, "feature_dim": self.feature_dim,
                "extractor": getattr(self.extractor, "metadata", lambda: {"name": type(self.extractor).__name__})()}
