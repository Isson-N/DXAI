"""Сборка прогноза сервиса из частей: геометрия поясницы, CNN, отсутствующие веса."""
from dataclasses import dataclass

import numpy as np

from dxaqc.geometry import SpineGeometry
from dxaqc.labels import (REGION_HIP, REGION_SPINE, V_AXIS, V_FOREIGN, V_POSITIONING, V_ROI)
from dxaqc.service_model import ServiceModel, angle_to_probability


@dataclass
class FakeImage:
    columns: int = 300
    rows: int = 300
    pixels: np.ndarray = None

    def __post_init__(self):
        if self.pixels is None:
            self.pixels = np.zeros((self.rows, self.columns), dtype=np.float32)


class FakeSpine:
    def __init__(self, crest_out, angle):
        self.states = {"crest_left": "out_of_frame" if crest_out else "in_frame",
                       "crest_right": "in_frame"}
        self.geometry = SpineGeometry(angle, angle, angle, 1.0, 1.0, 100.0, 6)

    @property
    def crest_out_of_frame(self):
        return "out_of_frame" in self.states.values()


class FakeKeypoints:
    def __init__(self, crest_out=False, angle=1.0):
        self.result = FakeSpine(crest_out, angle)

    def predict(self, image):
        return self.result


class FakeCNN:
    def __init__(self, **probs):
        self.probs = probs

    def probabilities(self, image):
        return self.probs


def test_crest_out_of_frame_marks_positioning():
    model = ServiceModel(keypoints=FakeKeypoints(crest_out=True, angle=1.0))
    p = model.predict(FakeImage())
    assert p.region == REGION_SPINE
    assert p.violations[V_POSITIONING] == 1.0
    assert V_POSITIONING in p.violation_list
    assert p.quality_class == 1


def test_axis_probability_crosses_half_at_threshold():
    assert angle_to_probability(5.0) == 0.5
    assert angle_to_probability(8.0) > 0.85   # 2 ширины сигмоиды от порога
    assert angle_to_probability(1.0) < 0.1
    assert angle_to_probability(float("nan")) == 0.0


def test_axis_decided_by_angle_not_by_cnn():
    """Геометрия по оси приоритетнее CNN: у CNN на этой метке AUC 0,57."""
    model = ServiceModel(keypoints=FakeKeypoints(angle=9.0),
                         cnn=FakeCNN(region=0.1, spine_axis=0.01, spine_foreign=0.9, spine_any=0.3))
    p = model.predict(FakeImage())
    assert p.violations[V_AXIS] > 0.9 and V_AXIS in p.violation_list
    assert p.violations[V_FOREIGN] == 0.9


def test_hip_uses_cnn_heads():
    model = ServiceModel(cnn=FakeCNN(region=0.9, hip_pos=0.8, hip_roi=0.1, hip_any=0.7))
    p = model.predict(FakeImage(columns=280))
    assert p.region == REGION_HIP
    assert p.violations[V_POSITIONING] == 0.8 and p.violations[V_ROI] == 0.1
    assert p.quality_prob >= 0.8


def test_missing_weights_do_not_crash_and_are_noted():
    model = ServiceModel()
    p = model.predict(FakeImage())
    assert p.quality_class == 0 and p.violation_list == []
    assert any("размеру кадра" in note for note in model.notes)
    assert any("ключевых точек" in note for note in model.notes)
