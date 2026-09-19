import math

import numpy as np

from dxaqc.geometry import AXIS_THRESHOLD_DEG, crest_rule, spine_geometry, to_millimetres


def test_vertical_column_has_zero_angle():
    points = [(100, 20 + 30 * i) for i in range(6)]
    g = spine_geometry(points)
    assert g.n_points == 6
    assert g.angle_chord < 1e-6 and g.angle_ls < 1e-6
    assert g.max_dev_chord < 1e-6
    assert math.isclose(g.curvature, 0.0, abs_tol=1e-6)


def test_tilted_column_angle_matches_geometry():
    # Наклон ровно 10° в миллиметрах: сдвиг по x = tan(10°) · длина по y.
    dy_mm = 30 * 0.606
    dx_mm = math.tan(math.radians(10)) * dy_mm
    points = [(100 + i * dx_mm / 0.600, 20 + i * 30) for i in range(6)]
    g = spine_geometry(points)
    assert math.isclose(g.angle_chord, 10.0, abs_tol=0.05)
    assert math.isclose(g.angle_ls, 10.0, abs_tol=0.05)
    assert g.angle_chord > AXIS_THRESHOLD_DEG


def test_arc_gives_curvature_and_small_chord_angle():
    """Сколиоз: точки на дуге, концы по вертикали. Хорда почти вертикальна, кривизна большая."""
    ys = np.linspace(0, 150, 6)
    xs = 100 + 12 * np.sin(np.pi * ys / 150)
    g = spine_geometry(list(zip(xs, ys)))
    assert g.angle_chord < 1.0
    assert g.curvature > 5.0
    assert g.max_dev_chord > 5.0


def test_points_are_ordered_top_to_bottom():
    points = [(100, 90), (100, 30), (100, 60)]
    ordered = to_millimetres(points)
    assert list(ordered[:, 1]) == sorted(ordered[:, 1])


def test_too_few_points_is_not_an_error():
    g = spine_geometry([(10, 10)])
    assert g.n_points == 1 and math.isnan(g.angle_chord)


def test_crest_rule_counts_only_out_of_frame():
    assert crest_rule("out_of_frame", "in_frame")
    assert crest_rule("in_frame", "out_of_frame")
    assert not crest_rule("partial", "in_frame")
    assert not crest_rule("in_frame", "in_frame")
