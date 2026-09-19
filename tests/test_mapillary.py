import math

import pytest

from road_defect.data.mapillary import MAX_TILE_DEG, iter_tiles


def test_tiles_cover_the_area_without_exceeding_it():
    tiles = iter_tiles(30.50, 50.43, 30.56, 50.47, 0.009)
    assert min(t[0] for t in tiles) == pytest.approx(30.50)
    assert min(t[1] for t in tiles) == pytest.approx(50.43)
    assert max(t[2] for t in tiles) == pytest.approx(30.56)
    assert max(t[3] for t in tiles) == pytest.approx(50.47)


def test_every_tile_is_within_the_api_size_limit():
    for lon0, lat0, lon1, lat1 in iter_tiles(30.50, 50.43, 30.56, 50.47, 0.009):
        assert lon1 - lon0 < MAX_TILE_DEG
        assert lat1 - lat0 < MAX_TILE_DEG


def test_tile_count_matches_the_grid():
    tiles = iter_tiles(0.0, 0.0, 0.05, 0.02, 0.01 - 1e-9)
    assert len(tiles) == math.ceil(0.05 / (0.01 - 1e-9)) * math.ceil(0.02 / (0.01 - 1e-9))


def test_partial_edge_tiles_are_clipped_to_the_area():
    # 0.025 wide with 0.009 tiles leaves a 0.007 remainder on the last column.
    tiles = iter_tiles(0.0, 0.0, 0.025, 0.009, 0.009)
    widths = sorted({round(t[2] - t[0], 6) for t in tiles})
    assert widths == [0.007, 0.009]


def test_tiles_do_not_overlap():
    tiles = iter_tiles(0.0, 0.0, 0.027, 0.009, 0.009)
    spans = sorted((t[0], t[2]) for t in tiles)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert next_start >= prev_end - 1e-9


def test_rejects_tile_size_at_or_above_api_limit():
    with pytest.raises(ValueError):
        iter_tiles(0.0, 0.0, 0.05, 0.05, MAX_TILE_DEG)


def test_rejects_non_positive_tile_size():
    with pytest.raises(ValueError):
        iter_tiles(0.0, 0.0, 0.05, 0.05, 0.0)


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        iter_tiles(30.56, 50.43, 30.50, 50.47, 0.009)


def test_single_tile_area():
    assert len(iter_tiles(0.0, 0.0, 0.005, 0.005, 0.009)) == 1
