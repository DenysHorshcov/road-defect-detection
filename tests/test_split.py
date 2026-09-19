from road_defect.data.split import assign_blocks

STEMS = [f"Japan_{i:06d}" for i in range(1000)]
RATIOS = (0.7, 0.2, 0.1)


def test_every_stem_is_assigned_exactly_once():
    result = assign_blocks(STEMS, block_size=25, ratios=RATIOS, seed=42)
    assigned = [s for stems in result.values() for s in stems]
    assert sorted(assigned) == sorted(STEMS)


def test_splits_are_disjoint():
    result = assign_blocks(STEMS, block_size=25, ratios=RATIOS, seed=42)
    train, val, test = (set(result[k]) for k in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test


def test_consecutive_frames_stay_in_the_same_split():
    """The whole point of block splitting: near-duplicate frames must not leak."""
    block_size = 25
    result = assign_blocks(STEMS, block_size=block_size, ratios=RATIOS, seed=42)
    split_of = {stem: split for split, stems in result.items() for stem in stems}

    for start in range(0, len(STEMS), block_size):
        block = STEMS[start : start + block_size]
        assert len({split_of[s] for s in block}) == 1


def test_split_sizes_roughly_match_ratios():
    result = assign_blocks(STEMS, block_size=25, ratios=RATIOS, seed=42)
    assert abs(len(result["train"]) / len(STEMS) - 0.7) < 0.05
    assert abs(len(result["val"]) / len(STEMS) - 0.2) < 0.05
    assert abs(len(result["test"]) / len(STEMS) - 0.1) < 0.05


def test_assignment_is_deterministic_for_a_given_seed():
    assert assign_blocks(STEMS, 25, RATIOS, 42) == assign_blocks(STEMS, 25, RATIOS, 42)


def test_different_seeds_produce_different_assignments():
    assert assign_blocks(STEMS, 25, RATIOS, 1) != assign_blocks(STEMS, 25, RATIOS, 2)


def test_handles_stem_count_not_divisible_by_block_size():
    stems = STEMS[:107]
    result = assign_blocks(stems, block_size=25, ratios=RATIOS, seed=42)
    assigned = [s for stems_ in result.values() for s in stems_]
    assert sorted(assigned) == sorted(stems)
