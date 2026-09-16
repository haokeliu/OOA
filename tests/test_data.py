from edcr.data import (
    deterministic_split,
    image_labels,
    mapping_from_categories,
    maximum_noncrowd_area_ratios,
    maximum_noncrowd_areas,
    remove_instances,
    select_relations,
)


def test_non_contiguous_category_mapping_and_multi_instance_removal():
    mapping = mapping_from_categories([{"id": 9, "name": "b"}, {"id": 3, "name": "a"}])
    annotations = [
        {"id": 1, "image_id": 10, "category_id": 3},
        {"id": 2, "image_id": 10, "category_id": 3},
        {"id": 3, "image_id": 10, "category_id": 9},
    ]
    assert mapping["original_to_contiguous"] == {"3": 0, "9": 1}
    one_removed = [ann for ann in annotations if ann["id"] != 1]
    assert image_labels(one_removed, mapping)[10] == [0, 1]
    all_target_removed = remove_instances(annotations, image_id=10, category_id=3)
    assert image_labels(all_target_removed, mapping)[10] == [1]


def test_split_is_deterministic_complete_and_disjoint():
    image_ids = list(range(101))
    first = deterministic_split(image_ids, seed=20260909, proportions=[0.7, 0.15, 0.075, 0.075])
    second = deterministic_split(
        reversed(image_ids), seed=20260909, proportions=[0.7, 0.15, 0.075, 0.075]
    )
    assert first == second
    assert set(first) == set(image_ids)
    groups = [
        {key for key, value in first.items() if value == name} for name in set(first.values())
    ]
    assert sum(len(group) for group in groups) == len(set().union(*groups))
    assert sorted(map(len, groups)) == [7, 8, 15, 71]


def test_relation_selection_removes_reciprocals_and_prefers_target_support():
    rows = [
        {"source_id": 0, "target_id": 1, "n10": 10, "smoothed_lift": 9.0},
        {"source_id": 1, "target_id": 0, "n10": 20, "smoothed_lift": 9.0},
        {"source_id": 2, "target_id": 1, "n10": 30, "smoothed_lift": 8.0},
        {"source_id": 1, "target_id": 2, "n10": 40, "smoothed_lift": 8.0},
    ]
    selected = select_relations(rows, ["a", "b", "c"], pair_count=2)
    assert [(row["source_id"], row["target_id"]) for row in selected] == [(1, 0), (1, 2)]


def test_maximum_noncrowd_area_uses_largest_instance_and_skips_crowds():
    mapping = mapping_from_categories([{"id": 3, "name": "a"}])
    annotations = [
        {"image_id": 10, "category_id": 3, "area": 20, "iscrowd": 0},
        {"image_id": 10, "category_id": 3, "area": 30, "iscrowd": 0},
        {"image_id": 10, "category_id": 3, "area": 100, "iscrowd": 1},
    ]
    assert maximum_noncrowd_areas(annotations, mapping) == {(10, 0): 30.0}
    ratios = maximum_noncrowd_area_ratios(
        annotations, [{"id": 10, "width": 10, "height": 20}], mapping
    )
    assert ratios == {(10, 0): 0.15}
