import numpy as np

from edcr.interventions import (
    decode_union,
    degrade_target,
    dilate_mask,
    inpaint_rgb,
    stable_rank,
)


def test_polygon_masks_are_decoded_and_unioned():
    annotations = [
        {"segmentation": [[0, 0, 0, 2, 2, 2, 2, 0]]},
        {"segmentation": [[3, 3, 3, 5, 5, 5, 5, 3]]},
    ]
    mask = decode_union(annotations, 6, 6)
    assert mask.dtype == np.bool_
    assert mask.any()
    assert mask[0:2, 0:2].any()
    assert mask[3:5, 3:5].any()


def test_candidate_ranking_is_stable():
    assert stable_rank(1, "a_to_b", 10) == stable_rank(1, "a_to_b", 10)
    assert stable_rank(1, "a_to_b", 10) != stable_rank(1, "a_to_b", 11)


def test_dilation_and_degradation_only_change_masked_region():
    image = np.full((40, 40, 3), 127, dtype=np.uint8)
    image[10:30, 10:30] = 240
    mask = np.zeros((40, 40), dtype=bool)
    mask[15:25, 15:25] = True
    dilated = dilate_mask(mask, 2)
    assert dilated.sum() > mask.sum()
    changed = degrade_target(image, mask, "contrast", "medium")
    assert np.array_equal(changed[~mask], image[~mask])
    assert not np.array_equal(changed[mask], image[mask])


def test_inpaint_shape_and_outside_mask():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[8:12, 8:12] = 255
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    result = inpaint_rgb(image, mask)
    assert result.shape == image.shape
    assert np.array_equal(result[~mask], image[~mask])
    assert inpaint_rgb(image, mask, method="ns").shape == image.shape
