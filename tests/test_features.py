from PIL import Image

from edcr.features import make_views, view_boxes


def test_five_view_geometry_and_order():
    boxes = view_boxes(100, 80, 0.6)
    assert boxes == [
        (0, 0, 100, 80),
        (0, 0, 60, 48),
        (40, 0, 100, 48),
        (0, 32, 60, 80),
        (40, 32, 100, 80),
    ]
    views = make_views(Image.new("RGB", (100, 80)), 0.6)
    assert [view.size for view in views] == [(100, 80)] + [(60, 48)] * 4
