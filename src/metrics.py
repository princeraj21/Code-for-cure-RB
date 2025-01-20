def compute_iou(box_a, box_b):
    # if boxes don't intersect
    if _boxes_intersect(box_a, box_b) is False:
        return 0
    inter_area = _get_intersection_area(box_a, box_b)
    union = _get_union_areas(box_a, box_b, inter_area=inter_area)
    iou = inter_area / union  # intersection over union
    assert iou >= 0
    return iou


def _boxes_intersect(box_a, box_b):
    if box_a[0] > box_b[2]:
        return False  # box_a is right of box_b
    if box_b[0] > box_a[2]:
        return False  # box_a is left of box_b
    if box_a[3] < box_b[1]:
        return False  # box_a is above box_b
    if box_a[1] > box_b[3]:
        return False  # box_a is below box_b
    return True


def _get_intersection_area(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)


def _get_union_areas(box_a, box_b, inter_area=None):
    area_a = _get_area(box_a)
    area_b = _get_area(box_b)
    if inter_area is None:
        inter_area = _get_intersection_area(box_a, box_b)
    return float(area_a + area_b - inter_area)


def _get_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

