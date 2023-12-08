def bbox_iou_ymin_xmin_ymax_xmax(bbox1, bbox2):
    ymin1, xmin1, ymax1, xmax1 = bbox1
    ymin2, xmin2, ymax2, xmax2 = bbox2
    return compute_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)

def bbox_iou_xmin_ymin_xmax_ymax(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    return compute_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)

def bbox_iou_cx_cy_w_h(bbox1, bbox2):
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2
    xmin1, ymin1, xmax1, ymax1 = cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2
    xmin2, ymin2, xmax2, ymax2 = cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2
    return compute_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)


def compute_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xi1 = max(xmin1, xmin2)
    yi1 = max(ymin1, ymin2)
    xi2 = min(xmax1, xmax2)
    yi2 = min(ymax1, ymax2)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    bbox1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    bbox2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou
