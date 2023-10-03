# IoU tells about goodness of a predicted bounding box

import torch


def iou(pred_bboxes, true_bboxes, box_format):
    """
    Calculates IoU metric

    :param pred_bboxes: Predictions of bounding boxes (BATCH_SIZE, 4)
    :type pred_bboxes: tensor
    :param true_bboxes: True bounding boxes (BATCH_SIZE, 4)
    :type true_bboxes: tensor
    :param box_format: midpoint (x, y, w, h) / corners (x1, y1, x2, y2)
    :type box_format: str
    :return: tensor: IoU over all examples (BATCH_SIZE, 1)
    """
    box1_x1, box1_y1, box1_x2, box1_y2, box2_x1, box2_y1, box2_x2, box2_y2 = [0.]*8
    if box_format == "midpoint":  # (x, y, w, h)
        # slicing the lists to keep the dimensionality
        # ... deals with dimensions before the coordinates (YOLO has (bsize, s, s, 4))
        # predicted box
        #                  x_mid                  w/2
        box1_x1 = pred_bboxes[..., 0:1] - (pred_bboxes[..., 2:3]/2)
        box1_y1 = pred_bboxes[..., 1:2] - (pred_bboxes[..., 3:4]/2)
        box1_x2 = pred_bboxes[..., 0:1] + (pred_bboxes[..., 2:3]/2)
        box1_y2 = pred_bboxes[..., 1:2] + (pred_bboxes[..., 3:4]/2)
        # true box
        box2_x1 = true_bboxes[..., 0:1] - (true_bboxes[..., 2:3] / 2)
        box2_y1 = true_bboxes[..., 1:2] - (true_bboxes[..., 3:4] / 2)
        box2_x2 = true_bboxes[..., 0:1] + (true_bboxes[..., 2:3] / 2)
        box2_y2 = true_bboxes[..., 1:2] + (true_bboxes[..., 3:4] / 2)
    elif box_format == "corners":  # (x1, y1, x2, y2)
        # predicted box
        box1_x1 = pred_bboxes[..., 0:1]
        box1_y1 = pred_bboxes[..., 1:2]
        box1_x2 = pred_bboxes[..., 2:3]
        box1_y2 = pred_bboxes[..., 3:4]
        # true box
        box2_x1 = true_bboxes[..., 0:1]
        box2_y1 = true_bboxes[..., 1:2]
        box2_x2 = true_bboxes[..., 2:3]
        box2_y2 = true_bboxes[..., 3:4]

    # finding coordinates of intersection box
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)  # .clamp(0) deals with no intersection case
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection + 1e-8

    return intersection/union



