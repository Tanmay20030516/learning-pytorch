# For each class:
#   first drop all boxes with probability score below a threshold
#   from the remaining boxes, select the box with the highest probability score
#   calculate IoU of that box with all other boxes, and drop the other boxes that have IoU greater than a threshold

import torch
from metric_intersection_over_union import iou


def nms(bboxes: list, iou_threshold: float, prob_threshold: float, box_format: str):
    """
    Performs Non Max Suppression
    :param bboxes: list of lists containing all bboxes with each bboxes specified as [class_pred, prob_score, x1, y1, x2, y2]
    :param iou_threshold: threshold where predicted bboxes is correct
    :param prob_threshold: threshold to remove predicted bboxes (independent of IoU)
    :param box_format: "midpoint" or "corners" format used to specify bboxes
    :return: list: bboxes after performing NMS given a specific IoU threshold
    """

    # bboxes -> [[class, probability score, x1, y1, x2, y2], ..., ..., ...]
    bboxes = [box for box in bboxes if box[1] > prob_threshold]  # drop boxes below a probability threshold
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)  # decreasing sort wrt probability score
    bboxes_after_nms = []

    while len(bboxes) > 0:
        chosen_box = bboxes.pop(0)  # get the 1st bbox
        # dropping redundant boxes == keeping unique boxes
        bboxes = [  # dropping the boxes if of same class as chosen box or, has an IoU greater than iou_threshold
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # keep the box if classes don't match
            # keep the box if it has a lower IoU than iou_threshold
            or iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

