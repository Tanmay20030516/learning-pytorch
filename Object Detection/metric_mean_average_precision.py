# https://www.v7labs.com/blog/mean-average-precision

import torch
from collections import Counter
from metric_intersection_over_union import iou


def AP(pred_boxes, true_boxes, iou_threshold, box_format, num_classes):

    average_precisions = []  # list to store APs for all classes for a given iou threshold
    epsilon = 1e-9  # numerical stability
    for class_label in range(num_classes):
        detections = []  # storing all predicted bboxes for current class
        ground_truths = []  # storing all true bboxes for current class

        for detection in pred_boxes:  # pred_boxes = [[image_index, class_label, probability, x1, y1, x2, y2], ..., ...]
            if detection[1] == class_label:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == class_label:
                ground_truths.append(true_box)

        # we find the number of ground truth boxes associated with each image_index
        num_bboxes = Counter([ground_truth[0] for ground_truth in ground_truths])
        # we get a dict with total no. of bboxes for each image

        # {img1:3, img2:2, ...} -> {img1:torch.tensor([0,0,0]), img2:torch.tensor([0,0])}
        for k, v in num_bboxes.items():
            num_bboxes[k] = torch.zeros(v)

        detections.sort(key=lambda x: x[2], reverse=True)  # decreasing sort wrt probability score
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_boxes = len(ground_truths)  # TP + FN

        if total_true_boxes == 0:
            # skip if no true boxes exist for current class
            continue

        for detection_idx, detection in enumerate(detections):
            # taking ground_truths that have same image_index as detection
            # i.e. collecting all true boxes for an image
            ground_truth_boxes_in_image = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_ground_truths = len(ground_truth_boxes_in_image)
            best_iou = 0
            best_gt_idx = None

            for idx, ground_truth_box in enumerate(ground_truth_boxes_in_image):
                # checking IoU score between ground truth box (in curr image) and detection/pred box (of curr class)
                iou_score = iou(
                    torch.tensor(detection[3:]),
                    torch.tensor(ground_truth_box[3:]),
                    box_format=box_format,
                )

                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if num_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1  # true positive
                    num_bboxes[detection[0]][best_gt_idx] = 1  # add this true bounding box to seen
                else:  # the box is already seen
                    FP[detection_idx] = 1

            else:  # if IOU is lower than the threshold, the detection is a false positive
                FP[detection_idx] = 1

        # cumulative sum (like prefix sum array)
        TP_cumsum = torch.cumsum(TP, dim=0)  # [0,1,0,1,1] -> [0,1,1,2,3]
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_boxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # adding points for numerical integration (x-axis = recall, y-axis = precision)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz (trapezoidal rule) for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


if __name__ == "__main__":
    iou_threshold_range = list(torch.range(start=0.5, end=0.9, step=0.05))
    AP_list = []
    for iou_th in iou_threshold_range:
        AP_list.append(
            AP(
                pred_boxes=...,
                true_boxes=...,
                iou_threshold=iou_th,
                box_format="corners",
                num_classes=20
            )
        )

    mAP = sum(AP_list)/len(AP_list)
    print(f"Mean Average Precision mAP is {mAP}:0.5:0.9:0.05")







