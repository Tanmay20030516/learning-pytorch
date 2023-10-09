import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def mAP(pred_boxes, true_boxes, iou_threshold, box_format, num_classes):

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


def plot_image(image, boxes):
    """
    Plots predicted bounding boxes on the image
    """
    im = np.array(image)
    height, width, _ = im.shape

    # create figure and axes
    fig, ax = plt.subplots(1)
    # displaying the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # creating a rectangle box
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            xy=(upper_left_x * width, upper_left_y * height),
            width=box[2] * width,
            height=box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # adding the box to the axes
        ax.add_patch(rect)

    plt.show()


def save_checkpoint(state, filename="checkpoint.pth", epoch=None):
    print("=> Saving checkpoint")
    # filename = f"epoch{epoch}_" + filename
    # filename = "checkpoints/" + filename
    torch.save(state, filename)  # saving the checkpoint


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def get_bboxes(loader, model, iou_threshold, prob_threshold, pred_format="cells", box_format="midpoint", device="cuda"):
    """
    Get true and prediction bounding boxes after passing model and data
    :param loader: train/test data loader
    :param model: object detection model
    :param iou_threshold: IoU threshold req for non-max suppression
    :param prob_threshold: threshold probability score for object's presence
    :param pred_format: prediction format for bboxes (wrt "cells" or "boxes" (image))
    :param box_format: "midpoint" or "corners"
    :param device: "cuda" if available else "cpu"
    :type iou_threshold: float
    :type prob_threshold: float
    :return: lists of all true bboxes and prediction boxes
    """
    all_pred_boxes = []
    all_true_boxes = []

    # set model in eval mode before predicting bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)  # coordinates wrt cells

        with torch.no_grad():  # don't keep track of gradients
            predictions = model(x)

        batch_size = x.shape[0]
        # change from wrt cells to wrt image
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
                box_format=box_format,
            )

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()  # set model back to train mode
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes



