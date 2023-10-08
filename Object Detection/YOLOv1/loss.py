# YOLOv1 paper: https://arxiv.org/pdf/1506.02640.pdf
# Article: https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
import torch
import torch.nn as nn
from utils import iou


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        """
        :param S: number of splits
        :param B: number of detections per cell
        :param C: number of classes
        """
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        # from paper
        self.lambda_noobj = 0.5  # to decrease impact of loss from no object cells
        self.lambda_coord = 5  # emphasise more on localisation loss

        self.mse = nn.MSELoss(reduction="sum")  # just sum the losses

    def forward(self, predictions, target):
        # reshaping since predictions of shape (batch, S*S*(C + B*5))
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # calculate IoU of two predicted boxes with target box
        iou1 = iou(predictions[..., 21:25], target[..., 21:25], box_format="midpoint")
        iou2 = iou(predictions[..., 26:30], target[..., 21:25], box_format="midpoint")
        ious = torch.cat([torch.unsqueeze(iou1, dim=0), torch.unsqueeze(iou2, dim=0)], dim=0)

        iou_max, best_box = torch.max(ious, dim=0)  # val, idx
        # target[..., 20].shape -> (-1, 7, 7)
        # torch.unsqueeze(target[..., 20], dim=3).shape -> (-1, 7, 7, 1)
        # the index 20 corresponds to pc (i.e. 1 or 0)
        # Iobj_i: extracting objectedness score from target
        exists_box = torch.unsqueeze(target[..., 20], dim=3)  # shape (-1, S, S, 1)

        # BOX COORDINATES (1st two lines of loss fn)
        box_predictions = exists_box * (
                best_box * predictions[..., 26:30] +  # when best_box is 1 (i.e. second box)
                (1-best_box) * predictions[..., 21:25]  # when best_box is 0 (i.e. first box)
        )  # shape (-1, S, S, 4)
        # taking square root of w, h      this is done to incorporate correct sign of gradient (lost due to torch.abs())
        #                                         |
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-9)  # to avoid sqrt of -ve, and infinite gradient of sqrt fn at 0
        )

        box_targets = exists_box * target[..., 21:25]  # shape (-1, S, S, 4)
        # taking square root of w, h
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        true = torch.flatten(box_targets, end_dim=-2)  # shape (-1, S*S, 4)
        pred = torch.flatten(box_predictions, end_dim=-2)  # shape (-1, S*S, 4)
        box_loss = self.mse(pred, true)

        # OBJECT LOSS (3rd line of loss fn)
        # confidence score for box with highest IoU
        pred_box = best_box * predictions[..., 25:26] + (1-best_box) * predictions[..., 20:21]  # shape (-1, S, S, 1)

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # NO OBJECT LOSS (4th line of loss fn)
        # shape (-1, S, S, 1) -> (-1, S*S)
        no_object_box1 = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1-exists_box)*target[..., 20:21], start_dim=1)
        )
        no_object_box2 = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        )
        no_object_loss = no_object_box1 + no_object_box2

        # CLASS LOSS (5th line of loss fn)
        # shape (-1, S, S, NUM_CLASSES) -> (-1, S*S, NUM_CLASSES)
        classification_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        # COMBINING ALL LOSS COMPONENTS
        complete_loss = (
            + self.lambda_coord*box_loss
            + object_loss
            + self.lambda_noobj*no_object_loss
            + classification_loss
        )

        return complete_loss


