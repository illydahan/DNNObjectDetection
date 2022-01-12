import torch
import torch.nn as nn
from utils import IOU

class ObjectDetectionLoss(nn.Module):
    
    def __init__(self, S=7, B=2, C=8):
        super().__init__()
        
        self.S = S
        self.C = C
        self.B = B
        
        self.lambda_cord = 5
        self.lambda_noobj = 0.5
        
        self.mse = nn.MSELoss(reduction="sum")
    
    def forward(self, pred, target):
        
        
        pred = torch.reshape(pred, (-1, self.S, self.S, self.B * 5 + self.C))
        is_obj = target[..., 0].unsqueeze(3) # 1_i in the article
        
        # find the best bounding box
        
        iou1 = IOU(pred[..., 1:5], target[..., 1:5])
        iou2 = IOU(pred[..., 6:10], target[..., 1:5])
        
        ious = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0)
        
        max_iou, best_box = torch.max(ious, dim=0)
        
        
        
        
        box_target = is_obj*target[..., 1:5]

        box_pred = is_obj * (
            best_box * pred[..., 6:10] + 
            (1 - best_box) * pred[..., 1:5]
        )
        
        box_pred[..., 3:5] = torch.sign(box_pred[..., 3:5]) * torch.sqrt(torch.abs(box_pred[..., 3:5] + 1e-6)) 
        box_target[..., 3:5] = torch.sqrt(box_target[..., 3:5])
        
        
        cords_loss = self.mse(
            torch.flatten(box_pred, end_dim=-2),
            torch.flatten(box_target, end_dim= - 2)
        )
            
        
        pred_box = (
            best_box * pred[..., 10:11] + (1 - best_box) * pred[..., 0:1]
        )
        
        obj_loss = self.mse(
            torch.flatten(is_obj * pred_box),
            torch.flatten(is_obj * target[..., 0:1]),
        )
            
        
        no_object_loss = self.mse(
            torch.flatten((1 - is_obj) * pred[..., 0:1], start_dim=1),
            torch.flatten((1 - is_obj) * target[..., 0:1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - is_obj) * pred[..., 5:6], start_dim=1),
            torch.flatten((1 - is_obj) * target[..., 0:1], start_dim=1)
        )
        
        class_loss = self.mse(
            torch.flatten(is_obj * pred[..., 10:], end_dim=-2,),
            torch.flatten(is_obj * target[..., 10:], end_dim=-2,),
        )
        
        loss = (
            self.lambda_cord * cords_loss +
            obj_loss +
            self.lambda_noobj * no_object_loss +
            class_loss
        )
        
        # import math
        # if math.isnan(loss.item()):
        #     raise Exception("We shouldnt be here")
        return loss


def test_loss():
    
    pred = torch.rand((8, 882))
    target = torch.rand((8, 7, 7, 18))
    
    criterion = ObjectDetectionLoss()
    loss = criterion(pred, target)
    print(loss)

#test_loss()