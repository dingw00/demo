import math

import torch
import torch.nn as nn
import numpy as np

from Tools.HDA.yolo_utils import *
from Tools.Utils.test_utils import to_cpu, min_max_scale

def mse(x,x_a):
    loss = (x_a - x)**2
    return torch.mean(loss,dim=[1,2,3])

def psnr(x,x_a):
    mse_loss = torch.mean((x_a - x) ** 2, dim=[1,2,3])
    return 20 * torch.log10(1.0 / torch.sqrt(mse_loss))

def compute_loss_yolo(predictions, targets, model):
    """
    For yolov3
    """
    # Check which device was used
    device = targets.device

    # Add placeholder varables for the different losses
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # Build yolo targets
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets

    # Define different loss functions classification
    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    # Calculate losses for each yolo layer
    for layer_index, layer_predictions in enumerate(predictions):
        # Get image ids, anchors, grid index i and j for each target in the current yolo layer
        b, anchor, grid_j, grid_i = indices[layer_index]
        # Build empty object target tensor with the same shape as the object prediction
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj
        # Get the number of targets for this layer.
        # Each target is a label box with some scaling and the association of an anchor box.
        # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
        num_targets = b.shape[0]
        # Check if there are targets for this batch
        if num_targets:
            # Load the corresponding values from the predictions for each of the targets
            ps = layer_predictions[b, anchor, grid_j, grid_i]

            # Regression of the box
            # Apply sigmoid to xy offset predictions in each cell that has a target
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            # Build box out of xy and wh
            pbox = torch.cat((pxy, pwh), 1)
            # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
            lbox += (1.0 - iou).mean()  # iou loss

            # Classification of the objectness
            # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)  # Use cells with iou > 0 as object targets

            # Classification of the class
            # Check if we need to do a classification (number of classes > 1)
            if ps.size(1) - 5 > 1:
                # Hot one class encoding
                t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                t[range(num_targets), tcls[layer_index]] = 1
                # Use the tensor to calculate the BCE loss
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        # Classification of the objectness the sequel
        # Calculate the BCE loss between the on the fly generated target and the network prediction
        lobj += BCEobj(layer_predictions[..., 4], tobj) # obj loss

    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    # Merge losses
    loss = lbox + lobj + lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))

def pred_loss(x,x_class,model):
    with torch.no_grad():
        y = model(x)
        y2 = 1 - y
        if x_class == 1:
            result = y2 -y
            result, _ = result.max(dim=1)
            return result

        else:

            result = y -y2
            result, _ = result.max(dim=1)
            return result
        
def fitness_score(x,y,x_a,model,local_op,alpha):
    loss = pred_loss(x_a,y,model)
    # 适应度得分：遗传算法主要是针对obj的值，来进行AE的生成

    if local_op == 'None':
        op = None
        obj = min_max_scale(loss)
        return obj, loss, op

    elif local_op == 'mse':
        op = -mse(x,x_a)
    elif local_op == 'psnr':
        op = psnr(x,x_a)
    elif local_op == 'ms_ssim':
        op = ms_ssim_module(x,x_a)
    else:
        raise Exception("Choose the support local_p from None, mse, psnr, ms_ssim")

    # if torch.sum(loss>0)/len(loss) < 0.6 :
    #     obj = min_max_scale(loss)
    # else:
    #     obj = min_max_scale(loss) + alpha * min_max_scale(op)

    obj = min_max_scale(loss)

    return obj, loss, op
def fitness_score_yolo(x, y, x_a, model, local_op, alpha, conf_thres, nms_thres, device="cpu", batch_size=1):

    x_input = torch.split(x_a, batch_size)
    loss = []
    model.eval()
    with torch.no_grad():
        for x_batch in x_input:          
            y_pred = model(x_batch.to(device))
            y_pred = non_max_suppression(y_pred, conf_thres, nms_thres) #(y_batch, conf_thres, nms_thres)
            y_loss = s_mis_detect(y_pred, y.to(device))
      
            loss.extend(torch.tensor(y_loss).cpu())

    #TODO: We are wasting the memory here.
    loss = torch.stack(loss)
    # loss = loss.squeeze(1).size()

    if local_op == 'None':
        op = None
        obj = min_max_scale(loss)
        return obj, loss, op

    elif local_op == 'mse':
        op = -mse(x,x_a)
    elif local_op == 'psnr':
        op = psnr(x,x_a)
    elif local_op == 'ms_ssim':
        op = ms_ssim_module(x,x_a)
    else:
        raise Exception("Choose the support local_p from None, mse, psnr, ms_ssim")
    
    if torch.sum(loss>0)/len(loss) < 0.6 :
        obj = min_max_scale(loss)
    else:
        obj = min_max_scale(loss) +  alpha * min_max_scale(op.cpu())

    return obj, loss, op

def cal_gradient(model,images,labels):
    # 修改了
    loss = nn.BCELoss()
    images.requires_grad = True
    outputs = model(images)
    cost = loss(outputs.squeeze().float(), labels.float())
    grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
    grad_norm = torch.norm(grad,p = np.inf, dim = [1,2,3])
    return grad_norm

def cal_gradient_yolo(model,images,labels):
    images.requires_grad = True
    outputs = model(images)
    cost, loss_components = compute_loss_yolo(outputs, labels, model)
    grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
    grad_norm = torch.norm(grad,p = np.inf, dim = [1,2,3])
    return grad_norm
