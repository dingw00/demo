import torch
import torch.nn as nn
from torchvision.models import Inception_V3_Weights, inception_v3
from Tools.Models.yolo import Darknet
import os

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
def load_model(model_config, weight_path=None, device="cpu"):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    if "yolov3" in model_config.lower():
        model = Darknet(model_config).to(device)
        model.apply(weights_init_normal)
    elif "inception_v3" in model_config.lower():
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights)

        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Sequential(
            nn.Linear(num_ftrs_aux, 1),
            nn.Sigmoid()
        )
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        model = model.to(device)

    # Load checkpoint weights
    if weight_path:
        assert os.path.exists(weight_path), f"Weight file {weight_path} does not exist!"   
        model.load_state_dict(torch.load(weight_path, map_location=device)) # 448_V3_without_norm.pth
       
    return model
