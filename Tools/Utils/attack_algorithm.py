import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim

from Tools.HDA.loss import compute_loss_yolo

def generate_random_noise_attack(image, epsilon=0.05):
    noise = torch.FloatTensor(*image.shape).uniform_(-epsilon, epsilon).to(image.device)
    adversarial_image = torch.clamp(image + noise, 0.0, 1.0)

    return adversarial_image

# PGD attack to generate AE
def pgd_whitebox_v3(model, X, y, epsilon=0.05, num_steps=1, step_size=1/255, random=False, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    for _ in range(num_steps):

        optimizer.zero_grad()

        with torch.enable_grad():
            outputs, aux_outputs = model(X_pgd)
            loss1 = criterion(outputs.squeeze(), y.float())
            loss2 = criterion(aux_outputs.squeeze(), y.float())
            loss = loss1 + 0.4 * loss2


        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X_pgd

def pgd_whitebox_yolo(model, imgs, targets, epsilon=8 / 255, num_steps=10, step_size=2 / 255, random_start=True,
                       device='cpu'):
    imgs_pgd = Variable(imgs.data, requires_grad=True)

    if random_start:
        random_noise = torch.FloatTensor(*imgs_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        imgs_pgd = Variable(imgs_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([imgs_pgd], lr=1e-3)
        opt.zero_grad()

        # change here: For computing the loss of yolov3 model
        with torch.enable_grad():
            outputs = model(imgs_pgd)
            loss, loss_component = compute_loss_yolo(outputs, targets, model)

        loss.backward()
        eta = step_size * imgs_pgd.grad.data.sign()
        imgs_pgd = Variable(imgs_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(imgs_pgd.data - imgs.data, -epsilon, epsilon)
        imgs_pgd = Variable(imgs.data + eta, requires_grad=True)
        imgs_pgd = Variable(torch.clamp(imgs_pgd, 0, 1.0), requires_grad=True)

    return imgs_pgd

def fgsm_attack(model, images, targets, epsilon=8 / 255,device='cpu'):
    images.requires_grad = True
    images = images.to(device) # non_blocking=True
    outputs = model(images)
    loss, _ = compute_loss_yolo(outputs, targets, model)

    # # Forward pass
    # loss, _ = compute_loss_yolo(outputs, targets, model)
    #
    # # Backward pass
    model.zero_grad()
    loss.backward()

    # Collect the gradient of the input image
    data_grad = images.grad.data

    # Generate perturbed image using FGSM formula
    perturbed_images = images + epsilon * data_grad.sign()

    # Clip perturbed image to be within valid image range [0, 1]
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images

