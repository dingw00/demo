import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import torch.nn as nn
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from Tools.Utils.attack_algorithm import pgd_whitebox_v3
from Tools.Utils.test_utils import to_cpu

from Tools.Utils.test_utils import provide_determinism
from Tools.Utils.dataloader import load_dataset
from Tools.Utils.logger import create_logger
from Tools.Utils.attack_algorithm import pgd_whitebox_v3, pgd_whitebox_yolo, fgsm_attack, generate_random_noise_attack
from Tools.HDA.loss import compute_loss_yolo
from Tools.HDA.eval import evaluate_yolo
from Tools.Models.utils import load_model


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#### For VAE model
def _train_vae(data_loader, model, optimizer, epoch, logger, device="cpu"):
    """
    This function trains the VAE model for one epoch.
    TODO: CUDA Memory Problem here!
    """
    train_loss = []
    recon_loss = []
    kld_loss = []
    model.train()
    start_time = time.time()

    for x, y in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        x = x.to(device)

        optimizer.zero_grad()
        x_tilde, kl_d = model(x)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, x)
        loss = loss_recons + 0.001 * kl_d
        loss.backward()
        optimizer.step()

        train_loss.append(loss.cpu().item())
        recon_loss.append(loss_recons.cpu().item())
        kld_loss.append(kl_d.cpu().item())

    avg_train_loss = np.asarray(train_loss).mean(0)

    logger.info('\tLoss: {:7.6f}   Reconstruction Loss: {:7.6f}   KLD Loss: {:7.6f}  Time: {:5.3f} s'.format(
        np.asarray(train_loss).mean(0),
        np.asarray(recon_loss).mean(0),
        np.asarray(kld_loss).mean(0),
        time.time() - start_time
    ))
    return avg_train_loss

def _test_vae(data_loader, model, logger, device="cpu"):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        loss_recons, loss_kld = 0., 0.
        for x, y in tqdm(data_loader, desc="Testing"):
            x = x.to(device)
            x_tilde, kl_d = model(x)
            loss_recons += F.mse_loss(x_tilde, x)
            loss_kld += kl_d

        loss_recons /= len(data_loader)
        loss_kld /= len(data_loader)

    logger.info('Validation:\tReconstruction Loss: {:7.6f} Time: {:5.3f} s'.format(
        np.array(loss_recons.item()),
        time.time() - start_time
    ))
    return loss_recons.cpu().item(), loss_kld.cpu().item()

def train_vae(train_loader, val_loader, vae_model, optimizer="adam", lr=1e-3, weight_decay=0, 
              n_epochs=10, logger=None, device="cpu"):
    """
    This function trains the VAE model. The validation loss is checked after each epoch and the 
    model is saved if the loss is the best so far. The train/validation loss - epoch line chart 
    is saved and updated after each epoch.
    """

    root_path_split = train_loader.dataset.root.split("/")
    dataset_name = root_path_split[root_path_split.index("Dataset")+1]
    save_dir = os.path.join("Results", "vae")
    os.makedirs(save_dir, exist_ok=True)

    for f in os.listdir(save_dir):
        if dataset_name in f:
            os.remove(os.path.join(save_dir, f))

    if logger is None:
        logger = create_logger("vae_train", os.path.join(save_dir, "vae_train.log"))
    logger.info(f"Start training VAE model. Dataset: {dataset_name}. Device: {device}")

    if str(device) == "cuda":
        torch.cuda.empty_cache()

    if optimizer == "adam":
        _optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer} not supported!")

    best_loss = -1.
    last_saved = -1
    train_losses = []
    recon_losses = []
    
    for epoch in range(n_epochs):
        logger.info(f"Epoch={epoch}")
        train_loss = _train_vae(train_loader, vae_model, _optimizer, epoch, logger, device=device)
        print("Train loss:", train_loss)
        recon_loss, kld_val_loss = _test_vae(val_loader, vae_model, logger, device=device)

        train_losses.append(train_loss)
        recon_losses.append(recon_loss)

        plt.figure()
        plt.plot(range(epoch+1), train_losses, label="Train Loss")
        plt.plot(range(epoch+1), recon_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"vae_loss_{dataset_name}.png"))
        plt.close("all")
        
        vae_model.generate_reconstructions(val_loader, save_dir=save_dir, device=device)

        if (epoch == 0) or (recon_loss < best_loss):
            logger.info(f"Saving model. Best loss: {best_loss}, Current loss: {recon_loss}")
            best_loss = recon_loss
            last_saved = epoch
            torch.save(vae_model.state_dict(), os.path.join(save_dir, f"{dataset_name}_vae_epoch_{epoch}.pt"))
        else:
            logger.info("Not saving model! Last saved: {}\n".format(last_saved))


#### For Inception V3 model
def train_V3(model, train_loader, val_loader, start_from=0, num_epochs=10, lr=0.001,
             device="cpu"):
    
    save_dir = os.path.join("Results", "adv_train")
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    
    for epoch in range(start_from+1, num_epochs+start_from+1):
        model.train()
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, aux_outputs = model(inputs)
            
            predicted = (outputs > 0.5).squeeze().long()
            correct += (predicted == labels.long()).sum().item()
            total += labels.size(0)
            train_acc = correct / total

            loss1 = criterion(outputs.squeeze(), labels.float())
            loss2 = criterion(aux_outputs.squeeze(), labels.float())
            loss = loss1 + 0.4 * loss2
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)
                val_predicted = (val_outputs > 0.5).squeeze().long()
                val_correct += (val_predicted == val_labels.long()).sum().item()
                val_total += val_labels.size(0)

            val_acc = val_correct / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_dir, f'inception_v3_epoch_{epoch}.pth'))

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Train Acc: {train_acc}, Val Acc: {val_acc}")

def test_V3(model, test_loader, criterion, device="cpu"):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            test_loss += loss.item()
            predicted = (outputs > 0.5).squeeze().long()
            test_total += labels.size(0)
            test_correct += (predicted == labels.long()).sum().item()

    test_loss /= len(test_loader)
    test_acc = test_correct / test_total

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return test_loss, test_acc

def test_V3_much(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            test_loss += loss.item()

            predicted = (outputs > 0.5).squeeze().long()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()

    # 计算各项指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    specificity = tn / (tn + fp)

    print(f'Confusion Matrix: \nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n'
          f'Test Loss: {test_loss:.4f}\n'
          f'Test Accuracy: {accuracy:.4f}\n'
          f'Precision: {precision:.4f}\n'
          f'Recall: {recall:.4f}\n'
          f'F1 Score: {f1:.4f}\n'
          f'Specificity: {specificity:.4f}')
    
    return test_loss, accuracy, precision, recall, f1, specificity

def ad_train_V3(model, train_loader, val_loader, attacker="pgd", 
                start_from=0, num_epochs=1, lr=0.001,
                eps=0.05, num_steps=10, step_size=8/255,
                optimizer="adam", 
                logger=None, device="cpu"):

    save_dir = os.path.join("Results", "adv_train")
    os.makedirs(save_dir, exist_ok=True)
    if logger is None:
        logger = create_logger("inception_v3_train_adv", os.path.join(save_dir, "yolo_adv_train.log"))
    logger.info(f"Start adversarial training for Inception V3 model. Device: {device}")
    logger.info(f"Attacker: {attacker}. Epsilon: {eps}. Num steps: {num_steps}. Step size: {step_size}. Optimizer: {optimizer}")
    # start with normal model training
    model = model.to(device)
    criterion = nn.BCELoss()
    assert optimizer in ["adam"] 
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0  # Initialize the best validation accuracy
    for epoch in range(start_from+1, num_epochs+start_from+1):
        model.train()
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # adversarial training
            if attacker == "pgd":
                inputs = pgd_whitebox_v3(model, inputs, labels, epsilon=eps, 
                                         num_steps=num_steps, step_size=step_size,
                                        device=device)
            else:
                raise ValueError(f"Attacker {attacker} not supported!")
            optimizer.zero_grad()
            outputs, aux_outputs = model(inputs)

            predicted = (outputs > 0.5).squeeze().long()
            correct += (predicted == labels.long()).sum().item()
            total += labels.size(0)
            train_acc = correct / total

            loss1 = criterion(outputs.squeeze(), labels.float())
            loss2 = criterion(aux_outputs.squeeze(), labels.float())
            loss = loss1 + 0.4 * loss2
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)
                val_predicted = (val_outputs > 0.5).squeeze().long()
                val_correct += (val_predicted == val_labels.long()).sum().item()
                val_total += val_labels.size(0)

            val_acc = val_correct / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_dir, f"inception_v3_adv_epoch_{epoch}.pth"))

        logger.info(f"Epoch {epoch}, Loss: {loss.item()}, Train Acc: {train_acc}, Val Acc: {val_acc}")

    logger.info("Adversarial Training finished.")

### For YOLO model
def ad_train_yolo(model, train_loader, val_loader, class_names, start_from=0, 
                  num_epochs=1, attacker=None,
                  eps = 8/255, num_steps=10, step_size=8/255, 
                  iou_thres=0.5, conf_thres=0.1, nms_thres=0.5,
                  ckpt_interval=1, eval_interval=1,
                  logger=None, verbose=False, device="cpu"):
    
    output_dir = os.path.join("Results", "adv_train")
    if logger is None:
        logger = create_logger("yolo_train_adv", os.path.join(output_dir, "yolo_v3_adv_train.log"))
    
    # ################
    # Create optimizer
    # ################
    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = torch.optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = torch.optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        logger.error("Unknown optimizer. Please choose between (adam, sgd).")

    logger.info(f"Start adversarial training for YOLOv3 model. Device: {device}")
    logger.info(f"Attacker: {attacker}. Epsilon: {eps}. Num steps: {num_steps}. Step size: {step_size}. Optimizer: {optimizer}")

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(start_from+1, start_from+num_epochs+1):
        criterion_kl = torch.nn.KLDivLoss(reduction="sum")
        model.train()  # Set model to training mode

        for batch_i, (imgs, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            batches_done = len(train_loader) * epoch + batch_i

            # easy version
            imgs = imgs.to(device) # non_blocking=True
            targets = targets.to(device)

            output = model(imgs)
            loss, loss_components = compute_loss_yolo(output, targets, model)
            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.info(f"train/learning_rate, {lr}, {batches_done}")
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                optimizer.zero_grad()

            if attacker is not None:
                # PGD whitebox yolo", "Random noise", "FGSM"
                if attacker == "pgd":
                    data_adv = pgd_whitebox_yolo(model, imgs, targets, epsilon=eps,
                                                  num_steps=num_steps, step_size=step_size, 
                                                  device=device).float()
                elif attacker == "rand_noise":
                    data_adv = generate_random_noise_attack(imgs, epsilon=eps).float()
                elif attacker == "fgsm":
                    data_adv = fgsm_attack(model, imgs, targets, epsilon=eps, device=device).float()
                else:
                    raise ValueError(f"Attacker {attacker} not supported!")
                
                # create adverarial example --- FGSM                
                output_adv = model(data_adv)
                loss2, loss_components2 = compute_loss_yolo(output_adv, targets, model)
                loss2.backward()

                if batches_done % model.hyperparams['subdivisions'] == 0:
                    # Adapt learning rate
                    # Get learning rate defined in cfg
                    lr = model.hyperparams['learning_rate']
                    if batches_done < model.hyperparams['burn_in']:
                        # Burn in
                        lr *= (batches_done / model.hyperparams['burn_in'])
                    else:
                        # Set and parse the learning rate to the steps defined in the cfg
                        for threshold, value in model.hyperparams['lr_steps']:
                            if batches_done > threshold:
                                lr *= value
                    # Set learning rate
                    for g in optimizer.param_groups:
                        g['lr'] = lr

                    # Run optimizer
                    optimizer.step()
                    optimizer.zero_grad()
    
            if verbose:
                logger.info("\n"+AsciiTable(
                    [
                        ["Type", "Origin_loss", "Adv_loss"],
                        ["IoU loss", float(loss_components[0]), float(loss_components2[0])],
                        ["Object loss", float(loss_components[1]), float(loss_components2[1])],
                        ["Class loss", float(loss_components[2]), float(loss_components2[2])],
                        ["Loss", float(loss_components[3]), float(loss_components2[3])],
                        ["Batch loss", to_cpu(loss).item(), to_cpu(loss2).item()],
                    ]).table)
              
            model.seen += imgs.size(0)

        # Save model to checkpoint file
        if epoch % ckpt_interval == 0 or (epoch == (start_from+num_epochs)):
            checkpoint_path = os.path.join(output_dir, 
                                           f"yolov3_{attacker}_epoch_{epoch}.pth")
            logger.info(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)
        

        if (epoch % eval_interval == 0) or (epoch == (start_from+num_epochs)):
            logger.info("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = evaluate_yolo(
                model,
                val_loader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=iou_thres,
                conf_thres=conf_thres,
                nms_thres=nms_thres,
                verbose=verbose,
                device=device
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output

                logger.info("\n"+AsciiTable(
                    [
                        ["validation/precision", precision.mean()],
                        ["validation/recall", recall.mean()],
                        ["validation/mAP", AP.mean()],
                        ["validation/f1", f1.mean()],
                    ]).table)

### General Use
def ad_train_model(model, train_loader, val_loader, num_epochs, lr=0.001,
                   attacker="pgd", eps=8/255, step_size=8/255, num_steps=10, 
                   iou_thres=0.5, conf_thres=0.1, nms_thres=0.5, optimizer="adam",
                   ckpt_interval=1, eval_interval=1, 
                   start_from=0, rand_seed=-1,
                   logger=None, verbose=False, device="cpu"):
                   
    if rand_seed != -1:
        provide_determinism(rand_seed)
    save_dir = os.path.join("Results", "adv_train")
    os.makedirs(save_dir, exist_ok=True)
       
    if logger is None:
        logger = create_logger(f"adv_train_{model._get_name()}", os.path.join(save_dir, f"adv_train_{model._get_name()}.log"))

    if str(device) == "cuda":
        torch.cuda.empty_cache()
    # Start adv training
    if model._get_name() == "Darknet":
        class_names = train_loader.dataset.class_names
        ad_train_yolo(model, train_loader, val_loader, class_names, start_from=start_from, 
                      num_epochs=num_epochs, attacker=attacker,
                      eps=eps, num_steps=num_steps, step_size=step_size,
                      iou_thres=iou_thres, conf_thres=conf_thres, nms_thres=nms_thres,
                      ckpt_interval=ckpt_interval, eval_interval=eval_interval,
                      logger=logger, verbose=verbose, device=device)
    elif model._get_name() == "Inception3":
        ad_train_V3(model, train_loader, val_loader, attacker=attacker, 
                    start_from=start_from, num_epochs=num_epochs, lr=lr,
                    eps=eps, num_steps=num_steps, step_size=step_size,
                    optimizer=optimizer, 
                    logger=logger, device=device)
    else:
        raise ValueError(f"Model {model._get_name()} not supported!")        
