import torch
from torch.distributions.categorical import Categorical
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
import os

from Tools.HDA.loss import fitness_score, fitness_score_yolo, cal_gradient, cal_gradient_yolo
from Tools.HDA.utils import mutation, calculate_fid
from Tools.Utils.attack_algorithm import *
from Tools.Utils.test_utils import provide_determinism, min_max_scale

from Tools.Utils.logger import create_logger
from Tools.HDA.yolo_utils import detect, non_max_suppression
from Tools.HDA.utils import load_images_from_folder
from Tools.Models.inception import InceptionV3
from Tools.HDA.GaussianKDE import GaussianKDE

def two_step_ga(model, x_seed, y_seed, eps=0.03, local_op="mse", conf_thres=0.1, nms_thres=0.5, 
                n_particles = 300, n_mate = 20, max_itr = 50, alpha = 1.00, batch_size=1, device="cpu"):
    
    adv_images = x_seed.repeat(n_particles,1,1,1)
    delta = torch.empty_like(adv_images).normal_(mean=0.0,std=0.01)
    delta = torch.clamp(delta, min=-eps, max=eps)
    adv_images = torch.clamp(x_seed + delta, min=0, max=1).detach()

    for i in range(max_itr):
        if str(device) == "cuda":
            torch.cuda.empty_cache()

        if model._get_name() == "Darknet": # yolov3
            obj, loss, op_loss = fitness_score_yolo(x_seed, y_seed, adv_images, model, 
                                                    local_op, alpha, conf_thres=conf_thres, 
                                                    nms_thres=nms_thres, 
                                                    device=device, 
                                                    batch_size=batch_size)
        elif model._get_name() == "Inception3":
            obj, loss, op_loss = fitness_score(x_seed, y_seed, adv_images, model, local_op, alpha)
        else:
            raise NotImplementedError(f"Model {model._get_name()} is not supported.")
        
        sorted_, indices = torch.sort(obj, dim=-1, descending=True)
        parents = adv_images[indices[:n_mate]]
        obj_parents = (sorted_[:n_mate]).to(device)

        # Generating next generation using crossover
        m = Categorical(logits=obj_parents)
        parents_list = m.sample(torch.Size([2*n_particles]))
        parents1 = parents[parents_list[:n_particles]]
        parents2 = parents[parents_list[n_particles:]]
        pp1 = obj_parents[parents_list[:n_particles]]
        pp2 = obj_parents[parents_list[n_particles:]]
        pp2 = pp2 / (pp1+pp2)
        pp2 = pp2[(..., ) + (None,)*3]

        mask_a = torch.empty_like(parents1).uniform_() > pp2
        mask_b = ~mask_a
        parents1[mask_a] = 0.0
        parents2[mask_b] = 0.0
        children = parents1 + parents2

        # add some mutations to children and genrate test set for next generation
        children = mutation(x_seed, children, eps, p=0.2)
        adv_images = torch.cat([children,parents], dim=0)

    if model._get_name() == "Darknet": # yolov3
        obj, loss, op_loss = fitness_score_yolo(x_seed, y_seed, adv_images, model, 
                                                local_op, alpha, conf_thres, nms_thres, 
                                                device=device,
                                                batch_size=batch_size)
    else:
        obj, loss, op_loss = fitness_score(x_seed, y_seed, adv_images, model, local_op, alpha)
    sorted_, indices = torch.sort(loss, dim=-1, descending=True)
    return adv_images[indices[:10]], loss[indices[:10]], op_loss[indices[:10]]

def select_seeds(vae, model, data_loader, n_seeds, density_mdl="kde", rand_seed=-1,
                 conf_thres=0.1, nms_thres=0.5, logger=None, device="cpu",
                 save_dir=None, remove_old=True):
    
    data_set = data_loader.dataset
    root_path_split = data_set.root.split("/")
    dataset_name = root_path_split[root_path_split.index("Dataset")+1]
    n = len(data_set)
    if save_dir is None:
        save_dir = os.path.join("Results", "hda_test", 
                                f"{dataset_name}", f"{model._get_name()}",
                                f"seeds_{density_mdl}")
    os.makedirs(save_dir, exist_ok=True)

    if logger is None:
        logger = create_logger("hda_test", os.path.join(os.path.dirname(os.path.dirname(save_dir)), 
                                                        "hda_test.log"))
    logger.info(f"Dataset {dataset_name} size: {n}")

    vae.eval()
    model.eval()

    if rand_seed != -1:
        provide_determinism(rand_seed)

    # delete previous image files
    if remove_old:
        for filename in os.listdir(save_dir):
            if filename.endswith(".png"):
                os.remove(os.path.join(save_dir, filename))
        aes_dir = save_dir.replace("seeds", "AEs")
        if os.path.exists(aes_dir):
            for filename in os.listdir(aes_dir):
                if filename.endswith(".png"):
                    os.remove(os.path.join(aes_dir, filename))
        logger.info("Previous seed & AE images have been deleted")

    
    grad_norm = None
    if density_mdl.lower() == "kde":
        grad_norm = []
        logger.info("Calculating grad_norm.")
        for data in tqdm(data_loader):
            x = data[0].to(device)
            y = data[1].to(device)
            # outputs = model(data)
            # _, loss_components = compute_loss_yolo(outputs, targets, model)
            if model._get_name() == "Darknet": # yolov3
                model.train()
                grad_batch = cal_gradient_yolo(model, x, y)
            elif model._get_name() == "Inception3":
                model.eval()
                grad_batch = cal_gradient(model, x, y)
            else:
                raise NotImplementedError(f"Model {model._get_name()} is not supported.")
            grad_norm.append(grad_batch.cpu())
        grad_norm = torch.cat(grad_norm, dim=0)

    # Test and obtain data distribution
    model.eval()
    x_test = torch.empty(0, dtype=torch.float32) # (n, vae.input_dim, data_set.img_size, data_set.img_size)
    x_mu = torch.empty(0, dtype=torch.float32) # (n, vae.z_dim)
    x_std = torch.empty(0, dtype=torch.float32) # (n, vae.z_dim)
    y_test = []
    correct_test = []

    for x, targets in data_loader:
        
        x_test = torch.cat([x_test, x])
        # [(,img_id + y_true + 4 box dimensions)]
        x = x.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            mu, log_var = vae.encode(x)
            # print(mu.shape, log_var.shape)
            x_mu = torch.cat([x_mu, mu[:, 0:vae.z_dim].cpu()])
            x_std = torch.cat([x_std, torch.exp(0.5 * log_var[:, 0:vae.z_dim]).cpu()])
            
            outputs = model(x)
            if model._get_name() == "Darknet": # yolov3
                # targets_pred is a list of [(n_bbox, 4 box dimensions + conf + y_pred + 80 class scores)]
                # each image's output is an element of the list
                targets = targets[:, :2]
                targets_pred = non_max_suppression(outputs, conf_thres, nms_thres)
                for img_i, target_pred in enumerate(targets_pred):
                    y_pred = -1
                    if len(target_pred) > 0:
                        # use the max confidence box & label
                        y_pred = target_pred[0, 5]

                    i_target = torch.where(targets[:, 0] == img_i)[0]
                    y_true = targets[i_target, 1]
                    corr = y_pred in y_true

                    y_test.append(int(y_pred))
                    correct_test.append(corr)

            elif model._get_name() == "Inception3":
                targets = targets.long()
                targets_pred = (outputs > 0.5).squeeze().long()

                y_test += targets_pred.tolist()
                correct_test += (targets_pred == targets).tolist()
            else:
                raise NotImplementedError(f"Model {model._get_name()} is not supported.")

    y_test = torch.tensor(y_test, dtype=torch.int)
    x_test = x_test[correct_test]
    y_test = y_test[correct_test]
    x_mu = x_mu[correct_test]
    x_std = x_std[correct_test]
    if density_mdl.lower() == "kde":
        grad_norm = grad_norm[correct_test]

    # Estimate data distribution (Gaussian KDE)
    kde = GaussianKDE(x_mu, x_std)
    pd = kde.score_samples(x_mu)

    # Select seeds based on the probability density
    if density_mdl.lower() == "kde":
        grad_aux = min_max_scale(grad_norm)
        aux_inf = pd * grad_aux
        sorted_, indices = torch.sort(aux_inf, dim=-1, descending=True)
        indices = indices[:n_seeds]

    elif density_mdl.lower() == "random":
        # compare with random seeds density
        indices = torch.randperm(len(y_test))[:n_seeds]
        # indices = torch.randperm(len(pd))[:n_seeds]

    # if len(indices) != n_seeds:
    #     raise IndexError("len(indices) != n_seeds")
    
    x_seeds = x_test[indices]
    y_seeds = y_test[indices]
    op = pd[indices]

    prob_density = (sum(op)/sum(pd)).item()

    logger.info('-----------------------------------------------------------')
    logger.info(f"Model: {model._get_name()}")
    logger.info(f'Dataset: {dataset_name}')
    logger.info('################# SEEDs SELECTION ###########################')
    logger.info(f'No. of test seeds selected: {n_seeds}')
    logger.info(f'Total No. of correctly detected seeds in test set: {len(x_mu)}')
    logger.info(f'Global Seeds Probability Density (normalized) - {density_mdl}: {prob_density}')

    # Save test seed images ({id}_{true label}.png)
    for i, (x_seed, y_seed) in enumerate(zip(x_seeds,y_seeds)):
        filename = f'{i+1}_{int(y_seed.item())}.png'
        vutils.save_image(x_seed,
                          os.path.join(save_dir, filename),
                          normalize=False)
    
    return dict(prob_density=prob_density)

def generate_aes(seeds_dir, model, eps, local_op, ga_params, 
                 conf_thres=0.1, nms_thres=0.5,
                 batch_size=1, device="cpu", remove_old=True, logger=None):

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    aes_dir = os.path.join(seeds_dir.replace("seeds", "AEs"), f"{local_op}_{eps}")
    os.makedirs(aes_dir, exist_ok=True)
    if logger is None:
        logger = create_logger("hda_test", 
                               os.path.join(os.path.dirname(os.path.dirname(seeds_dir)), 
                                                        "hda_test.log"))

    logger.info('\nStart to generate AEs.')
    logger.info(f'Norm ball radius: {eps}')

    # delete previous png picture in the file
    if remove_old:
        for filename in os.listdir(aes_dir):
            if filename.endswith(".png"):
                os.remove(os.path.join(aes_dir, filename))
        logger.info("Previous AE images have been deleted")

    # Load previously selected seeds
    _, x_seeds, y_seeds = load_images_from_folder(seeds_dir)
    if len(x_seeds) == 0:
        logger.error("No seed images found.")
        return dict(attack_success_rate=np.nan, average_prediction_loss=np.nan)

    logger.info("Generating AEs using GA on seeds images.")
    test_set = []
    loss_set = []
    with torch.no_grad():
        for i, (x_seed, y_seed) in tqdm(enumerate(zip(x_seeds, y_seeds))):
            # x_seed = x_seed / 2 + 0.5
            # start running GA on seeds input
            x_seed = torch.tensor(x_seed).to(device)
            y_seed = torch.tensor(y_seed).to(device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ae, loss, op_loss = two_step_ga(model, x_seed, y_seed, eps, local_op=local_op,
                                            conf_thres=conf_thres, nms_thres=nms_thres,
                                            n_particles=ga_params["n_particles"], 
                                            n_mate=ga_params["n_mate"], 
                                            max_itr=ga_params["max_itr"], alpha=ga_params["alpha"],
                                            batch_size=batch_size, device=device)
            test_set.append(ae.cpu())

            # Check if the AE is successfully generated
            idx = torch.where(loss>=0)[0]
            if len(idx)>0:
                # Test the DNN model with the generated AE
                ae = ae[idx[0]]
                ae_loss = loss[idx[0]]
                
                ae_pred = model(ae.unsqueeze(0).to(device))
                if model._get_name() == "Darknet": # yolov3
                    ae_pred = non_max_suppression(ae_pred, conf_thres=conf_thres, iou_thres=nms_thres)[0]
                elif model._get_name() == "Inception3":
                    ae_pred = (ae_pred > 0.5).squeeze().long()
                else:
                    raise NotImplementedError(f"Model {model._get_name()} is not supported.")

                loss_set.append(ae_loss.cpu().unsqueeze(0))

                if model._get_name() == "Darknet": # yolov3
                    ae_pred_cls = ae_pred[:,5].to(int).cpu()
                    ae_pred_unique  = list(set([int(cl) for cl in ae_pred_cls]))
                    order_dict = {int(value): len(ae_pred_cls)-1-order for order, value  # value: order
                                in enumerate(reversed(ae_pred_cls)) if (value in ae_pred_unique)}
                    if len(ae_pred_unique) == 0:
                        ae_pred_unique_sorted = ["na"]
                    else:
                        ae_pred_unique_sorted = sorted(ae_pred_unique, key=lambda x: order_dict[x])
                elif model._get_name() == "Inception3":
                    ae_pred_unique_sorted = [ae_pred.cpu().item()]
                
                # Save the generated AE (true_label=y_seed.item())
                filename = '{no}_{pred}.png'.format(no=i+1, pred="_".join(map(str, ae_pred_unique_sorted)))
                vutils.save_image(
                    ae,
                    os.path.join(aes_dir, filename),
                    normalize=False)

            else:
                filename = '{no}_failed.png'.format(no=i+1)
                vutils.save_image(
                    torch.ones_like(x_seed),
                    os.path.join(aes_dir, filename),
                    normalize=False)
                
    if len(loss_set) == 0:
        logger.info("No AE images generated.")
        loss_set = torch.tensor(loss_set)
    else:
        loss_set = torch.cat(loss_set)
        
        # Write the test report
        logger.info('################# AEs GENERATION ###########################')
        logger.info('Model: {}'.format(model._get_name()))
        logger.info('Norm ball radius: {}'.format(eps))
        logger.info('Attack success rate: {}'.format(len(loss_set)/len(x_seeds)))
        logger.info('Avg. prediction loss of AEs: {}'.format(torch.mean(loss_set).item()))
        
    return dict(attack_success_rate=len(loss_set)/len(x_seeds), 
                average_prediction_loss=torch.mean(loss_set).item())

def eval_aes(seeds_dir, aes_dir, device="cpu", logger=None):
    
    save_dir = os.path.dirname(os.path.dirname(seeds_dir))
    os.makedirs(save_dir, exist_ok=True)
    if logger is None:
        logger = create_logger("hda_test", os.path.join(save_dir, "hda_test.log"))

    idx_seeds, x_seeds, _ = load_images_from_folder(seeds_dir)
    idx_aes, aes, _ = load_images_from_folder(aes_dir)

    x_seeds_ = []
    for id in idx_aes:
        if id in idx_seeds: 
            x_seeds_.append(x_seeds[idx_seeds.index(id)])

    if len(x_seeds_) == 0 or len(aes) == 0:
        logger.info("No seeds or AE images found.")
        return dict(avg_perb_amount=np.nan, fid=np.nan)
    
    seed_set = torch.concat([x_seed.unsqueeze(0) for x_seed in x_seeds_])
    adv_set = torch.concat([ae.unsqueeze(0) for ae in aes])

    # calculate l_inf norm beween seeds and AEs
    epsilon = torch.norm(adv_set-seed_set,p=float('inf'),dim=(1,2,3))
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)

    if len(seed_set) > 1 and len(adv_set) > 1:
        with torch.no_grad():
            seed_set = seed_set.to(device)
            adv_set = adv_set.to(device)
            adv_mu = inception(adv_set)[0]
            seed_mu = inception(seed_set)[0]
            adv_mu= torch.flatten(adv_mu, start_dim = 1)
            seed_mu = torch.flatten(seed_mu, start_dim = 1)

        fid = calculate_fid(np.array(adv_mu.cpu()),np.array(seed_mu.cpu()))
    else:
        fid = np.nan
        
    # Generate a test report
    logger.info('################# Local AEs Perceptual Quality ###########################')
    logger.info(f'Avg. Perturbation Amount: {torch.mean(epsilon).item()}')
    logger.info('FID (adv.): {:.3f}'.format(fid))

    return dict(avg_perb_amount=torch.mean(epsilon).item(), fid=fid)

def hda_test(vae, model, data_loader,
               eps, n_seeds, density_mdl, local_op,
               conf_thres=0.1, nms_thres=0.5,
               ga_params=dict(n_particles=100, n_mate=20, max_itr=10, alpha=1.00),
               logger=None, rand_seed=-1, batch_size=1,
               device="cpu"):
    
    data_set = data_loader.dataset
    root_path_split = data_set.root.split("/")
    dataset_name = root_path_split[root_path_split.index("Dataset")+1]
    n = len(data_set)
    seeds_dir = os.path.join("Results", "hda_test", 
                            f"{dataset_name}", f"{model._get_name()}",
                            f"seeds_{density_mdl}")
    aes_dir =  os.path.join(seeds_dir.replace("seeds", "AEs"), f"{local_op}_{eps}")
    save_dir = os.path.dirname(os.path.dirname(seeds_dir))
    os.makedirs(save_dir, exist_ok=True)
    if logger is None:
        logger = create_logger("hda_test", os.path.join(save_dir, "hda_test.log"))
    logger.info(f"Start HDA test on {model._get_name()} model. Dataset {dataset_name}. Device: {device}.")
    logger.info(f"n_seeds: {n_seeds}, density_mdl: {density_mdl}, local_op: {local_op}, eps: {eps}, ga_params: {ga_params}")
    
    select_seeds(vae, model, data_loader, n_seeds, density_mdl, rand_seed,
                 conf_thres=conf_thres, nms_thres=nms_thres, logger=logger, 
                 device=device, save_dir=seeds_dir)
    generate_aes(seeds_dir, model, eps, local_op, ga_params, conf_thres=conf_thres, nms_thres=nms_thres,
                 batch_size=batch_size, device=device, logger=logger)
    eval_aes(seeds_dir, aes_dir, device=device, logger=logger)

    return 0

