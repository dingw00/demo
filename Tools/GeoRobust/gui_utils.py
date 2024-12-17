import os
import torch
import numpy as np
import time
import shutil
import torchvision.transforms as tfs
import PIL.Image as Image

from Tools.Utils.dataloader import load_dataset
from Tools.Models.utils import load_model
from Tools.Utils.logger import create_logger
from Tools.GeoRobust.direct_with_lb import LowBoundedDIRECT, LowBoundedDIRECT_POset_full_parrallel
from Tools.GeoRobust.geo_transf_verifications import GeometricVarification, \
    AffineTransf, make_theta, cw_loss, cw_loss_2

SAVE_DIR = os.path.join("Results", "geo_robust")
def update_report_fn():
    """
    This function reads the GeoRobust test log and returns its content.
    """
    # Read the GeoRobust test log
    content = ""
    report_filepath = os.path.join(SAVE_DIR, "geo_robust.log")
    if os.path.exists(report_filepath):
        with open(report_filepath, 'r') as f:
            content = f.read()
    return content

def test_model_fn(model_cfg, model_weights, img_size, dataset_name, idx, 
                                            img_seed0, y_seed0, dev):
    """
    This function tests the model classification and confidence on the selected image seed.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = create_logger("geo_robust", os.path.join(SAVE_DIR, "geo_robust.log"),
                       remove_old=True)
    logger.info("Testing the model on the selected image seed.")
    logger.info(f"Model: {model_cfg}. Weights: {model_weights}. Device: {dev}")
    device = torch.device(dev)
    # Load test image
    if img_seed0 is None:
        data_set, data_loader = load_dataset(dataset_name, subset="test", 
                                             img_size=img_size, shuffle=False)
        img_seed = data_set[idx][0]
        y_seed = data_set[idx][1]
    else:
        img_seed0 = Image.fromarray(img_seed0, mode='RGB')
        transforms = tfs.Compose([tfs.Resize((img_size, img_size)), tfs.ToTensor()])
        img_seed = transforms(img_seed0)
        y_seed = y_seed0
        idx = 0
        dataset_name = "uploaded"

    # Load model
    model = load_model(model_cfg, model_weights, device=device)
    model.eval()

    # evaluate the output
    ori_out = model(img_seed.unsqueeze(0).to(device))
    ori_conf = cw_loss(ori_out, y_seed).item()
    if (ori_out > 0.5).long().item() == y_seed: # correct classification
        logger.info(f'The origin confidence on class 1 is {ori_conf:.4f}')
        logger.info(f'{dataset_name}: {idx}, correctness: True, ori_conf: {ori_conf:.6f}')
    else: # incorrect classification
        logger.info(f'Image seed is misclassified. Please select another seed.')
        logger.info(f'{dataset_name}: {idx}, correctness: False, ori_conf: {ori_conf:.6f}')
    return tfs.ToPILImage()(img_seed), f"True label: {y_seed}. Predicted label: {(ori_out > 0.5).long().item()}. Confidence: {ori_conf:.4f}"

def solve_direct_fn(model_cfg, model_weights, img_size, dataset_name, idx, 
                    img_seed0, y_seed0, max_iter, max_deep, max_eval, tol,
                    po_set_size, dev, rot, max_rot, trans, max_trans, scale, max_scale):
    """
    This function solves the optimization problem using DIRECT method.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = create_logger("geo_robust", os.path.join(SAVE_DIR, "geo_robust.log"),
                       remove_old=False)
    device = torch.device(dev)

    # Prepare for the transform
    max_rot = max_rot if rot else 0.0
    max_trans = max_trans if trans else 0.0
    max_scale = max_scale if scale else 0.0

    transf = []
    bound = []
    location_dist = {}
    if max_rot != 0:
        transf.append('angle')
        bound.append([-np.pi*max_rot/180.0, np.pi*max_rot/180.0])
    if max_trans != 0:
        transf.append('h_shift'); transf.append('v_shift')
        bound.append([-max_trans, max_trans])
        bound.append([-max_trans, max_trans])
    if max_scale != 0:
        transf.append('scale')
        bound.append([1-max_scale, 1+max_scale])
    if len(bound) == 0:
        logger.info('No transform is allowed. Please select at least one transform.')
        return None, None
    
    # Load model
    model = load_model(model_cfg, model_weights, device=device)
    model.eval()

    # Load test image
    if img_seed0 is None:
        data_set, data_loader = load_dataset(dataset_name, subset="test", 
                                             img_size=img_size, shuffle=False)
        img_seed = data_set[idx][0]
        y_seed = data_set[idx][1]
    else:
        img_seed0 = Image.fromarray(img_seed0, mode='RGB')
        transforms = tfs.Compose([tfs.Resize((img_size, img_size)), tfs.ToTensor()])
        img_seed = transforms(img_seed0)
        y_seed = y_seed0
        dataset_name = "uploaded"
        idx = 0

    # Set task and solver
    # cw_loss or cw_loss_2
    data_size = tuple(img_seed.shape)
    task = GeometricVarification(model, img_seed, data_size, y_seed, cw_loss, device, transf, **location_dist)
    object_func = task.set_problem()
    direct_solver = LowBoundedDIRECT_POset_full_parrallel(object_func, len(bound), bound, max_iter,
                                                        max_deep, max_eval, tol, po_set_size, debug=False)
    start_time = time.time()
    direct_solver.solve()
    end_time = time.time()

    # Get the optimal result and transform the image
    opt_theta = make_theta(transf, direct_solver.optimal_result())
    optimal_transf = AffineTransf(opt_theta)
    optimal_img = optimal_transf(img_seed.unsqueeze(0))
    # save the result image
    img_seed_pil = tfs.ToPILImage()(img_seed.squeeze(0))
    optimal_img_pil = tfs.ToPILImage()(optimal_img.squeeze(0))
    output_dir = os.path.join(SAVE_DIR, 'sample_images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path0 = os.path.join(output_dir, f'img_seed_{dataset_name}_{idx}.png')
    img_seed_pil.save(save_path0)
    save_path = os.path.join(output_dir, f'after_geo_{dataset_name}_{idx}.png')
    optimal_img_pil.save(save_path)

    # get the confidence after georobust
    opt_out = model(optimal_img.to(device))
    logger.info(f'The confidence now on class 1 is {opt_out.item():.4f}')

    post_correctness = (opt_out > 0.5).long().item() == y_seed
    logger.info(f'{dataset_name}:{idx}, post_correctness: {post_correctness}')
    logger.info(f'rcd_min: {direct_solver.rcd.minimum:.6f}, rcd_last_center: {direct_solver.rcd.last_center + 1}, rcd_best_idx: {direct_solver.rcd.best_idx}')
    logger.info(f'local_low_bound: {direct_solver.local_low_bound:.6f}, largest_slope: {direct_solver.get_largest_slope():.1f}, opt_size: {direct_solver.get_opt_size()}, largest_po_size: {direct_solver.get_largest_po_size()}')
    logger.info(f'time spent: {(end_time - start_time):.2f}')
    logger.info(f'optimal_result: {list(direct_solver.optimal_result())}')

    return optimal_img_pil, f"True label: {y_seed}. Predicted label: {(opt_out > 0.5).long().item()}. Confidence: {opt_out.item():.4f}"

def stop_fn():
    """
    This function writes a message indicating the cancel of the optimization process.
    """
    logger = create_logger("geo_robust", os.path.join(SAVE_DIR, "geo_robust.log"),
                       remove_old=False)
    logger.info("The solver's optimization process is stopped.")

def reset_fn():
    """
    This function resets the optimization process.
    """
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR)
        logger = create_logger("geo_robust", os.path.join(SAVE_DIR, "geo_robust.log"))
        logger.info("Reset GeoRobust.")
