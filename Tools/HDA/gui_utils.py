import torch
import os
import gradio as gr
import pandas as pd
import shutil

from Tools.HDA.hda_utils import select_seeds, generate_aes, eval_aes
from Tools.Utils.logger import create_logger
from Tools.HDA.model_train import train_vae, ad_train_model
from Tools.Utils.dataloader import load_dataset, load_classes
from Tools.Models.vae import load_vae
from Tools.Models.utils import load_model
from Tools.HDA.yolo_utils import draw_bounding_boxes

ATTACKER_NAMES_MAP = {"PGD whitebox": "pgd", "Random noise": "rand_noise",
                      "FGSM": "fgsm", "No attack": None}

### VAE TRAIN
def start_train_vae_fn(dataset_name, img_size, num_channel, hidden_dim, z_dim, 
                       upload_pth, optimizer, lr, weight_decay, 
                       batch_size, n_epochs, device, 
                       progress=gr.Progress(track_tqdm=True)):
    """
    This function trains the VAE model.
    """
    # Build logger
    os.makedirs(os.path.join("Results", "vae"), exist_ok=True)
    logger = create_logger("vae_train", os.path.join("Results", "vae", "vae_train.log"), 
                           remove_old=False)
    # Load dataset
    train_set, train_loader = load_dataset(dataset_name, subset="train", batch_size=batch_size, 
                                       img_size=img_size,)
    val_set, val_loader = load_dataset(dataset_name, subset="val", batch_size=batch_size,
                                        img_size=img_size)
    # Initialize VAE model
    vae = load_vae(num_channel, hidden_dim, z_dim, weight_path=upload_pth, device=device)

    # Start training
    train_vae(train_loader, val_loader, vae, optimizer, lr=lr, weight_decay=weight_decay, 
              n_epochs=n_epochs, logger=logger, device=device)

def get_vae_train_report_fn(dataset_text):
    """
    This function returns the VAE training log and the loss-epoch & sample reconstruction figures.
    """
    content = ""
    filepath = os.path.join("Results", "vae", "vae_train.log")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
    vae_loss = os.path.join("Results", "vae", f"vae_loss_{dataset_text}.png")
    if not os.path.exists(vae_loss):
        vae_loss = None
    vae_recons = os.path.join("Results", "vae", f"vae_reconstructions_{dataset_text}.png")
    if not os.path.exists(vae_recons):
        vae_recons = None
    return content, vae_loss, vae_recons

def stop_train_vae_fn():
    """
    This function writes a stop signal to the VAE training log.
    """
    logger = create_logger("vae_train", os.path.join("Results", "vae", "vae_train.log"), 
                           remove_old=False)
    logger.info("VAE training cancelled.")

def gen_recon_fn(dataset_name, num_channel, hidden_dim, z_dim, weights, img_size, batch_size,
                 device="cpu",
                  progress=gr.Progress(track_tqdm=True)):
    """
    This function generates image reconstructions using the VAE model.
    """
    save_dir = os.path.join("Results", "vae")
    # Build logger
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger("vae_train", os.path.join("Results", "vae", "vae_train.log"), 
                           remove_old=False)
    # Load dataset
    val_set, val_loader = load_dataset(dataset_name, subset="val", batch_size=batch_size,
                                        img_size=img_size)    
    # Initialize VAE model
    vae = load_vae(num_channel, hidden_dim, z_dim, device=device)

    # load from last time trained file
    if weights != None:
        vae.load_state_dict(torch.load(weights, map_location=device))
        logger.info(f"Loading pretrained weight {weights}.")
    else:
        vae.load_state_dict(torch.load(os.path.join("Checkpoints", f"{dataset_name}_vae.pt"), 
                                         map_location=device))
        logger.info(f"Loading default weight {os.path.join('Checkpoints', f'{dataset_name}_vae.pt')}.")

    logger.info("Generating reconstructions.")
    vae.generate_reconstructions(val_loader, save_dir=save_dir, device=device)

def get_vae_weight_fn(dataset_name):
    """
    This function extracts the trained VAE weights.
    """
    files = []
    save_dir = os.path.join("Results", "vae")
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            if (f.endswith(".pt")) and (dataset_name in f):
                files.append(os.path.join(save_dir, f))
    return files

def reset_vae_train_fn():
    save_dir = os.path.join("Results", "vae")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        logger = create_logger("vae_train", os.path.join(save_dir, "vae_train.log"))
        logger.info("Reset VAE training results.")


### HDA TEST
def update_hda_report_fn(dataset_text):
    """
    This function updates the HDA test logs, reads and displays the HDA test evaluation table.
    """
    save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
    # Read the HDA test log
    content = ""
    report_filepath = os.path.join(save_dir, "hda_test.log")
    if os.path.exists(report_filepath):
        with open(report_filepath, 'r') as f:
            content = f.read()
    # Read the HDA test evaluation table
    eval_filepath = os.path.join(save_dir, "hda_eval.csv")
    df_hda_eval = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], [], [], [], []], 
                               names=["model_arch", "model_pth", "density_mdl", "local_op", "eps"]))
    if os.path.exists(eval_filepath):
        df_hda_eval = pd.read_csv(eval_filepath)
    return content, df_hda_eval

def select_seeds_fn(img_size, num_channel, hidden_dim, z_dim, vae_weight_text,
                        model_cfg_text, model_pth_text, conf_thres, nms_thres,
                        batch_size, rand_seed, n_seeds, density_mdl, local_op, eps,
                        dataset_text, device,
                        progress=gr.Progress(track_tqdm=True)):
    """
    This function samples seeds for the HDA test (based on density estimation).
    """
    # Get eval records
    save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
    seeds_dir = os.path.join(save_dir, 
                             f"{model_cfg_text.replace('.cfg', '')}_{model_pth_text.replace('.pth', '')}",
                             f"seeds_{density_mdl}")
    os.makedirs(seeds_dir, exist_ok=True)

    df_hda_eval = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], [], [], [], []], 
                               names=["model_arch", "model_pth", "density_mdl", "local_op", "eps"]))
    if os.path.exists(os.path.join(save_dir, "hda_eval.csv")):
        df_hda_eval = pd.read_csv(os.path.join(save_dir, "hda_eval.csv")).set_index(
                                  ["model_arch", "model_pth", "density_mdl", "local_op", "eps"]).copy()
    else:
        df_hda_eval.to_csv(os.path.join(save_dir, "hda_eval.csv"))

    # Build logger
    logger = create_logger("hda_test", os.path.join(save_dir, "hda_test.log"), remove_old=False)
    logger.info(f"-> Sampling seeds.")
    logger.info("Dataset: {}\n".format(dataset_text))
    logger.info(f"Model: {model_cfg_text}, Weight: {model_pth_text}")
    logger.info('Total No. of test seeds: {}\n'.format(n_seeds))
    # Load dataset
    test_set, test_loader = load_dataset(dataset_text, subset="test", batch_size=batch_size, 
                                         img_size=img_size)
    # Load DNN model
    model_cfg_path = os.path.join("Checkpoints", model_cfg_text)
    model_weight_path = os.path.join("Checkpoints", model_pth_text)
    model = load_model(model_cfg_path, model_weight_path, device=device)
    # Initialize VAE model
    vae_weight_path = os.path.join("Checkpoints", vae_weight_text)
    vae = load_vae(num_channel, hidden_dim, z_dim, weight_path=vae_weight_path, device=device)

    # Select seeds
    rlt = select_seeds(vae, model, test_loader, n_seeds, density_mdl, rand_seed,
                        conf_thres=conf_thres, nms_thres=nms_thres, logger=logger, 
                        device=device, save_dir=seeds_dir)
    # Save eval results
    df_hda_eval.at[(f"{model_cfg_text}", f"{model_pth_text}", f"{density_mdl}", f"{local_op}", eps), 
                   "prob_density"] = rlt["prob_density"]
    df_hda_eval.to_csv(os.path.join(save_dir, "hda_eval.csv"))

def generate_aes_fn(model_cfg_text, model_pth_text, eps,
                    n_particles, n_mate, max_itr, alpha, conf_thres, nms_thres, 
                    dataset_text, local_op, density_mdl, batch_size, device,
                    progress=gr.Progress(track_tqdm=True)):
    """
    This function generates adversarial examples using the Genetic Algorithm.
    """
    save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
    seeds_dir = os.path.join(save_dir, 
                             f"{model_cfg_text.replace('.cfg', '')}_{model_pth_text.replace('.pth', '')}", 
                             f"seeds_{density_mdl}")

    df_hda_eval = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], [], [], [], []], 
                               names=["model_arch", "model_pth", "density_mdl", "local_op", "eps"]))
    if os.path.exists(os.path.join(save_dir, "hda_eval.csv")):
        df_hda_eval = pd.read_csv(os.path.join(save_dir, "hda_eval.csv")).set_index(
            ["model_arch", "model_pth", "density_mdl", "local_op", "eps"]).copy()
    else:
        df_hda_eval.to_csv(os.path.join(save_dir, "hda_eval.csv"))

    # Build logger
    logger = create_logger("hda_test", os.path.join(save_dir, "hda_test.log"),
                           remove_old=False)

    logger.info(f"-> Generating AEs. Model: {model_cfg_text}, Weight: {model_pth_text}.")
    logger.info(f"Dataset: {dataset_text}")
    ga_params = dict(n_particles=n_particles, n_mate=n_mate, max_itr=max_itr, alpha=alpha)

    # Load DNN model
    model_cfg_path = os.path.join("Checkpoints", model_cfg_text)
    model_pth_path = os.path.join("Checkpoints", model_pth_text)
    model = load_model(model_cfg_path, model_pth_path, device=device)
    # Generate AEs
    rlt = generate_aes(seeds_dir, model, eps, local_op, ga_params, 
                        conf_thres=conf_thres, nms_thres=nms_thres,
                        batch_size=batch_size, device=device, remove_old=True, logger=logger)
    # Save eval results
    df_hda_eval.at[(f"{model_cfg_text}", f"{model_pth_text}", f"{density_mdl}", f"{local_op}", eps), 
                   "attack_success_rate"] = rlt["attack_success_rate"]
    df_hda_eval.at[(f"{model_cfg_text}", f"{model_pth_text}", f"{density_mdl}", f"{local_op}", eps), 
                   "average_prediction_loss"] = rlt["average_prediction_loss"]
    df_hda_eval.to_csv(os.path.join(save_dir, "hda_eval.csv"))

def hda_eval_fn(model_cfg_text, model_pth_text, dataset_text, local_op, eps, density_mdl, 
                device, progress=gr.Progress(track_tqdm=True)):
    """
    This function evaluates the adversarial examples generated by the HDA test.
    """
    save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
    seeds_dir = os.path.join(save_dir, 
                             f"{model_cfg_text.replace('.cfg', '')}_{model_pth_text.replace('.pth', '')}", 
                             f"seeds_{density_mdl}")
    aes_dir =  os.path.join(seeds_dir.replace("seeds", "AEs"), f"{local_op}_{eps}")

    df_hda_eval = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], [], [], [], []], 
                                names=["model_arch", "model_pth", "density_mdl", "local_op", "eps"]))
    if os.path.exists(os.path.join(save_dir, "hda_eval.csv")):
        df_hda_eval = pd.read_csv(os.path.join(save_dir, "hda_eval.csv")).set_index(["model_arch", "model_pth", "density_mdl", "local_op", "eps"]).copy()
    else:
        df_hda_eval.to_csv(os.path.join(save_dir, "hda_eval.csv"))
    # Build logger
    logger = create_logger("hda_test", os.path.join(save_dir, "hda_test.log"),
                           remove_old=False)
    logger.info(f"-> Evaluating AEs. Model: {model_cfg_text}, Weight: {model_pth_text}.")
    logger.info(f"Dataset: {dataset_text}")

    # Evaluate AEs
    rlt = eval_aes(seeds_dir, aes_dir, logger=logger, device=device)
    
    # Record eval results
    df_hda_eval.at[(f"{model_cfg_text}", f"{model_pth_text}", f"{density_mdl}", f"{local_op}", eps), "epsilon"] = \
        rlt["avg_perb_amount"]
    df_hda_eval.at[(f"{model_cfg_text}", f"{model_pth_text}", f"{density_mdl}", f"{local_op}", eps), "fid"] = rlt["fid"]
    df_hda_eval.to_csv(os.path.join(save_dir, "hda_eval.csv"))

def _get_images(path):
    """
    This function returns the images in a given directory.
    """
    images = []
    if os.path.exists(path):
        for file in sorted(os.listdir(path)):
            if file.endswith(".png"):
                images.append((os.path.join(path, file), file))
    return images

def draw_img_seeds_fn(dataset_text, model_cfg_text, model_pth_text, density_mdl):
    """
    This function loads the seed images from target directory.
    """
    save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
    seeds_dir = os.path.join(save_dir, 
                             f"{model_cfg_text.replace('.cfg', '')}_{model_pth_text.replace('.pth', '')}",
                             f"seeds_{density_mdl}")
    seeds = _get_images(seeds_dir)
    return seeds

def draw_img_aes_fn(dataset_text, density_mdl, model_cfg_text, model_pth_text, local_op, eps):
    """
    This function loads the adversarial examples (AEs) from target directory.
    """
    save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
    aes_dir = os.path.join(save_dir, 
                            f"{model_cfg_text.replace('.cfg', '')}_{model_pth_text.replace('.pth', '')}",
                            f"AEs_{density_mdl}", f"{local_op}_{eps}")
    aes = _get_images(aes_dir)
    return aes

def _draw_box(img_dir, out_dir, classes, model, conf_thres, nms_thres, 
              batch_size=1, device="cpu"):
    """
    This function draws bounding boxes on the images of a specified directory.
    """
    # Skip the drawing if the files already exist
    if os.path.exists(out_dir):
        images_origin = sorted(os.listdir(img_dir))
        if sorted(os.listdir(out_dir)) == images_origin:
            return None
        else:
            shutil.rmtree(out_dir)
    # Detect objects and draw bounding boxes
    draw_bounding_boxes(img_dir, classes, model, conf_thres, nms_thres, 
                        batch_size=batch_size, out_dir=out_dir, device=device)

def draw_seeds_bd_fn(model_cfg_text, model_pth_text, dataset_text, density_mdl, 
                   conf_thres, nms_thres, batch_size, device):
    """
    This function draws bounding boxes on the seed images.
    """
    if "yolo" in model_cfg_text.lower():
        model_cfg_path = os.path.join("Checkpoints", model_cfg_text)
        model_pth_path = os.path.join("Checkpoints", model_pth_text)
        model = load_model(model_cfg_path, model_pth_path)

        save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
        seeds_dir = os.path.join(save_dir, 
                                f"{model_cfg_text.replace('.cfg', '')}_{model_pth_text.replace('.pth', '')}",
                                f"seeds_{density_mdl}")
        out_dir = seeds_dir+"_bd"
        classes = load_classes(os.path.join("Dataset", dataset_text, "classes.names"))
        _draw_box(seeds_dir, out_dir, classes, model, conf_thres, nms_thres, batch_size, device=device)
        seeds_bd = _get_images(out_dir)
        return seeds_bd
    else:
        return []

def draw_aes_bd_fn(model_cfg_text, model_pth_text, dataset_text, density_mdl, 
                   conf_thres, nms_thres, local_op, eps, batch_size, device):
    """
    This function draws bounding boxes on the adversarial examples (AEs).
    """
    if "yolo" in model_cfg_text.lower():
        model_cfg_path = os.path.join("Checkpoints", model_cfg_text)
        model_pth_path = os.path.join("Checkpoints", model_pth_text)
        model = load_model(model_cfg_path, model_pth_path)

        save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
        aes_dir = os.path.join(save_dir, 
                                f"{model_cfg_text.replace('.cfg', '')}_{model_pth_text.replace('.pth', '')}",
                                f"AEs_{density_mdl}", f"{local_op}_{eps}")
        out_dir = aes_dir+"_bd"
        classes = load_classes(os.path.join("Dataset", dataset_text, "classes.names"))
        _draw_box(aes_dir, out_dir, classes, model, conf_thres, nms_thres, batch_size, device=device)
        aes_bd = _get_images(out_dir)
        return aes_bd
    else:
        return []

def stop_hda_test_fn(dataset_text):
    """
    This function writes a stop signal to the HDA test log.
    """
    save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
    logger = create_logger("hda_test", os.path.join(save_dir, "hda_test.log"), 
                           remove_old=False)
    logger.info("Test cancelled.")

def reset_hda_test_fn(dataset_text):
    """
    This function removes all files in the target directory and writes a reset signal to the HDA test log.
    """
    save_dir = os.path.join("Results", "hda_test", f"{dataset_text}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        logger = create_logger("hda_test", os.path.join(save_dir, "hda_test.log"))
        logger.info(f"Reset HDA test results for {dataset_text}.")
        df_hda_eval = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], [], [], [], []], 
                                names=["model_arch", "model_pth", "density_mdl", "local_op", "eps"]))
        df_hda_eval.to_csv(os.path.join(save_dir, "hda_eval.csv"))
        
### ADV TRAIN MODEL
def update_adv_train_report(model_cfg):
    """
    This function reads the adversarial training log.
    """
    content = ""
    filepath = os.path.join("Results", "adv_train", 
                            f"adv_train_{model_cfg.replace('.cfg', '')}.log")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
    return content
def adv_train_fn(model_cfg, model_pth, dataset_name, img_size, batch_size, lr,
                 n_epochs, verbose, n_cpu, ckpt_interval, 
                multiscale, eval_interval, iou_thres, conf_thres, nms_thres, attacker, optimizer,
                eps, step_size, num_steps, rand_seed, device):
    """
    This function starts adversarial training for the DNN model.
    """
    save_dir = os.path.join("Results", "adv_train")
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger("adv_train", os.path.join(save_dir, 
                            f"adv_train_{model_cfg.replace('.cfg', '')}.log"))
    logger.info(f"Start adversarial training for model {model_cfg}. Weight: {model_pth}. Dataset: {dataset_name}.")
    
    # Load dataset
    train_set, train_loader = load_dataset(dataset_name, subset="train", batch_size=batch_size, 
                                        multiscale=multiscale, img_size=img_size, num_workers=n_cpu)
    val_set, val_loader = load_dataset(dataset_name, subset="val", batch_size=batch_size,
                            multiscale=multiscale, img_size=img_size, num_workers=n_cpu) 
    
    # Load DNN model
    model_cfg = os.path.join("Checkpoints", model_cfg) # choose from [yolov3-tiny.cfg, inception_v3]
    # model_pth = os.path.join("Checkpoints", model_pth) # choose from [yolov3_ckpt_297.pth, inception_v3_wo_norm_448.pth]
    print(model_pth)
    model = load_model(model_cfg, model_pth, device=device)
    # Start adv training
    ad_train_model(model, train_loader, val_loader, n_epochs, optimizer=optimizer,
                   ckpt_interval=ckpt_interval, lr=lr,
                   eval_interval=eval_interval,
                   attacker=ATTACKER_NAMES_MAP[attacker], eps=eps, step_size=step_size, num_steps=num_steps,
                   iou_thres=iou_thres, conf_thres=conf_thres, nms_thres=nms_thres, 
                   device=device, verbose=verbose, rand_seed=rand_seed, logger=logger)    
def get_adv_model_weight_fn(attacker):
    """
    This function extracts the AT model weights.
    """
    file_paths = []
    _attacker = ATTACKER_NAMES_MAP[attacker]
    save_dir = os.path.join("Results", "adv_train")
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            if _attacker in filename:
                file_paths.append(os.path.join(save_dir, filename))
    return sorted(file_paths, key=lambda p: int(p.split("_")[-1].replace(".pth", "")), reverse=True)
def remove_adv_model_weight_fn(attacker):
    """
    This function removes all AT model weights.
    """
    _attacker = ATTACKER_NAMES_MAP[attacker]
    save_dir = os.path.join("Results", "adv_train")
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            if _attacker in filename:
                os.remove(os.path.join(save_dir, filename))
    return get_adv_model_weight_fn(attacker)