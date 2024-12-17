import unittest

import os
import torch
from Tools.Utils.dataloader import load_dataset
from Tools.Models.utils import load_model
from Tools.Models.vae import load_vae, VAE

from Tools.HDA.model_train import train_vae, ad_train_model
from Tools.HDA.hda_utils import hda_test

@unittest.skip("Skip TestTrainVAE")
class TestTrainVAE(unittest.TestCase):
    def setUp(self):
        self.optimizer = "adam"
        self.n_epochs = 2
        self.lr = 1e-3
        self.weight_decay = 0

        self.input_dim = 3
        self.hidden_dim = 256
        self.z_dim = 4

        self.batch_size = 5
        self.multiscale = False
        self.n_cpu = 1
    def test_train_vae_coco128(self):
        # Load dataset
        dataset_name = "coco128" # choose from ["coco128", "railway_track_fault_detection"]
        img_size = 256 # choose from [256, 448]
        train_set, train_loader = load_dataset(dataset_name, subset="train", batch_size=self.batch_size, 
                                            multiscale=self.multiscale, img_size=img_size, num_workers=self.n_cpu)
        val_set, val_loader = load_dataset(dataset_name, subset="val", batch_size=self.batch_size,
                                   multiscale=self.multiscale, img_size=img_size, num_workers=self.n_cpu) 
        for device in ["cpu"]: # "cuda"
            with self.subTest(device=device):
                vae = load_vae(self.input_dim, self.hidden_dim, self.z_dim, device=device)
                train_vae(train_loader, val_loader, vae, optimizer=self.optimizer, lr=self.lr, 
                        weight_decay=self.weight_decay, n_epochs=self.n_epochs, device=device)
    def test_train_vae_railway(self):
        # Load dataset
        dataset_name = "railway_track_fault_detection" # choose from ["coco128", "railway_track_fault_detection"]
        img_size = 448 # choose from [256, 448]
        train_set, train_loader = load_dataset(dataset_name, subset="train", batch_size=self.batch_size, 
                                            multiscale=self.multiscale, img_size=img_size, num_workers=self.n_cpu)
        val_set, val_loader = load_dataset(dataset_name, subset="val", batch_size=self.batch_size,
                                   multiscale=self.multiscale, img_size=img_size, num_workers=self.n_cpu) 
        for device in ["cpu"]: # "cuda"
            with self.subTest(device=device):
                vae = load_vae(self.input_dim, self.hidden_dim, self.z_dim, device=device)
                train_vae(train_loader, val_loader, vae, optimizer=self.optimizer, lr=self.lr, 
                        weight_decay=self.weight_decay, n_epochs=self.n_epochs, device=device)

# @unittest.skip("Skip TestHDATest")
class TestHDATest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.n_cpu = 1
        self.n_seeds = 5
    def test_hda_test_yolo(self):

        dataset_name = "coco128" # choose from ["coco128", "railway_track_fault_detection"]
        img_size = 256 # choose from [256, 448]
        eps = 8/255
        ga_params = dict(n_particles=100, n_mate=20, max_itr=10, alpha=1.00)
        conf_thres = 0.1
        nms_thres = 0.5

        # VAE encoder params
        input_dim = 3
        hidden_dim = 256
        z_dim = 4

        test_set, test_loader = load_dataset(dataset_name, subset="val", batch_size=self.batch_size, 
                                             img_size=img_size, num_workers=self.n_cpu)

        for device in ["cuda", "cpu"]: # "cuda"
            if device == "cuda":
                torch.cuda.empty_cache()
            # Load DNN model
            model_cfg = os.path.join("Checkpoints", "yolov3-tiny.cfg") # choose from [yolov3-tiny.cfg, inception_v3]
            model_pth = os.path.join("Checkpoints", "yolov3_ckpt_297.pth") # choose from [yolov3_ckpt_297.pth, inception_v3_wo_norm_448.pth]
            model = load_model(model_cfg, model_pth, device=device)
            # Load VAE model
            vae_pth = os.path.join("Checkpoints", "coco128_vae.pt") # coco128_vae.pt, railway_track_fault_detection_vae.pt
            vae = load_vae(input_dim, hidden_dim, z_dim, weight_path=vae_pth, device=device)

            for density_mdl in ["kde", "random"]:
                for local_op in ["mse", "psnr", "ms_ssim"]:
                    hda_test(vae, model, test_loader, eps=eps, n_seeds=self.n_seeds, density_mdl=density_mdl, 
                            local_op=local_op, conf_thres=conf_thres, nms_thres=nms_thres, 
                            ga_params=ga_params, batch_size=self.batch_size, device=device)
    def test_hda_test_inception(self):

        dataset_name = "railway_track_fault_detection" # choose from ["coco128", "railway_track_fault_detection"]
        img_size = 448 # choose from [256, 448]
        eps = 0.1
        ga_params = dict(n_particles=150, n_mate=30, max_itr=40, alpha=1.00)

        # VAE encoder params
        input_dim = 3
        hidden_dim = 256
        z_dim = 4

        test_set, test_loader = load_dataset(dataset_name, subset="val", batch_size=self.batch_size, 
                                             img_size=img_size, num_workers=self.n_cpu)

        for device in ["cpu"]: # , "cuda"
            if device == "cuda":
                torch.cuda.empty_cache()
            # Load DNN model
            model_cfg = os.path.join("Checkpoints", "inception_v3") # choose from [yolov3-tiny.cfg, inception_v3]
            model_pth = os.path.join("Checkpoints", "inception_v3_wo_norm_448.pth") # choose from [yolov3_ckpt_297.pth, inception_v3_wo_norm_448.pth]
            model = load_model(model_cfg, model_pth, device=device)
            # Load VAE model
            vae_pth = os.path.join("Checkpoints", "railway_track_fault_detection_vae.pt") # coco128_vae.pt, railway_track_fault_detection_vae.pt
            vae = load_vae(input_dim, hidden_dim, z_dim, weight_path=vae_pth, device=device)
            for density_mdl in ["kde", "random"]:
                for local_op in ["mse", "psnr", "ms_ssim"]:
                    hda_test(vae, model, test_loader, eps=eps, n_seeds=self.n_seeds, density_mdl=density_mdl, 
                             local_op=local_op, ga_params=ga_params, batch_size=self.batch_size, device=device)

# @unittest.skip("Skip TestAdvTrainModel")
class TestAdvTrainModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.multiscale = False
        self.n_cpu = 1
        self.n_epochs = 2
    def test_adv_train_yolo(self):

        conf_thres = 0.1
        nms_thres = 0.5
        iou_thres = 0.5

        eps = 8/255
        step_size = 8/255
        num_steps = 2

        # Load dataset
        dataset_name = "coco128" # choose from ["coco128", "railway_track_fault_detection"]
        img_size = 256 # choose from [256, 448]
        train_set, train_loader = load_dataset(dataset_name, subset="train", batch_size=self.batch_size, 
                                            multiscale=self.multiscale, img_size=img_size, num_workers=self.n_cpu)
        val_set, val_loader = load_dataset(dataset_name, subset="val", batch_size=self.batch_size,
                                multiscale=self.multiscale, img_size=img_size, num_workers=self.n_cpu) 
        for device in ["cpu", "cuda"]:
            with self.subTest(device=device):
                # Load DNN model
                model_cfg = os.path.join("Checkpoints", "yolov3-tiny.cfg") # choose from [yolov3-tiny.cfg, inception_v3]
                model_pth = os.path.join("Checkpoints", "yolov3_ckpt_297.pth") # choose from [yolov3_ckpt_297.pth, inception_v3_wo_norm_448.pth]
                model = load_model(model_cfg, model_pth, device=device)
                for attacker in ["pgd", "fgsm", "rand_noise"]:
                    ad_train_model(model, train_loader, val_loader, self.n_epochs,
                                    attacker=attacker, eps=eps, step_size=step_size, num_steps=num_steps,
                                    iou_thres=iou_thres, conf_thres=conf_thres, nms_thres=nms_thres, device=device)
    def test_adv_train_inception(self):
        
        eps = 0.05
        num_steps = 1
        step_size = 1/255

        # Load dataset
        dataset_name = "railway_track_fault_detection" # choose from ["coco128", "railway_track_fault_detection"]
        img_size = 448 # choose from [256, 448]
        train_set, train_loader = load_dataset(dataset_name, subset="train", batch_size=self.batch_size, 
                                            multiscale=self.multiscale, img_size=img_size, num_workers=self.n_cpu)
        val_set, val_loader = load_dataset(dataset_name, subset="val", batch_size=self.batch_size,
                                multiscale=self.multiscale, img_size=img_size, num_workers=self.n_cpu) 
        for device in ["cpu", "cuda"]:
            with self.subTest(device=device):
                # Load DNN model
                model_cfg = os.path.join("Checkpoints", "inception_v3") # choose from [yolov3-tiny.cfg, inception_v3]
                model_pth = os.path.join("Checkpoints", "inception_v3_wo_norm_448.pth") # choose from [yolov3_ckpt_297.pth, inception_v3_wo_norm_448.pth]
                model = load_model(model_cfg, model_pth, device=device)
                for attacker in ["pgd"]:
                    optimizer = "adam"
                    lr = 1e-3

                    ad_train_model(model, train_loader, val_loader, self.n_epochs, optimizer=optimizer, lr=lr, 
                                attacker=attacker, eps=eps, step_size=step_size, num_steps=num_steps, device=device)

if __name__ == "__main__":
    # Redirect stdout to a file
    unittest.main(verbosity=2)