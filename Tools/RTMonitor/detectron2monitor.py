import torch
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
import gdown

import numpy as np
import os
import pickle
import tqdm
import pandas as pd
import sys
from prettytable import PrettyTable
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
from Tools.Models.regnet import build_regnet_fpn_backbone
from .data.setup_datasets import *

# from utils_clustering import *
from .feature_extraction import feature_extraction
from .monitor_construction import features_clustering, monitor_construction_from_features

benchmark = {"kitti-resnet":[16.02, 7.07, 18.01], "kitti-regnet":[12.51, 5.31, 14.91], "voc-resnet":[47.47, 55.48], "voc-regnet":[47.75, 52.65], "bdd-resnet": [46.19, 35.84, 49.01], "bdd-regnet":[36.95, 25.94, 40.35], "nu-resnet":[33.48, 19.00, 45.49], "nu-regnet":[33.17, 15.20, 47.68]}
weights = {"model_final_kitti_resnet.pth":"https://drive.usercontent.google.com/download?id=1_0zqxtxvzZ_ApURotSYOC9YduS9CwTyD&export=download&authuser=0&confirm=t&uuid=3e2b7ba1-9ef1-465d-93c7-46515cd50241&at=APZUnTX_UZFYSXYvpJ4CqxB9uM_W:1710180925558",
           "model_final_kitti_regnet.pth": "https://drive.usercontent.google.com/download?id=1KJamkVG1g2Hh0AHo_0pJ9Avv07909i5T&export=download&authuser=0&confirm=t&uuid=76b3dc04-0d9e-40b7-94f9-c955b2edcf7c&at=APZUnTX4_0U_jd59AKCx6uek0MuC:1710181012335"}

DATA_FOLDER = os.path.join("Dataset")
WEIGHT_FOLDER = os.path.join("Checkpoints")
OUTPUT_FOLDER = os.path.join("Results/RTMonitor")
FEATS_FOLDER = os.path.join(OUTPUT_FOLDER, "feats")
MONITORS_FOLDER = os.path.join(OUTPUT_FOLDER, "monitors")

DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAND_SEED = 0

class Detectron2Monitor():
    def __init__(self, id, backbone):
        self.id = id
        self.backbone = backbone
        self.eval_list = [['openimages_ood_val', f'voc_custom_val','coco_ood_val']] if self.id == "voc" else ['openimages_ood_val','voc_ood_val',f'{self.id}_custom_val','coco_ood_val_bdd']
        if not os.path.exists(WEIGHT_FOLDER):
            os.makedirs(WEIGHT_FOLDER)
        if not os.path.exists(os.path.join(WEIGHT_FOLDER, f"model_final_{self.id}_{self.backbone}.pth")):
            gdown.download(weights[f"model_final_{self.id}_{self.backbone}.pth"], 
                           os.path.join(WEIGHT_FOLDER, f"model_final_{self.id}_{self.backbone}.pth"))

    def _get_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(f"Tools/Models/configs/FX_{self.id}_{self.backbone}.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(WEIGHT_FOLDER, f"model_final_{self.id}_{self.backbone}.pth")
        cfg.MODEL.DEVICE='cuda' if torch.cuda.is_available() else "cpu"
        return cfg
    
    def _setup_dataset(self):
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
        path_dataset = {'bdd_custom_val': os.path.join(DATA_FOLDER, "bdd"),
                        'kitti_custom_val': os.path.join(DATA_FOLDER, "kitti"),
                        'kitti_custom_train': os.path.join(DATA_FOLDER, "kitti"),
                        'voc_ood_val': os.path.join(DATA_FOLDER, "voc-ood"),
                        'openimages_ood_val': os.path.join(DATA_FOLDER, "OpenImages"),
                        'coco_ood_val_bdd': os.path.join(DATA_FOLDER, "coco-2017")}
        DATASET_SETUP_FUNCTIONS = {
        'bdd_custom_val': setup_bdd_dataset,
        'kitti_custom_train': setup_kitti_dataset,
        # 'kitti_custom_val': setup_kitti_dataset,
        'coco_ood_val_bdd': setup_coco_ood_bdd_dataset,
        'openimages_ood_val': setup_openim_odd_dataset,
        'voc_ood_val': setup_voc_ood_dataset
        }
        for dataset_name in DATASET_SETUP_FUNCTIONS:
            setup_dataset = DATASET_SETUP_FUNCTIONS[dataset_name]
            setup_dataset(path_dataset[dataset_name])
    
    def _feature_extraction(self, dataset_name, batch_size=1):
        self.cfg = self._get_cfg()
        return feature_extraction(self.id, self.backbone, self.cfg, dataset_name, batch_size)
    
    def _construct(self, tau):
        with open(os.path.join(FEATS_FOLDER, 
                               f"{self.id}/{self.backbone}/{self.id}_custom_train.pkl"), 'rb') as f:
            feats_dict = pickle.load(f)
        dir_path = os.path.join(MONITORS_FOLDER, f"{self.id}/{self.backbone}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        monitor_dict = {}
        for class_, feats in tqdm.tqdm(feats_dict.items(), desc="Contructing monitors"):
            if feats.shape[0] == 0:
                continue
            clustering_results  = features_clustering(feats, [tau])
            monitor_dict[class_] = monitor_construction_from_features(feats, clustering_results)
        with open(f"{dir_path}/{tau}.pkl" , 'wb') as f:
            pickle.dump(monitor_dict, f)

        counts_instances = dict()
        for k, v in feats_dict.items():
            counts_instances[k] = v.shape[0]
        counts_clusters = dict()
        for k, v in monitor_dict.items():
            counts_clusters[k] = len(v.good_ref)
        # Create a PrettyTable object
        table = PrettyTable()

        # Add columns to the table
        table.field_names = ["category", "#instances", "#clusters"]
        data = []
        # Add rows to the table
        for label, feats in counts_instances.items():
            row = [label, feats, counts_clusters[label]]
            table.add_row(row)
            data.append(row)
        df = pd.DataFrame(data, columns=["category", "#instances", "#clusters"])
        table.sortby = "#instances"
        # Print the table
        filename = f"{dir_path}/{tau}.txt"
        with open(filename, 'w') as f:
            f.write(str(table))
        return df
    
    def _evaluate(self, tau):
        with open(os.path.join(MONITORS_FOLDER, f"{self.id}/{self.backbone}/{tau}.pkl"), 'rb') as f:
            monitors_dict = pickle.load(f)
        # ID evaluation
        dataset_name = f"{self.id}_custom_val"
        with open(os.path.join(FEATS_FOLDER, f"{self.id}/{self.backbone}/{dataset_name}.pkl"), 'rb') as f:
            feats_dict = pickle.load(f)
        classes = []
        nb_class = []
        acc_class = []
        acc_rate = []
        for k, v in feats_dict.items():
            if v.shape[0] == 0:
                continue
            classes.append(k)
            nb_class.append(v.shape[0])
            verdict = monitors_dict[k].make_verdicts(v)
            acc_class.append(np.sum(verdict))
            rate = round(np.sum(verdict)/len(verdict)*100, 2)
            acc_rate.append(rate)
        # make a dataframe
        df_id = pd.DataFrame(list(zip(classes, nb_class, acc_class, acc_rate)), 
                    columns =['class', f'nb', f'accepted', f'TPR(%)'])
        df_id.loc['Total']= df_id.sum(numeric_only=True, axis=0)
        df_id.loc['Total', 'class'] = 'Total'
        df_id.loc['Total',  f'TPR(%)'] = f"{df_id.loc['Total', f'accepted']/df_id.loc['Total', f'nb']*100:.2f}"
        df_id[f'nb'] = df_id[f'nb'].astype(int)
        df_id[f'accepted'] = df_id[f'accepted'].astype(int)
        
        # OOD evaluation
        df_per_ds = []
        data_ood = []
        i = 0
        eval_list = [eval_name for eval_name in self.eval_list if eval_name != dataset_name]
        for eval_name in tqdm.tqdm(eval_list, desc="Evaluation on OOD data"):
            accept_sum = 0
            total = 0
            classes = []
            nb_class = []
            acc_class = []
            acc_rate = []
            with open(os.path.join(FEATS_FOLDER, f"{self.id}/{self.backbone}/{eval_name}.pkl"), 'rb') as f:
                feats_dict = pickle.load(f)
            for label in monitors_dict:
                classes.append(label)
                feats = feats_dict[label]
                if len(feats) != 0:
                    nb_class.append(feats.shape[0])
                    verdict = monitors_dict[label].make_verdicts(feats)
                    acc_class.append(np.sum(verdict))
                    rate = round(np.sum(verdict)/len(verdict)*100, 2)
                    acc_rate.append(rate)
                    total += len(verdict)
                    accept_sum += np.sum(verdict)
                else:
                    nb_class.append(0)
                    verdict = []
                    acc_class.append(np.sum(verdict))
                    acc_rate.append(0)
                    total += len(verdict)
                    accept_sum += np.sum(verdict)
            # make a dataframe
            df = pd.DataFrame(list(zip(classes, nb_class, acc_class, acc_rate)), 
                columns =['class', f'nb', f'accepted', f'FPR (%)'])
            df.loc['Total']= df.sum(numeric_only=True, axis=0)
            df.loc['Total', 'class'] = 'Total'
            df.loc['Total',  f'FPR (%)'] = f"{df.loc['Total', f'accepted']/df.loc['Total', f'nb']*100:.2f}"
            df[f'nb'] = df[f'nb'].astype(int)
            df[f'accepted'] = df[f'accepted'].astype(int)
            df_per_ds.append(df)
            FPR =  round((accept_sum / total*100), 2)
            data_ood.append([eval_name, str(FPR)])
            i += 1
        # prepare dataframes
        df_ood = pd.DataFrame(data_ood, columns=["Dataset", "FPR(%)"])
        df_ood["Dataset"] = ["COCO", "Open Images"] if self.id == "voc" else ["COCO", "Open Images", "VOC-OOD"]
        df_ood["Benchmark(%)"] = benchmark[f"{self.id}-{self.backbone}"]
        return df_id, df_ood, df_per_ds[0], df_per_ds[1], df_per_ds[2]

    
