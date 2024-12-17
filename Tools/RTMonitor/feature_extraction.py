import torch
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import numpy as np
from detectron2.data import build_detection_test_loader, MetadataCatalog
import tqdm
from prettytable import PrettyTable
from collections import defaultdict
import time
import os

OUTPUT_FOLDER = os.path.join("Results/RTMonitor")
FEATS_FOLDER = os.path.join(OUTPUT_FOLDER, "feats")
from .data.setup_datasets import *

def feature_extraction(id, backbone, cfg, dataset_name, batch_size=1):
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    val_loader = build_detection_test_loader(cfg, dataset_name, batch_size=batch_size)
    label_list = MetadataCatalog.get(f"{id}_custom_train").thing_classes
    label_dict = {i: class_ for i, class_ in enumerate(label_list)}

    feats_label_dict = defaultdict(list)
    t0 = time.time()
    for batch in tqdm.tqdm(val_loader):
        feat_label, feats = model(batch)
        labels = [label_dict[feat_label[i]] for i in range(len(feat_label))]
        for i in range(len(labels)):
            feats_label_dict[labels[i]].append(feats[i])
    print(f"Total time: {time.time()-t0}s")

    
    # Create a table with headers
    table = PrettyTable(['Label', 'Count'])
    for k, v in feats_label_dict.items():
        v = np.array(v)
        feats_label_dict[k] = v
        table.add_row([k, v.shape[0]])
    print(table)

    # save feats_label_dict
    import pickle
    dir_path =os.path.join(FEATS_FOLDER, f'{id}/{backbone}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(f"{dir_path}/{dataset_name}.pkl", "wb") as f:
        pickle.dump(feats_label_dict, f)
    # save count table
    file = open(f"{dir_path}/{dataset_name}.txt", 'w')
    file.write(str(table))

    # Close the file
    file.close()

    print(f"Extracted features information saved to file: {dataset_name}.txt")