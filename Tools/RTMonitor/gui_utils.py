import time
import gradio as gr
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data import DatasetCatalog

from .detectron2monitor import Detectron2Monitor

##################### Runtime Monitor GUI functions #####################

def fx_gradio(id, backbone, progress=gr.Progress(track_tqdm=True)):
    d2m = Detectron2Monitor(id, backbone)
    if f'{id}_custom_train' not in DatasetCatalog.list():
        d2m._setup_dataset()
    t0 = time.time()
    d2m._feature_extraction(f'{id}_custom_train')
    minutes, seconds = divmod(time.time()-t0, 60)
    return f"Total feature extraction time: {int(minutes):02d}:{int(seconds):02d}"

def construct_gradio(id, backbone, tau, progress=gr.Progress(track_tqdm=True)):
    d2m = Detectron2Monitor(id, backbone)
    t0 = time.time()
    df = d2m._construct(tau)
    minutes, seconds = divmod(time.time()-t0, 60)
    return f"Total monitor construction time: {int(minutes):02d}:{int(seconds):02d}\n", df

def fx_eval_gradio(id, backbone, progress=gr.Progress(track_tqdm=True)):
    d2m = Detectron2Monitor(id, backbone)
    if f'{id}_custom_train' not in DatasetCatalog.list():
        d2m._setup_dataset()
    t0 = time.time()
    for dataset_name in progress.tqdm(d2m.eval_list, desc="EXtracting features from evaluation datasets"):
        d2m._feature_extraction(dataset_name)
    minutes, seconds = divmod(time.time()-t0, 60)
    return f"Total evaluation data preparation time: {int(minutes):02d}:{int(seconds):02d}"

def eval_gradio(id, backbone, tau, progress=gr.Progress(track_tqdm=True)):
    d2m = Detectron2Monitor(id, backbone)
    df_id, df_ood, df_coco, df_open, df_voc = d2m._evaluate(tau)
    return df_id, df_ood, df_coco, df_open, df_voc