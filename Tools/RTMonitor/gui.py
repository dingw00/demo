import torch
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog
setup_logger()

import numpy as np
import time
import gradio as gr
import pandas as pd
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .detectron2monitor import Detectron2Monitor

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

with gr.Blocks(theme='soft') as demo:
    gr.Markdown("# Runtime Monitoring Computer Vision Models")
    gr.Markdown(
        """
This interactive demo presents an approach to monitoring neural networks-based computer vision models using box abstraction-based techniques. Our method involves abstracting features extracted from training data to construct monitors. The demo walks users through the entire process, from monitor construction to evaluation. 
The interface is divided into several basic modules:

- **In-distribution dataset and backbone**: This module allows users to select their target model and dataset.
- **Feature extraction**: Neuron activation pattern are extracted from the model's intermediate layers using training data. These features represent the good behaviors of the model.
- **Monitor construction**: Extracted features are grouped using different clustering techniques. These clusters are then abstracted to serve as references for the monitors. 
- **Evaluation preparation**: To facilate the evalution, the features should be extracted from evaluation datasets prior to monitor evalution. 
- **Monitor Evaluation**: The effectiveness of monitors in detecting Out-of-Distribution (OoD) objects are assessed. One of our core metric is FPR 95, which represents the false positive (incorrectly detected objects) rate when the true positive rate for ID is set at 95%. 
    """
    )
    with gr.Tab("Object Detection"):
        id = gr.Radio(['voc', 'bdd', 'kitti', 'nu'], label="Dataset")
        backbone = gr.Radio(['regnet', 'resnet'], label="Backbone")
        with gr.Tab("Feature extraction"):
            extract_btn = gr.Button("Extract features")
            output1 = gr.Textbox(label="Output")
        with gr.Tab("Monitor construction"):
            construct_btn = gr.Button("Monitor Construction")
            tau = gr.Number(value=0.001, label="Tau")
            output2 = gr.Textbox(label="Output")
            df_construct = gr.Dataframe(type="pandas", label="Monitor construction summary")
        with gr.Tab("Evaluation preparation"):
            prep_btn = gr.Button("Evaluation Data Preparation")
            prep_output = gr.Textbox(label="Output")
        with gr.Tab("Evaluation results"):
            tau_eval = gr.Number(value=0.001, label="Tau for Evaluation")
            eval_btn = gr.Button("Monitor Evaluation")
            gr.Markdown("## ID performance")
            eval_id = gr.Dataframe(type="pandas")
            gr.Markdown("## OOD performance")
            with gr.Tab("Overall"):
                eval_ood = gr.Dataframe(type="pandas")
            with gr.Tab("COCO-ODD"):
                eval_coco = gr.Dataframe(type="pandas")
            with gr.Tab("OpenImages-ODD"):
                eval_open = gr.Dataframe(type="pandas")
            with gr.Tab("VOC-ODD"):
                eval_voc = gr.Dataframe(type="pandas")
    extract_btn.click(fn=fx_gradio, inputs=[id, backbone], outputs=[output1])
    construct_btn.click(fn=construct_gradio, inputs=[id, backbone, tau], outputs=[output2, df_construct])
    prep_btn.click(fn=fx_eval_gradio, inputs=[id, backbone], outputs=[prep_output])
    eval_btn.click(fn=eval_gradio, inputs=[id, backbone, tau_eval], outputs=[eval_id, eval_ood, eval_coco, eval_open, eval_voc])

def render_gui():
    return demo