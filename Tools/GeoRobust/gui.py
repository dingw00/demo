import gradio as gr
from Tools.GeoRobust.gui_utils import *

with gr.Blocks(theme="soft") as demo:
    with gr.Tab("Geo-Robust Attack"):
        with gr.Row():
            with gr.Column(min_width=400):
                # Model panel
                with gr.Column(variant="panel", min_width=400):
                    gr.Markdown(f"#### Model")                                              
                    with gr.Column(variant="compact"):
                        with gr.Row():
                            model_cfg = gr.Radio(["inception_v3"], label="Architecture", value="inception_v3")
                            model_weights = gr.File(label="Model weights", value="Checkpoints/inception_v3_wo_norm_448.pth",
                                                    interactive=True)
                # Solver settings panel
                with gr.Column(variant="panel", min_width=400):
                    gr.Markdown("#### Solver Settings")
                    with gr.Column(variant="compact"):
                        with gr.Row():
                            max_iter = gr.Number(label="Max iterations", value=20, minimum=1, precision=0, step=1, interactive=True)
                            max_deep = gr.Number(label="Max deep", value=6, minimum=0, precision=0, step=1, interactive=True)
                            max_eval = gr.Number(label="Max evaluation", value=5000, minimum=1, precision=0, step=1, interactive=True)
                            tol = gr.Number(label="Tolerance", value=1e-4, minimum=1e-6, precision=6, step=1e-6, interactive=True)
                            po_set_size = gr.Number(label="Population set size", value=2, minimum=1, precision=0, step=1, interactive=True)
                            dev = gr.Dropdown(label="Device", choices=["cpu", "cuda"], value="cpu",
                                            interactive=True)

            # Dataset panel
            with gr.Column(variant="panel", min_width=400):
                gr.Markdown("#### Original Image")
                img_size = gr.Number(label="Image size", value=224, minimum=10, precision=0, step=1, 
                                     min_width=50, interactive=True)
                with gr.Tab("Choose from dataset"):
                    with gr.Row():
                        idx = gr.Number(label="Index", value=0, minimum=0, precision=0, step=1, min_width=50, interactive=True)
                        data_set = gr.Radio(label="Dataset", choices=[ "railway_track_fault_detection", "coco128"], min_width=200,
                                                value="railway_track_fault_detection", interactive=True)
                with gr.Tab("Upload an image"):
                    img_seed0 = gr.Image(label="Image seed", interactive=True)
                    y_seed0 = gr.Number(label="Label", value=0, minimum=0, maximum=1, precision=0, step=1, interactive=True)
                test_model_btn = gr.Button(value="Test on Model")
                
        with gr.Row():
            # Transforms panel
            with gr.Column(variant="panel"):
                gr.Markdown("#### Transforms")
                with gr.Row(variant="compact"):
                    with gr.Row():
                        with gr.Row(variant="compact"):
                            rot = gr.Checkbox(label="Rotation", value=True, min_width=50)
                            max_rot = gr.Slider( minimum=0, maximum=180, value=90, step=1, 
                                                interactive=True, show_label=False, min_width=900)
                        with gr.Row(variant="compact"):
                            trans = gr.Checkbox(label="Translation", value=False, min_width=50)
                            max_trans = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01,
                                    interactive=True, show_label=False, min_width=900)
                        with gr.Row(variant="compact"):
                            scale = gr.Checkbox(label="Scale", value=False, min_width=50)
                            max_scale = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01,
                                    interactive=True, show_label=False, min_width=900)
            
        with gr.Row():
            start_btn = gr.Button(value="Start Attack", min_width=500)
            stop_btn =  gr.Button(value="Stop", min_width=50)
            reset_btn =  gr.Button(value="Reset", min_width=50)
        
        # Results panel
        with gr.Column(variant="panel", min_width=900):
            gr.Markdown("#### Results")
            with gr.Column(variant="compact"):
                with gr.Row():
                    with gr.Column(variant="compact"):
                        img_seed = gr.Image(label="Seed Image", interactive=False)
                        conf_seed = gr.Textbox(show_label=False, interactive=False)
                    with gr.Column(variant="compact"):
                        img_tfd = gr.Image(label="Geo-Trasformed Image", interactive=False)
                        conf_tfd = gr.Textbox(show_label=False, interactive=False)
                report = gr.TextArea(label="Report", lines=8, interactive=False)

        # Add functions
        timer_report = gr.Timer(3, active=False)
        timer_report.tick(update_report_fn, outputs=[report])
        test_model_process = test_model_btn.click(test_model_fn, inputs=[model_cfg, model_weights, img_size, data_set, idx, 
                                                    img_seed0, y_seed0, dev], outputs=[img_seed, conf_seed]).then(
                                                      lambda: gr.Timer(active=True), None, timer_report)   
        direct_process = start_btn.click(solve_direct_fn, inputs=[model_cfg, model_weights, img_size, data_set, idx, 
                                            img_seed0, y_seed0, max_iter, max_deep, max_eval, tol,
                                            po_set_size, dev, rot, max_rot, trans, max_trans, scale, max_scale],
                                            outputs=[img_tfd, conf_tfd]).then(
                                            lambda: gr.Timer(active=True), None, timer_report)
        stop_btn.click(stop_fn, queue=False, cancels=[test_model_process, direct_process]).then(
            lambda: gr.Timer(active=False), None, timer_report).then(
                        update_report_fn, outputs=[report])
        reset_btn.click(reset_fn, queue=False, cancels=[test_model_process, direct_process]).then(
            lambda: gr.Timer(active=False), None, timer_report).then(
                        update_report_fn, outputs=[report])

def render_gui():
    return demo