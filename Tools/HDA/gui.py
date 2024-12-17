import gradio as gr
from Tools.HDA.gui_utils import *

##################### Layout #####################
dev = gr.Dropdown(label="Device", choices=["cpu", "cuda"], value="cpu", render=False)

with gr.Blocks(theme="soft") as demo:
    with gr.Row():
        with gr.Column("HDA Attacker", min_width=1350):
            with gr.Tab("Inroduction to HDA") as hda_intro:
                gr.Markdown("# Hierarchical Distribution-Aware Testing of Deep Learning")
                gr.Markdown("The Hierarchical Distribution-Aware (HDA) testing implements the \
                            distribution-aware in the whole testing process, including test seeds \
                            selection and local test cases generation. The robustness and global \
                            distribution is combined to guide the test seeds selection, while a novel \
                            two-step Genetic Alogorithm (GA) based test case generation is developed \
                            to search Adeversarial Examples (AEs) in the balance of local distribution \
                            and prediction loss.")
                
                gr.Markdown("## Train VAE encoder")
                gr.Markdown("To train VAE encoder for seeds selection, please choose your dataset \
                            (e.g. coco128), VAE model architecture, and customize \
                            your training settings. Click on the 'Start Training' button and see the \
                            report and training loss - epoch plot, as well as the image reconstruction \
                            demo using the VAE model trained in each epoch. The trained weights can be \
                            extracted and downloaded by clicking the 'Extract Weight' button.")
                
                gr.Markdown("## Generate AEs with HDA attacker")
                gr.Markdown("You can quickly run the HDA testing on coco128 dataset by selecting the number \
                                of test seeds, seeds' density model, DNN model and VAE encoder. The GA \
                            parameters are proposed by default but can also be customized. \
                            By default, we use 'mse' as local perceptual quality metrics.")
                gr.Markdown("Please click the 'Start Sampling' -> 'Start AE Generation' -> 'Start Evaluation' \
                            buttons step by step. The sampled seeds will be displayed, followed by generated AE \
                            samples.")
                
                gr.Markdown("## Train robust model")
                gr.Markdown("You can adapt several robustness training methods (PGD, FGSM, random noise, etc.) \
                            to train the robust model for model evaluation and comparison. \
                            The training settings and attacker setups can be selected and customized.")
                
                gr.Markdown("## Model comparison")
                gr.Markdown("It is possible to evaluate and compare two models (e.g. normal model and AT model) \
                            under the HDA attacker. Just click the 'Start Sampling' -> 'Start AE Generation' -> \
                            'Start Evaluation' buttons and see the evaluation results.")

            with gr.Tab("Train VAE Encoder") as train_vae_encoder:
                with gr.Column(variant="compact"):
                    gr.Markdown("#### Dataset")
                    with gr.Row():
                        with gr.Column(min_width=700):
                            dataset_choice1 = gr.Radio(choices=["coco128", "railway_track_fault_detection", "other"], 
                                                      value="coco128", show_label=False, min_width=700)
                            dataset_text1 = gr.Textbox(value="coco128", show_label=False, interactive=True)
                        img_size = gr.Number(label="Image size", value=256, minimum=10, precision=0, step=1, min_width=50)
                
                with gr.Column(variant="compact"):
                    gr.Markdown("#### VAE model")
                    with gr.Row():
                        num_channel1 = gr.Dropdown(label="N_channel", choices=[3, 1], value=3)
                        hidden_dim1 = gr.Number(label="Hidden dimension", value=256, minimum=10, precision=0, step=1)
                        z_dim1 = gr.Number(label="Z dimension", value=4, minimum=1, precision=0, step=1)
                    with gr.Accordion(label="Upload pretrained weight", open=False):
                        upload_pth = gr.File(label="Pretrained weight", value="Checkpoints/coco128_vae.pt", height=50)
                        gen_recons_btn = gr.Button(value="Generate reconstructions")

                with gr.Column(variant="compact"):
                    gr.Markdown("#### Training settings")
                    with gr.Row():
                        optimizer1 = gr.Radio(label="Optimizer", choices=["adam"], value="adam")
                        lr1 = gr.Number(label="Learning rate", value=1e-3, minimum=0, step=1e-3)
                        weight_decay = gr.Number(label="Weight decay", value=0, minimum=0, step=1e-3)
                        batch_size1 = gr.Number(label="Batch size", value=64, minimum=1, precision=0, step=1)
                        n_epochs1 = gr.Number(label="N_epochs", value=100, minimum=1, precision=0, step=1)
                        
                with gr.Row():
                    vae_train_btn = gr.Button(value="Start Training")
                    stop_btn1 = gr.Button(value="Stop Training")
                    reset_btn1 = gr.Button(value="Reset")
                    
                with gr.Row(variant="compact"):
                    with gr.Column():
                        with gr.Tab("Report") as report_tab:
                            report = gr.TextArea(show_label=False)
                        with gr.Tab("Loss - epoch plot") as loss_epoch_plot_tab:
                            loss = gr.Image(label="Loss - epoch plot")
                    with gr.Column():
                        with gr.Tab("Image reconstruction demo"):
                            recon = gr.Image(label="Image reconstruction demo")

                with gr.Column(variant="compact"):
                    with gr.Row():
                        with gr.Column(min_width=700):
                            gr.Markdown("#### VAE model weight",)
                        get_vae_weight_btn = gr.Button(value="Extract Weight", min_width=20)
                        # set_weight_btn = gr.Button(value="Upload", min_width=50)
                    with gr.Row():
                        download_pths = gr.File(height=50, value=None, label="Training output")
                
                # choose training dataset
                dataset_choice1.change(lambda dt: "" if dt=="other" else dt, inputs=dataset_choice1, outputs=dataset_text1)
                vae_train_process = vae_train_btn.click(start_train_vae_fn, inputs=[dataset_text1, img_size, num_channel1, hidden_dim1, 
                                                            z_dim1, upload_pth, optimizer1, lr1, weight_decay, 
                                                            batch_size1, n_epochs1, dev])
                # setup and update the training report every 5 seconds
                timer_vae_train_report = gr.Timer(5, active=False)
                timer_vae_train_report.tick(get_vae_train_report_fn, dataset_text1, [report, loss, recon])
                vae_train_btn.click(lambda: gr.Timer(active=True), None, timer_vae_train_report)
                stop_btn1.click(stop_train_vae_fn, cancels=[vae_train_process], queue=False).then(
                    lambda: gr.Timer(active=False), None, timer_vae_train_report).then(
                    get_vae_train_report_fn, dataset_text1, [report, loss, recon])
                gen_recons_btn.click(gen_recon_fn, inputs=[dataset_text1, num_channel1, hidden_dim1, z_dim1, 
                                                           upload_pth, img_size, batch_size1, dev]).then(
                                                           get_vae_train_report_fn, dataset_text1, [report, loss, recon])
                get_vae_weight_btn.click(get_vae_weight_fn, dataset_text1, outputs=download_pths)
                reset_btn1.click(stop_train_vae_fn, cancels=[vae_train_process], queue=False).then(
                    reset_vae_train_fn).then(get_vae_train_report_fn, dataset_text1, [report, loss, recon])

            with gr.Tab("HDA Test") as hda_test_tab:
                with gr.Row():
                    with gr.Column(min_width=750):
                        # Dataset panel
                        with gr.Column(variant="panel"):
                            gr.Markdown("#### Dataset")
                            with gr.Row():
                                with gr.Column(min_width=700):
                                    dataset_choice2 = gr.Radio(choices=["coco128", "railway_track_fault_detection", "other"], min_width=700,
                                                              show_label=False, value="coco128")
                                    dataset_text2 = gr.Textbox(value="coco128", show_label=False, interactive=True)
                                img_size = gr.Number(label="Image size", value=256, minimum=10, precision=0, step=1, min_width=50, interactive=True)
                            with gr.Row():   
                                density_mdl = gr.Radio(label="Seeds density model", choices=["KDE", "random"], value="KDE", interactive=True)

                                with gr.Column():
                                    n_seeds = gr.Number(label="N_seeds", minimum=1, maximum=128, value=5, step=1, 
                                                        precision=0, interactive=True)
                                with gr.Column():
                                    rand_seed = gr.Number(label="Random_seed", minimum=0, value=0, step=1, 
                                                            precision=0, interactive=True)
                        # VAE encoder panel        
                        with gr.Column(variant="panel"):
                            gr.Markdown("#### VAE encoder")
                            with gr.Row(variant="compact"):
                                with gr.Row():
                                    num_channel = gr.Dropdown(label="n_channel", choices=[3, 1], value=3, interactive=True)
                                    hidden_dim = gr.Number(label="hidden_dim", value=256, minimum=10, precision=0, step=1, interactive=True)
                                    z_dim = gr.Number(label="z_dim", value=4, minimum=1, precision=0, step=1, interactive=True)
                                with gr.Column():
                                    vae_weight_choice = gr.Radio(choices=["coco128_vae.pt", "railway_track_fault_detection_vae.pt", "other"], label="Weight", 
                                                                 value="coco128_vae.pt")
                                    vae_weight_text = gr.Textbox(value=vae_weight_choice.value,
                                                                    show_label=False, interactive=True)

                        # update text box values according to choices
                        dataset_choice2.select(lambda ds: f"{'' if ds=='other' else ds}", inputs=[dataset_choice2], outputs=[dataset_text2])
                        vae_weight_choice.select(lambda vae_weight: f"{'' if vae_weight=='other' else vae_weight}", inputs=[vae_weight_choice], 
                                                 outputs=[vae_weight_text])

                        # GA parameters panel           
                        with gr.Column(variant="panel"):
                            gr.Markdown("#### GA parameters")
                            with gr.Row():
                                eps = gr.Number(label="Epsilon", value=0.03, minimum=0, step=0.002, interactive=True)        
                                local_op = gr.Dropdown(["mse"], label="Local_op", value="mse", interactive=True)  
                                n_particles = gr.Number(label="N_particles", value=100, minimum=0, step=1, interactive=True, precision=0)
                                n_mate = gr.Number(label="N_mate", value=20, minimum=0, step=1, interactive=True, precision=0)
                                max_itr = gr.Number(label="Max_itr", value=10, minimum=0, step=1, interactive=True, precision=0)
                                alpha = gr.Number(label="Alpha", value=1.00, minimum=0, step=0.2, interactive=True)
                                batch_size2 = gr.Number(label="Batch size", value=64, minimum=1, precision=0, step=1, interactive=True)
                            with gr.Accordion(label="Object Detection Parameters", open=False):
                                with gr.Row():
                                    conf_thres = gr.Slider(minimum=0, maximum=1, value=0.1, label="Confidence threshold", interactive=True)
                                    nms_thres = gr.Slider(minimum=0, maximum=1, value= 0.5, label="NMS threshold", interactive=True)
                        
                        # DNN model panel
                        with gr.Column(variant="panel"):
                            gr.Markdown("#### Model")    
                            # Use dynamic layout for multiple models
                            n_models = gr.State(1)
                            with gr.Row():
                                add_model_btn = gr.Button("Add Model")
                                delete_model_btn = gr.Button("Delete Model")
                            add_model_btn.click(lambda count: count + 1, n_models, n_models)
                            delete_model_btn.click(lambda count: max(0, count - 1), n_models, n_models)
                                
                            @gr.render(inputs=n_models)
                            def render_models(count):
                                model_cfg_choices = []
                                model_cfg_texts = []
                                model_pth_choices = []
                                model_pth_texts = []
                                
                                with gr.Row():
                                    for i in range(count):
                                        with gr.Column(variant="panel", min_width=600):      
                                            gr.Markdown(f"#### Model {i+1}")                                              
                                            with gr.Row(variant="compact"):
                                                with gr.Column():
                                                    model_cfg_choice = gr.Radio(["yolov3-tiny.cfg", "inception_v3", "other"], 
                                                                                label="Architecture", value="yolov3-tiny.cfg", key=f"model_arch{i}")
                                                    model_cfg_text = gr.Textbox(value=model_cfg_choice.value, show_label=False, 
                                                                                interactive=True, key=f"model_arch_txt{i}")
                                                with gr.Column():
                                                    model_pth_choice = gr.Radio(["yolov3_ckpt_297.pth", "yolov3_pgd_ckpt_46.pth", 
                                                                                 "inception_v3_wo_norm_448.pth", "other"], label="Weight", 
                                                                                 value="yolov3_ckpt_297.pth", key=f"model_weight{i}")
                                                    model_pth_text = gr.Textbox(value=model_pth_choice.value, show_label=False, 
                                                                                interactive=True, key=f"model_weight_txt{i}")

                                        model_cfg_choices.append(model_cfg_choice)
                                        model_cfg_texts.append(model_cfg_text)
                                        model_pth_choices.append(model_pth_choice)
                                        model_pth_texts.append(model_pth_text)

                                model_infos = []
                                img_seeds_tabs = []
                                img_seeds_list = []
                                img_seeds_bd_tabs = []
                                img_seeds_bd_list = []
                                img_aes_tabs = []
                                img_aes_list = []
                                img_aes_bd_tabs = []
                                img_aes_bd_list = []
                                with gr.Column():
                                    gr.Markdown("#### Generated AE samples")
                                    with gr.Row():
                                        for i in range(count):
                                            with gr.Column(variant="panel", min_width=600):
                                                model_info = gr.Markdown(f"#### Model {i+1}: arch={model_cfg_texts[i].value}, weight={model_pth_texts[i].value}")
                                                with gr.Column():
                                                    with gr.Tab(label="Seeds") as img_seeds_tab:
                                                        img_seeds = gr.Gallery(show_label=False, columns=5, height=200/count, 
                                                                               interactive=False, key=f"img_seeds{i}")
                                                    with gr.Tab(label="+Bounding box") as img_seeds_bd_tab:
                                                        img_seeds_bd = gr.Gallery(show_label=False, columns=5, height=200/count, 
                                                                                  interactive=False, key=f"img_seeds_bd{i}")
                                                with gr.Column():
                                                    with gr.Tab(label="AEs") as img_aes_tab:
                                                        img_aes = gr.Gallery(show_label=False, columns=5, height=200/count, 
                                                                             interactive=False, key=f"img_aes{i}")
                                                    with gr.Tab(label="+Bounding box") as img_aes_bd_tab:
                                                        img_aes_bd = gr.Gallery(show_label=False, columns=5, height=200/count, 
                                                                                interactive=False, key=f"img_aes_bd{i}")
                                            model_infos.append(model_info)
                                            img_seeds_tabs.append(img_seeds_tab)
                                            img_seeds_bd_tabs.append(img_seeds_bd_tab)
                                            img_seeds_list.append(img_seeds)
                                            img_seeds_bd_list.append(img_seeds_bd)
                                            img_aes_tabs.append(img_aes_tab)
                                            img_aes_bd_tabs.append(img_aes_bd_tab)
                                            img_aes_list.append(img_aes)
                                            img_aes_bd_list.append(img_aes_bd)

                                with gr.Row():      
                                    select_seeds_btn = gr.Button(value="1 - Select Seeds")
                                    gen_aes_btn = gr.Button(value="2 - Start AE Generation")
                                    eval_aes_btn = gr.Button(value="3 - Evaluate Generated AEs", min_width=300)
                                    stop_btn = gr.Button(value="Stop", min_width=50)
                                    reset_btn = gr.Button(value="Reset", min_width=50)

                                df_hda_eval = gr.Dataframe(headers=["Metrics"])
                                hda_report = gr.TextArea(label="Report")

                                # set up timer for updating the report
                                timer_hda = gr.Timer(3, active=False)
                                timer_hda.tick(update_hda_report_fn, dataset_text2, [hda_report, df_hda_eval])

                                # Add functions
                                select_seeds_processes = []
                                generate_aes_processes = []
                                eval_aes_processes = []
                                for i, (model_cfg_choice, model_cfg_text, model_pth_choice, model_pth_text, model_info, \
                                    img_seeds_tab, img_seeds_bd_tab, img_aes_tab, img_aes_bd_tab, \
                                    img_seeds, img_seeds_bd, img_aes, img_aes_bd) in \
                                    enumerate(zip(model_cfg_choices, model_cfg_texts, model_pth_choices, model_pth_texts, model_infos,
                                        img_seeds_tabs, img_seeds_bd_tabs, img_aes_tabs, img_aes_bd_tabs,
                                        img_seeds_list, img_seeds_bd_list, img_aes_list, img_aes_bd_list)):

                                    # Update text box values and labels according to choices
                                    model_cfg_choice.select(lambda cfg: f"{'' if cfg=='other' else cfg}", inputs=model_cfg_choice, 
                                                            outputs=model_cfg_text)
                                    model_pth_choice.select(lambda pth: f"{'' if pth=='other' else pth}", inputs=model_pth_choice, 
                                                            outputs=model_pth_text)
                                    model_cfg_text.change(lambda cfg, pth: f"#### Model {i+1}: arch={cfg}, weight={pth}", 
                                                          inputs=[model_cfg_text, model_pth_text], outputs=model_info)
                                    model_pth_text.change(lambda cfg, pth: f"#### Model {i+1}: arch={cfg}, weight={pth}", 
                                                          inputs=[model_cfg_text, model_pth_text], outputs=model_info)
                                    
                                    # Add image demo to the timer for updating
                                    timer_hda.tick(draw_img_seeds_fn, inputs=[dataset_text2, model_cfg_text, model_pth_text, density_mdl], 
                                                    outputs=[img_seeds])
                                    timer_hda.tick(draw_img_aes_fn, inputs=[dataset_text2, density_mdl, model_cfg_text, 
                                                                               model_pth_text, local_op, eps],
                                                                             outputs=[img_aes])
                                    # Select seeds
                                    select_seeds_btn.click(lambda: gr.Timer(active=True), None, timer_hda)
                                    select_seeds_process = select_seeds_btn.click(select_seeds_fn, inputs=[img_size, num_channel, 
                                                                                                           hidden_dim, z_dim, vae_weight_text,
                                                                                    model_cfg_text, model_pth_text, conf_thres, nms_thres,
                                                                                    batch_size2, rand_seed, n_seeds, density_mdl, local_op, eps,
                                                                                    dataset_text2, dev]).then(
                                                                                    draw_img_seeds_fn, inputs=[dataset_text2, model_cfg_text, 
                                                                                                              model_pth_text, density_mdl], 
                                                                                    outputs=[img_seeds])
                                    select_seeds_processes.append(select_seeds_process)
                                    # Generate AEs
                                    gen_aes_btn.click(lambda: gr.Timer(active=True), None, timer_hda)
                                    generate_aes_process = gen_aes_btn.click(generate_aes_fn, inputs=[model_cfg_text, model_pth_text, eps,
                                                                                n_particles, n_mate, max_itr, alpha, conf_thres, nms_thres, 
                                                                                dataset_text2, local_op, density_mdl, batch_size2, dev]).then(
                                                      draw_img_aes_fn, inputs=[dataset_text2, density_mdl, model_cfg_text, 
                                                                              model_pth_text, local_op, eps],
                                                                              outputs=[img_aes])
                                    generate_aes_processes.append(generate_aes_process)
                                    # Evaluate AEs
                                    eval_aes_btn.click(lambda: gr.Timer(active=True), None, timer_hda)
                                    eval_aes_process = eval_aes_btn.click(hda_eval_fn, inputs=[model_cfg_text, model_pth_text, dataset_text2, 
                                                                            local_op, eps, density_mdl, dev])
                                    eval_aes_processes.append(eval_aes_process)
                                    
                                    # Demonstrate seed images
                                    img_seeds_tab.select(draw_img_seeds_fn, inputs=[dataset_text2, model_cfg_text, model_pth_text, density_mdl], 
                                                         outputs=[img_seeds])
                                    # Draw bounding boxes on seed images
                                    img_seeds_bd_tab.select(draw_seeds_bd_fn, inputs=[model_cfg_text, model_pth_text, dataset_text2, density_mdl, 
                                                                                      conf_thres, nms_thres, batch_size2, dev], 
                                                                                      outputs=[img_seeds_bd])
                                    # Demonstrate AE images
                                    img_aes_tab.select(draw_img_aes_fn, inputs=[dataset_text2, density_mdl, model_cfg_text, 
                                                                               model_pth_text, local_op, eps],
                                                                             outputs=[img_aes])
                                    # Draw bounding boxes on AE images
                                    img_aes_bd_tab.select(draw_aes_bd_fn, inputs=[model_cfg_text, model_pth_text, dataset_text2, density_mdl, 
                                                                                  conf_thres, nms_thres, local_op, eps, batch_size2, dev], 
                                                                                  outputs=[img_aes_bd])
                                
                                # Stop all processes
                                stop_btn.click(stop_hda_test_fn, inputs=dataset_text2, # stop the report update timer
                                                cancels=select_seeds_processes+generate_aes_processes+eval_aes_processes, 
                                                queue=False).then(update_hda_report_fn, dataset_text2, [hda_report, df_hda_eval]).then(
                                                # Stop the report update timer
                                                lambda: gr.Timer(active=False), None, timer_hda)
                                # Reset the HDA test and delete all files
                                reset_btn.click(reset_hda_test_fn, inputs=dataset_text2, 
                                                cancels=select_seeds_processes+generate_aes_processes+eval_aes_processes, 
                                                queue=False).then(update_hda_report_fn, dataset_text2, [hda_report, df_hda_eval]).then(
                                                # Stop the report update timer
                                                lambda: gr.Timer(active=False), None, timer_hda)
                                                    
            with gr.Tab("Train Robust Model") as train_rob_model:
                with gr.Column(variant='compact'):
                    gr.Markdown("#### Model & Dataset")
                    with gr.Row():
                        model_choice = gr.Radio(label="Model", choices=["yolov3-tiny.cfg", "inception_v3","other"], 
                                                value="yolov3-tiny.cfg", min_width=450)
                        model_text = gr.Textbox(label="", interactive=True, value="yolov3-tiny.cfg", min_width=50, visible=False)
                        pretrained_weights = gr.File(label="Pretrained weights", value="Checkpoints/yolov3_ckpt_297.pth")
                    with gr.Row():
                        dataset_choice3 = gr.Radio(label="Dataset", choices=["coco128", "railway_track_fault_detection", "other"], value="coco128", min_width=800)
                        dataset_text3 = gr.Textbox(label="", interactive=True, value="coco128", min_width=50, visible=False)
                        img_size = gr.Number(label="Image size", value=256, minimum=10, precision=0, step=1, min_width=50, interactive=True)
                with gr.Column(variant='compact'):
                    gr.Markdown("#### Training settings")
                    with gr.Row():
                        optimizer = gr.Radio(label="Optimizer", choices=["adam"], value="adam")
                        lr = gr.Number(label="Learning rate", value=1e-3, minimum=0, step=1e-3)
                        epochs = gr.Number(label="Epochs", minimum=1, precision=0, value=300, min_width=30)
                        batch_size3 = gr.Number(label="Batch size", value=64, minimum=1, precision=0, step=1, interactive=True)
                        n_epochs = gr.Number(label="N_epochs", value=100, minimum=1, precision=0, step=1)
                        rand_seed = gr.Number(label="Random seed", minimum=0, precision=0, value=0, min_width=30)
                        n_cpu = gr.Number(label="N_cpu", minimum=0, precision=0, value=1, min_width=10)
                        
                        with gr.Column(min_width=20):
                            multiscale = gr.Checkbox(label="Multiscale training", min_width=20)
                            verbose =  gr.Checkbox(label="Verbose", min_width=20)
                    with gr.Accordion(label="Object Detection Parameters", open=False):
                        with gr.Row():
                            conf_thres = gr.Slider(minimum=0, maximum=1, value=0.1, label="Confidence threshold", interactive=True)
                            nms_thres = gr.Slider(minimum=0, maximum=1, value= 0.5, label="NMS threshold", interactive=True)

                with gr.Column(variant='compact'):
                    gr.Markdown("#### Evaluation settings")
                    with gr.Row():   
                        iou_thres = gr.Number(value=0.5, label="IoU threshold", minimum=0, maximum=1, min_width=270)
                        eval_interval = gr.Number(label="Evaluation interval", minimum=1, precision=0, value=5, min_width=270)
                        ckpt_interval = gr.Number(label="Save checkpoint interval", minimum=1, precision=0, value=1, min_width=270)
                        
                with gr.Column(variant='compact'):
                    gr.Markdown("#### Attacker settings")
                    with gr.Row():
                        attacker = gr.Radio(label="Attacker", choices=["PGD whitebox", "Random noise", "FGSM", "No attack"], 
                                            value="PGD whitebox")
                    with gr.Row():
                        epsilon = gr.Number(label="Epsilon", value=8/255, minimum=0)
                        step_size = gr.Number(label="Step size", value=2/255, minimum=0)
                        num_steps = gr.Number(label="Num_steps", value=10, precision=0, minimum=1)
                        
                    with gr.Row():
                        start_adv_train_btn = gr.Button(value="Start Training")
                        stop_adv_train_btn = gr.Button(value="Stop Training")

                with gr.Column(variant='compact'):
                    report = gr.TextArea(label="Report", lines=12)
                    with gr.Row():
                        extract_weight_btn = gr.Button(value="Extract Weights")
                        remove_weight_btn =  gr.Button(value="Clear Weights")

                    weight_file = gr.File(label="Weight file", height=200)
                
                # Add functions
                dataset_choice3.change(lambda ds: gr.Textbox(value="", visible=True) if ds=="other" else 
                                    gr.Textbox(value=ds, visible=False), inputs=dataset_choice3, outputs=dataset_text3)
                model_choice.change(lambda ml: gr.Textbox(value="", visible=True) if ml=="other" else 
                                    gr.Textbox(value=ml, visible=False), inputs=model_choice, outputs=model_text)
                
                # set up timer to update training report periodically
                timer_adv_train = gr.Timer(3, active=False)
                timer_adv_train.tick(update_adv_train_report, inputs=[model_text], outputs=[report])
                start_adv_train_btn.click(lambda: gr.Timer(active=True), None, timer_adv_train)
                adv_train_process = start_adv_train_btn.click(adv_train_fn, 
                                        inputs=[model_text, pretrained_weights, dataset_text3, img_size, batch_size3, lr,
                                                n_epochs, verbose, n_cpu, ckpt_interval, 
                                                multiscale, eval_interval, iou_thres, conf_thres, nms_thres, attacker, 
                                                optimizer,
                                                eps, step_size, num_steps, rand_seed, dev])
                
                # stop the training process
                stop_adv_train_btn.click(lambda r: r+"\n Training cancelled.", inputs=[report], 
                    outputs=[report], cancels=[adv_train_process], queue=False).then(
                        lambda: gr.Timer(active=False), None, timer_adv_train)
                
                extract_weight_btn.click(get_adv_model_weight_fn, inputs=attacker, outputs=weight_file, queue=False)
                remove_weight_btn.click(remove_adv_model_weight_fn, inputs=attacker, outputs=weight_file, queue=False)
        with gr.Column(min_width=50):              
            dev.render()

# TODO: Initialize all output files in Results/ directory when loading the page

def render_gui():
    return demo
