from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

import gradio as gr
import os
import Tools.HDA.gui as hda_gui
import Tools.RTMonitor.gui as rt_monitor_gui
import Tools.GeoRobust.gui as geo_robust_gui


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("<ul><li><a href='/hda_test/'>HDA Test</a></li>\
                        <li><a href='/runtime_monitor/'>Runtime Monitor</a></li>\
                        <li><a href='/geo_robust'>GeoRobust</a></li></ul>")

hda_demo = hda_gui.render_gui()
rt_monitor_demo = rt_monitor_gui.render_gui()
geo_robust_demo = geo_robust_gui.render_gui()

app = gr.mount_gradio_app(app, hda_demo, path="/hda_test")
app = gr.mount_gradio_app(app, rt_monitor_demo, path="/runtime_monitor")
app = gr.mount_gradio_app(app, geo_robust_demo, path="/geo_robust")