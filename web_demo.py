import gradio as gr
from matplotlib import pyplot as plt
import sys
import os
import cv2
import numpy as np
from infersam2 import run_inference

sys.path.append(".")

# points color and marker
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 3]

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">SAM-2</h1>""")
    with gr.Row():
        gr.Markdown(
            '''# Segment Anything-2!🚀
                Copyright (c)
            '''
        )
        with gr.Row():
            # select model
            model_type = gr.Dropdown(
                choices=["tiny", "small", "base", "large"],
                value=0,
                label="Select Model",
            )

    # Segment image
    with gr.Tab(label='Image'):
        with gr.Row():
            with gr.Column():
                # input image
                original_image = gr.State(
                    value=None
                )  # store original image without points, default None
                input_image = gr.Image(type="numpy")
                # point prompt
                with gr.Column():
                    selected_points = gr.State([])  # store points
                    with gr.Row():
                        gr.Markdown(
                            'You can click on the image to select prompt mode. Default: foreground_point.'
                        )
                    radio = gr.Radio(
                        ['foreground_point', 'background_point', 'Box_Mode'],
                        label='Prompt Mode',
                        value='foreground_point',
                    )
                with gr.Row():
                    undo_button1 = gr.Button('Undo point')
                    undo_button2 = gr.Button('Undo boxes')
                    undo_button3 = gr.Button('Clear!')

                # run button
                button = gr.Button("Mask generation!")

            # show the image with mask
            with gr.Column():
                with gr.Tab(label='Image+Mask'):
                    output_image = gr.Image(type='numpy')
                # show only mask
                with gr.Tab(label='Mask'):
                    output_mask = gr.Image(type='numpy')
                with gr.Row():
                    image_save = gr.Button('Save image')

                    mask_save = gr.Button('Save mask')

    # once user upload an image, the original image is stored in `original_image`
    def store_img(img):
        return (
            img,
            [],
        )  # when new image is uploaded, `selected_points` should be empty

    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points],
    )

    # global global_boxes
    global_boxes = []

    def get_point_boxes(img, sel_pix, point_type, evt: gr.SelectData):
        sel_boxes = []
        if point_type == 'foreground_point':
            sel_pix.append(
                {'type': 'points', 'loc': evt.index, 'label': 1}
            )  # append the foreground_point
        elif point_type == 'background_point':
            sel_pix.append(
                {'type': 'points', 'loc': evt.index, 'label': 0}
            )  # append the background_point
        elif point_type == 'Box_Mode':
            sel_pix.append({'type': 'boxes', 'loc': evt.index, 'label': None})
        else:
            sel_pix.append(
                {'type': 'points', 'loc': evt.index, 'label': 1}
            )  # default foreground_point
        for data in sel_pix:
            # print(data)
            if data['type'] == 'points':
                cv2.drawMarker(
                    img,
                    data['loc'],
                    colors[data['label']],
                    markerType=markers[data['label']],
                    markerSize=20,
                    thickness=5,
                )
            elif data['type'] == 'boxes':

                sel_boxes.append(data['loc'])
                # print(sel_boxes)
        global global_boxes
        global_boxes = sel_boxes
        if len(sel_boxes) % 2 == 0:
            # Draw rectangles
            ind = len(sel_boxes) // 2
            for jnd in range(ind):
                top_left = (sel_boxes[2 * jnd][0], sel_boxes[2 * jnd][1])
                bottom_right = (
                    sel_boxes[2 * jnd + 1][0],
                    sel_boxes[2 * jnd + 1][1],
                )
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

        if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img if isinstance(img, np.ndarray) else np.array(img)

    input_image.select(
        get_point_boxes,
        [input_image, selected_points, radio],
        [input_image],
    )

    # undo the selected point
    def undo_points(orig_img, sel_pix):

        temp = orig_img.copy()
        # remove last point or box

        if len(sel_pix) != 0:
            for i in range(len(sel_pix) - 1, -1, -1):
                if sel_pix[i]['type'] == 'points':
                    sel_pix.pop(i)
                    break

        # draw points
        for data in sel_pix:
            if data['type'] == 'points':
                cv2.drawMarker(
                    temp,
                    data['loc'],
                    colors[data['label']],
                    markerType=markers[data['label']],
                    markerSize=20,
                    thickness=5,
                )

        global global_boxes
        sel_boxes = global_boxes
        if len(sel_boxes) % 2 == 0:
            # Draw rectangles
            ind = len(sel_boxes) // 2

            for jnd in range(ind):
                top_left = (sel_boxes[2 * jnd][0], sel_boxes[2 * jnd][1])
                bottom_right = (
                    sel_boxes[2 * jnd + 1][0],
                    sel_boxes[2 * jnd + 1][1],
                )
                cv2.rectangle(temp, top_left, bottom_right, (0, 255, 0), 2)

        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return temp if isinstance(temp, np.ndarray) else np.array(temp)

    def undo_boxes(orig_img, sel_pix):

        temp = orig_img.copy()

        # remove last box
        global global_boxes
        sel_boxes = global_boxes
        if len(sel_boxes) != 0:
            if len(sel_boxes) % 2 == 0:
                sel_boxes.pop()
                sel_boxes.pop()
                k = 0
                for i in range(len(sel_pix) - 1, -1, -1):
                    if sel_pix[i]['type'] == 'boxes':
                        sel_pix.pop(i)
                        k += 1
                        if k == 2:
                            break
            elif len(sel_boxes) == 1:
                sel_boxes.pop()
                for i in range(len(sel_pix) - 1, -1, -1):
                    if sel_pix[i]['type'] == 'boxes':
                        sel_pix.pop(i)
            else:
                sel_boxes.pop()
                sel_boxes.pop()
                sel_boxes.pop()
                k = 0
                for i in range(len(sel_pix) - 1, -1, -1):
                    if sel_pix[i]['type'] == 'boxes':
                        sel_pix.pop(i)
                        k += 1
                        if k == 3:
                            break
        global_boxes = sel_boxes
        for data in sel_pix:
            if data['type'] == 'points':
                cv2.drawMarker(
                    temp,
                    data['loc'],
                    colors[data['label']],
                    markerType=markers[data['label']],
                    markerSize=20,
                    thickness=5,
                )

        if len(sel_boxes) % 2 == 0:

            ind = len(sel_boxes) // 2

            for jnd in range(ind):
                top_left = (sel_boxes[2 * jnd][0], sel_boxes[2 * jnd][1])
                bottom_right = (
                    sel_boxes[2 * jnd + 1][0],
                    sel_boxes[2 * jnd + 1][1],
                )
                cv2.rectangle(temp, top_left, bottom_right, (0, 255, 0), 2)

        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return temp if isinstance(temp, np.ndarray) else np.array(temp)

    def undo(orig_img, sel_pix):

        temp = orig_img.copy()
        # remove last point or box
        sel_pix.clear()

        global global_boxes
        global_boxes.clear()

        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return temp if isinstance(temp, np.ndarray) else np.array(temp)

    undo_button1.click(
        undo_points,
        [original_image, selected_points],
        [input_image],
    )
    undo_button2.click(
        undo_boxes,
        [original_image, selected_points],
        [input_image],
    )
    undo_button3.click(
        undo,
        [original_image, selected_points],
        [input_image],
    )

    def saveimage(output_image):
        temp = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./sample_outputs/output.png', temp)
        gr.Info("Image save success!", duration=1)

    def savemask(output_mask):
        temp = cv2.cvtColor(output_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./sample_outputs/mask.png', temp)
        gr.Info("Mask save success!", duration=1)

    image_save.click(saveimage, inputs=[output_image])
    mask_save.click(savemask, inputs=[output_mask])

    button.click(
        run_inference,
        inputs=[original_image, selected_points, model_type],
        outputs=[output_image, output_mask],
    )

demo.queue().launch()
