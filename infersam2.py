import os
import cv2
import sys
import numpy as np

sys.path.append(".")
from samexporter.sam2_onnx import SegmentAnything2ONNX2

image_examples = [
    [os.path.join(os.path.dirname(__file__), "./images/plants.png"), 0, []],
    [os.path.join(os.path.dirname(__file__), "./images/truck.jpg"), 1, []],
]


def generator_inference(input_x, selected_points, model_type):
    if len(selected_points) != 0:

        prompt_points = []
        prompt_boxes = []

        k = 0
        temp_data = []
        for data in selected_points:
            if data['type'] == 'points':
                prompt_points.append(
                    {
                        'type': "point",
                        'data': data['loc'],
                        'label': data['label'],
                    }
                )
            else:
                temp_data.append(data['loc'][0])
                temp_data.append(data['loc'][1])
                k += 1
                if k == 2:
                    prompt_boxes.append(
                        {'type': 'rectangle', 'data': temp_data}
                    )
                    temp_data = []
                    k = 0
        prompt = prompt_points + prompt_boxes
    else:
        print("All mask mode! 还没有实现")
        exit()

    if model_type == 'small':
        encoder_model_path = 'checkpoints/sam2_hiera_small.encoder.onnx'
        decoder_model_path = 'checkpoints/sam2_hiera_small.decoder.onnx'
    elif model_type == 'base':
        encoder_model_path = 'checkpoints/sam2_hiera_base_plus.encoder.onnx'
        decoder_model_path = 'checkpoints/sam2_hiera_base_plus.decoder.onnx'
    elif model_type == 'large':
        encoder_model_path = 'checkpoints/sam2_hiera_large.encoder.onnx'
        decoder_model_path = 'checkpoints/sam2_hiera_large.decoder.onnx'
    else:
        encoder_model_path = 'checkpoints/sam2_hiera_tiny.encoder.onnx'
        decoder_model_path = 'checkpoints/sam2_hiera_tiny.decoder.onnx'

    exit()
    sam = SegmentAnything2ONNX2(
        encoder_model_path=encoder_model_path,
        decoder_model_path=decoder_model_path,
    )

    image = input_x

    embedding = sam.encode(input_x)
    masks = sam.predict_masks(embedding, prompt)

    # Merge masks
    mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
    for m in masks[0, :, :, :]:
        mask[m > 0.5] = [255, 0, 0]
    # Binding image and mask
    visualized = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
    visualized = visualized.astype(np.uint8)

    return visualized, mask


def run_inference(
    input_x,
    selected_points=[],
    model_type='small',
):
    if isinstance(input_x, int):
        input_x = cv2.imread(image_examples[input_x][0])
        input_x = cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB)

    print('Inference')
    return generator_inference(input_x, selected_points, model_type)
