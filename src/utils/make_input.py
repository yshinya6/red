import os
import pdb

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers.image_utils import load_image


def get_make_input_func(model_path):
    model_name = model_path.split("/")[-1]
    make_input_func = None
    if model_name.startswith("gemma-3"):
        make_input_func = make_input_for_gemma3
    elif model_name.startswith("Qwen"):
        make_input_func = make_input_for_qwen
    elif model_name.startswith("InternVL"):
        make_input_func = make_input_for_internvl
    else:
        raise NotImplementedError()
    return make_input_func


def make_input_for_gemma3(processor, image_paths, prompts, tokenize=True):
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    assert len(prompts) > 0
    message_batch = []
    for image_path, prompt in zip(image_paths, prompts):
        contexts = []
        if image_path:
            contexts += [{"type": "image", "image": image_path}]
        contexts += [{"type": "text", "text": prompt}]
        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": contexts,
            },
        ]
        message_batch.append(message)
    input_dict = processor.apply_chat_template(
        message_batch,
        padding=True,
        add_generation_prompt=True,
        tokenize=tokenize,
        return_dict=True,
        return_tensors="pt",
        do_pan_and_scan=True,  # for captureing local features
        pan_and_scan_min_ratio_to_activate=0.5,
        pan_and_scan_max_num_crops=32,
        pan_and_scan_min_crop_size=4,
    )
    return input_dict


def make_input_for_qwen(processor, image_paths, prompts, tokenize=True):
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    assert len(prompts) > 0
    message_batch = []
    for image_path, prompt in zip(image_paths, prompts):
        contexts = []
        if image_path:
            contexts += [{"type": "image", "image": image_path}]
        contexts += [{"type": "text", "text": prompt}]
        message = [
            {
                "role": "user",
                "content": contexts,
            }
        ]
        message_batch.append(message)
    input_dict = processor.apply_chat_template(
        message_batch,
        padding=True,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return input_dict


def make_input_for_internvl(processor, image_paths, prompts, text_only=False, tokenize=True):
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    assert len(prompts) > 0
    message_batch = []
    for image_path, prompt in zip(image_paths, prompts):
        contexts = []
        if image_path:
            contexts += [{"type": "image", "image": image_path}]
        contexts += [{"type": "text", "text": prompt}]
        message = [
            {
                "role": "user",
                "content": contexts,
            }
        ]
        message_batch.append(message)
    input_dict = processor.apply_chat_template(
        message_batch,
        padding=True,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_dict["pad_token_id"] = processor.tokenizer.eos_token_id
    return input_dict
