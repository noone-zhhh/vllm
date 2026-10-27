# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from PIL import Image

from vllm.transformers_utils.processors.deepseek_ocr import (
    ImageTransform,
    dynamic_preprocess,
)


@pytest.mark.core_model
@pytest.mark.parametrize(
    ("model_id", "image_size", "model_env", "load_format"),
    [
        pytest.param(
            "deepseek-ai/DeepSeek-OCR",
            640,
            "DEEPSEEK_OCR_TEST_MODEL",
            "dummy",
            id="ocr1",
        ),
        pytest.param(
            "deepseek-ai/DeepSeek-OCR-2",
            768,
            "DEEPSEEK_OCR2_TEST_MODEL",
            "auto",
            id="ocr2",
        ),
    ],
)
def test_deepseek_ocr_balanced_tiled_image(
    vllm_runner, model_id, image_size, model_env, load_format
):
    image = Image.new("RGB", (1536, 768), "white")
    image.paste("black", (768, 0, 1536, 768))

    crops, crop_ratio = dynamic_preprocess(image, image_size=image_size)
    patches = torch.stack([ImageTransform()(crop) for crop in crops])
    assert crop_ratio == (2, 1)
    assert patches.numel() > 0
    assert patches.sum().item() == 0

    model = os.environ.get(model_env, model_id)
    with vllm_runner(
        model,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        load_format=load_format,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            ["<image>\nFree OCR."], max_tokens=8, images=[[image]]
        )

    assert len(outputs) == 1
    assert len(outputs[0][0]) > 0
