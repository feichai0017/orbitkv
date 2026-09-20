"""Identify a GPU consistently across an inference process and Cache Manager."""

import os

import torch


def resolve_device_id() -> int:
    local_id = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return local_id
    slots = [slot.strip() for slot in visible.split(",") if slot.strip()]
    try:
        return int(slots[local_id])
    except (IndexError, ValueError):
        return local_id
