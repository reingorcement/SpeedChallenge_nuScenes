"""Observational runtime identity; no changes to tensors or training policy."""
import os
import subprocess
import torch


def runtime_gpu(device):
    device=torch.device(device)
    result=dict(process_device=str(device),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if device.type!='cuda':return result
    # Initialize the selected context before CUDA allocator telemetry is reset.
    torch.cuda.set_device(device)
    logical=device.index if device.index is not None else torch.cuda.current_device()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    selector=visible.split(',')[logical].strip() if visible else str(logical)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True,timeout=10)
    for line in rows.splitlines():
        index,uuid,name=[v.strip() for v in line.split(',',2)]
        if selector in {index,uuid}:
            result.update(physical_index=int(index),uuid=uuid,name=name)
            return result
    raise ValueError('Cannot resolve the assigned CUDA device to a physical GPU/UUID')


def peak_memory(device):
    if torch.device(device).type!='cuda':return dict(cuda_peak_allocated_bytes=0,cuda_peak_reserved_bytes=0)
    return dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
