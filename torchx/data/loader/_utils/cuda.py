r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
copy samples fetched from dataset to GPUs.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
from torch._six import container_abcs, string_classes, int_classes


def to_cuda(data, stream=None, device=None):
    if stream is None:
        stream = torch.cuda.current_stream()

    # NOTE: Just convert Sequence & Mapping to list & dict,
    # since we don't know how to construct the original objects.
    if hasattr(data, "cuda"):
        with torch.cuda.stream(stream):
            return data.cuda(device, non_blocking=True)
    elif isinstance(data, container_abcs.Sequence):
        return [to_cuda(d, stream) for d in data]
    elif isinstance(data, container_abcs.Mapping):
        return {k: to_cuda(v, stream) for k, v in data.items()}
    else:
        return data


def record_stream(data, stream=None):
    if stream is None:
        stream = torch.cuda.current_stream()

    if hasattr(data, "record_stream"):
        data.record_stream(stream)
    elif isinstance(data, container_abcs.Sequence):
        for d in data:
            record_stream(d, stream)
    elif isinstance(data, container_abcs.Mapping):
        for v in data.values():
            record_stream(v, stream)
    else:
        pass
