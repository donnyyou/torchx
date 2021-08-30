r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import re
from torch._six import container_abcs, string_classes, int_classes

from ..tensor_list import TensorList
from .worker import get_worker_info

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if all(e.shape == elem.shape for e in batch):
            return torch.stack(batch, 0, out=out)
        else:
            return TensorList(batch, contiguous=True, buffer=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def default_decollate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    r"""Puts data batch tensor into data fields"""

    if isinstance(batch, (float, int_classes, string_classes)):
        return batch    # for single element, just return itself
    elif isinstance(batch, TensorList):
        return batch.tensors
    elif isinstance(batch, torch.Tensor):
        if batch.dim() == 1:
            return batch.tolist()
        else:
            return [elem for elem in batch]
    elif isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
        return [type(batch)(**attrs) for attrs in default_decollate(batch._asdict())]
    elif isinstance(batch, container_abcs.Sequence):
        if isinstance(batch[0], string_classes):
            return batch
        else:
            return list(zip(*(default_decollate(field) for field in batch)))
    elif isinstance(batch, container_abcs.Mapping):
        fields = {key: None for key in batch}
        for key, value in batch.items():
            fields[key] = default_decollate(value)
        # check to make sure that the elements in batch have consistent size
        nums = [len(value) for value in fields.values()]
        if not all(num == nums[0] for num in nums):
            raise RuntimeError('each element in list of batch should be of equal size')
        return [{key: value[i] for key, value in fields.items()} for i in range(nums[0])]

    raise TypeError(default_collate_err_msg_format.format(type(batch)))
