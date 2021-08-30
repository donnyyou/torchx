import functools
import logging
import torch

logger = logging.getLogger(__name__)


# NOTE currently we use normal tensor ops.
# TODO use Storage to implement the functions below.


def flatten_tensors(tensors, out=None):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
        out (Tensor): dense tensor for output, will create new one if None.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0, out=out)
    return flat


def unflatten_tensors(flat, shapes):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Size]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for shape in shapes:
        shape = torch.Size(shape)
        numel = shape.numel()
        outputs.append(flat.narrow(0, offset, numel).view(shape))
        offset += numel
    return tuple(outputs)


def same_storage(x, y):
    return x.storage().data_ptr() == y.storage().data_ptr()


class TensorList:
    """
    Hold a list of tensors with different size.
    Can flatten tensors into contiguous memory and verse visa.
    Also can transfer these tensors to pinned memory or GPU.
    Currently tensors must have same data type.
    """

    def __init__(self, tensors, contiguous=False, buffer=None):
        assert len(tensors) > 0, "At least 1 tensor in `tensors`"

        self.tensors = tensors
        self.shapes = tuple(tensor.shape for tensor in tensors)
        if buffer is not None or contiguous:
            self._do_contiguous(buffer)
        else:
            self.flattened = None

    def _do_contiguous(self, buffer=None):
        if buffer is None:
            buffer = self.tensors[0].new_empty(self.numel())
        self.flattened = flatten_tensors(self.tensors, buffer)
        self.tensors = unflatten_tensors(self.flattened, self.shapes)

    def contiguous(self, out=None):
        # If hasn't be flattened or out is a different tensor, do flatten
        if self.flattened is None or (out is not None and not same_storage(out, self.flattened)):
            self._do_contiguous(out)
        return self.flattened

    def numel(self):
        return sum(t.numel() for t in self.tensors)

    def __getattr__(self, method):
        if method in ("pin_memory", "cuda", "to", "share_memory_"):
            inplace = method[-1] == "_"

            def _wrapper(*args, **kwargs):
                new = self if inplace else TensorList(self.tensors)
                if self.flattened is not None:
                    new.flattened = getattr(self.flattened, method)(*args, **kwargs)
                    new.tensors = unflatten_tensors(new.flattened, new.shapes)
                else:
                    new.tensors = tuple([getattr(t, method)(*args, **kwargs) for t in self.tensors])
                return new

            return _wrapper
        elif method in ("is_shared", "is_pinned"):
            def _wrapper(*args, **kwargs):
                return getattr(self.tensors[0], method)(*args, **kwargs)

            return _wrapper()
        elif method == "record_stream":
            def _wrapper(stream):
                if self.flattened is not None:
                    # if flattened, just record buffer tensor
                    self.flattened.record_stream(stream)
                else:
                    # we have to record each tensor
                    for tensor in self.tensors:
                        tensor.record_stream(stream)

            return _wrapper
        else:
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__.__name__, method))

    def __getitem__(self, item):
        return self.tensors[item]

    def __iter__(self):
        return iter(self.tensors)
