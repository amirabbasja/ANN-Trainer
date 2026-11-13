import torch


def resolve_cuda_devices(selection, *, unavailable):
    """Resolve the configured GPU order to explicit CUDA devices."""
    if not torch.cuda.is_available():
        raise ValueError(unavailable)
    if not isinstance(selection, list):
        device = torch.device(selection)
        if device.type != "cuda":
            raise ValueError(unavailable)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return (device,)
    count = torch.cuda.device_count()
    for index in selection:
        if not 0 <= index < count:
            raise ValueError(f"device selects unavailable cuda:{index}; {count} CUDA device(s) visible")
    return tuple(torch.device("cuda", index) for index in selection)
