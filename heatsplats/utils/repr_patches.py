try:
    import torch

    original_tensor_repr = torch.Tensor.__repr__

    def custom_tensor_repr(tensor):
        info = "tensor: "
        info += f"{str(list(tensor.shape))}, "
        info += f"{tensor.device}, "
        info += f"{str(tensor.dtype)}, "
        info += f"requires_grad={tensor.requires_grad}"

        original_str = original_tensor_repr(tensor)

        return f"{info}\n{original_str}"

    torch.Tensor.__repr__ = custom_tensor_repr

    print("Applied torch.Tensor representation patches by importing 'repr_patches'.")

except ImportError:
    pass
