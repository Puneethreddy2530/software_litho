import torch

def mask_image_border(img_tensor: torch.Tensor, border_width: int = 40, is_normalized: bool = True) -> dict:
    """
    Applies a deterministic mask to the border of an image tensor.
    Never modifies the original DCID files on disk.
    
    Args:
        img_tensor (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W).
        border_width (int): The width of the border to mask in pixels (default: 40).
        is_normalized (bool): If True, masks with the ImageNet-normalized equivalent of pure black (RGB=0).
        
    Returns:
        dict: Containing:
            'intervened_image': The masked tensor
            'image_size': Tuple of (H, W)
            'mask_width': The border width applied
            'pixels_modified_percentage': Percentage of pixels masked
    """
    intervened = img_tensor.clone()
    device = intervened.device
    
    # Calculate the exact normalized black value: (0 - mean) / std
    if is_normalized:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        mask_val = (0.0 - mean) / std
    else:
        mask_val = torch.tensor([0.0, 0.0, 0.0], device=device).view(3, 1, 1)
    
    if intervened.dim() == 3:
        # Shape: (C, H, W)
        C, H, W = intervened.shape
        intervened[:, :border_width, :] = mask_val       # Top
        intervened[:, -border_width:, :] = mask_val      # Bottom
        intervened[:, :, :border_width] = mask_val       # Left
        intervened[:, :, -border_width:] = mask_val      # Right
    elif intervened.dim() == 4:
        # Shape: (B, C, H, W)
        B, C, H, W = intervened.shape
        mask_val = mask_val.unsqueeze(0) # (1, 3, 1, 1)
        intervened[:, :, :border_width, :] = mask_val    # Top
        intervened[:, :, -border_width:, :] = mask_val   # Bottom
        intervened[:, :, :, :border_width] = mask_val    # Left
        intervened[:, :, :, -border_width:] = mask_val   # Right
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {intervened.dim()}D")
        
    total_pixels = H * W
    # Calculate mask area avoiding double counting the corners
    masked_pixels = (2 * border_width * W) + (2 * border_width * (H - 2 * border_width))
    percentage = (masked_pixels / total_pixels) * 100.0
    
    return {
        'intervened_image': intervened,
        'image_size': (H, W),
        'mask_width': border_width,
        'pixels_modified_percentage': percentage
    }
