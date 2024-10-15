import torch

def custom_median_filter(image, kernel_size):
    padding = kernel_size // 2

    _, H, W = image.size()
    
    image_padded = torch.zeros((1, H + 2 * padding, W + 2 * padding))
    image_padded[:, padding:padding + H, padding:padding + W] = image
    
    image_padded[:, :padding, padding:-padding] = image[:, :padding, :].flip(1)
    image_padded[:, -padding:, padding:-padding] = image[:, -padding:, :].flip(1)
    
    image_padded[:, padding:-padding, :padding] = image[:, :, :padding].flip(2)
    image_padded[:, padding:-padding, -padding:] = image[:, :, -padding:].flip(2)
    
    image_padded[:, :padding, :padding] = image[:, :padding, :padding].flip(1, 2)
    image_padded[:, -padding:, :padding] = image[:, -padding:, :padding].flip(1, 2)
    image_padded[:, :padding, -padding:] = image[:, :padding, -padding:].flip(1, 2)
    image_padded[:, -padding:, -padding:] = image[:, -padding:, -padding:].flip(1, 2)
    
    output_image = torch.zeros_like(image)
    
    for i in range(H):
        for j in range(W):
            patch = image_padded[:, i:i + kernel_size, j:j + kernel_size]
            
            patch_flattened = patch.flatten()
            
            output_image[:, i, j] = patch_flattened.median()
    
    return output_image


