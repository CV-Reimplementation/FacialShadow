import cv2
import numpy as np

def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def get_illumination_map(img):
    """Convert RGB image to illumination map using max RGB values"""
    return np.max(img, axis=2)

def bilateral_decomposition(img, sigma_s=16, sigma_r=0.1):
    """Decompose image into large-scale and detail layers using bilateral filter"""
    # Convert to float32 for processing
    img_float = img.astype(np.float32) / 255.0
    
    # Get illumination map
    illum_map = get_illumination_map(img_float)
    
    # Apply bilateral filter to get large-scale layer
    large_scale = cv2.bilateralFilter(illum_map, -1, sigma_r, sigma_s)
    
    # Get detail layer
    detail = illum_map - large_scale
    
    return large_scale, detail

def detect_shadow_mask(large_scale, threshold=0.5):
    """Detect shadow regions using large-scale layer"""
    # Normalize large-scale layer
    normalized = normalize_image(large_scale)
    
    # Apply threshold to get binary shadow mask
    shadow_mask = (normalized < threshold).astype(np.float32)
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((5,5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    
    return shadow_mask

def adjust_illumination(large_scale, shadow_mask):
    """Adjust illumination in shadow regions"""
    # Get mean illumination of non-shadow regions
    non_shadow_mean = np.mean(large_scale[shadow_mask < 0.5])
    
    # Adjust shadow regions
    adjusted = large_scale.copy()
    adjusted[shadow_mask > 0.5] = non_shadow_mean
    
    return adjusted

def remove_shadows(img, sigma_s=16, sigma_r=0.1, shadow_threshold=0.5):
    """Main function for shadow removal"""
    # Convert image to float32
    img_float = img.astype(np.float32) / 255.0
    
    # Decompose image into large-scale and detail layers
    large_scale, detail = bilateral_decomposition(img_float, sigma_s, sigma_r)
    
    # Detect shadow regions
    shadow_mask = detect_shadow_mask(large_scale, shadow_threshold)
    
    # Adjust illumination in shadow regions
    adjusted_large_scale = adjust_illumination(large_scale, shadow_mask)
    
    # Reconstruct image
    reconstructed = adjusted_large_scale + detail
    
    # Apply the same adjustment to each color channel
    result = np.zeros_like(img_float)
    for i in range(3):
        result[:,:,i] = img_float[:,:,i] * (reconstructed / large_scale)
    
    # Clip values and convert back to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result

# Example usage
def main():
    # Read input image
    image = cv2.imread('input_image.jpg')
    
    # Remove shadows
    result = remove_shadows(image)
    
    # Save result
    cv2.imwrite('output_image.jpg', result)

if __name__ == "__main__":
    main()