import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter

class ShadowRemoval:
    def __init__(self):
        self.n_clusters = 8
        self.patch_size = 16
        
    def get_features(self, img):
        """Extract features for shadow detection"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Normalize LAB channels
        l_chan = lab[:,:,0] / 255.0
        a_chan = (lab[:,:,1] - 128) / 127.0
        b_chan = (lab[:,:,2] - 128) / 127.0
        
        # Calculate texture features using gradient
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Stack features
        features = np.dstack([l_chan, a_chan, b_chan, gradient_mag])
        return features
    
    def segment_image(self, features):
        """Segment image into regions using K-means clustering"""
        h, w, d = features.shape
        reshaped_features = features.reshape(-1, d)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(reshaped_features)
        
        # Reshape labels back to image dimensions
        segments = labels.reshape(h, w)
        return segments
    
    def find_paired_regions(self, features, segments):
        """Find paired regions (shadow/non-shadow pairs)"""
        pairs = []
        n_segments = np.max(segments) + 1
        
        # Calculate region statistics
        region_stats = []
        for i in range(n_segments):
            mask = (segments == i)
            if np.sum(mask) == 0:
                continue
            
            region_features = features[mask]
            stats = {
                'mean': np.mean(region_features, axis=0),
                'std': np.std(region_features, axis=0),
                'mask': mask,
                'id': i
            }
            region_stats.append(stats)
        
        # Find potential pairs
        for i in range(len(region_stats)):
            for j in range(i+1, len(region_stats)):
                if self.is_potential_pair(region_stats[i], region_stats[j]):
                    pairs.append((i, j))
        
        return pairs, region_stats
    
    def is_potential_pair(self, region1, region2):
        """Check if two regions form a potential shadow/non-shadow pair"""
        # Compare intensity difference
        intensity_diff = abs(region1['mean'][0] - region2['mean'][0])
        
        # Compare color similarity in a,b channels
        color_diff = np.sqrt(np.sum((region1['mean'][1:3] - region2['mean'][1:3])**2))
        
        # Compare texture similarity
        texture_diff = abs(region1['mean'][3] - region2['mean'][3])
        
        return (intensity_diff > 0.2 and color_diff < 0.1 and texture_diff < 0.1)
    
    def recover_illumination(self, img, shadow_mask):
        """Recover illumination in shadow regions"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Get non-shadow region statistics
        non_shadow_mask = shadow_mask < 0.5
        if np.sum(non_shadow_mask) > 0:
            mean_non_shadow = np.mean(lab[non_shadow_mask, 0])
        else:
            mean_non_shadow = 128
        
        # Adjust shadow regions
        shadow_mask_3d = np.dstack([shadow_mask] * 3)
        lab_adjusted = lab.copy()
        
        # Adjust L channel in shadow regions
        lab_adjusted[:,:,0] = lab[:,:,0] * (1 - shadow_mask) + \
                             (lab[:,:,0] * (mean_non_shadow / np.maximum(lab[:,:,0], 1e-6))) * shadow_mask
        
        # Convert back to BGR
        recovered = cv2.cvtColor(lab_adjusted.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return recovered
    
    def remove_shadows(self, img):
        """Main function for shadow detection and removal"""
        # Extract features
        features = self.get_features(img)
        
        # Segment image
        segments = self.segment_image(features)
        
        # Find paired regions
        pairs, region_stats = self.find_paired_regions(features, segments)
        
        # Create shadow mask
        shadow_mask = np.zeros_like(segments, dtype=np.float32)
        for pair in pairs:
            region1 = region_stats[pair[0]]
            region2 = region_stats[pair[1]]
            
            # Determine which region is shadow
            if region1['mean'][0] < region2['mean'][0]:
                shadow_region = region1
            else:
                shadow_region = region2
                
            shadow_mask[shadow_region['mask']] = 1.0
        
        # Smooth shadow mask
        shadow_mask = gaussian_filter(shadow_mask, sigma=3)
        
        # Recover illumination
        recovered = self.recover_illumination(img, shadow_mask)
        
        return recovered, shadow_mask

def main():
    # Read input image
    img = cv2.imread('input_image.jpg')
    if img is None:
        raise ValueError("Could not read input image")
    
    # Initialize shadow removal
    shadow_remover = ShadowRemoval()
    
    # Remove shadows
    recovered_img, shadow_mask = shadow_remover.remove_shadows(img)
    
    # Save results
    cv2.imwrite('recovered_image.jpg', recovered_img)
    cv2.imwrite('shadow_mask.jpg', (shadow_mask * 255).astype(np.uint8))

if __name__ == "__main__":
    main()