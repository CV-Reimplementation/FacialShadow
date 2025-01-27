#!/usr/bin/env python3
"""
Learning to Remove Soft Shadows (Simplified Prototype)

Reference:
  M. Gryka, M. Terry, and G. Brostow. "Learning to Remove Soft Shadows."
  ACM Transactions on Graphics (TOG), 2015.

High-Level Pipeline:
  1) Data Generation/Acquisition (training)
  2) Feature Extraction + Patch Alignment
  3) Train Random Forest Regressor (patch appearance → matte patch)
  4) Test-time initialization (inpainting or other guess)
  5) Patch-based inference with MRF (TRW-S, graph cut, or similar)
  6) Color channel optimization
  7) Output final unshadowed image (and shadow matte)
"""

import numpy as np
import cv2
import os
from skimage.filters import threshold_otsu

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# 1) HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def load_training_images(shadow_dir, noshadow_dir):
    """
    Load paired shadowed and shadow-free images from disk.
    Returns lists of (shadowed, noshadow) pairs as NumPy arrays.
    """
    shadows = sorted(os.listdir(shadow_dir))
    noshadows = sorted(os.listdir(noshadow_dir))

    shadow_imgs = []
    noshadow_imgs = []

    for s_name, ns_name in zip(shadows, noshadows):
        s_path = os.path.join(shadow_dir, s_name)
        ns_path = os.path.join(noshadow_dir, ns_name)
        s_img = cv2.imread(s_path, cv2.IMREAD_COLOR)  # BGR, 8-bit
        ns_img = cv2.imread(ns_path, cv2.IMREAD_COLOR)

        if s_img is None or ns_img is None:
            continue

        # Convert to float32 [0,1] for convenience
        s_img = s_img.astype(np.float32) / 255.0
        ns_img = ns_img.astype(np.float32) / 255.0

        shadow_imgs.append(s_img)
        noshadow_imgs.append(ns_img)

    return shadow_imgs, noshadow_imgs


def compute_shadow_matte(shadow_im, noshadow_im, epsilon=1e-6):
    """
    Given shadowed image and shadow-free image (aligned, same size),
    compute matte = shadowed / noshadowed (element-wise).
    Clamp values to [0,1].
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        matte = np.divide(shadow_im, (noshadow_im + epsilon))
    matte = np.clip(matte, 0.0, 1.0)
    return matte


def align_patch(patch, template_size=32):
    """
    In practice, Gryka et al. search over small integer translations/rotations
    to align the patch with a template to reduce rotational variance.
    Here, we just center-crop or pad with zeros to a fixed size.
    """
    h, w, c = patch.shape
    aligned = np.zeros((template_size, template_size, c), dtype=np.float32)
    # Simple center fit for demonstration:
    min_h = min(h, template_size)
    min_w = min(w, template_size)
    aligned[:min_h, :min_w, :] = patch[:min_h, :min_w, :]
    return aligned


def extract_features(aligned_patch, init_matte_patch, user_mask_dist):
    """
    Construct a feature vector from:
      - the local patch intensities
      - gradients
      - init matte patch
      - distance from shadow boundary, etc.
    Returns a 1D feature vector.
    """
    # Flatten intensities (optionally subtract mean intensity to reduce variance)
    patch_mean = aligned_patch.mean()
    patch_norm = aligned_patch - patch_mean  # shape (H, W, C)

    # Gradients (x and y) in each channel
    grad_x = cv2.Sobel(aligned_patch, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(aligned_patch, cv2.CV_32F, 0, 1, ksize=3)

    # Flatten everything
    feat_intensity = patch_norm.flatten()
    feat_gradx = grad_x.flatten()
    feat_grady = grad_y.flatten()
    feat_init_matte = init_matte_patch.flatten()

    # Distance from boundary (scalar)
    feat_dist = np.array([user_mask_dist], dtype=np.float32)

    # Concatenate
    feature_vec = np.concatenate([
        feat_intensity, feat_gradx, feat_grady, feat_init_matte, feat_dist
    ], axis=0)

    return feature_vec


# ------------------------------------------------------------------------------
# 2) TRAINING DATA PREPARATION
# ------------------------------------------------------------------------------

def prepare_training_data(shadow_imgs, noshadow_imgs, patch_size=16, template_size=32, num_samples=2000):
    """
    Randomly sample patches from the shadow region of each training pair,
    compute the ground-truth matte patches, and build up a list of (feature, label).
    """
    features = []
    labels = []

    # Simple random sampling across all images
    all_indices = list(range(len(shadow_imgs)))
    np.random.shuffle(all_indices)
    # We'll gather ~ num_samples from the entire set as a small example
    max_per_image = max(1, num_samples // len(shadow_imgs))

    for idx in all_indices:
        s_img = shadow_imgs[idx]
        ns_img = noshadow_imgs[idx]

        # For demonstration, pick random seeds
        h, w, _ = s_img.shape
        matte_gt = compute_shadow_matte(s_img, ns_img)

        # Suppose we want random 16x16 patches from the entire image
        # A real approach would sample only in the shadow (user or auto mask).
        for _ in range(max_per_image):
            row = np.random.randint(0, h - patch_size)
            col = np.random.randint(0, w - patch_size)

            patch_shadow = s_img[row:row+patch_size, col:col+patch_size, :] 
            patch_matte = matte_gt[row:row+patch_size, col:col+patch_size, :]

            # Expand to 32x32 for alignment (fake "align" step)
            aligned_patch = align_patch(patch_shadow, template_size=template_size)
            aligned_matte = align_patch(patch_matte, template_size=template_size)

            # We can't truly compute real "distance from boundary" without a real mask,
            # so let's just fill in a random placeholder for demonstration
            user_mask_dist = np.random.rand()

            # For the initial matte guess in training, assume naive inpainting or
            # simple average
            init_guess = np.ones((template_size, template_size, 3), dtype=np.float32)

            feat_vec = extract_features(aligned_patch, init_guess, user_mask_dist)
            # The label is the matte patch we want to predict (red channel only)
            # per Gryka et al., they focus on a single channel and reconstruct the remaining
            label_vec = aligned_matte[...,0].flatten()  # shape: (32*32,)

            features.append(feat_vec)
            labels.append(label_vec)

    features_np = np.array(features, dtype=np.float32)
    labels_np   = np.array(labels, dtype=np.float32)

    return features_np, labels_np


# ------------------------------------------------------------------------------
# 3) TRAIN RANDOM FOREST REGRESSOR
# ------------------------------------------------------------------------------

def train_regressor(features_np, labels_np, pca_dim=4, max_trees=10, max_depth=20):
    """
    Train a random forest to map from patch appearance → patch matte (red channel).
    Using PCA on the label side to reduce label dimensionality at training time,
    then store full label data for nonparametric retrieval (simplified approach).
    """
    # For demonstration, we do a standard “y in R^p” regression, not a nonparametric label retrieval.
    # Full approach in the paper does a specialized random forest that references
    # entire label distributions. Here we do an approximation: we train a standard
    # multi-output regressor (in scikit-learn, that means shape (n_samples, p) for “y”).
    # The result won't replicate the paper's approach exactly, but it’s a starting point.

    n_samples = features_np.shape[0]
    label_dim = labels_np.shape[1]

    # Optional PCA on label side
    pca = PCA(n_components=pca_dim)
    label_proj = pca.fit_transform(labels_np)  # shape: (n_samples, pca_dim)

    # Train forest
    forest = RandomForestRegressor(
        n_estimators=max_trees,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42
    )
    forest.fit(features_np, label_proj)

    return forest, pca


# ------------------------------------------------------------------------------
# 4) TEST-TIME: INFERENCE PIPELINE
# ------------------------------------------------------------------------------

def inpaint_region(img_bgr, mask):
    """
    Simple inpainting using OpenCV for the user-selected shadow region.
    This is just a placeholder. In practice, PatchMatch or advanced guided inpainting
    could be used to produce a better initial guess.
    """
    # cv2.inpaint expects a single-channel mask in {0,255}
    mask_uint8 = (mask*255).astype(np.uint8)
    inpainted = cv2.inpaint(
        (img_bgr*255).astype(np.uint8),
        mask_uint8,
        3,
        cv2.INPAINT_TELEA
    )
    return (inpainted.astype(np.float32)/255.0)


def mrf_solve_patchwise(forest, pca, img_bgr, mask, patch_size=16, template_size=32):
    """
    Patchwise MRF-based inference as described in the paper, simplified:
      1) Divide masked region into grid of patches
      2) For each patch, extract feature, run regressor to get matte patch
      3) MRF to ensure consistency across patch boundaries
    Here, we skip the real MRF construction and do a simple per-patch feed-forward.
    """
    # Convert to float
    h, w, _ = img_bgr.shape
    matte_red = np.ones((h, w), dtype=np.float32)

    # A simple grid over mask bounding box
    # Real code handles partial patches near edges
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return matte_red  # no shadow region

    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()

    for row in range(min_y, max_y, patch_size):
        for col in range(min_x, max_x, patch_size):
            # Check if patch is within mask
            patch_mask = mask[row:row+patch_size, col:col+patch_size]
            if patch_mask.sum() < 1: 
                continue  # skip patches mostly outside the user mask

            patch_img = img_bgr[row:row+patch_size, col:col+patch_size, :]

            # Expand to template_size for alignment
            aligned_patch = align_patch(patch_img, template_size=template_size)

            # Dummy “distance from boundary” for demonstration
            user_mask_dist = 0.5

            # We skip a real init matte patch for simplicity
            init_guess = np.ones((template_size, template_size, 3), dtype=np.float32)

            feat_vec = extract_features(aligned_patch, init_guess, user_mask_dist)
            feat_vec = feat_vec.reshape(1, -1)  # shape: (1, #features)

            # Predict PCA space label
            label_pca = forest.predict(feat_vec)  # shape: (1, pca_dim)
            # Project back to original dimension
            label_est = pca.inverse_transform(label_pca)  # shape: (1, 32*32)

            # Reshape to patch
            matte_est_32x32 = label_est.reshape(template_size, template_size)
            # Realign (just a center-crop for this code)
            matte_est_16x16 = matte_est_32x32[:patch_size, :patch_size]

            # Paste into the output matte
            output_shape = matte_red[row:row+patch_size, col:col+patch_size].shape
            matte_red[row:row+patch_size, col:col+patch_size] = matte_est_16x16[:output_shape[0], :output_shape[1]]

    # A real approach would also have an MRF step to unify patch seams and
    # do pairwise consistency. This is just a placeholder.
    matte_red = np.clip(matte_red, 0.0, 1.0)
    return matte_red


def color_channel_optimization(img_bgr, matte_red, mask):
    """
    Optimize scaling factors σg, σb to keep color distribution consistent.
    Simplified version: we just compute an average in shadow area and do a quick ratio.
    """
    # Copy original to avoid destructive changes
    output_bgr = img_bgr.copy()

    # Estimate scale factors by comparing mean intensities in shadow region
    # vs. the no-shadow region, etc. Here, we do something simpler:
    r_shadow = (img_bgr[...,0]*mask).mean()
    g_shadow = (img_bgr[...,1]*mask).mean() + 1e-5
    b_shadow = (img_bgr[...,2]*mask).mean() + 1e-5

    # Suppose we want the ratio r:g:b to remain the same
    # but the final matte is in red channel only, etc. 
    # We do a naive scale so that each channel’s shadow attenuation is consistent
    # with the red channel. This is a big simplification.
    scale_g = r_shadow / g_shadow
    scale_b = r_shadow / b_shadow

    # Final unshadowed image, channel by channel
    # Iu = Is / Im  => for red:   out[...,0] = in[...,0] / matte_red
    out_r = output_bgr[...,0] / np.clip(matte_red, 1e-4, 1.0)
    # For green
    matte_g = np.clip(matte_red * scale_g, 0.0, 1.0)
    out_g = output_bgr[...,1] / np.clip(matte_g, 1e-4, 1.0)
    # For blue
    matte_b = np.clip(matte_red * scale_b, 0.0, 1.0)
    out_b = output_bgr[...,2] / np.clip(matte_b, 1e-4, 1.0)

    output_bgr[...,0] = out_r
    output_bgr[...,1] = out_g
    output_bgr[...,2] = out_b

    output_bgr = np.clip(output_bgr, 0.0, 1.0)
    return output_bgr, (matte_red, matte_g, matte_b)


# ------------------------------------------------------------------------------
# 5) DEMO MAIN
# ------------------------------------------------------------------------------

def mask_generator(im_s, im_f):
    # convert im_s and im_f to 255
    im_s = (im_s * 255).astype(np.uint8)
    im_f = (im_f * 255).astype(np.uint8)

    im_s = cv2.cvtColor(im_s, cv2.COLOR_BGR2GRAY)
    im_f = cv2.cvtColor(im_f, cv2.COLOR_BGR2GRAY)
    diff = (im_f - im_s)
    L = threshold_otsu(diff)
    # if diff > L, it is shadow region, else not, convert the binary mask to 0-255 not bool
    mask = (diff > L).astype(np.uint8) * 255
    # mask = diff > L
    # mask = mask.astype(np.uint8)
    return mask


def main_demo():
    """
    Demonstration of a simplified pipeline in one go.

    1) Load some training data of shadow/unshadow pairs. 
    2) Prepare features, train a random forest regressor (approx).
    3) For a new test image + user shadow mask: 
       - inpaint to get initial guess
       - run patch-based inference
       - color optim
       - show or save result
    """

    # --- 5a) Load training data and train a model ---
    # You must supply directories with shadow and noshadow images
    # for a small example. Adjust the paths to your own data.
    shadow_dir = "input"
    noshadow_dir = "target"

    print("Loading training images...")
    shadow_imgs, noshadow_imgs = load_training_images(shadow_dir, noshadow_dir)

    if len(shadow_imgs) == 0:
        print("No training images loaded. Please check paths.")
        return

    print("Preparing training data...")
    feats, labs = prepare_training_data(shadow_imgs, noshadow_imgs,
                                        patch_size=16,
                                        template_size=32,
                                        num_samples=2000)

    print(f"Feature shape: {feats.shape}, Labels shape: {labs.shape}")

    print("Training random forest regressor (this may take a while)...")
    forest, pca = train_regressor(feats, labs, 
                                  pca_dim=4, 
                                  max_trees=10, 
                                  max_depth=20)
    print("Training complete.")

    # test on the same dirs
    test_shadow_path = "input"
    test_noshadow_path = "target"

    shadow_imgs, noshadow_imgs = load_training_images(shadow_dir, noshadow_dir)


    for i in range(len(shadow_imgs)):
        test_img = shadow_imgs[i]
        test_noshadow = noshadow_imgs[i]

        # read user mask from file (white = shadow region)
        user_mask = mask_generator(test_img, test_noshadow)
        user_mask = user_mask.astype(np.float32)/255.0
        user_mask = (user_mask > 0.5).astype(np.float32)

        # Inpaint to get initial shadow-free guess
        inpainted_img = inpaint_region(test_img, user_mask)

        # Patch-based inference from random forest
        matte_red_approx = mrf_solve_patchwise(forest, pca, inpainted_img, user_mask,
                                            patch_size=16,
                                            template_size=32)

        # Color channel optimization
        unshadowed, (mr, mg, mb) = color_channel_optimization(test_img, matte_red_approx, user_mask)

        # Convert to display range
        matte_out = (mr*255).astype(np.uint8)
        unshadowed_out = (unshadowed*255).astype(np.uint8)

        # Save results
        cv2.imwrite(os.path.join('result', f"{i}.jpg"), unshadowed_out)
        # cv2.imwrite(os.path.join('result', f"{i}.jpg"), matte_out)



if __name__ == "__main__":
    main_demo()