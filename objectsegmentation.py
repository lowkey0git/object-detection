import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Simulate a "toy image" (just for visualization, 100x100 pixels)
image = np.ones((100, 100, 3))  # White background

# Simulate an "object" — a red square
image[30:70, 40:80] = [1, 0, 0]  # Red square = the object

# Simulate a segmentation mask
segmentation_mask = np.zeros((100, 100))
segmentation_mask[30:70, 40:80] = 1  # 1 for the object, 0 for background

# Simulate object detection output (bounding box)
bbox = (40, 30, 40, 40)  # (x, y, width, height)

# --- Plotting the comparison ---
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Original image
axs[0].imshow(image)
axs[0].set_title("Original Image")
axs[0].axis("off")

# Object Detection
axs[1].imshow(image)
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                         linewidth=2, edgecolor='cyan', facecolor='none')
axs[1].add_patch(rect)
axs[1].set_title("Object Detection (BBox)")
axs[1].axis("off")

# Image Segmentation
axs[2].imshow(image)
axs[2].imshow(segmentation_mask, alpha=0.5, cmap='gray')
axs[2].set_title("Image Segmentation (Mask)")
axs[2].axis("off")

plt.tight_layout()
plt.show()
