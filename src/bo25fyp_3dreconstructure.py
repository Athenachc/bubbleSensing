import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to compute the gradient for a specific color channel
def compute_color_gradient(image, channel_index):
    channel = image[:, :, channel_index]
    gradient_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=5)  # Gradient in x-direction
    gradient_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=5)  # Gradient in y-direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

# Load the original and pressed images
original_image = cv2.imread('img1.jpg')  # No pressure image
pressed_image = cv2.imread('img3.jpg')    # Pressed image

# Calculate gradients for each color channel
red_gradient = compute_color_gradient(pressed_image, 2)  # Red channel
green_gradient = compute_color_gradient(pressed_image, 1)  # Green channel
blue_gradient = compute_color_gradient(pressed_image, 0)  # Blue channel

# Combine gradients into a single image
combined_gradient = np.maximum(red_gradient, np.maximum(green_gradient, blue_gradient))

# Normalize combined gradient for depth estimation
combined_gradient_normalized = cv2.normalize(combined_gradient, None, 0, 255, cv2.NORM_MINMAX)

# Step 3: Estimate depth based on combined gradient
def estimate_depth(gradient):
    points_3D = []
    max_depth = 10  # Maximum depth to simulate (adjust as needed)
    
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            if gradient[i, j] > 10:  # Threshold for significant gradients
                z = (1 - (gradient[i, j] / gradient.max())) * max_depth  # Normalize to depth
                points_3D.append((j, i, z))  # Store x, y, z coordinates
    
    return np.array(points_3D)

# Estimate 3D points from the combined gradient
points_3D = estimate_depth(combined_gradient)

# Visualize the 3D points
if len(points_3D) > 0:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Reconstruction from Color Gradients')
    plt.show()
else:
    print("No valid points for reconstruction.")
