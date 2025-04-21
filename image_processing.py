import cv2
import matplotlib.pyplot as plt
import os

# Path to your images folder
image_folder = "images"
processed_folder = "processed_images"

# Create processed images folder if not exists
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

# Get all image filenames
image_files = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]

for img_name in image_files:
    # Read image
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image to 256x256 for consistency
    resized_img = cv2.resize(gray_img, (256, 256))

    # Save preprocessed image
    save_path = os.path.join(processed_folder, f"gray_{img_name}")
    cv2.imwrite(save_path, resized_img)

    # Display original and processed images side-by-side
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Grayscale Image')
    plt.imshow(resized_img, cmap='gray')
    plt.axis('off')

    plt.show()
