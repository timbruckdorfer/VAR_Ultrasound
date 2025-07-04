import cv2
import numpy as np
import os
import glob

def resize_image(input_path, output_path, target_size=(256, 256)):
    """
    Resize an image to the target size while maintaining aspect ratio with padding.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path where the resized image will be saved
        target_size (tuple): Target size as (width, height)
    """
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return False
    
    # Get original dimensions
    h, w = img.shape[:2]
    
    # Calculate scaling factor to fit the image within target size
    scale = min(target_size[0] / w, target_size[1] / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas of target size
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Calculate position to center the resized image
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    # Place the resized image on the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Save the resized image
    cv2.imwrite(output_path, canvas)
    return True

def process_directory(directory_path, target_size=(256, 256)):
    """
    Process all images in a directory, resizing them to the target size.
    
    Args:
        directory_path (str): Path to the directory containing images
        target_size (tuple): Target size as (width, height)
    """
    # Get all image files (PNG and JPG)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    print(f"Found {len(image_files)} images in {directory_path}")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Create a temporary resized filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        temp_resized_path = os.path.join(directory_path, f"{base_name}_temp_resized.png")
        
        try:
            # Resize the image
            success = resize_image(image_path, temp_resized_path, target_size)
            
            if success:
                # Remove the original image
                os.remove(image_path)
                
                # Rename the resized image to the original filename
                os.rename(temp_resized_path, image_path)
                print(f"  ✓ Successfully resized: {os.path.basename(image_path)}")
            else:
                print(f"  ✗ Failed to resize: {os.path.basename(image_path)}")
                # Clean up temporary file if it exists
                if os.path.exists(temp_resized_path):
                    os.remove(temp_resized_path)
                    
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(image_path)}: {str(e)}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_resized_path):
                os.remove(temp_resized_path)
    
    print(f"Completed processing {directory_path}")

def main():
    """
    Main function to resize all images in the preprocessed datasets.
    """
    # Define the directories to process
    directories = [
        "Datasets_preprocessed/Kidney_preprocessed",
        "Datasets_preprocessed/Fetal_head_preprocessed", 
        "Datasets_preprocessed/Thyroid_preprocessed/DDTI_preprocessed",
        "Datasets_preprocessed/Thyroid_preprocessed/Gland_preprocessed",
        "Datasets_preprocessed/Thyroid_preprocessed/Nodules_preprocessed"
    ]
    
    target_size = (256, 256)
    
    print(f"Starting to resize all images to {target_size[0]}x{target_size[1]} pixels...")
    
    # Process each directory
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nProcessing directory: {directory}")
            process_directory(directory, target_size)
        else:
            print(f"Directory not found: {directory}")
    
    print("\nAll images have been resized to 256x256 pixels!")

if __name__ == "__main__":
    main()
