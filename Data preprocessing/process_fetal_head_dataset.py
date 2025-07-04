import os
import sys
from crop_ultrasound_fan import crop_ultrasound_fan
import glob

def process_fetal_head_dataset():
    """
    Process all images in the Fetal_head_preprocessed dataset by cropping them
    and replacing the original images with the cropped versions.
    """
    # Path to the Fetal_head_preprocessed dataset
    dataset_path = "Datasets_preprocessed/Fetal_head_preprocessed"
    
    # Get all PNG files in the dataset directory
    image_files = glob.glob(os.path.join(dataset_path, "*.png"))
    
    # Filter out files that are already cropped (contain "_cropped" in filename)
    original_images = [f for f in image_files if "_cropped" not in os.path.basename(f)]
    
    print(f"Found {len(original_images)} original images to process")
    
    # Process each image
    for i, image_path in enumerate(original_images, 1):
        print(f"Processing {i}/{len(original_images)}: {os.path.basename(image_path)}")
        
        # Create a temporary cropped filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        temp_cropped_path = os.path.join(dataset_path, f"{base_name}_temp_cropped.png")
        
        try:
            # Apply the cropping function
            crop_ultrasound_fan(image_path, temp_cropped_path)
            
            # Check if cropping was successful (file exists and has size > 0)
            if os.path.exists(temp_cropped_path) and os.path.getsize(temp_cropped_path) > 0:
                # Remove the original image
                os.remove(image_path)
                
                # Rename the cropped image to the original filename
                os.rename(temp_cropped_path, image_path)
                print(f"  ✓ Successfully cropped and replaced: {os.path.basename(image_path)}")
            else:
                print(f"  ✗ Cropping failed for: {os.path.basename(image_path)}")
                # Clean up temporary file if it exists
                if os.path.exists(temp_cropped_path):
                    os.remove(temp_cropped_path)
                    
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(image_path)}: {str(e)}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_cropped_path):
                os.remove(temp_cropped_path)
    
    # Clean up any existing "_cropped" files that might be left over
    cropped_files = glob.glob(os.path.join(dataset_path, "*_cropped.png"))
    for cropped_file in cropped_files:
        print(f"Removing leftover cropped file: {os.path.basename(cropped_file)}")
        os.remove(cropped_file)
    
    print(f"\nProcessing complete! All images in {dataset_path} have been cropped.")

if __name__ == "__main__":
    process_fetal_head_dataset() 