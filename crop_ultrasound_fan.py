import cv2
import numpy as np
import sys
import os

def crop_ultrasound_fan(input_path, output_path):
    """
    Crop an ultrasound image to remove the black background and keep only the fan-shaped ultrasound content.
    
    Args:
        input_path (str): Path to the input ultrasound image
        output_path (str): Path where the cropped image will be saved
    """
    # Load the ultrasound image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    # Convert the image to grayscale for easier processing
    # Ultrasound images often have a dark background that we want to remove
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding to separate the ultrasound content from the background
    # Pixels with intensity > 10 become white (255), others become black (0)
    # This creates a binary mask where the ultrasound fan is white and background is black
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    # Contours are the boundaries of connected components (in this case, the ultrasound fan)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"Error: No contours found in {input_path}")
        return
    
    # Find the largest contour, which should be the main ultrasound fan
    # This assumes the ultrasound content is the largest object in the image
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask with the same size as the original image
    mask = np.zeros_like(gray)
    
    # Fill the largest contour (ultrasound fan) with white pixels
    # This creates a mask where only the ultrasound content is white
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Apply the mask to the original image using bitwise AND operation
    # This keeps only the pixels that are white in both the original image and the mask
    # Effectively removing the black background
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Find the bounding rectangle of the largest contour
    # This gives us the coordinates to crop the image tightly around the ultrasound fan
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image to the bounding rectangle dimensions
    # This removes any remaining black areas around the ultrasound fan
    cropped = result[y:y+h, x:x+w]
    
    # Save the cropped image to the specified output path
    cv2.imwrite(output_path, cropped)
    print(f"Saved cropped image to {output_path}")

if __name__ == "__main__":
    # Check if the correct number of command line arguments is provided
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <input_image> <output_image>")
        sys.exit(1)
    
    # Call the cropping function with the provided input and output paths
    crop_ultrasound_fan(sys.argv[1], sys.argv[2]) 