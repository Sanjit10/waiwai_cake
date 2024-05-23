from PIL import Image
import numpy as np

def extract_brown_region(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Convert the image to HSV
    hsv_image = image.convert('HSV')
    hsv_data = np.array(hsv_image)
    
    # Define the lower and upper bounds for the brown color in HSV space
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([30, 255, 200])
    
    # Create a mask for the brown color
    mask = ((hsv_data[:,:,0] >= lower_brown[0]) & (hsv_data[:,:,0] <= upper_brown[0]) &
            (hsv_data[:,:,1] >= lower_brown[1]) & (hsv_data[:,:,1] <= upper_brown[1]) &
            (hsv_data[:,:,2] >= lower_brown[2]) & (hsv_data[:,:,2] <= upper_brown[2]))
    
    # Create an output image that only includes the brown regions
    output_image_data = np.zeros_like(hsv_data)
    output_image_data[mask] = np.array(image)[mask]
    
    # Convert the output array back to an image
    output_image = Image.fromarray(output_image_data, 'RGB')
    
    return output_image

# Example usage
image_path = 'path_to_your_image.jpg'
brown_regions = extract_brown_region(image_path)

# Display the result
brown_regions.show()
