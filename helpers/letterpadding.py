import os
from PIL import Image

def resize_image_with_letterbox(image, target_size):
    # Resize the image using letterbox method
    width, height = image.size
    target_width, target_height = target_size
    aspect_ratio = min(target_width / width, target_height / height)
    resized_width = int(width * aspect_ratio)
    resized_height = int(height * aspect_ratio)
    resized_image = image.resize((resized_width, resized_height), Image.LANCZOS)

    # Create a new blank image of the target size
    new_image = Image.new("RGB", target_size, (0, 0, 0))

    # Paste the resized image onto the new image
    new_image.paste(resized_image, (0, 0))

    return new_image

# Folder containing original images
folder_path = "/home/swordlord/crimson_tech/Data_Augmentation/test_copy/"

# Target size for letterbox method
target_size = (416, 416)

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        # Load original image
        original_image = Image.open(os.path.join(folder_path, filename))

        # Resize the image using letterbox method
        resized_image = resize_image_with_letterbox(original_image, target_size)

        # Save the resized image with the same name as the original image
        resized_image.save(os.path.join(folder_path, filename))
