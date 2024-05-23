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

    return new_image, aspect_ratio

def adjust_annotations(annotations, aspect_ratio):
    adjusted_annotations = []
    for annotation in annotations:
        # Split annotation into x, y coordinates and tag
        coordinates = annotation.strip().split(',')[:-1]  # Exclude the last element (tag)
        coordinates = [int(coord) for coord in coordinates]

        # Adjust coordinates based on aspect ratio
        adjusted_coordinates = [int(coord * aspect_ratio) for coord in coordinates]
        adjusted_annotation = ','.join(map(str, adjusted_coordinates))

        # Add tag back to the adjusted annotation
        adjusted_annotation += ',' + annotation.strip().split(',')[-1]
        adjusted_annotations.append(adjusted_annotation)

    return adjusted_annotations

# Folder containing original images and annotations
folder_path = "/home/swordlord/crimson_tech/Data_Augmentation/train/"

# Target size for letterbox method
target_size = (416, 416)

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        # Load original image
        original_image = Image.open(os.path.join(folder_path, filename))

        # Resize the image using letterbox method
        resized_image, aspect_ratio = resize_image_with_letterbox(original_image, target_size)

        # Load annotation file
        annotation_path = os.path.join(folder_path, os.path.splitext(filename)[0] + ".txt")
        with open(annotation_path, 'r') as file:
            annotations = file.readlines()

        # Adjust annotations based on aspect ratio
        adjusted_annotations = adjust_annotations(annotations, aspect_ratio)

        # Save the resized image
        resized_image.save(os.path.join(folder_path, filename))

        # Save the adjusted annotations back to the file
        with open(annotation_path, 'w') as file:
            file.writelines("\n".join(adjusted_annotations))
