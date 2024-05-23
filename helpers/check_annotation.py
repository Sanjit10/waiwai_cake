import os
import random
import argparse
from PIL import Image, ImageDraw

def draw_bounding_boxes(image, annotations, yolo_version='yolov8'):
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        # Detect delimiter
        delimiter = ',' if ',' in annotation else ' '

        # Split annotation into parts based on delimiter
        parts = annotation.strip().split(delimiter)

        # Process coordinates based on YOLO version
        if yolo_version == 'yolov8':
            coordinates = [float(coord) for coord in parts[1:]]
            # Convert normalized coordinates to absolute coordinates
            x1 = int(coordinates[0] * image.width)
            y1 = int(coordinates[1] * image.height)
            x2 = int(coordinates[2] * image.width)
            y2 = int(coordinates[3] * image.height)
            x3 = int(coordinates[4] * image.width)
            y3 = int(coordinates[5] * image.height)
            x4 = int(coordinates[6] * image.width)
            y4 = int(coordinates[7] * image.height)
        elif yolo_version == 'yolov5':
            coordinates = list(map(int, parts[:-1]))
            x1, y1, x2, y2, x3, y3, x4, y4 = coordinates

        # Draw bounding box
        draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline="red")

    return image

def create_collage(images, collage_width, collage_height):
    collage = Image.new("RGB", (collage_width * 416, collage_height * 416))
    for i, image in enumerate(images):
        # Calculate position for the image in the collage
        x = (i % collage_width) * 416
        y = (i // collage_width) * 416
        collage.paste(image, (x, y))
    return collage

def main():
    parser = argparse.ArgumentParser(description="Display annotated images as collages")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--label_path", type=str, required=True, help="Path to the folder containing labels")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to display (default: 16)")
    parser.add_argument("--yolo_version", type=str, choices=['yolov5', 'yolov8'], default='yolov5', help="Specify YOLO version (default: yolov5)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise ValueError("Image path provided does not exist. Please provide a valid path.")

    if not os.path.exists(args.label_path):
        raise ValueError("Label path provided does not exist. Please provide a valid path.")

    # Get list of image file names
    image_filenames = [f for f in os.listdir(args.image_path) if f.endswith('.jpg')]

    if not image_filenames:
        raise ValueError("No image files found in the specified image path.")

    # Randomly choose specified number of photos
    num_images = len(image_filenames)
    num_samples = min(args.num_samples, num_images) if args.num_samples is not None else num_images

    random_image_filenames = random.sample(image_filenames, num_samples)

    # Create a list to store the images with bounding boxes
    images_with_boxes = []

    # Load images in batches
    batch_size = min(num_samples, 16)
    for i in range(0, num_samples, batch_size):
        # Load a batch of images
        batch_image_paths = [os.path.join(args.image_path, filename) for filename in random_image_filenames[i:i+batch_size]]
        batch_images = [Image.open(image_path) for image_path in batch_image_paths]

        # Load annotations for the batch
        batch_annotations = []
        for image_path in batch_image_paths:
            label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
            label_path = os.path.join(args.label_path, label_filename)
            with open(label_path, 'r') as file:
                batch_annotations.append(file.readlines())

        # Draw bounding boxes for the batch
        batch_images_with_boxes = []
        for image, annotations in zip(batch_images, batch_annotations):
            batch_images_with_boxes.append(draw_bounding_boxes(image, annotations, args.yolo_version))

        images_with_boxes.extend(batch_images_with_boxes)

    # Determine collage dimensions
    collage_width = min(4, num_samples)
    collage_height = min((num_samples + 3) // 4, 4) if num_samples > 16 else 4

    # Create collage(s) and display
    start = 0
    end = 16
    while start < num_samples:
        collage_images = images_with_boxes[start:end]
        collage = create_collage(collage_images, collage_width, collage_height)
        collage.show()
        start = end
        end += 16

if __name__ == "__main__":
    main()



"""
Command to run the script:
python check_annotation.py --image_path /home/swordlord/crimson_tech/Data_Augmentation/test --label_path /home/swordlord/crimson_tech/Data_Augmentation/test --num_samples 16 --yolo_version yolov5

python check_annotation.py --image_path /home/swordlord/crimson_tech/Data_Augmentation/tags_obb.v1i.yolov5-obb/test/images --label_path /home/swordlord/crimson_tech/Data_Augmentation/tags_obb.v1i.yolov5-obb/test/label --num_samples 16 --yolo_version yolov8

"""