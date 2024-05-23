import os
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_folder, annotation_folder, output_folder):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.output_folder = output_folder

    def load_annotations(self, annotation_path, image_shape):
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        annotations = []
        for line in lines:
            parts = line.strip().split(' ')
            class_id = int(parts[0])
            corners = np.array(parts[1:], dtype=float).reshape((4, 2))
            # Denormalize the corners
            corners[:, 0] *= image_shape[1]  # Width
            corners[:, 1] *= image_shape[0]  # Height
            annotations.append((class_id, corners))
        return annotations

    def crop_and_save_images(self):
        # Get all image files in the directory
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

        # Process each image file
        for image_file in image_files:
            # Load the image
            image_path = os.path.join(self.image_folder, image_file)
            image = cv2.imread(image_path)

            # Load annotations
            image_shape = image.shape
            annotation_path = os.path.join(self.annotation_folder, os.path.splitext(image_file)[0] + '.txt')
            if not os.path.exists(annotation_path):
                print(f"No annotation file found for {image_file}, skipping.")
                continue
            annotations = self.load_annotations(annotation_path, image_shape)

            # Ensure the output directory exists
            output_dir = os.path.join(self.output_folder, os.path.splitext(image_file)[0])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Process each annotation
            for idx, (_, corners) in enumerate(annotations):
                # Ensure corners is a numpy array of type float32
                if not isinstance(corners, np.ndarray) or corners.dtype != np.float32:
                    corners = np.array(corners, dtype=np.float32)

                # Get the rotated rectangle from the points
                rect = cv2.minAreaRect(corners)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                # Get the width and height of the bounding box
                width = int(rect[1][0])
                height = int(rect[1][1])

                # Get the rotation matrix
                src_pts = box.astype("float32")
                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # Warp the image to get the rotated rectangle
                warped = cv2.warpPerspective(image, M, (width, height))

                # Save the cropped image
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_crop_{idx}.png")
                cv2.imwrite(output_path, warped)

# Example usage
processor = ImageProcessor('data/dataset_images', 
                           'data/dataset_images', 
                           'data/cropped_images')
processor.crop_and_save_images()