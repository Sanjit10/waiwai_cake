from skimage.color import rgb2lab, deltaE_ciede2000
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class ColorAnalyzer:
    def __init__(self, reference_image_path: str):
        """
        Initialize the ColorAnalyzer with a reference image.
        """
        self.reference_image = cv2.imread(reference_image_path)
        if self.reference_image is None:
            raise ValueError(f"Failed to load reference image from {reference_image_path}")

    def analyze(self, test_image_path: str):
        """
        Analyze the color difference between the reference image and a test image.
        """
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            raise ValueError(f"Failed to load test image from {test_image_path}")

        test_image_lab = rgb2lab(test_image)

        # Resize the reference image to match the test image and convert to LAB color space
        resized_image = cv2.resize(self.reference_image, (test_image_lab.shape[1], test_image_lab.shape[0]))
        reference_image_lab = rgb2lab(resized_image)

        # Calculate the color difference between the two images
        diff = deltaE_ciede2000(reference_image_lab, test_image_lab)
        return {"Color analysis result": diff}

    def calculate_mean_color(self, image_path: str):
        """
        Calculate the mean color of an image, excluding outliers.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        rgb_values = image.reshape(-1, 3)

        # Exclude outliers
        rgb_values = rgb_values[(np.abs(stats.zscore(rgb_values)) < 2).all(axis=1)]
        mean_color = np.mean(rgb_values, axis=0)

        plt.figure()
        plt.imshow([[mean_color / 255]])
        plt.axis('off')
        plt.show()

        return mean_color
