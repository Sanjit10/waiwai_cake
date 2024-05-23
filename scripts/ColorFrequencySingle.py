import cv2
import numpy as np
from collections import Counter
from skimage.color import rgb2lab, deltaE_ciede2000
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from io import BytesIO

class ImageColorComparator:
    def __init__(self, reference_image):
        """
        Initialize with a reference image.

        Parameters:
        reference_image (np.array or str): The reference image array or file path.
        """
        self.reference_image = self._load_image(reference_image)
        self.reference_colors = self._get_dominant_colors(self.reference_image)

    def _load_image(self, image):
        """
        Load an image from a file path or numpy array.

        Parameters:
        image (np.array or str): The image array or file path.

        Returns:
        np.array: The loaded image.
        """
        if isinstance(image, str):
            image = Image.open(image)
            image = np.array(image)
        return image

    def _get_dominant_colors(self, image, top_colors=5):
        """
        Get the dominant colors in an image.

        Parameters:
        image (np.array): The image array.
        top_colors (int): The number of dominant colors to return.

        Returns:
        list: The dominant colors in the image.
        """
        pixels = image.reshape(-1, 3)
        counter = Counter(map(tuple, pixels))
        most_common = counter.most_common(top_colors)
        dominant_colors = [color for color, count in most_common]
        return dominant_colors

    def _calculate_color_difference(self, colors1, colors2):
        """
        Calculate the color difference between two sets of colors.

        Parameters:
        colors1 (list): The first set of colors.
        colors2 (list): The second set of colors.

        Returns:
        list: The color differences.
        """
        assert len(colors1) == len(colors2), "Color lists must be of the same length."
        color_diff = []
        for color1, color2 in zip(colors1, colors2):
            lab1 = rgb2lab(np.uint8([[color1]]))
            lab2 = rgb2lab(np.uint8([[color2]]))
            diff = deltaE_ciede2000(lab1[0][0], lab2[0][0])
            color_diff.append(diff)
        return color_diff

    def compare_with_image(self, compare_image):
        """
        Compare the colors of the reference image with another image.

        Parameters:
        compare_image (np.array or str): The image array or file path to compare.

        Returns:
        list: The color differences.
        """
        compare_image = self._load_image(compare_image)
        compare_colors = self._get_dominant_colors(compare_image)
        color_diff = self._calculate_color_difference(self.reference_colors, compare_colors)
        return color_diff
    
    def plot_color_differences(self, color_diff):
        """
        Plot a bar graph of the color differences.

        Parameters:
        color_diff (list): The color differences to plot.

        Returns:
        BytesIO: The plot image.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(color_diff)), color_diff, color='blue')
        ax.set_title('Color Differences')
        ax.set_xlabel('Color Pair Index')
        ax.set_ylabel('Color Difference (CIEDE2000)')
        ax.grid(True)

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Return the BytesIO object
        return buf

if __name__ == "__main__":
    # Initialize the comparator with the reference image
    reference_image_path = 'data/good_cake/3_crop_12.png'
    comparator = ImageColorComparator(reference_image_path)
    
    # Compare with another image
    compare_image_path = 'data/good_cake/2_crop_26.png'
    color_diff = comparator.compare_with_image(compare_image_path)
    
    # Plot the color differences
    plot_image = comparator.plot_color_differences(color_diff)

    # Display the plot image
    plot = Image.open(plot_image)
    plot.show()
    # Print the color differences
    print(f"Color differences: {color_diff}")
