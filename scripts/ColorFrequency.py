import cv2
import numpy as np
from collections import Counter
from skimage.color import rgb2lab, deltaE_ciede2000
from PIL import Image, ImageDraw
import os, random
import matplotlib.pyplot as plt
from io import BytesIO

class DominantColorAnalyzer:
    def get_dominant_colors(self, image, top_colors=5):
        """
        Get the dominant colors in an image.

        Parameters:
        image (np.array or str): The image array or file path.
        top_colors (int): The number of dominant colors to return.

        Returns:
        list: The dominant colors in the image.
        """
        # If image is a string, assume it's a file path and load the image
        if isinstance(image, str):
            image = Image.open(image)
            image = np.array(image)

        # Flatten the image to get a list of pixels
        pixels = image.reshape(-1, 3)

        # Count the frequency of each color
        counter = Counter(map(tuple, pixels))

        # Get the most common colors
        most_common = counter.most_common(top_colors)

        # Extract the RGB values of the most common colors
        dominant_colors = [color for color, count in most_common]
        return dominant_colors

    def visualize_colors(self, colors, width=100, height=50):
        """
        Visualize the dominant colors in an image.

        Parameters:
        colors (list): The dominant colors.
        width (int): The width of the output image.
        height (int): The height of the output image.

        Returns:
        PIL.Image: The output image.
        """
        assert len(colors) > 0, "No colors to visualize."

        band_height = height // len(colors)
        result_image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(result_image)
        for i, color in enumerate(colors):
            draw.rectangle([0, i * band_height, width, (i + 1) * band_height], fill=color)
        return result_image

    def calculate_color_difference(self, colors1, colors2):
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

    def get_image_files(self, directory):
        """
        Get all image files in a directory.

        Parameters:
        directory (str): The directory path.

        Returns:
        list: The image files in the directory.
        """
        assert os.path.isdir(directory), f"{directory} is not a valid directory."

        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def create_file_pairs(self, files):
        """
        Create pairs of files.

        Parameters:
        files (list): The list of files.

        Returns:
        list: The pairs of files.
        """
        assert len(files) > 1, "Not enough files to create pairs."

        random.shuffle(files)
        return [(files[i], files[i+1]) for i in range(0, len(files)-1, 2)]

    def analyze_image_pairs(self, file_pairs, good_cake_files, bad_cake_files):
        """
        Analyze pairs of images and categorize them based on their color differences.

        Parameters:
        file_pairs (list): The pairs of files.
        good_cake_files (list): The list of good cake files.
        bad_cake_files (list): The list of bad cake files.

        Returns:
        dict: The categorized color differences.
        """
        result_dict = {'good-good': [], 'good-bad': [], 'bad-bad': []}
        for file1, file2 in file_pairs:
            colors1 = self.get_dominant_colors(file1)
            colors2 = self.get_dominant_colors(file2)
            color_diff = self.calculate_color_difference(colors1, colors2)
            if file1 in good_cake_files and file2 in good_cake_files:
                result_dict['good-good'].append(color_diff)
            elif file1 in bad_cake_files and file2 in bad_cake_files:
                result_dict['bad-bad'].append(color_diff)
            else:
                result_dict['good-bad'].append(color_diff)
        return result_dict

    def flatten_list_of_lists(self, lol):
        """
        Flatten a list of lists into a single list.

        Parameters:
        lol (list): The list of lists.

        Returns:
        list: The flattened list.
        """
        return [item for sublist in lol for item in sublist]

    def visualize_color_differences(self, result_dict):
        """
        Visualize the color differences.

        Parameters:
        result_dict (dict): The categorized color differences.

        Returns:
        None
        """
        categories = list(result_dict.keys())
        data = [self.flatten_list_of_lists(result_dict[category]) for category in categories]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data, patch_artist=True, vert=True, tick_labels=categories)
        ax.set_title('Color Differences by Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Color Difference (CIEDE2000)')
        ax.grid(True)

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

    def flatten_list_of_lists(self, lol):
        """
        Flatten a list of lists into a single list.

        Parameters:
        lol (list): The list of lists.

        Returns:
        list: The flattened list.
        """
        return [item for sublist in lol for item in sublist]

    def visualize_color_differences(self, result_dict):
        """
        Visualize the color differences.

        Parameters:
        result_dict (dict): The categorized color differences.

        Returns:
        BytesIO: The plot image.
        """
        categories = list(result_dict.keys())
        data = [self.flatten_list_of_lists(result_dict[category]) for category in categories]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data, patch_artist=True, vert=True, tick_labels=categories)
        ax.set_title('Color Differences by Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Color Difference (CIEDE2000)')
        ax.grid(True)

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

    def analyze_and_visualize(self, good_cake_dir, bad_cake_dir):
        """
        Analyze and visualize the color differences in good and bad cakes.

        Parameters:
        good_cake_dir (str): The directory of good cake images.
        bad_cake_dir (str): The directory of bad cake images.

        Returns:
        tuple: The result dictionary and the plot image.
        """
        good_cake_files = self.get_image_files(good_cake_dir)
        bad_cake_files = self.get_image_files(bad_cake_dir)
        all_files = good_cake_files + bad_cake_files
        file_pairs = self.create_file_pairs(all_files)
        result_dict = self.analyze_image_pairs(file_pairs, good_cake_files, bad_cake_files)
        plot_image = self.visualize_color_differences(result_dict)
        return result_dict, plot_image

if __name__ == "__main__":
    analyzer = DominantColorAnalyzer()
    good_cake_dir = 'data/good_cake'
    bad_cake_dir = 'data/bad_cake'
    result_dict, plot_image = analyzer.analyze_and_visualize(good_cake_dir, bad_cake_dir)
    # Display the plot image
    plot = Image.open(plot_image)
    plot.show()
    # Save the plot image if needed
    plot.save('color_difference_plot.png')       