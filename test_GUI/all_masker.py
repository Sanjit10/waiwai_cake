import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Path to the .pkl file, we obtain this file after using the HSV_highlighter. it provides the list of lower and upper bound
# hsv values for the required mask
PKL_FILE = 'test_GUI/hsv_highlighter.pkl'


class mask_image:
    def __init__(self, path_to_pkl):
        self.pkl_path = path_to_pkl

        # Open the .pkl file in read-binary mode to obtain its content
        with open(self.pkl_path, 'rb') as file:
            self.mask_list = pickle.load(file)

    def highlighter(self,image, color):

        img_copied = image.copy()
        masks = [inner_list for i, inner_list in enumerate(
          self.mask_list) if inner_list not in self.mask_list[:i]]
        combined_mask = np.zeros_like(img_copied[:, :, 0])
        for mask in masks:
            low = np.array(mask[0])
            up = np.array(mask[1])
            hsv_image = cv2.cvtColor(img_copied, cv2.COLOR_BGR2HSV)
            temp_mask = cv2.inRange(hsv_image, low, up)
            combined_mask = cv2.bitwise_or(combined_mask, temp_mask)
        green_image = np.zeros_like(img_copied)
        green_image[:] = color

        # Here we could implement for masking operation for the multiple images
        masked_image = cv2.bitwise_and(
            img_copied, img_copied, mask=combined_mask)
        green_image = cv2.bitwise_or(
            green_image, green_image, mask=combined_mask)

        return masked_image, green_image, combined_mask
    


m1 = mask_image(PKL_FILE)
image = cv2.imread('data/day_chauchau.jpeg') 
masked_image, green_image, combined_mask = m1.highlighter(image, (0, 255, 0))
cv2.imshow('masked_image', masked_image)
cv2.imshow('green_image', green_image)
cv2.imshow('combined_mask', combined_mask)
cv2.waitKey(0)