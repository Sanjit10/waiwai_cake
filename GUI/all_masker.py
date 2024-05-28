import cv2
import numpy as np, random
import pickle

class mask_image:
    def __init__(self, path_to_pkl):
        self.pkl_path = path_to_pkl

        # Open the .pkl file in read-binary mode to obtain its content
        self.mask_list=[[[13, 47, 29], [29, 250, 199]]]
        # with open(self.pkl_path, 'rb') as file:
            # self.mask_list = pickle.load(file)
            
            

    def highlighter(self,image):

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
            
        
        #Randomly select a tag for the combined mask between 'Good' and 'Bad'
        result = random.choice(['Good', 'Bad'])
        return combined_mask, result
