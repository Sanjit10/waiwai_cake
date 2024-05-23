#rename script
import os

def rename_files(folder_path):
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print("Error: Not a valid directory path.")
        return
    
    # Get all files in the directory
    files = os.listdir(folder_path)
    # Sort files to ensure consistency
    files.sort()
    
    # Counter to track renaming
    counter = 1
    
    for file in files:
        # Split the file name and extension
        name, ext = os.path.splitext(file)
        # Check if the file is an image file
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            # Rename the image file
            new_name = str(counter) + ext
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
            
            # Check if corresponding annotation file exists
            annotation_file = os.path.join(folder_path, name + '.txt')
            if os.path.exists(annotation_file):
                # Rename the annotation file
                os.rename(annotation_file, os.path.join(folder_path, str(counter) + '.txt'))
            
            # Increment counter
            counter += 1

# Example usage:
folder_path = 'data/noodles_cake_detection/'
rename_files(folder_path)
