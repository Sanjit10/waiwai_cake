import os

def replace_first_element(folder_path):
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print("Error: Not a valid directory path.")
        return
    
    # Get all files in the directory
    files = os.listdir(folder_path)
    
    # Filter out only .txt files
    txt_files = [file for file in files if file.endswith('.txt')]
    
    for txt_file in txt_files:
        # Read the contents of the text file
        with open(os.path.join(folder_path, txt_file), 'r') as f:
            lines = f.readlines()
        
        # Process each line and replace the first element with '0'
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = '0'
            updated_line = ' '.join(parts) + '\n'
            updated_lines.append(updated_line)
        
        # Write the updated lines back to the text file
        with open(os.path.join(folder_path, txt_file), 'w') as f:
            f.writelines(updated_lines)

# Example usage:
folder_path = '/home/swordlord/crimson_tech/Data_Augmentation/train_Detection_Dataset'
replace_first_element(folder_path)
