import os

def convert_to_comma_separated(folder_path):
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
        
        # Process each line, round the numbers to integers, omit the last element (class label), and replace spaces with commas
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            rounded_parts = [str(round(float(part))) if part.replace('.', '').isdigit() else part for part in parts[:-1]]
            updated_line = ','.join(rounded_parts)
            updated_lines.append(updated_line + '\n')
        
        # Write the updated lines back to the text file
        with open(os.path.join(folder_path, txt_file), 'w') as f:
            f.writelines(updated_lines)

# Example usage:
folder_path = '/home/swordlord/crimson_tech/Data_Augmentation/good_tags'
convert_to_comma_separated(folder_path)
