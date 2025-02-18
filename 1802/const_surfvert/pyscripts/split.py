import os
import shutil

# Function to create folders and split files equally into three folders
def split_txts_equally(folder_path):
    # Paths for the new folders
    txts_1_path = os.path.join(folder_path, "pro_newtxts_1")
    txts_2_path = os.path.join(folder_path, "pro_newtxts_2")
    txts_3_path = os.path.join(folder_path, "pro_newtxts_3")

    # Create directories if they don't exist
    os.makedirs(txts_1_path, exist_ok=True)
    os.makedirs(txts_2_path, exist_ok=True)
    os.makedirs(txts_3_path, exist_ok=True)

    # Iterate through files in the original folder
    for i, file_name in enumerate(os.listdir(folder_path)):
        if file_name.endswith(".txt"):
            try:
                # Determine the destination folder based on the file index
                if i % 3 == 0:
                    destination_folder = txts_1_path
                elif i % 3 == 1:
                    destination_folder = txts_2_path
                else:
                    destination_folder = txts_3_path

                # Copy the file to the appropriate folder
                shutil.copy(os.path.join(folder_path, file_name), destination_folder)

            except Exception as e:
                print(f"Skipping file due to error: {file_name}, Error: {e}")

# Example usage
folder_path = "protein_20_txts"  # Replace with the path to your folder
split_txts_equally(folder_path)
