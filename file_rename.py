import os
import tkinter as tk
from tkinter import filedialog

def rename_files_in_folder():
    # Open dialog to select folder
    root = tk.Tk()
    root.withdraw()  # Hide main window
    folder_path = filedialog.askdirectory(title="Select Folder Containing Files to Rename")

    if not folder_path:
        print("No folder selected.")
        return

    base_name = "WyvernGale_FullArena1"
    files = sorted(os.listdir(folder_path))

    counter = 1
    for file_name in files:
        full_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_path):
            file_ext = os.path.splitext(file_name)[1]
            new_name = f"{base_name}{counter}{file_ext}"
            new_full_path = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(full_path, new_full_path)
            print(f"Renamed: {file_name} -> {new_name}")
            counter += 1

    print("Renaming complete!")

if __name__ == "__main__":
    rename_files_in_folder()
