import os
import hashlib
from pathlib import Path

wd = Path("./data")


def calculate_md5(file_path):
    """
    Calculate the MD5 hash of a file.
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def compare_folders(original_npz_folder, new_npz_folder):
    """
    Compare files in two folders using MD5 hashes.
    """
    original_files = set(os.listdir(original_npz_folder))
    new_npz_files = set(os.listdir(new_npz_folder))

    common_files = original_files.intersection(new_npz_files)

    for index , file_name in enumerate(common_files):
        file1_path = os.path.join(original_npz_folder, file_name)
        file2_path = os.path.join(new_npz_folder, file_name)

        md5_1 = calculate_md5(file1_path)
        md5_2 = calculate_md5(file2_path)

        if md5_1 == md5_2:
            print(f"{index +1} File '{file_name}' is the same in both folders.")
        else:
            print(f"{index +1} File '{file_name}' is different in both folders.")

original_npz_folder = wd / "original_cubexy"
new_npz_folder = wd / "cubesxy"

compare_folders(original_npz_folder, new_npz_folder)
