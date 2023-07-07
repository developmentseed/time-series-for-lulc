import os
import hashlib
from pathlib import Path
import boto3

wd = Path("./data")

# Get the AWS_PROFILE environment variable
aws_profile_name = os.getenv("AWS_PROFILE")
# Create a session using your AWS profile from the environment variable
session = boto3.Session(profile_name=aws_profile_name)
s3 = session.resource("s3")


def download_file_from_s3(bucket_name, s3_file_name, local_file_name):
    if local_file_name.exists():
        return
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket_name, s3_file_name, local_file_name)
    except Exception as e:
        print(f"{'#'*20} File s3://{bucket_name}/{s3_file_name} not exist")

def calculate_md5(file_path):
    """
    Calculate the MD5 hash of a file.
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def compare_folders(s3_cubesxy, new_npz_folder):
    """
    Compare files in two folders using MD5 hashes.
    """
    original_files = set(os.listdir(s3_cubesxy))
    new_npz_files = set(os.listdir(new_npz_folder))

    common_files = original_files.intersection(new_npz_files)

    for index, file_name in enumerate(common_files):
        file1_path = os.path.join(s3_cubesxy, file_name)
        file2_path = os.path.join(new_npz_folder, file_name)

        md5_1 = calculate_md5(file1_path)
        md5_2 = calculate_md5(file2_path)

        if md5_1 == md5_2:
            print(f"{index +1} File '{file_name}' is the same in both folders.")
        else:
            print(f"{index +1} File '{file_name}' is different in both folders {s3_cubesxy} and {new_npz_folder}.")


s3_cubesxy = wd / "s3_cubesxy"
new_npz_folder = wd / "cubesxy"

# Download original files from S3.
if not os.path.exists(s3_cubesxy):
    os.makedirs(s3_cubesxy)
npzFiles = list(wd.glob("cubesxy/*.npz"))
npzFiles.sort()
for npzFile in npzFiles:
    local_file_name = wd / "s3_cubesxy" / f"{npzFile.stem}.npz"
    download_file_from_s3(
        "ds-labs-lulc", f"cubesxy/{npzFile.stem}.npz", local_file_name
    )

compare_folders(s3_cubesxy, new_npz_folder)
