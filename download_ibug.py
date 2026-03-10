import os
import urllib.request
import zipfile
import shutil

def download_audioear():
    """
    Downloads and extracts the AudioEar2D Dataset from Zenodo.
    Contains 2,000 images with 55 landmarks.
    """
    # Direct Zenodo download link for AudioEar2D
    URL = "https://zenodo.org/record/7581758/files/AudioEar2D.zip?download=1"
    ZIP_NAME = "AudioEar2D.zip"
    EXTRACT_DIR = "./data_audioear"

    print(f"Downloading AudioEar2D dataset from Zenodo...")
    try:
        urllib.request.urlretrieve(URL, ZIP_NAME)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading from Zenodo: {e}")
        print("Please visit: https://zenodo.org/record/7581758 and download AudioEar2D.zip manually.")
        return

    print("Extracting...")
    with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    
    # Cleanup zip
    os.remove(ZIP_NAME)
    
    print(f"Dataset extracted to {EXTRACT_DIR}")
    print("Please follow the README.md in the parent folder to organize this into 'images/' and 'landmarks/'.")

if __name__ == "__main__":
    download_audioear()
