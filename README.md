# Custom Ear Landmark Detection Model

This is a standalone training repository for high-precision ear landmarking.

## Standalone Setup (GitHub & Kaggle)

1.  **Create a New Repo**: Create a new empty repository on GitHub.
2.  **Initialize & Push**:
    ```bash
    cd src/models/ear-landmarker
    git init
    git add .
    git commit -m "Initial commit"
    git remote add origin https://github.com/YOUR_USERNAME/ear-landmarker.git
    git push -u origin main
    ```
3.  **Train on Kaggle**:
    - Upload `kaggle_train.ipynb` to a New Notebook on Kaggle.
    - Set the **Accelerator** to **GPU P100** or **T4 x2**.
    - Run the cells to train. It will download the 2,000 photos and train in minutes.

## Local Dataset Setup

1.  Download the **AudioEar2D** dataset (2,000 images, 55 landmarks) from Zenodo:
    [AudioEar 2D Dataset - Zenodo](https://zenodo.org/record/7581758)
2.  Alternatively, run our downloader script:
    ```bash
    python download_ibug.py  # Now updated to fetch AudioEar2D
    ```
3.  Organize the data as follows:
    ├── images/
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── landmarks/
        ├── 001.txt
        ├── 002.txt
        └── ...
    ```
3.  Ensure landmark text files contain normalized (0.0 to 1.0) coordinates for all points.

## Training

1.  Create a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```
2.  Run the training script (supports dynamic ROI cropping):
    ```bash
    python train.py
    ```

## Verification

After training, run the verification script to see predictions on actual images:
```bash
python verify_model.py
```
This will save result images to the `results/` folder for your inspection.

## Conversion to TensorFlow.js

1.  Install the converter:
    ```bash
    pip install tensorflowjs
    ```
2.  Convert the `ear_landmarker_final.keras` model:
    ```bash
    tensorflowjs_converter --input_format keras ear_landmarker_final.keras ./tfjs_model
    ```
3.  Copy the contents of `tfjs_model` to `public/models/ear-landmarker/`.
