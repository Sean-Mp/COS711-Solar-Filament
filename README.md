# COS711 Assignment 2: Solar Filament Segmentation

Initial Google Colab project setup for the COS711 solar filament segmentation assignment

Notebook 01 prepares the MAGFiLO competition data for filament segmentation. It loads the COCO annotations, groups images into observations, creates a date-blocked 85/15 train/validation split, normalises solar-disk images, and builds cached instance and five-channel segmentation targets. It then defines a PyTorch dataset with crop sampling and augmentation and checks the resulting batches.

## Run the notebook

Open [`notebooks/01_data_preprocessing.ipynb`](notebooks/01_data_preprocessing.ipynb) in Google Colab, select a GPU runtime if available, and run the cells from top to bottom. The notebook mounts the `COS 711` shared drive, extracts `filament-segmentation-2026.zip` into `/content/data`, and writes preprocessing caches to `/content/cache`. Drive access and the archive are required; Colab runtime files are temporary.

## Files

- `notebooks/01_data_preprocessing.ipynb` — data setup, preprocessing, target generation, and PyTorch dataset setup.
- `requirements.txt` — dependencies for optional local use; the notebook installs its additional preprocessing packages in Colab.
- `.gitignore` — excludes local data, archives, environment files, and notebook checkpoints.
