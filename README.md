# COS711 Assignment 2: Solar Filament Segmentation

Initial Google Colab project setup for the COS711 solar filament segmentation assignment. At this stage, the project only prepares the Colab environment, checks GPU availability, extracts the official competition data from the course shared drive, and inventories the files. It does not create a model, train, or split the data.

## Run the setup notebook

1. Open [`notebooks/00_colab_data_setup.ipynb`](notebooks/00_colab_data_setup.ipynb) in Google Colab.
2. Make sure you have access to the `COS 711` shared drive in Google Drive, and that it contains `filament-segmentation-2026.zip`.
3. (Recommended) Select a GPU runtime via **Runtime → Change runtime type**.
4. Run the notebook cells from top to bottom. The notebook:
   - prints Python, platform, PyTorch, and GPU/CUDA information,
   - mounts Google Drive and unzips the dataset into `/content/data`,
   - lists the extracted file names relative to `/content/data/filament-segmentation-2026`.

When mounting Drive, Colab will prompt you to authorise access to your Google account.

The data is extracted to `/content/data` in the Colab runtime. It is not saved in this repository, and Colab's runtime storage is temporary; re-run the mount and unzip cells when starting a fresh runtime.

## Files

- `notebooks/00_colab_data_setup.ipynb` — installs dependencies, checks GPU availability, mounts Google Drive, extracts the competition data, and prints a file inventory.
- `requirements.txt` — lists dependencies for optional local use. The notebook installs them directly in Colab.
- `.gitignore` — excludes local data, archives, environment files, and notebook checkpoints from Git.

No Python virtual environment is needed when running the notebook in Colab. If you later run the project locally, you can install the dependencies from `requirements.txt` in your preferred environment. Note that the Drive mount and unzip steps are Colab-specific, so locally you would need to point the data path at your own copy of the dataset.

### Preprocessing:
1. Parse and group
2. Split by observation.
3. Downsampling
4. Image normalisation
5. Build targets