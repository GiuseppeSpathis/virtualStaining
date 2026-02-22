# Virtual Staining (H&E → CK7 IHC) — Research Project

This repository contains code and experiments for virtual staining in computational pathology: generating CK7 immunohistochemistry (IHC)-like images from standard H&E kidney cancer tiles using a diffusion-based model (PixCell) with lightweight LoRA adaptation in an unpaired setting (no aligned H&E/IHC pairs). 
### PixCell Fork Modifications

This project uses a custom fork of [PixCell as a submodule](https://github.com/GiuseppeSpathis/PixCell.git). The following files have been added or modified to support this project:

* **`run_pixcell_virtual_stain_npz.py`**: Added to perform inference and generate images. This is a Python script version of the original `virtual_staining.ipynb` notebook, adapted to run non-interactively on a remote server. It includes modifications to handle the custom dataset and to speed up the inference process.
* **`npz_dataset.py`**: Added to load and manage the custom `.npz` dataset.
* **`train_lora.py`**: Slightly modified from the original to support the custom dataset and to make the training phase faster.

For a complete description of the methodology, dataset/preprocessing, and evaluation, see the project [report](https://github.com/GiuseppeSpathis/virtualStaining/blob/main/report.pdf).
