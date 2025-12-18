# Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

This repository contains the code for the paper - Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge

## Installation Instructions

Follow these steps to set up the environment and run the code:

1.  **Create a Conda Environment:**

    Open your terminal and create a new Conda environment with Python 3.x:

    ```
    conda create --name lddbm python=3.8 -y && conda activate lddbm
    ```

2.  **Install Dependencies:**

    Navigate to the root directory of this repository and install the required packages using pip:

    ```
    pip install -r requirements.txt
    ```

3.  **Download Datasets:**
    #### Multi-view to 3D - ShapeNet
       Download the ShapeNet dataset from the following URL:
       ```
       https://github.com/fomalhautb/3D-RETR/archive/refs/heads/main.zip 
       ```
    
       Unzip the files into the below folder within your project. 
       ```
       lddbm/datasets/shapenet 
       ```
    
    #### Zero-shot Super Resolution - Celebs and Flicker.
    Download the datasets VoxCelebs and Flicker50k 
    ```
    https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
    https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
    ```
    and place them in the same folder
    ```
    lddbm/datasets/sr 
    ```
    
    The loading of the files happens in the '__init__.py' file of the datasets folder:
    ```
    train_paths = sorted([str(p) for p in glob(f'{data_path}/Flicker50k' + '/*.png')])
    trainset = CelebaDataset(train_paths, lr_transforms=lr_transforms, hr_transforms=hr_transforms, train=True)

    image_paths = sorted([str(p) for p in glob(f'{data_path}/celebsA_HQ/celeba_hq_256' + '/*.jpg')])
    _, valid_paths = train_test_split(image_paths, test_size=5000, shuffle=True, random_state=42)
     ```
    make sure folder postfix are alike.

       
4. **Run Training and Evaluation:**
    Execute the training and evaluation scripts using the following command:

    For multi-view to 3D task:
    ```
    python scripts/main.py --config_name multi2shape --data_path lddbm/datasets/shapenet "
    ```
   For super resolution task:
    ```
    python scripts/main.py --config_name sr --data_path lddbm/datasets/sr "
    ```

## License

This project is licensed under the AGPL 3.0 License - see the [LICENSE](LICENSE) file for details.

