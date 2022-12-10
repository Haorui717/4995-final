# Editing from Fake Images to Real Images

This is the GitHub repository for COMS4995:006 Deep Learning for Computer Vision final project.
Group Members: Haorui Song (hs3374), Haoqing Wang (hw2888), Ziggy Chen (xc2605) in letter order.

## Environment Setup
+ We run the project on Linux operating system equipped with a Tesla K80 GPU.
+ CUDA is required, the version of which is 11.2.
+ The full environment requirement is specified in `environment.yaml`. In particular,
the pytorch version is 1.10. 
+ `Ninja` is required because the project needs the C++ cuda extension. To install ninja please run
    ```shell
    sudo apt install ninja-build
    pip install ninja
    ```

## Quick Start

+ Clone the repository first:
    ```shell
    git clone https://github.com/Haorui717/4995-final.git
    cd 4995-final
    ```
+ Our editing method requires a pretrained styleGAN weights. We used <a href="https://github.com/eladrich/pixel2style2pixel">pixel2style2pixel</a>
encoder in the project, which has integrated the styleGAN model. Therefore, please download the
<a href="https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing">pretrained weights</a> of the pSp encoder.
+ Run the `edit_image.py` to edit some sample images. We have put the semantic directions
we calculated in the `directions` folder. `--real` means to edit real images rather than fake images.
`--attr_idx` is the index of the attribute to be edited. Here `39` is the index for `Young` attribute.
    ```shell
    python edit_image.py --ckptpath path_to_psp_weight \
                         --direction_dir ./directions \
                         --real \
                         --attr_idx 39 \
                         --image_path ./sample_images \
                         --save_dir folder_to_save_outputs
    ```

## Train your own directions

### Train the classifiers
+ We have preprocessed the CelebA dataset to get the CelebA-HQ dataset. Please download the zipped
dataset, which contains a folder of 30,000 images, `image_list.txt` and `list_attr_celeba.txt`. The
label of each image is obtained according to the two files.
+ Then run the `split_dataset.py` to split the dataset into training set and test set. Here we randomly
selected 27,000 images as the training set and 3,000 images as the test set.
  ```shell
  python split_dataset.py --dataset_dir path_to_celeba_hq_all \
                          --train_dir path_to_new_training_set \
                          --test_dir path_to_new_test_set
  ```
+ please run the `train_classifier.py` to train 40 binary classifiers for all label.
  ```shell
  python train_classifier.py --trainset_path path_to_train_set \
  --testset_path path_to_test_set \
  --image_list_path path_to_image_list.txt \
  --list_attr_celeba_path path_to_list_attr_celeba.txt \
  --ckpt_dir path_to_save_the_classifier_checkpoints \
  --log_dir path_to_log_direction
  ```

### Calculate Semantic Directions
+ Run `gen_labelled_code.py` to randomly generate latents and label them.
  ```shell
  python gen_labelled_code.py --ckptpath path_to_psp_weight \
                              --cls_ckpt_dir path_to_direction_of_classifier_weights \
                              --save_dir path_to_folder_to_save_latents
  ```
+ Run `learn_direction.py` to learn the semantic direction.
```shell
python learn_direction.py --latent_dir path_to_latents \
                          --log_dir path_to_log_dir \
                          --direction_dir direction_to_save_new_directions
```