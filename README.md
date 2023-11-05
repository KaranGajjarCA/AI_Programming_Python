# Image Classifier

## Prerequisites
The Code is written in Python
To install required packages

```
pip install numpy pandas matplotlib pil torch
```
or [conda](https://anaconda.org/anaconda/python)
```
conda install numpy pandas matplotlib pil torch
```

## Command Line Application
* Train a new model ```train.py```
  * Basic Usage : ```python train.py --data_dir data_directory```
  * Options:
    * Set direcotry to save checkpoints: ```python train.py --data_dir data_directory --save_dir save_directory```
    * Choose Arch (densenet121 or vgg16): ```pytnon train.py --data_dir data_directory --arch "vgg16"```
    * Set hyperparameters: ```python train.py --data_dir data_directory --learning_rate 0.001 --hidden_units 500 --epochs 3 ```
    * Use GPU: ```python train.py --data_dir data_directory --gpu gpu```
    
* Predict flower name from an image with ```predict.py``` 
  * Basic usage: ```python predict.py --image_path /path/to/image --checkpoint checkpoint```
  * Options:
    * top_k :``` python predict.py --image_path /path/to/image --checkpoint checkpoint ---top_k 3```
    * category_name : ```python predict.py --image_path /path/to/image --checkpoint checkpoint --category_names cat_To_name.json```
    * Use GPU: ```python predict.py --image_path /path/to/image --checkpoint checkpoint --gpu```
