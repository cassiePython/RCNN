# Implementation of R-CNN

This is a Implementation of R-CNN. (R-CNN: Regions with Convolutional Neural Network Features)

## Prerequisites
- Python 3.5
- Pytorch 0.4.1
- sklearn 0.18.1
- cv2 3.4.0

You can run the code in Windows/Linux with CPU/GPU. 

## Dataset

For simplicity, rather than use PASCAL VOC 2012 dataset, I choose a samll dataset 17flowers. It can be downloaded from my Google Drive: 

https://drive.google.com/open?id=1qAkLZ-Wa8mca2xz-RZI3XW1Syjk2JfOB

## Structure

The project is structured as follows:

```
├── checkpoints/
├── data/
|   ├── dataset_factory.py    
|   ├── dataset_factory.py    
├── generate/
├── loss/
|   ├── losses.py  
├── models/
|   ├── model_factory.py    
|   ├── models.py  
├── networks/
|   ├── network_factory.py    
|   ├── networks.py 
├── options/
|   ├── base_options.py    
|   ├── test_options.py 
|   ├── train_options.py
├── sample_dataset/
|   ├── 2flowers   
|   ├── 17flowers 
│   ├── fine_tune_list.txt
|   ├── train_list.txt
├── utils/
|   ├── selectivesearch.py    
|   ├── util.py 
├── evaluate.py
├── train_step1.py
├── train_step2.py
├── train_step3.py
├── train_step4.py
```

## Getting started

### Supervised Train

In this step, we use pre-trained AlexNet of Pytorch and train it using 17flowers dataset. 

```
$ python train_step1.py 
```

You can directly run it with default parameters.

### Finetune

In this step, we finetune the model on the 2flowers dataset.

```
$ python train_step2.py --batch_size 128 --load_alex_epoch 100 --options_dir finetune
```

Here we use the pre-trained AlexNet with 100 training epoch in the first step. "Options_dir" decides where to save parameters.

### Train SVM

In this step, we use features extracted from the last step to train SVMs. As stated in the paper, class-specific SVMs are trained in this step. Here there are two SVMs.

```
$ python train_step3.py --load_finetune_epoch 100 --options_dir svm
```

Here we adopt the finetuned AlexNet with 100 epoch in last step as feature extracter.

### Box Regression

In this step, we train a regression network to refine bounding boxes.

```
$ python train_step4.py --decay_rate 0.5 --options_dir regression --batch_size 512
```

### Evaluate

```
$ python evaluate.py --load_finetune_epoch 100 --load_reg_epoch 40 --img_path ./sample_dataset/2flowers/jpg/1/image_1281.jpg
```

![](https://github.com/cassiePython/RCNN/blob/master/Figure_1.png)

## References

- Selective-search: https://github.com/AlpacaDB/selectivesearch
- R-CNN with Tensorflow: https://github.com/bigbrother33/Deep-Learning

Many codes related to image proposals are copied directly from "R-CNN with Tensorflow", and this project has shown more details about R-CNN in chinese.
