# Python Kernel for Ship Detection in Satellite Images

This repository houses the Python code developed as part of a project to detect and segment ships in satellite imagery, leveraging the power of machine learning with Keras and the Mask R-CNN architecture. It builds on the work found in [MaskRCNN by Matterport](https://github.com/matterport/Mask_RCNN) and uses data from the [Airbus Ship Detection Competition hosted in kaggle](https://www.kaggle.com/c/airbus-ship-detection) on Kaggle.

## Our Approach

Our project doesn't modify the original [MaskRCNN by Matterport](https://github.com/matterport/Mask_RCNN) implementation exactly. Instead, we focus on preparing and processing the data necessary to fit the model. We also adapt the output post-processing steps to better suit our needs, drawing inspiration from [Matterport's MASKRCNN implementation](https://github.com/matterport/Mask_RCNN).

## Preparing the Data

1. Download the Kaggle DataSet From the [Airbus Ship Detection Competition page](https://www.kaggle.com/c/airbus-ship-detection).

2. Extract all of them inside the `data` folder. You should have two folder, `train_v2` and `test_v2` containing the training and testing image dataset. Also you will have two csv files, one is `train_ship_segmentation_v2.csv` and another is `sample_submission_v2.csv`.

## Training

Since our training data is ready inside our `data` folder, it's time to train the model. For our convinience, we have added a config folder inside the keras_maskrcnn and placed a config.ini, where we have changed the anchor size, because ship instances were mostly small and we wanted to detect all the small ships.

Assuming we have the annotation.csv, class_ids.csv, we can now proceed to the training.

1. Download the Coco weight provided from [COCO dataset](https://cocodataset.org/#home). We used the
   mask_rcnn_coco.h5

## References:

Lots of references of codes and implemntation were taken from the various sources, which are not least but listed below:

1. [https://github.com/matterport/Mask_RCNN] Many of the utility functions were actually the modified version available in the `utility.py` of matterport's Mask RCNN.
2. [Kaggle Airbus Ship Detection competition](https://www.kaggle.com/c/airbus-ship-detection), Kernels.
