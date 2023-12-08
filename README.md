# Detector Cleanse

This repository is an unofficial implementation of Detector Cleanse, extending the functionality of the [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/tree/master) to include a method for identifying potentially poisoned input in object detection models.

## Getting Started

To use this tool, first download the clean feature images. These images are used by the Detector Cleanse algorithm to determine if an input image has been tampered with.

### Download Clean Feature Images

The clean feature images used in this repository are composed of bounding box regions from the VOC2007 dataset's test set, converted into images. You can download these prepared images from the following link:
[Clean Feature Images](https://drive.google.com/drive/folders/1Ao5X3ZYSMYwfxApmgSleotgbtRxeCV14?usp=drive_link)

Alternatively, you may also use your own folder of JPG images as clean feature images. Ensure that the images are representative of the types of objects and scenes your model will process.

### Setting up the Environment

Ensure Python is installed along with the necessary packages. Use the provided `requirements.txt` file to install the required packages:

```bash
pip install -r requirements.txt

```

### Usage

To run the Detector Cleanse, use the following command-line arguments:

- `--n`: Number of features to randomly select from the clean feature images.
- `--m`: Detection mean for identifying poison.
- `--delta`: Detection threshold for identifying poison.
- `--image_path`: Path to the image that needs to be analyzed.
- `--clean_feature_path`: Path to the clean feature image folder. Default is 'clean_feature_images'.
- `--weight`: Path to the weight file of the model.

```bash
python main.py --n 100 --m 0.51 --delta 0.16 --image_path 'path/to/image.jpg' --clean_feature_path 'path/to/clean_feature_images' --weight 'path/to/model/weight.pth'
```

### Additional Customization

Since this implementation of Detector Cleanse is based on a specific repository ([simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/tree/master)), additional modifications might be necessary to test it with your own model. To accommodate different object detection models or different setups, you may need to make changes in the code.

For detailed instructions on how to customize this implementation for your specific needs, please refer to the [Customization Guide](CustomizationGuide.md).
