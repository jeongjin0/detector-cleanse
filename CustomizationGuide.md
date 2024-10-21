# Customization Guide for Detector Cleanse

This guide provides instructions on how to customize the Detector Cleanse implementation to fit your specific object detection model and setup. Below are the key areas you may need to modify:

## 1. Using the Correct Bounding Box Format

Different models may output bounding boxes in various formats such as `[xmin, ymin, xmax, ymax]`, `[ymin, xmin, ymax, xmax]`, or `[center_x, center_y, width, height]`. In `bbox_tool.py`, IoU calculation functions for these different formats are defined. Use the appropriate function based on your model's bounding box format.

Example:
```python
# Adjust the IoU calculation based on your bbox format
ious = box_iou_ymin_xmin_ymax_xmax(torch.tensor(bbox).unsqueeze(0), torch.tensor(perturbed_bboxes))
```
## 2. Adjusting the Model's Output

For our implementation, we have modified the `predict` function of the Faster R-CNN model to output class-wise probabilities separately. Depending on your model, you may need to adjust how the output, specifically the class-wise probabilities is processed. Ensure that the probability extraction aligns with your model's output format.

Example:
```python
probs = model_output['class_probabilities']  # This is an example, modify as per your model's output
```

## 3. Modifying the Prediction Call
The way you call your model to make a prediction might be different. Modify the prediction call in the Detector Cleanse process to align with your model's API.

Example:

```python
prediction = model([ori_img])  # Adjust this call to fit your model's prediction method
```


## 4. Generating Poisoning Data

To generate poisoning data for testing your Detector Cleanse implementation, you can use the following code as a starting point:
data.dataset, utils.config are from our base repository [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)

```python
from data.dataset import Dataset, TestDataset
from torch.utils import data as data_
import torchvision.transforms as transforms
import os
from backdoor_utils import backdoor_inject
from utils.config import opt

def generate_poisoning_data(n=10):
    testset = TestDataset(opt)  # or Dataset(opt)
    test_dataloader = data_.DataLoader(testset, batch_size=1, num_workers=2, shuffle=False)
    image_dir = 'test_images'
    os.makedirs(image_dir, exist_ok=True)
    
    for i, (image, bbox, label, scale) in enumerate(test_dataloader):
        if i >= n:
            break

        image = transforms.ToPILImage()(image.squeeze(0))
        clean_path = os.path.join(image_dir, f'clean_{i+1}.jpg')
        image.save(clean_path)
        print(f"Saved clean image {i+1} to {clean_path}")

        poisoned_image = backdoor_inject(image, bbox, label)
        poisoned_path = os.path.join(image_dir, f'modified_{i+1}.jpg')
        poisoned_image.save(poisoned_path)
        print(f"Saved poisoned image {i+1} to {poisoned_path}")

# Call the function to generate data
generate_poisoning_data()
```

## 5. Additional Customizations
Depending on your specific use case, you might need to make additional customizations. This could include adjusting the image preprocessing steps, modifying the entropy calculation method, or changing the way results are interpreted and presented.

