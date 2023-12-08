# Customization Guide for Detector Cleanse

This guide provides instructions on how to customize the Detector Cleanse implementation to fit your specific object detection model and setup. Below are the key areas you may need to modify:

## 1. Loading the Model

Your object detection model might have a different architecture or require a unique loading method. Update the model loading section to instantiate and load your specific model.

Example:
```python
from model import YourModelClass

# Load your custom model
model = YourModelClass()
model.load_state_dict(torch.load(args.weight))
```
## 2. Using the Correct Bounding Box Format

Different models may output bounding boxes in various formats such as `[xmin, ymin, xmax, ymax]`, `[ymin, xmin, ymax, xmax]`, or `[center_x, center_y, width, height]`. In `bbox_tool.py`, IoU calculation functions for these different formats are defined. Use the appropriate function based on your model's bounding box format.

Example:
```python
# Adjust the IoU calculation based on your bbox format
ious = box_iou_ymin_xmin_ymax_xmax(torch.tensor(bbox).unsqueeze(0), torch.tensor(perturbed_bboxes))
```
## 3. Adjusting the Model's Output

For our implementation, we have modified the `predict` function of the Faster R-CNN model to output class-wise probabilities separately. Depending on your model, you may need to adjust how the output, specifically the class-wise probabilities is processed. Ensure that the probability extraction aligns with your model's output format.

Example:
```python
probs = model_output['class_probabilities']  # This is an example, modify as per your model's output
```

## 4. Modifying the Prediction Call
The way you call your model to make a prediction might be different. Modify the prediction call in the Detector Cleanse process to align with your model's API.

Example:

```python
prediction = model([ori_img])  # Adjust this call to fit your model's prediction method
```

## 5. Additional Customizations
Depending on your specific use case, you might need to make additional customizations. This could include adjusting the image preprocessing steps, modifying the entropy calculation method, or changing the way results are interpreted and presented.

