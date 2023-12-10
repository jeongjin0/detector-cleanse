import torch
import torchvision.transforms as T

import numpy as np
from skimage.transform import resize

from tqdm import tqdm

from PIL import Image


from bbox_tool import bbox_iou_ymin_xmin_ymax_xmax, bbox_iou_xmin_ymin_xmax_ymax, bbox_iou_cx_cy_w_h
from transform import preprocess


def calculate_entropy(scores):
    return -torch.sum(scores * torch.log2(scores), dim=0).mean()


def perturb_image(image, bbox, feature, alpha=0.5):
    perturbed_image = np.copy(image)

    ymin, xmin, ymax, xmax = map(int, bbox)

    feature_resized = resize(feature, (3, ymax - ymin,  xmax - xmin))
    perturbed_region = perturbed_image[:, ymin:ymax, xmin:xmax]

    blended_region = alpha * feature_resized + (1 - alpha) * perturbed_region

    perturbed_image[:, ymin:ymax, xmin:xmax] = blended_region

    return perturbed_image

def save_numpy_array_as_jpg(array, file_name):
    # Assuming the input array is in the format (3, w, h) and the values are scaled between 0 and 1
    array = array[[2, 1, 0], :, :]
    if array.max() <= 1:
        array = array * 255  # Scale to 0-255 range if not already

    # Convert to uint8
    array = array.astype(np.uint8)

    # Transpose the array to the format (h, w, 3)
    array = array.transpose(1, 2, 0)

    # Convert to an image and save
    image = Image.fromarray(array)
    image.save(file_name + '.jpg')

def detector_cleanse(img, size, model, clean_features, m, delta, iou_threshold=0.5):
    model = model.cuda()

    prediction = model.predict([img], [img.shape[1:]])
    _bboxes, _labels, _scores, probs = prediction
    print(_bboxes)
    print(_labels)
    print(_scores)
    print()

    # Assuming the size is a tuple (width, height)
    original_width, original_height = size

    # Calculate the scale factor
    # Assuming img is a tensor or ndarray with shape (channels, height, width)

    poisoned_flag = False
    coordinates = []
    j=0
    i=0

    for bbox in tqdm(_bboxes[0]):
        H_sum = 0.0
        num_tested = 0
        j+=1
        for feature in clean_features:
            perturbed_image = perturb_image(img, bbox, feature)

            save_numpy_array_as_jpg(perturbed_image.copy(),"test/"+str(j)+str(i))
            i+=1
            perturbed_prediction = model.predict([perturbed_image], [size])
            print(perturbed_prediction[3])
            perturbed_bboxes = perturbed_prediction[0][0][0]

            ious = bbox_iou_ymin_xmin_ymax_xmax(torch.tensor(bbox), torch.tensor(perturbed_bboxes))
            max_iou, max_index = torch.max(ious, dim=0)
            if max_iou.item() < iou_threshold:
                continue

            H_sum += calculate_entropy(torch.tensor(probs[0][max_index.item()]))
            num_tested += 1

        if num_tested == 0:
          print("pass")
        H_avg = H_sum / num_tested if num_tested != 0 else 0
        if H_avg <= m - delta or H_avg >= m + delta:
            poisoned_flag = True
            coordinates.append(bbox)

    return poisoned_flag, coordinates
