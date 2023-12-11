import torch
import torchvision.transforms as T

import numpy as np
from skimage.transform import resize

from PIL import Image

from tqdm import tqdm

from bbox_tool import bbox_iou_ymin_xmin_ymax_xmax, bbox_iou_xmin_ymin_xmax_ymax, bbox_iou_cx_cy_w_h

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


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
    array = array.transpose((1, 2, 0))
    array = std * array + mean
    array = np.clip(array, 0, 1)

    array = (array * 255).astype(np.uint8)

    image = Image.fromarray(array)
    image.save(file_name + '.jpg')


def detector_cleanse(img, model, clean_features, m, delta, alpha, iou_threshold=0.5):
    model = model.cuda()

    prediction = model.predict([img], [img.shape[1:]])
    _bboxes, _labels, _scores, probs = prediction

    poisoned_flag = False
    coordinates = []

    for bbox in _bboxes[0]:
        H_sum = 0.0
        num_tested = 0
        for feature in clean_features:
            perturbed_image = perturb_image(img, bbox, feature, alpha)

            perturbed_prediction = model.predict([perturbed_image], [img.shape[1:]])
            perturbed_bboxes = perturbed_prediction[0][0]

            
            if len(perturbed_bboxes) == 0:
              continue
            
            save_numpy_array_as_jpg(perturbed_image, "test/"+str(0))
            
            ious = list()

            for perturbed_bbox in perturbed_bboxes:
              ious.append(bbox_iou_ymin_xmin_ymax_xmax(torch.tensor(bbox), torch.tensor(perturbed_bbox)).item())
            
            max_iou, max_index = max(ious), np.argmax(ious)
            
            if max_iou < iou_threshold:
                continue

            H_sum += calculate_entropy(probs[0][max_index].clone().detach())            
            num_tested += 1

        if num_tested == 0:
          continue
        H_avg = H_sum / num_tested
        if H_avg <= m - delta or H_avg >= m + delta:
            poisoned_flag = True
            coordinates.append(bbox)

    return poisoned_flag, coordinates