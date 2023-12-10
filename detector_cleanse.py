import torch
import torchvision.transforms as T

import tqdm


from bbox_tool import bbox_iou_ymin_xmin_ymax_xmax, bbox_iou_xmin_ymin_xmax_ymax, bbox_iou_cx_cy_w_h

def calculate_entropy(scores):
    return -torch.sum(scores * torch.log2(scores), dim=0).mean()


def perturb_image(image, bbox, feature, alpha=0.5):
    perturbed_image = image.clone()
    feature_resized = T.functional.resize(feature, [bbox[3] - bbox[1], bbox[2] - bbox[0]])
    
    perturbed_region = perturbed_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    blended_region = alpha * feature_resized + (1 - alpha) * perturbed_region
    perturbed_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = blended_region
    return perturbed_image


def detector_cleanse(ori_img, model, clean_features, m, delta, iou_threshold=0.5):
    ori_img = T.ToTensor()(ori_img).cuda()
    model = model.cuda()

    prediction = model.predict([ori_img])
    _bboxes, _labels, _scores, probs = prediction

    poisoned_flag = False
    coordinates = []

    for bbox in tqdm(_bboxes[0]):
        H_sum = 0.0
        num_tested = 0
        for feature in clean_features:
            perturbed_image = perturb_image(ori_img, bbox, feature)
            perturbed_prediction = model.predict([perturbed_image])
            perturbed_bboxes = perturbed_prediction[0][0]

            ious = box_iou_ymin_xmin_ymax_xmax(torch.tensor(bbox).unsqueeze(0), torch.tensor(perturbed_bboxes))
            max_iou, max_index = torch.max(ious, dim=1)
            if max_iou.item() < iou_threshold:
                continue

            selected_prob = torch.tensor(probs[0][max_index.item()])
            H_sum += calculate_entropy(selected_prob)
            num_tested += 1

        H_avg = H_sum / num_tested
        if H_avg <= m - delta or H_avg >= m + delta:
            poisoned_flag = True
            coordinates.append(bbox)

    return poisoned_flag, coordinates
