import torch

import numpy as np

import argparse
from PIL import Image
import random
import glob

from detector_cleanse import detector_cleanse
from model import FasterRCNNVGG16
from transform import preprocess


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Detector Cleanse on an image')
    parser.add_argument('--n', type=int, required=True, help='Number of features to randomly select')
    parser.add_argument('--m', type=float, required=True, help='Detection mean')
    parser.add_argument('--delta', type=float, required=True, help='Detection threshold')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to be analyzed')
    parser.add_argument('--clean_feature_path', type=str ,default='clean_feature_images', help='Path to the clean_feature image folder')
    parser.add_argument('--weight', type=str, required=True, help='Path to weight of the model')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load and randomly select clean features
    print("Loading clean feature files...")
    clean_feature_files = glob.glob(f'{args.clean_feature_path}/*.jpg')  # Update the path as needed
    selected_features = random.sample(clean_feature_files, args.n)
    clean_features = [Image.open(feature_path) for feature_path in selected_features]
    
    for i in range(len(clean_features)):
        feature = clean_features[i].convert('RGB')
        feature = np.asarray(feature, dtype=np.float32)
        feature = feature.transpose((2, 0, 1))
        feature = preprocess(feature)
        clean_features[i] = feature

    print("Complete")

    # Load the image to be analyzed
    print("Loading image...")
    f = Image.open(args.image_path)
    ori_img = f.convert('RGB')
    ori_img = np.asarray(ori_img, dtype=np.float32)
    ori_img = ori_img.transpose((2, 0, 1))
    img = preprocess(ori_img)

    size = ori_img.shape[1:]

    print("Complete")


    print("Loading model...")
    model = FasterRCNNVGG16(n_fg_class=20)

    state_dict = torch.load(args.weight)
    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:  # legacy way, for backward compatibility
        model.load_state_dict(state_dict)
    print("Complete")

    print("Detecting...")
    poisoned, coordinates = detector_cleanse(img, size, model, clean_features, args.m, args.delta)
    print("Complete")

    if poisoned:
        print("Image is poisoned. Trigger coordinates:", coordinates)
    else:
        print("Image is clean.")

if __name__ == "__main__":
    main()