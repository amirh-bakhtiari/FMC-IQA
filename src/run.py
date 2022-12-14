# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[]
import torch
import numpy as np
from torchvision import models
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor

import DatasetHandler as dh
import pooling
import regression as reg
import SFVQA as sfv
import VideoUtility as vu


# -

def load_image(path, device, frame_size, center_crop, min_size=520):
    '''Load in and transform an image, then move it to the specified device ('cpu' or 'cuda')
    '''
    
    image = Image.open(path)
    
    # large images will slow down processing
    if max(image.size) > min_size:
        size = frame_size
    else:
        size = max(image.size)
    
    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.CenterCrop(center_crop),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                                 (0.229, 0.224, 0.225))])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = transform(image)[:3, :, :].unsqueeze(0).to(device)
    
    return image


def imageset_features(ffeats_extractor, device, dataset, frame_size, center_crop):
    '''Get images features in a dataset
    
    :param ffeats_extractor: feature extractor model object
    :param device: torch.device for torch tensors, which can be of cuda or cpu type
    :param dataset: VQA dataset, to extract its frames'features
    :prama frame_size: the size to which frames are resized
    :param center_crop: the size to crop a center patch from the resized frame
    :return: features of images, images' quality scores 
    '''
    
    # Get the list of images and corresponding scores and preprocessing module of dataset
    images_list, scores, images_path = dh.get_imageset_info(dataset=dataset)
    
    images_features = []
    for image in tqdm(images_list):
        # Load in the image
        img_path = Path(images_path) / image
        img = load_image(img_path, device, frame_size, center_crop)
        
        # Get features maps of all frames from the specified layers
        features = ffeats_extractor(img)
        image_gram_matrices = []
        for layer, feature_maps in features.items():
            # If the layer is used for getting style features
            if layer != 'avgpool' and layer != 'flatten':
                image_gram_matrices.extend(sfv.gram_matrix(feature_maps).cpu().numpy())
            else: # If the layer is used for getting content (CNN) features 
                image_gram_matrices.extend(feature_maps.flatten().cpu().numpy())
        
        images_features.append(image_gram_matrices)
        
    
    return np.array(images_features), scores


# + tags=[]
def init_iqa(model_name, vqa_dataset, cross_dataset=None):
    ''' Set the frame feature extractor model, VQA dataset and frame size
    
    :param model_name: name of the feature extractor model, 'inceptionv3', 'vgg19':
    :param vqa_dataset: VQA dataset to evaluate the method on, 'KONVID1K' , 'LIVE'
    :return: model, dataset and frame size and patch
    '''
    
    if model_name == 'vgg19':
        frame_size, center_crop = 255, 224
        # Specify the layers to get style features from vgg19
        layers = {
                  # '0': 'conv1_1',
                  '5': 'conv2_1', 
                  # '10': 'conv3_1', 
                  # '19': 'conv4_1',
                  }
        
    elif model_name == 'inceptionv3':
        frame_size, center_crop = 338, 299
        # Specify the layers to get style features from inceptionv3
        layers = {
                  # 'Conv2d_1a_3x3': 'conv1_1',
                  # 'Conv2d_3b_1x1': 'conv2_1', 
                  'Mixed_5b': 'Mixed_1', 
                  # 'Mixed_5c': 'Mixed_2',
                  'avgpool': 'avgpool'
                  }
    
    # EfficientNet B4 layers
    elif model_name == 'efficientnet':
        frame_size, center_crop = 423, 380  # EfficientNet_B4
        # frame_size, center_crop = 427, 384  # EfficientNet_V2_S
        # frame_size, center_crop = 534, 480  # Efficientnet_v2_M
        # Specify the layers to get style features from EfficientNet B4
        layers = {
                  'features.2.2.add': 'features.2.2.add',  
                  'features.3.3.add': 'features.3.3.add',
                  'features.4.4.add': 'features.4.4.add',
                  'features.5.5.add': 'features.5.5.add',
                  # 'avgpool': 'avgpool',
                  # 'flatten': 'flatten'
                  }
        
    # Check if there is a GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the frame features extractor model (VGG19)
    ffeats_extractor, _ = sfv.set_feats_extractor(device, model_name, layers)
                
    return ffeats_extractor, device, vqa_dataset, cross_dataset, frame_size, center_crop   


# -

# Entry point of the program
def main():
    '''Run the whole process of VQA
    '''
    # Initialize vqa
    ffeats_extractor, device, dataset, cross_dataset, frame_size,\
                                                        center_crop = init_iqa('efficientnet',
                                                                               'scid')
    # Get video level features by pooling frame level features
    images_features, scores = imageset_features(ffeats_extractor, device, dataset, frame_size, center_crop)
    
    if cross_dataset:
        # Get video level features by pooling features of consecutive frames' differences
        c_images_feats, c_scores = imageset_features(ffeats_extractor, device,
                                                     cross_dataset, frame_size, center_crop)
    else:
        c_images_feats, c_scores = None, None
        
    # Train a regressor using video level features and indicate how well it predicts the scores
    reg.regression(images_features, scores, c_images_feats, c_scores, 'svr', dataset, cross_dataset)               


# + tags=[]
# Run the main function if current file is the script, not a module
if __name__ == "__main__":
    main()
# -














