import numpy as np
from PIL import Image
import torch
from torchvision import models


def set_sf_model(device):
    '''Set the model to extract both the content and style features according to the
       style transfer paper:
    
    :param device: Determines the device to be used by model ('cuda' or 'cpu')
    :return: module of the model
    '''
    
    # get the "features" portion of VGG19 (we will not need the "classifier" portion)
    vgg = models.vgg19(pretrained=True).features

    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    vgg.to(device)
        
    return vgg

def get_frame_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'
                 }
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor, flat=True):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        
        :param tensor: input tensor
        :param flat: flatten the rsultant matrix if'True',
        :return: Gram Matrix
    """
    
    # get the batch_size, depth, height, and width of the Tensor
    b, d, h, w = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(b * d, h * w)
    
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    
    if flat == 'True':
        return torch.flatten(gram)
    else:
        return gram

def get_video_style_features(video, model, device, transform, layers: dict = None):
    '''For a given array of video frames, preprocess each frame, get its specified layers' feature maps,
       turn the feature maps of each layer into gram matrices which indicates the correlation between features
       in individual layers, i.e. how similar the features in a single layer are. Similarities will include
       the general colors, textures and curvatures found in that layer, according to the style transfer paper
       by Gatys et al (2016). Finally, flatten and concatenate these matrices as the final style features of a frame.
       
    :param video: an array of video frames
    :param model: feature extractor model
    :param layers: layers to extract features from
    :param device: 'torch.cuda' or 'torch.cpu'
    :param transform: torchvision preprocessing pipeline
    :return: an array of concatenated gram matrices for each frame in video
    '''
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '28': 'conv5_1'
                 }
        
    
    video_features = []
    for frame in video:
        
        # Convert the array image to PIL image
        frame = Image.fromarray(frame)
        # Convert the image array to a tensor, and go through the defined preprocessing
        # then add the batch dimension and transfer the tensor to the GPU (if available)
        frame = transform(frame).unsqueeze(0).to(device)
        
        # Get features maps of all frames from the specified layers
        features = get_frame_features(frame, model, layers)
        
        frame_gram_matrices = []
        # Get flattened gram matrix of each frame and concatenate them as the new frame features
        for feature_maps in features.values():
            frame_gram_matrices.extend(gram_matrix(feature_maps).cpu().numpy())
       
        # Add the new features of a the current frame to the video frame features
        video_features.append(frame_gram_matrices)
    
    # Check the shape of each layers's features for the last frame of the video
    for key, value in features.items():
        print(f'Layer {key} features dimension = {value.shape}')
    # Check the shape of the resultant matrices of the last frame
    print(f'Concatenated Gram matrices dimension of a frame = {np.array(frame_gram_matrices).shape}')
    
    return video_features


       
        
        
        
        
        
        
    