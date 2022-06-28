import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
from torchvision import transforms

import DatasetHandler as dh

def set_sf_model(device, fine_tune=False):
    '''Set the model to extract both the content and style features according to the
       style transfer paper:
    
    :param device: Determines the device to be used by model ('cuda' or 'cpu')
    :return: module of the model
    '''
    
    model = models.vgg19(pretrained=True)
    
    if fine_tune:        
        # Replace the last dense layer of classifier portion with one node 
        # to use it as a neural network regressor
        model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
        model.to(device)
        model = fine_tune_model(model, device)
    else:
        model.to(device)        
        
    model = model.features
    # freeze all VGG parameters since we're only optimizing the target image
    for param in model.parameters():
        param.requires_grad_(False)
        
    return model

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

    if flat == True:
        return torch.flatten(gram)
    else:
        return gram

def get_video_style_features(video, model, device, transform, layers: dict = None, hist_feat=False, bins=10):
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
    :param hist_feat: if True, get the histogram of the Gram matrices
    :param bins: number of bins for histogram
    :return: an array of concatenated gram matrices for each frame in video
    '''
    
    # Determine the layers to get style features from (style feats indicate color, texture and curvatures in an image)
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
        if hist_feat:
            # Get the histogram of the Gram matrices of a frame as frame-level features
            video_features.append(np.histogram(frame_gram_matrices, bins=bins)[0])
        else:
            # Get the Gram matrices of a frame as frame-level features
            video_features.append(frame_gram_matrices)
    
    # Check the shape of each layers's features for the last frame of the video
    # for key, value in features.items():
    #     print(f'Layer {key} features dimension = {value.shape}')
    # # Check the shape of the resultant features of the last frame
    # print(f'Concatenated Gram matrices dimension of a frame = {np.array(video_features[-1]).shape}')
    
    return video_features


def fine_tune_model(model, device, dataset='tid2013', features=True):
    '''Fine tune the input model
    
    :param model: pretrained model
    :param dataset: dataset the model is fine-tuned on
    :param features: only features portion of the model is returned if True 
    :return: fine-tuned model
    '''
    
    # Read the dataset info from the yaml file
    dataset_info = dh.read_yaml('iqa_dataset_info.yaml')
    annotations_file_1 = dataset_info.get(dataset).get('annotations_file_1')
    # Get the image directory
    img_dir = dataset_info.get(dataset).get('img_dir')
    
    # Define the preproccesing pipeline
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    
    # Obtain the training data
    train_data = dh.CustomImageDataset(annotations_file_1, img_dir, transform)

    # how many samples per batch to load
    batch_size = 8
    
    
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    
    # percentage of training set to use as validation
    valid_size = 0.2
    split = int(np.floor(num_train * (1 - valid_size)))
    train_idx, valid_idx = indices[:split], indices[split:]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               sampler=valid_sampler)
    
    # Specify loss function
    criterion = nn.MSELoss()
    # Specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Number of epochs
    epochs = 20
    
    # Train the model with the given dataset to fine tune it
    model = train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, device)

    return model
    
    
    
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    '''Train a network
    
    :param model: the model which is going to be fine tuned
    :param train_loader: loader of the training set
    :param valid_loader: loader of the validation set
    :param epochs: number of epochs for training
    :param lr: learning rate
    '''
    
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf # set initial "min" to infinity

    for e in range(epochs):
        # Monitor the loss
        train_loss = 0.0
        valid_loss = 0.0
        
        # Prep model for training
        model.train()
        for inputs, labels in train_loader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # Forward pass
            output = model(inputs)
            # calculate the loss
            loss = criterion(output.squeeze(), labels.float().squeeze())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()

            
        # Prep the model for evaluation
        model.eval()
        for inputs, labels in valid_loader:
            with torch.no_grad():
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                output = model(inputs)
                # calculate the loss
                loss = criterion(output.squeeze(), labels.float().squeeze())
                # update running validation loss
                valid_loss += loss.item()
            
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        
        print(f'Epoch {e+1} / {epochs}  ---  train loss = {train_loss:.4f}  --- validation loss = {valid_loss:.4f}')
                      
        if valid_loss <= valid_loss_min:
            print(f'validation loss decreased ({valid_loss_min:.4f} --> {valid_loss:.4f}). Saving model ...')
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
    
    # Load the best model which has been saved
    state_dict = torch.load('model.pt')
    model.load_state_dict(state_dict)
    return model
              
              
            
            
            
            
        
        
    
    
    
    
    
    
    
          
        
        
        
        
    