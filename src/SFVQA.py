def set_sf_model(device: torch.device) -> nn.Module:
    '''Set the model to extract both the content and style features according to the
    style transfer paper:
    
    :param device: Determines the device to be used by model ('cuda' or 'cpu')
    
    '''
    
    # get the "features" portion of VGG19 (we will not need the "classifier" portion)
    vgg = models.vgg19(pretrained=True).features

    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    vgg.to(device)
        
    return vgg