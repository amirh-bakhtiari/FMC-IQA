def simple_pooling(videos_features: list, pooling: str = 'max'):
    '''Get a list of videos with features for each frame, pool the features of all frames of 
       each video using the given method to have a single vector as the final video level features
       
    :param videos_features: a list containing frames features of all videos in video dataset
    :param pooling: specify the pooling method
    :return: an array of video level features of all videos
    '''
    
    videos_features = np.array(videos_features)
    
    if pooling == 'max':
        video_level_features = np.max(videos_features, axis=1)
    elif pooling == 'mean':
        video_level_features = np.mean(video_features, axis=1)
    elif pooling == 'min':
        video_level_features = np.min(video_features, axis=1)
        
    return video_level_features