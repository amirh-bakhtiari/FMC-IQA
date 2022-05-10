def get_csiq_info(file_path: str) -> list:
    '''Get the video dataset metadata file and extraxt video file names and their 
    corresponding DMOS
    
    :param file_path: the path to the video dataset metadata file
    :return: list of video names, and their DMOS
    '''
    with open(file_path, 'r') as f:
        videos ,dmos = [], []
        # Skip the 1st line of the file which includes titles
        for i, line in enumerate(f):
            if i == 0:
                continue
            videos.append(line.split()[0])
            dmos.append(float(line.split()[1]))
    
    return videos, dmos


def prepare_csiq_data(file_path: str, frame_size: int = 224) -> list:
    '''Get the videos' names and DMOS. Set the preprocessing type
    :param file_path: the path to the video dataset metadata file
    :return: list of video names, and their DMOS and preprocessing module
    '''
    # Get the list of videos and their DMOS
    videos, dmos = get_csiq_info(path)
    
    frm_transform = transforms.compose([transforms.Resize(frame_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                              (0.229, 0.224, 0.225))])
    
    return videos, dmos, frm_transform
    
    
    