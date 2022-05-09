def get_csiq_info(file_path: str) -> list:
    '''Get the video dataset metadata file and extraxt video file names and their 
    corresponding DMOS
    
    :param file_path: the path to the video dataset metadata file
    :return: video names, DMOS
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