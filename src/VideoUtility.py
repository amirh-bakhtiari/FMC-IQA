import skvideo.io

def get_frames(vid_path: str, vid_pix_fmt: str = "yuv420p", frame_color_mode: str = 'rgb', 
                   height: int = 432, width: int = 768) -> list:
    ''' Get an input path containing a video and return its frames
    
    :param path_in: path of a video to extract
    :param vid_pix_fmt: pixel format of a YUV video (not used for mp4 videos)
    :param frame_color_mode: extracted frames'pixel format ('rgb' or 'gray') (not used for mp4 videos)
    :param height: height of a YUV video frame (not used for mp4 videos)
    :param width: width of a YUV video frame (not used for mp4 videos)
    :return: an array of video frames of the size (num of frames * height * width * num of channels)
    '''
    
    # Get the video extension
    extension = vid_path.split('.')[-1]
    # Check the video type to set the proper params
    if extension == 'mp4':
        return skvideo.io.vread(vid_path)
    # Otherwise check the output frame color mode for YUV videos
    elif frame_color_mode == 'rgb':
        return skvideo.io.vread(vid_path, height, width, inputdict={"-pix_fmt": "yuv420p"})
    elif frame_color_mode == 'gray':
        return skvideo.io.vread(vid_path, height, width, inputdict={"-pix_fmt": "yuv420p"},
                            outputdict={"-pix_fmt": "gray"})
    else:
        return None
    

def get_csiq_info(file_path: str) -> list:
    '''Get the video dataset metadata file and extraxt video file names and their 
    corresponding DMOS
    
    :param file_path: the path to the video dataset metadata file
    :return: video names, DMOS
    '''
    with open(file_path, 'r') as f:
        videos ,dmos = [], []
        for line in f:
            videos.append(line.split()[0])
            dmos.append(line.split()[1])
    
    return videos, dmos
        