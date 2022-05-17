import skvideo.io

def get_frames(vid_path: str, vid_pix_fmt: str = "yuv420p", frame_color_mode: str = 'rgb', 
                   height: int = 0, width: int = 0):
    ''' Get an input path containing a video and return its frames
    
    :param path_in: path of a video to extract
    :param vid_pix_fmt: pixel format of a YUV video (not used for mp4 videos)
    :param frame_color_mode: extracted frames'pixel format ('rgb' or 'gray') (not used for mp4 videos)
    :param height: height of a YUV video frame. Useful for raw inputs when video header does not exist.(not used for mp4      videos)
    :param width: width of a YUV video frame. Useful for raw inputs when video header does not exist. (not used for mp4        videos)
    :return: a generator of video frames of the size (num of frames * height * width * num of channels)
    '''
    
    # Get the video extension
    extension = vid_path.split('.')[-1]
    # Check the video type to set the proper params
    if extension == 'mp4':
        return skvideo.io.vreader(vid_path)
    # Otherwise check the output frame color mode for YUV videos
    elif frame_color_mode == 'rgb':
        return skvideo.io.vreader(vid_path, height, width, inputdict={"-pix_fmt": "yuv420p"})
    elif frame_color_mode == 'gray':
        return skvideo.io.vreader(vid_path, height, width, as_grey=True, inputdict={"-pix_fmt": "yuv420p"})
    else:
        return None
        