import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import Dataset
from torchvision import transforms
import yaml

from pathlib import Path
from PIL import Image

def get_csiq_info(file_path: str) -> list:
    '''Get CSIQ VQA dataset metadata file and extract video file names and their 
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
    dmos = np.array(dmos)
    return videos, dmos

def get_live_info(video_file: str, dmos_file: str) -> list:
    '''Get LIVE VQA dataset metadata files and extract video file names and their 
       corresponding DMOS
    
    :param video_file: the path to the video sequence names file
    :param dmos_file: the path to the dmos file
    :return: list of video names, and their DMOS
    '''
    
    with open(video_file, 'r') as vid_file, open(dmos_file, 'r') as dmos_file:
        # Get the name of video sequences
        videos = vid_file.read().strip().split('\n')
        # Get the DMOS of each video
        scores = dmos_file.read().strip().split('\n')
        dmos = [float(score.split('\t')[0]) for score in scores]
        dmos = np.array(dmos)
    
    return videos, dmos   

def get_konvid1k_info(file_path):
    '''Get Konvid1K VQA dataset annotation file and extract video file names and their
       correspondinf MOS ranging from 1 to 5
       
    :param file_path: the path to dataset info file
    :return: video sequence names and their MOS
    '''
    
    # Read the dataframe of the dataset which includes the video names and their MOS
    df = pd.read_csv(file_path)
    videos = df['flickr_id'].values
    mos = df['mos'].values
    
    # Convert the video names to string and concatanate the mp4 extension to names
    videos = [str(video) + '.mp4' for video in videos]
    
    return videos, mos

def get_live_vqc_info(file_path):
    '''Get LIVE-VQC dataset annotation file and extract video file names and their
       corresponding MOS ranging from 0 to 100
       
    :param file_path: the path to dataset info file
    :return: video sequence names and their MOS
    '''
    
    # Read the mat file which is a dict including the video names and their MOS
    mat = scipy.io.loadmat(file_path)
    mos = mat.get('mos').squeeze()
    vids_array = mat.get("video_list")
    
    # Convert the videos array of arrays to a list of strings
    videos = [video[0][0] for video in vids_array]
    
    return videos, mos


def get_videoset_info(dataset='LIVE', frame_size: int = 224, center_crop: int = 224, framework='pytorch'):
    '''Given the name of the video dataset, get the list of respective video names and their scores,
       and set the preprocessing method.
       
    :param dataset: name of a VQA dataset
    :param frame_size: frame size according to the input of the pretrained model
    :param center_crop: used to crop the frame if frame_size is bigger than input of the model
    :param framework: deep learning framework
    :return: list of video names, their scores, the preprocessing module, and the videos' directory
    '''
    
    dataset = dataset.lower()
    # Read the dataset info from the yaml file
    dataset_info = read_yaml('vqa_dataset_info.yaml')
    annotations_file_1 = dataset_info.get(dataset).get('annotations_file_1')
    annotations_file_2 = dataset_info.get(dataset).get('annotations_file_2')
    # Get the videos directory
    video_path = dataset_info.get(dataset).get('video_dir')
    
    if dataset == 'live':
        videos, scores = get_live_info(annotations_file_1, annotations_file_2)
    elif dataset == 'csiq':
        videos, scores = get_csiq_info(annotations_file_1)
    elif dataset == 'konvid1k':
        videos, scores = get_konvid1k_info(annotations_file_1)
    elif dataset == 'live_vqc':
        videos, scores = get_live_vqc_info(annotations_file_1)
        
    if framework == 'pytorch':
        # pytorch specific preprocessing
        # values of means and stds in Normalize are the ones used for ImageNet
        frm_transform = transforms.Compose([transforms.Resize(frame_size),
                                            transforms.CenterCrop(center_crop),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                 (0.229, 0.224, 0.225))])
    elif framework == 'keras':
        ## TODO
        pass
    
    
    return videos, scores, frm_transform, video_path

def get_koniq10k_info(file_path):
    '''Get KonIQ-10k IQA dataset annotation file and extract image file names and their
       corresponding MOS ranging from 1 to 5
       
    :param file_path: the path to dataset info file
    :return: image names and their MOS
    '''
    
    # Read the dataframe of the dataset which includes the images' names and their MOS
    df = pd.read_csv(file_path)
    mos = df.get('MOS').values
    images = df.get('image_name').values
    
    # Convert the ndarray to a list of strings
    images = [str(image) for image in images]
    
    return images, mos

def get_clive_info(image_file: str, mos_file: str):
    '''Get CLIVE in the wild IQA dataset annotation files and extract image file names and their
       corresponding MOS
    '''
    
    # Read the mat file of the dataset which includes the images' names and their MOS
    mos = scipy.io.loadmat(mos_file)
    images = scipy.io.loadmat(image_file)
    
    mos = mos['AllMOS_release'][0]
    images = [item[0][0] for item in images['AllImages_release']]
    
    return images, mos
    

def get_imageset_info(dataset='koniq10k', frame_size: int = 224, center_crop: int = 224, framework='pytorch'):
    '''Given the name of the image dataset, get the list of respective image names and their scores,
       and set the preprocessing method.
       
    :param dataset: name of a IQA dataset
    :param frame_size: frame size according to the input of the pretrained model
    :param center_crop: used to crop the frame if frame_size is bigger than input of the model
    :param framework: deep learning framework
    :return: list of video names, their scores, the preprocessing module, and the videos' directory
    '''
    dataset = dataset.lower()
    # Read the dataset info from the yaml file
    dataset_info = read_yaml('iqa_dataset_info.yaml')
    annotations_file_1 = dataset_info.get(dataset).get('annotations_file_1')
    annotations_file_2 = dataset_info.get(dataset).get('annotations_file_2')
    # Get the images directory
    images_path = dataset_info.get(dataset).get('img_dir')
    
    if dataset == 'koniq10k':
        images, scores = get_koniq10k_info(annotations_file_1)
    elif dataset == 'clive':
        images, scores = get_clive_info(annotations_file_1, annotations_file_2)
    
    if framework == 'pytorch':
        # pytorch specific preprocessing
        # values of means and stds in Normalize are the ones used for ImageNet
        transform = transforms.Compose([
                                        # transforms.Resize(frame_size),
                                        transforms.CenterCrop(center_crop),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                                 (0.229, 0.224, 0.225))])
    elif framework == 'keras':
        ## TODO
        pass
    
    return images, scores, transform, images_path


def read_yaml(file_path):
    '''Read a yaml file
    '''
    loader = yaml.SafeLoader
    with open(file_path) as f:
        return yaml.load(f, loader)

    

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = Path(self.img_dir) / self.img_labels.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label

