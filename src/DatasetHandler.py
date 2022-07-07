import pandas as pd
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
        videos = vid_file.read().split('\n')
        # Get the DMOS of each video
        scores = dmos_file.read().split('\n')
        dmos = [float(score.split('\t')[0]) for score in scores]
    
    return videos, dmos   

def get_konvid1k_info(file_path):
    '''Get Konvid1K VQA dataset data files and extract video file names and their
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


def read_yaml(file_path):
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

