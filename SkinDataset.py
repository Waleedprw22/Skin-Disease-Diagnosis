import os
from PIL import Image
from torch.utils.data import Dataset

class SkinDataset(Dataset):
    '''
    :param Dataset: Feed in image directory, csv dataframe and transformations'''
    def __init__(self, image_dir, csv_dataframe, transform = None):
        '''
        :param image_dir: Ham10000_images_part1 folder which has 5000 images
        :csv_dataframe: Read in dataframe that is passed into this class. Need to modify the dataframe before hand.
        :transform: Feed in image transformations.
        '''
        self.image_dir = image_dir
        self.annotations = csv_dataframe
        self.transform = transform
        self.label_map = {
            'nv': 0,
            'mel': 1,
            'bkl': 2,
            'bcc': 3,
            'akiec': 4,
            'vasc': 5,
            'df': 6
        }

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        '''
        :param idx: the index of the sample
        :return (image, label) for each sample
        '''
        img_name = os.path.join(self.image_dir, self.annotations.iloc[idx, 1] + '.jpg')
        image = Image.open(img_name)
        label = self.label_map[self.annotations.iloc[idx, 2]]

        if self.transform:
            image = self.transform(image)

        return image, label
