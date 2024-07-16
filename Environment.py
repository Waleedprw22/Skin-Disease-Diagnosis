import os
import torchvision.transforms as transforms
import pandas as pd
from SkinDataset import SkinDataset


# Construct path to the annotations and labels.
project = os.path.join('C:/', 'Users', 'walee','OneDrive', 'Documents', 'Project')
csv_path = os.path.join(project, 'HAM10000_metadata.csv')

# Turn the file into a dataframe.
df = pd.read_csv(csv_path)

# Due to hardware limitations, narrowing down data set to a range of images shown below
START_ID = 'ISIC_0024306'
END_ID = 'ISIC_0026305'

csv_dataframe = df[(df['image_id'] >= START_ID) & (df['image_id'] <= END_ID)]

# Specific transformations and values to ensure data is similarto pre-trained data in terms of normalization and size
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the folder that has all the images
image_directory = os.path.join(project, 'HAM10000_images_part_1')

# Initialize the skin dataset class
skin_dataset = SkinDataset(image_dir=image_directory, csv_dataframe = csv_dataframe, transform=transform)

# Model path
model_path = os.path.join(project, 'models', 'efficientnet_skin_lesion_classifier.pth')


