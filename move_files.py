import os
import shutil
import pandas as pd
from Environment import csv_dataframe

# Load the DataFrame
df = csv_dataframe

# Define the source directory
source_dir = r'Define your source_dir path here'

# Define the base target directory
target_base_dir = r'Define your target directory here'



# Move the images to the corresponding directories
for idx, row in df.iterrows():
    image_name = row['image_id']  
    label = row['dx']  
    

    source_path = os.path.join(source_dir, f'{image_name}.jpg')
    target_path = os.path.join(target_base_dir, label, f'{image_name}.jpg')

    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
        print(f'Moved {source_path} to {target_path}')
    else:
        print(f'{source_path} does not exist')

print('All images have been moved.')
