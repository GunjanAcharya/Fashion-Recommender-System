import pandas as pd

import numpy as np
from urllib.parse import urljoin
base_url = "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/fashion-dataset/images/"
from tensorflow.keras.layers import GlobalMaxPooling2D
import pickle

df=pd.read_csv('MF-styles.csv',error_bad_lines=False)
filenames=[]
for i in df['id']:
    image=str(i)+'.jpg'
    full_url = urljoin(base_url, image)
    filenames.append(full_url)

print(filenames[0:5])