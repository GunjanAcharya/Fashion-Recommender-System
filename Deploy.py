import pandas as pd
import streamlit as st
import os
import gdown
from PIL import Image
import numpy as np
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle

output_file = "features.pkl"
file_id='1JBLK6gEsoriUV6FIPqCUzIfk25hl2l5F'

if not os.path.exists(output_file):
    url =f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)

feature_list = np.array(pickle.load(open('features.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

st.title('Fashion Recommender System\n-----By GunjanAcharya')
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

model = tensorflow.keras.applications.resnet50.ResNet50(weights='imagenet',
                                                        include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
def feature_extraction(img_path,model):
    img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tensorflow.keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tensorflow.keras.applications.resnet.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join("Uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)
        # show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some Error Occured!")