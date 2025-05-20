import pickle
import numpy as np
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = np.array(pickle.load(open('features.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = tensorflow.keras.applications.resnet50.ResNet50(weights='imagenet',
                                                        include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = tensorflow.keras.preprocessing.image.load_img('Test_Data/1584.jpg',target_size=(224,224))
img_array = tensorflow.keras.preprocessing.image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = tensorflow.keras.applications.resnet.preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)
distances,indices = neighbors.kneighbors([normalized_result])

for i in indices[0][1:6]:
    temp_img = cv2.imread(filenames[i])
    cv2.imshow('output',cv2.resize(temp_img,(720,1280)))
    cv2.waitKey(0)
