##copyright belongs to Poter && Yang

import numpy as np
import os
import shutil
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

base_dir= os.path.dirname(os.path.abspath(__file__))

mode_path=os.path.join(base_dir,'model')
model_name=[f for f in os.listdir(mode_path) if f.endswith(('.h5'))]
mode_path=os.path.join(mode_path,model_name[0])

source_folder=os.path.join(base_dir,'image')
shutil.rmtree(os.path.join(base_dir,'res'))
os.mkdir(os.path.join(base_dir,'res'))

image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif','.jfif'))]

num=len(image_files)

categories = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
if_categories=[0 for i in range(len(categories))]
other=0

counter=0
model = tf.keras.models.load_model(mode_path)
res=np.zeros((num, 10))

for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif','.jfif')):

        path = os.path.join(source_folder, filename)
        img = tf.keras.preprocessing.image.load_img(path, target_size=(32, 32))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0) 
        x /= 255.0 
        predictions = model.predict(x)[0]
        for i in range(10):
            res[counter][i]=predictions[i]
        counter+=1

for i in range(counter):

    fig_one, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes = axes.flatten()
    probabilities = res[:][i]  
    ax = axes[1]  
    ax.bar(categories, probabilities)
    ax.set_title('Probability Distribution')
    max_index = np.argmax(probabilities)
    ax.set_xlabel(categories[max_index], color='blue')
    ax.set_title('Probability Distribution', color='blue')
    ax.set_ylabel('Probability', color='blue')

    ax = axes[0]
    image = Image.open(os.path.join(source_folder, image_files[i]))
    ax.imshow(image)
    ax.axis('off')

    plt.tight_layout()

    if probabilities[max_index] > 0.50 :
        if if_categories[max_index]==0 :
            os.mkdir(os.path.join(base_dir,'res',categories[max_index]))
        if_categories[max_index]+=1
        plt.savefig(os.path.join(base_dir,'res',categories[max_index],'num_'+str(if_categories[max_index])+'_pic_bar.png'))
    else:
        if other==0:
            os.mkdir(os.path.join(base_dir,'res','other'))
        other+=1
        plt.savefig(os.path.join(base_dir,'res','other','num_'+str(other)+'_pic_bar.png'))
    plt.close()

if other :
    os.rename(os.path.join(base_dir,'res','other'),os.path.join(base_dir,'res','other(num=='+str(other)+')'))
for i in range(len(if_categories)):
    if if_categories[i] :
        os.rename(os.path.join(base_dir,'res',categories[i]),os.path.join(base_dir,'res',categories[i]+'(num=='+str(if_categories[i])+')'))
