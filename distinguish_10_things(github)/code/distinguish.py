import numpy as np
import os
import tensorflow as tf
import shutil
from PIL import Image
import time
from importlib import reload
import matplotlib.pyplot as plt

base_dir='E:\\linear_algebre\\distinguish_10_things'

mode_path='E:\\linear_algebre\\distinguish_10_things\\mode\\2024_06_10_19_33_33(val_acc=0.77)\\mode.h5'##Give a specific address!

def main_body(base_dir,mode_path):
    categories_base = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    base_dir_1=os.path.join(base_dir,'source','test')

    ans = np.random.randint(0, 10, 10)
    ans_name=[categories_base[ans[i]] for i in range(10)]
    print(ans_name)
    rd_seq = np.random.randint(0, 1000, 10)

    folder_path = os.path.join(base_dir,'test_data','image_10')
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    counter=0

    for i in range(10):

        source_folder =os.path.join(base_dir_1,categories_base[ans[i]])
        destination_folder =folder_path

        image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

        image_to_copy = image_files[rd_seq[i]]
        new_file_name='test_num_'+str(i)+'.jpg'

        shutil.copy(os.path.join(source_folder, image_to_copy), os.path.join(destination_folder, new_file_name))


    model = tf.keras.models.load_model(mode_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            counter+=1

    res=np.zeros((counter, 10))
    counter=0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            path = os.path.join(folder_path, filename)
            img = tf.keras.preprocessing.image.load_img(path, target_size=(32, 32))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0) 
            x /= 255.0 
            predictions = model.predict(x)[0]
            for i in range(10):
                res[counter][i]=predictions[i]
            counter+=1



    categories = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    right=0
    false=0

    for i in range (counter):

        probabilities = res[:][i]
        max_index = np.argmax(probabilities)
        if max_index==ans[i]:
            right+=1
        else:
            false+=1

    sizes = [right,false]
    labels = ['right','false']

    #print(sizes)
    plt.figure(figsize=(30,16))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    nowtime = time.strftime('%m_%d_%H_%M_%S')
    temp=str(nowtime)+'(acc='+str(right*100/(right+false))+'%)'
    os.mkdir(os.path.join(base_dir,'test_data','res',temp))
    plt.savefig(os.path.join(base_dir,'test_data','res',temp,'percentage_graph.png'))
        

    fig_one, axes = plt.subplots(2, 5, figsize=(30, 16))


    axes = axes.flatten()

    for i in range(counter):
        probabilities = res[:][i]  
        ax = axes[i]  

        
        ax.bar(categories, probabilities)

        
        ax.set_title('Probability Distribution')
        max_index = np.argmax(probabilities)
        if max_index == ans[i]:
            ax.set_xlabel(categories[max_index], color='blue')
            ax.set_title('Probability Distribution', color='blue')
            ax.set_ylabel('Probability', color='blue')
        else:
            ax.set_xlabel(categories[max_index]+'('+categories[ans[i]]+")", color='red')
            ax.set_title('Probability Distribution', color='red')
            ax.set_ylabel('Probability', color='red')

        if i % 5 != 0:
            ax.set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,'test_data','res',temp,'pic_bar.png'))
    
    image_folder =folder_path

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    image_files = image_files[:10]

    num_rows, num_cols = 2, 5

    fig_two, axes_2 = plt.subplots(num_rows, num_cols, figsize=(15, 8))

    axes_2 = axes_2.flatten()

    for i, image_file in enumerate(image_files):
        image = Image.open(os.path.join(image_folder, image_file))
        axes_2[i].imshow(image)
        axes_2[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,'test_data','res',temp,'data_pic.png'))

for i in range(5):
    main_body(base_dir,mode_path)