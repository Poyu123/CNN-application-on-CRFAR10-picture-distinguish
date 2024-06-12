##'E:\\cifar-10-batches-py\\'

import os
# from scipy.misc import imsave
from imageio import imsave
 
 
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
 
 
filename = 'E:/cifar-10-batches-py/' #Path to the image crfar_10
meta = unpickle(filename + '/batches.meta')
label_name = meta[b'label_names']
 
for i in range(len(label_name)):
    file = label_name[i].decode()
    path = 'E:/cifar-10-batches-py/train/' + file
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

for i in range(len(label_name)):
    file = label_name[i].decode()
    path = 'E:/cifar-10-batches-py/test/' + file
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

content = unpickle(filename + '/test_batch') #Unzip each of the data_batch_ 
print('load data...')
print(content.keys())
print('test converting')
for j in range(10000):
    img = content[b'data'][j]
    img = img.reshape(3, 32, 32)
    img = img.transpose(1, 2, 0)
    img_name = 'E://cifar-10-batches-py//test//' + label_name[content[b'labels'][j]].decode() + '//test_' + str(0) + '_num_' + str(j) + '.jpg'
    imsave(img_name, img) 

for i in range(1, 6):
    content = unpickle(filename + '/data_batch_' + str(i)) #Unzip each of the data_batch_ 
    print('load data...')
    print(content.keys())
    print('tranfering data_batch' + str(i))
    for j in range(10000):
        img = content[b'data'][j]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        img_name = 'E:/cifar-10-batches-py/train/' + label_name[content[b'labels'][j]].decode() + '/batch_' + str(i) + '_num_' + str(j) + '.jpg'
        imsave(img_name, img)


