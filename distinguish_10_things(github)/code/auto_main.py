import os
import tensorflow as tf
from keras import datasets, layers, models, regularizers, callbacks
import matplotlib.pyplot as plt
import time
from itertools import product

#please go to line 252 to set initial data!!!


global base_dir
global base_dir_train
global log_name
global stop_acc
global epochs
global store_data
global batch_size_num
global patience_num
global min_delta_data
global kinds
global acc_proj
global if_conv_fixed
global if_neuron_fixed
global min_val_acc
global horizontal_flip_num

base_dir='E:\\linear_algebre\\distinguish_10_things'
base_dir_train=base_dir

kinds=10                # Classification categories
stop_acc = 0.95         # Set the maximum accurate value of the pause
epochs=75               #Set the maximum number of times
dense_num_max=3         #Set the maximum number of dense
batch_size_num=1024     #256,512,1024,2048,Optional, small has low performance requirements, large is effective
patience_num=5          #Setting the number of items that can not fall
min_delta_data=0.0040   #Setting the threshold for slow-growth pause
acc_proj=0.65           #Setting the minimum saved model accuracy value
if_conv_fixed=1         #0 is off, turn on for 1, 2, 3 modes See this line of code for details #conv_data_fixed=[[32,128],[64,128],[128,128]]
if_neuron_fixed=2       #0 is off, turn on for 1, 2, 3 modes See these multiple lines of code for details #neuron_data_fixed=[[256,32],[256,64],[256,256]](eg.sense_num==2)

min_val_acc=0.30        #The smallest value detected at epoch >= epoch_min_val_acc is 'val_categorical_accuracy'.
poch_min_val_acc=10     #see above

horizontal_flip_num=1   #Horizontal mirroring or not

class StopTrainingAtAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('categorical_accuracy') > stop_acc:
            print("\nAccuracy above"+ str(stop_acc*100)+ "%, stopping training.\n")
            self.model.stop_training = True
        elif logs.get('val_categorical_accuracy') <= min_val_acc and epoch >= (poch_min_val_acc-1) :
            print("\nless than"+ str(min_val_acc*100)+ "%, when epoch=="+str(epoch+1)+", stopping training.\n")
            self.model.stop_training = True

def mode_compile(dropout_data,l1_data,neuron_data,conv_data,dense_num):

    epochs_num=epochs
    stop_at_accuracy = StopTrainingAtAccuracy()
    train_dir=os.path.join(base_dir_train,'source','train')
    test_dir=os.path.join(base_dir_train,'source','test')

    if horizontal_flip_num :
        train_dategen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.00,
                                                                      horizontal_flip=True)
    else:
        train_dategen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.00)

    train_generator=train_dategen.flow_from_directory(
        train_dir,
        target_size=(32,32),
        batch_size=batch_size_num,
        class_mode='categorical'
    )

    test_dategen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.00)

    test_generator=test_dategen.flow_from_directory(
        test_dir,
        target_size=(32,32),
        batch_size=batch_size_num,
        class_mode='categorical'
    )

    model = models.Sequential()

    model.add(layers.Conv2D(conv_data[0],kernel_size=(3,3),padding='same',activation='relu',input_shape=(32,32,3)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(conv_data[1],kernel_size=(3,3),padding='same',activation='relu',input_shape=(32,32,3)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Flatten())

    counter=0

    model.add(layers.Dense(neuron_data[counter],activation='relu',kernel_regularizer=regularizers.l1(l1_data[counter])))
    model.add(layers.Dropout(dropout_data[counter]))
    counter+=1

    if dense_num>=2 :
        model.add(layers.Dense(neuron_data[counter],activation='relu',kernel_regularizer=regularizers.l1(l1_data[counter])))
        model.add(layers.Dropout(dropout_data[counter]))
        counter+=1

    if dense_num>=3 :
        model.add(layers.Dense(neuron_data[counter],activation='relu',kernel_regularizer=regularizers.l1(l1_data[counter])))
        model.add(layers.Dropout(dropout_data[counter]))
        counter+=1


    model.add(layers.Dense(kinds,activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=patience_num,
        min_delta=min_delta_data,
        restore_best_weights=True
    )

    history = model.fit(
                        train_generator,
                        epochs=epochs_num, 
                        validation_data=test_generator,
                        verbose=1,
                        callbacks=[stop_at_accuracy, early_stopping]
                        )

#    print(history.history)

    acc = history.history['categorical_accuracy']
    epochs_num=len(acc)
    val_acc = history.history['val_categorical_accuracy']

    if val_acc[epochs_num-1] >= acc_proj :

        if val_acc[epochs_num-1] < (stop_acc-0.05) :

            nowtime = time.strftime('%Y_%m_%d_%H_%M_%S')
            mode_dir=os.path.join(base_dir,'mode',str(nowtime)+'(val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+')')
            os.mkdir(mode_dir)
            model.save(os.path.join(mode_dir,'mode.h5'))
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs_range = range(epochs_num)
            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.savefig(os.path.join(mode_dir,'acc='+str(round(acc[epochs_num-1]*100)/100)+'  val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+'.png'))
            filename =os.path.join(mode_dir,'data.txt')
            with open(filename, 'w') as file:
                file.write('dropout_data == '+str(dropout_data) + '\n')
                file.write('l1_data == '+str(l1_data) + '\n')
                file.write('neuron_data == '+str(neuron_data) + '\n')
                file.write('conv_data == '+str(conv_data) + '\n')
                file.write('dense_num == '+str(dense_num) + '\n')
                file.write('epoch == '+str(epochs_num) + '\n\n')
                file.write('val_acc == '+str(val_acc[epochs_num-1]) + '\n')
                file.write('time == '+str(nowtime) + '\n')

        else:
            nowtime = time.strftime('%Y_%m_%d_%H_%M_%S')
            mode_dir=os.path.join(base_dir,'possible_mode',str(nowtime)+'(val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+')')
            os.mkdir(mode_dir)
            model.save(os.path.join(mode_dir,'mode.h5'))
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs_range = range(epochs_num)
            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.savefig(os.path.join(mode_dir,'acc='+str(round(acc[epochs_num-1]*100)/100)+'  val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+'.png'))
            filename =os.path.join(mode_dir,'data.txt')
            with open(filename, 'w') as file:
                file.write('dropout_data == '+str(dropout_data) + '\n')
                file.write('l1_data == '+str(l1_data) + '\n')
                file.write('neuron_data == '+str(neuron_data) + '\n')
                file.write('conv_data == '+str(conv_data) + '\n')
                file.write('dense_num == '+str(dense_num) + '\n')
                file.write('epoch == '+str(epochs_num) + '\n\n')
                file.write('val_acc == '+str(val_acc[epochs_num-1]) + '\n')
                file.write('time == '+str(nowtime) + '\n')

    else:
        nowtime = time.strftime('%Y_%m_%d_%H_%M_%S')
        mode_dir=os.path.join(base_dir,'mode',str(nowtime)+'(acc=less than '+str(acc_proj)+')')
        os.mkdir(mode_dir)   
        filename =os.path.join(mode_dir,'data.txt')
        with open(filename, 'w') as file:
            file.write('dropout_data == '+str(dropout_data) + '\n')
            file.write('l1_data == '+str(l1_data) + '\n')
            file.write('neuron_data == '+str(neuron_data) + '\n')
            file.write('conv_data == '+str(conv_data) + '\n')
            file.write('dense_num == '+str(dense_num) + '\n')
            file.write('epoch == '+str(epochs_num) + '\n\n')
            file.write('val_acc == '+str(val_acc[epochs_num-1]) + '\n')
            file.write('time == '+str(nowtime) + '\n')

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(epochs_num)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(mode_dir,'acc='+str(round(acc[epochs_num-1]*100)/100)+'  val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+'.png'))

    with open(os.path.join(base_dir,'data','cnn.txt'), 'a') as file:
        file.write('CNN=='+str(conv_data)+'\n')
        file.write('val_acc=='+str(val_acc[epochs_num-1])+'\n')
    with open(os.path.join(base_dir,'data','neuron.txt'), 'a') as file:
        file.write('neuron=='+str(neuron_data)+'\n')
        file.write('val_acc=='+str(val_acc[epochs_num-1])+'\n')
    with open(os.path.join(base_dir,'data','l1.txt'), 'a') as file:
        file.write('l1_data=='+str(l1_data)+'\n')
        file.write('val_acc=='+str(val_acc[epochs_num-1])+'\n')
    with open(os.path.join(base_dir,'data','dropout.txt'), 'a') as file:
        file.write('dropout_data=='+str(dropout_data)+'\n')
        file.write('val_acc=='+str(val_acc[epochs_num-1])+'\n')
    with open(os.path.join(base_dir,'data','dense_num.txt'), 'a') as file:
        file.write('dense_num=='+str(dense_num)+'\n')
        file.write('val_acc=='+str(val_acc[epochs_num-1])+'\n')

    plt.close()
            
def main(dense_num):
    if kinds <= 5:
        neuron_base=[16,32,64,128]
        dropout_base=[0.0,0.05,0.1,0.2,0.4]
        l1_base=[0.0,0.001,0.005,0.01,0.02]
    elif kinds<=10:
        neuron_base=[16,32,64,128,256]
        dropout_base=[0.05,0.1,0.2,0.4]
        l1_base=[0.001,0.005,0.01,0.02]
    elif kinds<=20 :
        neuron_base=[32,64,128,256,512]
        dropout_base=[0.05,0.1,0.2,0.4]
        l1_base=[0.001,0.005,0.01,0.02]
    elif kinds<=30 :
        neuron_base=[64,128,256,512]
        dropout_base=[0.05,0.1,0.2,0.4]
        l1_base=[0.001,0.005,0.01,0.02]

    conv_base=[32,64,128]
    conv_data_fixed=[[32,128],[64,128],[128,128]]

    if dense_num==1 :
        neuron_data_fixed=[[128],[256],[64]]
    elif dense_num==2 :
        neuron_data_fixed=[[256,32],[256,64],[256,256]]
    elif dense_num==3:
        neuron_data_fixed=[[256,64,32],[1024,256,64],[1024,256,256]]

    dropout_base_list= list(product(dropout_base, repeat=dense_num))
    l1_base_list= list(product(l1_base, repeat=dense_num))
    neuron_base_list= list(product(neuron_base, repeat=dense_num))
    conv_base_list= list(product(conv_base, repeat=2))

    num_sum=[0 for i in range(5)]
    num_sum[1]=len(dropout_base_list)
    num_sum[2]=len(l1_base_list)
    num_sum[3]=len(neuron_base_list)
    num_sum[4]=len(conv_base_list)

    if if_neuron_fixed :
        num_sum[3]=1
        store_data[3]=0
    if if_conv_fixed:
        num_sum[4]=1
        store_data[4]=0

    for i_1 in range (num_sum[1]):
        if store_data[1] <= i_1 :
            dropout_data=list(dropout_base_list[i_1])
            for i_2 in range (num_sum[2]):
                if store_data[2] <= i_2 :
                    l1_data=list(l1_base_list[i_2])
                    for i_3 in range (num_sum[3]):
                        if store_data[3] <= i_3 :
                            if if_neuron_fixed :
                                neuron_data=neuron_data_fixed[:][if_neuron_fixed-1]
                            else:
                                neuron_data=list(neuron_base_list[i_3])
                            if if_conv_fixed==0 :
                                for i_4 in range (num_sum[4]):
                                    if store_data[4] < i_4 :
                                        conv_data=list(conv_base_list[i_4])
                                        mode_compile(dropout_data,l1_data,neuron_data,conv_data,dense_num)
                                        time.sleep(1)
                                        filename =os.path.join(base_dir_train,'log',log_name+'.txt')
                                        with open(filename, 'w') as file:
                                            if if_neuron_fixed:
                                                file.write(str(dense_num)+'#'+str(i_1)+'#'+str(i_2)+'#'+str(0)+'#'+str(i_4))
                                            else:
                                                file.write(str(dense_num)+'#'+str(i_1)+'#'+str(i_2)+'#'+str(i_3)+'#'+str(i_4))
                                            file.write('\n'+base_dir)
                            else :
                                conv_data=conv_data_fixed[:][if_conv_fixed-1]
                                mode_compile(dropout_data,l1_data,neuron_data,conv_data,dense_num)
                                time.sleep(1)
                                filename =os.path.join(base_dir_train,'log',log_name+'.txt')
                                with open(filename, 'w') as file:
                                    if if_neuron_fixed:
                                        file.write(str(dense_num)+'#'+str(i_1)+'#'+str(i_2)+'#'+str(0)+'#'+str(0))
                                    else:
                                        file.write(str(dense_num)+'#'+str(i_1)+'#'+str(i_2)+'#'+str(i_3)+'#'+str(0))
                                    file.write('\n'+base_dir)

    for i in range (5):
        store_data[i]=0


if_log=0
user_input = input("Whether to read the log (0 for no, 1 for yes).")
if_log=int(user_input)
store_data=[0 for i in range(5)]
if_run=0

if if_log :
    log_name = input("Please enter the log of readings.")
    file_name_log=[f for f in os.listdir(os.path.join(base_dir_train,'log')) if f.endswith('.txt')]
    if log_name+'.txt' in file_name_log :
        if_run=1
    else:
        print("\nnot found  "+log_name+" !! ")
else :
    log_name = input("Please enter the stored logs.")
    file_name_log=[f for f in os.listdir(os.path.join(base_dir_train,'log')) if f.endswith('.txt')]
    if log_name+'.txt' in file_name_log :
        print("\npre-existing  "+log_name+" !! ")
    else:
        if_run=1

if if_run :
    if if_log :
        with open(os.path.join(base_dir_train,'log',log_name+'.txt'), 'r') as file:
            counter=0
            for line in file:
                if line!='\n' :
                    counter+=1
                if counter==1 :
                    parts = line.split('#')
                    for i in range(5):
                        store_data[i]=int(parts[i])
                    print(store_data)
                elif counter==2 :
                    mode_dir_name=line
                else:
                    break

        base_dir=mode_dir_name

    else:
        store_data=[0 for i in range(5)]
        nowtime = time.strftime('%Y_%m_%d_%H_%M')
        base_dir=os.path.join(base_dir,'train_auto',str(nowtime))
        os.mkdir(base_dir)
        os.mkdir(os.path.join(base_dir,'mode'))
        os.mkdir(os.path.join(base_dir,'possible_mode'))
        os.mkdir(os.path.join(base_dir,'data'))

    for i in range(dense_num_max):

        dense_num=i+1

        if if_log :
            if store_data[0] <= dense_num :
                main(dense_num)
        else:
            main(dense_num)
  