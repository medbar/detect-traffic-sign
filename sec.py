import matplotlib.pyplot as plt
import csv
import numpy

import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#from keras import backend as K
#K.set_session(sess)

from keras.preprocessing.image import load_img, array_to_img, img_to_array

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

from PIL import Image

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        gtReader.__next__()  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            with Image.open(prefix + row[0]) as img:
                img = img.resize([80, 80])
                imgarray = numpy.asarray(img)
                images.append(imgarray)  # the 1th column is the filename

            # images.append(plt.imread(prefix + row[0]))
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels


trainImages, trainLabels = readTrafficSigns('C:/Users/tonym/YandexDisk/python/CRT_testing_work/GTSRB/Training')
print(len(trainLabels), len(trainImages))



# Размер мини-выборки
batch_size = 32
# Количество классов изображений
nb_classes = 43
# Количество эпох для обучения
nb_epoch = 25
# Размер изображений
img_rows, img_cols = 80, 80
# Количество каналов в изображении: RGB
img_channels = 3



X_train = numpy.asarray(trainImages)
X_train = X_train.astype('float32')
X_train /= 255
Y_train = np_utils.to_categorical(trainLabels, nb_classes)





# Создаем последовательную модель
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
# Второй сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))

# Третий сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# Четвертый сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.5))
# Выходной полносвязный слой
model.add(Dense(nb_classes, activation='softmax'))




sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.1,
              shuffle=True,
              verbose=2)


model_json = model.to_json()
json_file = open("sec_model.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("sec_model.h5")
