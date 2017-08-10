

from keras.preprocessing.image import load_img, array_to_img, img_to_array
import matplotlib.pyplot as plt


img = load_img("TrainIJCNN2013/00/00001.ppm")
img2 = load_img("TrainIJCNN2013/00/00002.ppm")
plt.imshow([img], cmap='gray')
plt.show()



