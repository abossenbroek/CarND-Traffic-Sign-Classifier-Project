# Load pickled data
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical

from sklearn.model_selection import train_test_split
from skimage import img_as_float
import pickle
import numpy as np
import cv2

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = len(X_train[0])

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Let us do illumination correction on the images.
# We follow this post: http://stackoverflow.com/questions/24341114
def normalize_illumination(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return(final)

X_train_clahe = np.empty(X_train.shape)
for i in range(len(X_train)):
    X_train_clahe[i] = normalize_illumination(X_train[i])

X_test_clahe = np.empty(X_test.shape)
for i in range(len(X_test)):
    X_test_clahe[i] = normalize_illumination(X_test[i])

# Convert image to float.
X_float = np.empty(X_train.shape)
for i in range(len(X_train_clahe)):
    X_float[i] = img_as_float(X_train_clahe[i].astype(np.uint8))

mean_correction = list()
std_correction = list()
for ch in range(3):
    mean_correction.insert(ch, X_float[..., ch].mean())
    std_correction.insert(ch, X_float[..., ch].std())

# We see that the data is much less centered after applying CLAHE. We will recenter the data
# since CNN are very sensitive to data that is not centered.
X_norm = np.empty(X_float.shape)
X_test_norm = np.empty(X_test_clahe.shape)
for ch in range(3):
    X_norm[..., ch] = (X_float[..., ch] - mean_correction[ch])/ (std_correction[ch] + 1e-8)
    X_test_norm[..., ch] = ((img_as_float(X_test_clahe[..., ch].astype(np.uint8)) - mean_correction[ch]) /
                            (std_correction[ch] + 1e-8))

# TODO: change to stratisfied sampler to ensure that
X_input, X_validation_clahe, y_train, y_validation = train_test_split(X_norm,
                                                                      y_train,
                                                                      test_size=0.2,
                                                                      random_state=42,
                                                                      stratify=y_train)

y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

network = tflearn.input_data(shape=[None, image_shape, image_shape, 3])

# First layer is a convolution with filter size 32 and relu activation.
network = tflearn.conv_2d(network, 32, 3, activation='relu')
# Second layer is convolution with filter size of 7 and 100 depth.
network = tflearn.conv_2d(network, 100, 7, activation='relu')
# Maxpool with stride 2.
network = tflearn.max_pool_2d(network, 2)
# Fourth layer is convolution with filter size 4 and 150 depth and relu activation.
network = tflearn.conv_2d(network, 150, 4, activation='relu')
# Fifth layer is max pool with stride 2.
network = tflearn.max_pool_2d(network, 2)
# Sixth layer is convolution with filter size 4 and 250 depth and relu activation.
network = tflearn.conv_2d(network, 250, 4, activation='relu')
# Seventh layer is max pool with stride 2.
network = tflearn.max_pool_2d(network, 2)
# We include dropout layer with 50% drop-out probability.
network = tflearn.dropout(network, 0.5)
# Ninth layer is fully connected.
network = tflearn.fully_connected(network, 300, activation='relu')
# Fully connected layer with output size equal to 43, which is the number
# of classes we train on and softmax activation to increase the distance
# between the logit scores.
network = tflearn.fully_connected(network, n_classes, activation='softmax')

# Regression layer that will help us tune the neural network.
# Explanation of the parameter selection:
# * As an optimizer we select 'Adaptive Moment Estimation' (see https://arxiv.org/pdf/1412.6980.pdf)
#   this method has low memory requirements and only requires the first order derivatives
#   of the tensors. The latter requirement results in faster computations.
# * as learning_rate we set a low value to prevent the stochastic gradient descent algorithm
#   to jump too large distances at once.
# * as loss function we choose categorical_crossentropy since we seek to
#   categorize images. The categorical cross entropy function will compute errors
#   in the categorization task.
network = tflearn.regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=2,tensorboard_dir='/tmp/tflearn_logs/')

model.fit(X_input, y_train, n_epoch=50, shuffle=True,
          validation_set=(X_test_norm, y_test),
          show_metric=True, batch_size=240, run_id='traffic_cnn2')
model.save('models/traffic_cnn2.tflearn')
