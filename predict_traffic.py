import tensorflow as tf
import tflearn
import glob
import cv2
from skimage import img_as_float
import pickle
import numpy as np


training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_test = len(X_test)
image_shape = len(X_train[0])
n_classes = len(np.unique(y_train))

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

# Convert image to float.
X_float = np.empty(X_train.shape)
for i in range(len(X_train_clahe)):
    X_float[i] = img_as_float(X_train_clahe[i].astype(np.uint8))

mean_correction = list()
std_correction = list()
for ch in range(3):
    mean_correction.insert(ch, X_float[..., ch].mean())
    std_correction.insert(ch, X_float[..., ch].std())

new_photos = list()
image_name = list()
clahe_normalized = list()
final_images = list()
for (i,image_file) in enumerate(glob.iglob('new_traffic_signs/*.png')):
    image_name.append(image_file)
    image = cv2.imread(image_file)
    # Trun from BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r = 100.0 / image.shape[1]
    dim = (100, int(image.shape[0] *r))
    new_photos.append(cv2.resize(image,(32,32),dim,interpolation = cv2.INTER_AREA))
    # Perform CLAHE normalization.
    clahe_normalized.append(normalize_illumination(new_photos[-1]))

    # Perform mean and standard deviation correction.
    final_image = np.empty(clahe_normalized[-1].shape)
    for ch in range(3):
        final_image[..., ch] = ((clahe_normalized[-1][..., ch] - mean_correction[ch])
                                / (std_correction[ch] + 1e-8))
    final_images.append(final_image)


final_images = np.asarray(final_images)

with tf.Graph().as_default():
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
    model.load('models/traffic_cnn2.tflearn')

    predictions = model.predict(final_images)

    for i in range(len(predictions)):
        pred_category = np.asarray(range(44))[np.asarray(predictions[i]) > 0.99]
        print("For image %s the prediction is %s" %
              (image_name[i], pred_category))

