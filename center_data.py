from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# Let us inspect whether the data is centered.
for ch in range(3):
    print("for channel %s mean or clahe data: %s" %(
            ch, X_train[:,ch].mean()))

X_norm = np.copy(X_train)
for ch in range(3):
    X_norm[:, ch] = (X_norm[:, ch] - X_norm[:,ch].mean())/ X_norm[:, ch].std()

# Let us inspect our new mean.
for ch in range(3):
    print("for channel %s new mean for CLAHE data: %s new std: %s" % (
            ch, X_norm[:,ch].mean(), X_norm[:,ch].std()))


