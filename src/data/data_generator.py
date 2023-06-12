import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, path, subset_split = [], batch_size=16, dim=(64,480), n_channels=1, shuffle=True):
        'Initialization'
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = dataset
        if len(subset_split) != 0:
            self.list_IDs = self.list_IDs[subset_split]
        self.labels = {'right_hand':0, 'left_hand': 1, 'feet': 2, 'rest':3}
        self.n_channels = n_channels
        self.n_classes = len(self.labels)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        inds = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in inds]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            temp = np.load(self.path + ID + '.npy')
            X[i,] = np.resize(temp[:,:self.dim[1]], (*self.dim, self.n_channels))
            label = self.labels[ID.split('-')[-1]]
            # Store class
            y[i] = label

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

class EncoderDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, path, subset_split = [], batch_size=16, dim=(64,480), n_channels=1, shuffle=True):
        'Initialization'
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = dataset
        if len(subset_split) != 0:
            self.list_IDs = self.list_IDs[subset_split]
        self.labels = {'right_hand':0, 'left_hand': 1, 'feet': 2, 'rest':3}
        self.n_channels = n_channels
        self.n_classes = len(self.labels)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        inds = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in inds]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if ID.split('-')[1] == 'imagined':
              folder = 'imagery' + '/'
            else:
              folder = ID.split('-')[1] + '/'
            temp = np.load(self.path + folder + ID + '.npy')
            X[i,] = np.resize(temp[:,:self.dim[1]], (*self.dim, self.n_channels))
            label = self.labels[ID.split('-')[-1]]
            # Store class
            y[i] = label

        return X, y