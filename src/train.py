import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from pathlib import Path

from src.data.data_generator import DataGenerator, EncoderDataGenerator
from src.models.eegnet import eegnet_encoder, eegnet_classifier
from src.utils.utils import add_projection_head
from src.utils.supervised_contrastive_loss import SupervisedContrastiveLoss

def step_decay(epoch):
    if(epoch < 20):
        lr = 0.01
    elif(epoch < 50):
        lr = 0.001
    else:
        lr = 0.0001
    return lr
lrate = LearningRateScheduler(step_decay)

batch = 64
encoderBatch = 32
splits = KFold(n_splits=5,shuffle=True,random_state=42)
path='/home/hesaam/Hesaam/Projects/Thesis/supervised-contrastive-eegnet/data/'
dataset_files = np.array([str(p.stem) for p in Path(path + 'imagery/').glob("*.npy")])
dataset_pairs = np.array([str(p.stem) for p in Path(path + 'execution/').glob("*.npy")])
print(len(dataset_files))
data = []
for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset_files)))):
    pairs = np.concatenate((dataset_files[train_idx], dataset_pairs), axis=0)
    x_encoder = EncoderDataGenerator(path=path, dataset=pairs, subset_split=[], batch_size=encoderBatch)
    x_train = DataGenerator(path=path + 'imagery/', dataset=dataset_files, subset_split=train_idx, batch_size=batch)
    x_val = DataGenerator(path=path + 'imagery/', dataset=dataset_files, subset_split=val_idx, batch_size=batch)

    encoder = eegnet_encoder(kernLength=80)

    encoder_with_projection_head = add_projection_head(encoder)
    encoder_with_projection_head.compile(Adam(0.001), loss=SupervisedContrastiveLoss(0.05), )
    history_encoder = encoder_with_projection_head.fit(x=x_encoder, epochs=200)

    classifier = eegnet_classifier(encoder, trainable=False)
    adam_alpha = Adam(learning_rate=(0.0001))
    classifier.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics=['accuracy'])

    history_classifier = classifier.fit(x=x_train, validation_data=x_val, epochs=100, callbacks=[lrate], verbose=2)
    idx = np.argmax(history_classifier.history['val_accuracy'])
    print('Fold {:}\t{:.4f}\t{:.4f}'.format(fold, history_classifier.history['accuracy'][idx],
                                            history_classifier.history['val_accuracy'][idx]))
