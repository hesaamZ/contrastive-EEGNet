import keras
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
from keras.layers import Activation

def add_projection_head(encoder, out_dimension=128):
    inputs = keras.Input(shape=(64, 480, 1))
    features = encoder(inputs)
    outputs = keras.layers.Dense(out_dimension, activation="relu")(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="encoder_with_projection-head")
    return model

class MyLayer(Layer):

    def __init__(self):
        super(MyLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        print(input_shape)
        self._sigma = self.add_weight(name='sigma',
                                    shape=(input_shape[1],),
                                    initializer='uniform',
                                    trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        log_var = tf.math.exp(tf.reduce_sum(self._sigma))
        softmax = Activation('softmax', name='softmax')(log_var*x)
        return softmax