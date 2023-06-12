from keras.constraints import max_norm
from keras.layers.regularization import SpatialDropout2D, Dropout
from keras.layers import Conv2D, Dense
from keras import Model
from keras.layers import BatchNormalization, DepthwiseConv2D, Activation, AveragePooling2D, SeparableConv2D
from keras.layers.reshaping import Flatten
from keras.layers import Input

from src.utils.utils import MyLayer

def eegnet_encoder(Chans = 64, Samples = 480,
             dropoutRate = 0.5, kernLength = 64, F1 = 8,
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    data_input = Input(shape = (Chans, Samples, 1))
    block1_conv2d = Conv2D(F1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1), use_bias=False)(data_input)
    block1_conv2d_normalization = BatchNormalization()(block1_conv2d)
    block1_depthwise = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,
                                       depthwise_constraint=max_norm(1.))(block1_conv2d_normalization)
    block1_depthwise_normalization = BatchNormalization()(block1_depthwise)
    block1_depthwise_activation = Activation('elu')(block1_depthwise_normalization)
    block1_depthwise_pooling = AveragePooling2D((1, 8))(block1_depthwise_activation)
    block1_depthwise_dropout = dropoutType(dropoutRate)(block1_depthwise_pooling)
    block2_separable = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1_depthwise_dropout)
    block2_normalization = BatchNormalization()(block2_separable)
    block2_activation = Activation('elu')(block2_normalization)
    block2_pooling = AveragePooling2D((1, 4))(block2_activation)
    block2_dropout = dropoutType(dropoutRate)(block2_pooling)
    flatten = Flatten(name='flatten')(block2_dropout)
    model = Model(inputs=data_input, outputs=flatten, name="EEGNet-encoder")
    return model

def eegnet_classifier(encoder, Chans = 64, Samples = 480, nb_classes = 4, norm_rate = 0.25, trainable = True, uncerteinty = False):
    for layer in encoder.layers:
        layer.trainable = trainable
    data_input = Input(shape=(Chans, Samples, 1))
    features = encoder(data_input)
    dense_temp = Dense(128, name='denseTemp', kernel_constraint=max_norm(norm_rate))(features)
    dense_temp1 = Dense(32, name='denseTemp1', kernel_constraint=max_norm(norm_rate))(features)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(features)
    if uncerteinty == False:
      softmax = Activation('softmax', name='softmax')(dense)
    elif uncerteinty == True:
      softmax = MyLayer()(dense)
    model = Model(inputs=data_input, outputs=softmax, name="EEGNet-encoder")
    return model

# class EEGNet(Model):
#     def __init__(self, nb_classes=4, F1=8, D=2, kernLength=64, Chans=64,
#                  Samples=480, F2=16, dropoutRate=0.5, norm_rate=0.25,
#                  dropoutType='Dropout'):
#         super(EEGNet, self).__init__()
#
#         if dropoutType == 'SpatialDropout2D':
#             dropoutType = SpatialDropout2D
#         elif dropoutType == 'Dropout':
#             dropoutType = Dropout
#         else:
#             raise ValueError('dropoutType must be one of SpatialDropout2D '
#                              'or Dropout, passed as a string.')
#
#         self.block1_conv2d = Conv2D(F1, (1, kernLength), padding = 'same', input_shape = (Chans, Samples, 1), use_bias = False)
#         self.block1_conv2d_normalization = BatchNormalization()
#         self.block1_depthwise = DepthwiseConv2D((Chans, 1), use_bias=False,depth_multiplier=D ,depthwise_constraint=max_norm(1.))
#         self.block1_depthwise_normalization = BatchNormalization()
#         self.block1_depthwise_activation = Activation('elu')
#         self.block1_depthwise_pooling = AveragePooling2D((1, 8))
#         self.block1_depthwise_dropout = dropoutType(dropoutRate)
#
#         self.block2_separable = SeparableConv2D(F2, (1, 16),use_bias=False, padding='same')
#         self.block2_normalization = BatchNormalization()
#         self.block2_activation = Activation('elu')
#         self.block2_pooling = AveragePooling2D((1, 8))
#         self.block2_dropout = dropoutType(dropoutRate)
#
#         self.flatten = Flatten(name='flatten')
#
#         self.dense = Dense(nb_classes, name='dense',
#                       kernel_constraint=max_norm(norm_rate))
#         self.softmax = Activation('softmax', name='softmax')
#
#     def call(self, x):
#         # block1
#         x = self.block1_conv2d(x)
#         x = self.block1_conv2d_normalization(x)
#         x = self.block1_depthwise(x)
#         x = self.block1_depthwise_normalization(x)
#         x = self.block1_depthwise_activation(x)
#         x = self.block1_depthwise_pooling(x)
#         x = self.block1_depthwise_dropout(x)
#         # block2
#         x = self.block2_separable(x)
#         x = self.block2_normalization(x)
#         x = self.block2_activation(x)
#         x = self.block2_pooling(x)
#         x = self.block2_dropout(x)
#
#         x = self.flatten(x)
#         print(x.shape)
#         x = self.dense(x)
#         return self.softmax(x)

