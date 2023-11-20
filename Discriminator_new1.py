import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
#from SpectraNorm import SpectralNorm
import numpy as np


class Discriminator_new1(tf.keras.Model):

    def __init__(self,stride =2):
        super(Discriminator_new1,self).__init__()
        # [b , 100 , 100 , 1] => [b , 1]
        self.conv1 = layers.Conv2D(32 , kernel_size = 3 , strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(32 , kernel_size = 3 , strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(64, 3, 2, 'same')
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(64, 3, 1, 'same')
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(64, 3, 2, 'same')
        self.bn5 = layers.BatchNormalization()

        self.conv6 = layers.Conv2D(128, 3, 1, 'same')
        self.bn6 = layers.BatchNormalization()

        self.conv7 = layers.Conv2D(128, 3, 2, 'same')
        self.bn7 = layers.BatchNormalization()

        self.conv8 = layers.Conv2D(1, kernel_size=3, strides=1, padding='same')

    def call(self,inputs,training = None):
        #print('inputs:',inputs.shape)

        x1 = tf.nn.leaky_relu(self.bn1(self.conv1(inputs),training = training))
        x2 = tf.nn.leaky_relu(self.bn2(self.conv2(x1),training = training))

        x3 = tf.nn.leaky_relu(self.bn3(self.conv3(x2),training = training))
        x4 = tf.nn.leaky_relu(self.bn4(self.conv4(x3), training=training))

        x5 = tf.nn.leaky_relu(self.bn5(self.conv5(x4,training = training)))
        x6 = tf.nn.leaky_relu(self.bn6(self.conv6(x5, training=training)))

        x7 = tf.nn.leaky_relu(self.bn7(self.conv7(x6, training=training)))
        # x8 = tf.nn.leaky_relu(self.bn8(self.conv8(x7, training=training)))

        # x9 = tf.nn.leaky_relu(self.bn9(self.conv9(x8, training=training)))
        # x10 = tf.nn.leaky_relu(self.bn10(self.conv10(x9, training=training)))

        logits = self.conv8(x7)
        #logits = tf.sigmoid(logits)

        return logits


# model =Discriminator_new1()
# model.build(input_shape=(None, 100, 100, 1)) # 可选步骤，构建模型的网络结构
# model.summary()