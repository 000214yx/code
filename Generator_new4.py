import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add

class SkipBlock(tf.keras.Model):
    def __init__(self,filters):
        super(SkipBlock, self).__init__()

        self.conv7 = layers.Conv2D(filters // 8,kernel_size=1, strides=1,activation='leaky_relu',padding='same')

        self.conv4 = layers.Conv2D(filters // 8, kernel_size=3, strides=1,activation='leaky_relu',padding='same')
        self.conv5 = layers.Conv2D(filters // 8, kernel_size=5,dilation_rate=2, strides=1,activation='leaky_relu',padding='same')
        self.conv6 = layers.Conv2D(filters // 8, kernel_size=7,dilation_rate=3,strides=1,activation='leaky_relu',padding='same')

        self.conv1 = layers.Conv2D(filters // 8, kernel_size=3, strides=1,activation='leaky_relu',padding='same')
        self.conv2 = layers.Conv2D(filters, kernel_size=1, strides=1,activation='sigmoid')
        self.conv3 = layers.Conv2D(filters, kernel_size=1, strides=1,activation='sigmoid')
        #self.Dense1 = layers.Dense(filters // 4)
        # self.Dense2 = layers.Dense(filters // 8)
        # self.bn2 = layers.BatchNormalization()
        #self.Dense3 = layers.Dense(filters)

    def call(self, AC):
        #in_channel = AC.shape[-1]
        # x_a = layers.GlobalMaxPooling2D()(AC)
        # x_m = layers.GlobalAveragePooling2D()(AC)
        x = self.conv7(AC)
        x_x = layers.AveragePooling2D (pool_size=(1,100))(x)
        x_y = layers.AveragePooling2D (pool_size=(100,1))(x)
        x = layers.multiply([x_x,x_y])
        x1 = self.conv4(x)
        x2 = self.conv5(x)
        x3 = self.conv6(x)
        x = layers.add([x1, x2, x3,x])
        # x = layers.Reshape(target_shape=(4,4,in_channel // 16))(x)
        #x1 = self.conv4
        x = self.conv1(x)
        x_1 = self.conv2(x)
        x_2 = self.conv3(x)
        x = (x_1+x_2) * 0.5
        x = layers.multiply([AC,x])
        return x


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters,select,strides=1, use_attention = False, downsample = False):
        super(ResidualBlock, self).__init__(name='')
        self.use_attention = use_attention
        if select == 1:
            self.conv1 = Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)
            self.bn1 = BatchNormalization()
            self.Leaky_relu1 = Activation('leaky_relu')

            # 第二个卷积层
            self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)
            self.bn2 = BatchNormalization()
            self.Leaky_relu2 = Activation('leaky_relu')

            if use_attention:
                self.attention = SkipBlock(filters)

            # self.downsample = tf.keras.Sequential()
            # if strides != 1:
            #     self.downsample.add(Conv2D(filters, kernel_size=1, strides=strides, use_bias=False))
            #     self.downsample.add(BatchNormalization())

        if select == 2:
            self.conv1 = Conv2D(filters, kernel_size=5, strides=strides, padding='same', use_bias=False)
            self.bn1 = BatchNormalization()
            self.Leaky_relu1 = Activation('leaky_relu')

            self.conv2 = Conv2D(filters, kernel_size=5, strides=1, padding='same', use_bias=False)
            self.bn2 = BatchNormalization()
            self.Leaky_relu2 = Activation('leaky_relu')

            if use_attention:
                self.attention = SkipBlock(filters)

            # self.downsample = tf.keras.Sequential()
            # if strides != 1:
            #     self.downsample.add(Conv2D(filters, kernel_size=1, strides=strides, use_bias=False))
            #     self.downsample.add(BatchNormalization())


    def call(self, inputs):

        # 通过卷积操作扩展维度
        x = self.bn1(inputs)
        x = self.Leaky_relu1(x)
        x = self.conv1(x)

        # 执行剩余卷积
        x = self.bn2(x)
        x = self.Leaky_relu2(x)
        x = self.conv2(x)

        if self.use_attention:
            x = self.attention(x)

        x = layers.add([x, inputs])

        return x


class Generator_new4(tf.keras.Model):
    def __init__(self, stride = 1):
        super(Generator_new4, self).__init__()
        # [None ,100, 100, 1]
        self.conv1 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')
        self.bn3 = layers.BatchNormalization()

        # self.skipblock1 = SkipBlock(1)
        # self.skipblock3 = SkipBlock(1)
        #
        # self.skipblock2 = SkipBlock(2)
        # self.skipblock4 = SkipBlock(2)

        #left

        self.block1 = ResidualBlock(128,select=1,strides=1,use_attention=False)

        self.block3 = ResidualBlock(128,1, strides=1,use_attention=True)

        self.block5 = ResidualBlock(128,1, strides=1,use_attention=False)

        self.block7 = ResidualBlock(128, 1, strides=1, use_attention=True)


        #right

        self.block2 = ResidualBlock(128, 2, strides=1,use_attention=False)

        self.block4 = ResidualBlock(128, 2, strides=1,use_attention=True)

        self.block6 = ResidualBlock(128, 2, strides=1,use_attention=False)

        self.block8 = ResidualBlock(128, 2, strides=1, use_attention=True)

        #self.adapool = layers.GlobalAveragePooling2D()
        # self.conv4 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')
        # self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.bn5 = layers.BatchNormalization()
        self.conv6 = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')


    def call(self,inputs, training = None):
        x1 = tf.nn.leaky_relu(self.bn1(self.conv1(inputs),training = training))
        x2 = tf.nn.leaky_relu(self.bn2(self.conv2(x1),training = training))
        x3 = tf.nn.leaky_relu(self.bn3(self.conv3(x2), training=training))

        #左-----------------------------------------
        x = self.block1(x3)
        #x4_2 = self.skipblock1(x1)
        #x5 = Add()([x4_1, x4_2])
        x = self.block3(x)
        #x5_2 = self.skipblock3(x2)
        #x6 = Add()([x5_1, x5_2])
        x = self.block5(x)

        x7 = self.block7(x)
        #右------------------------------------------
        x = self.block2(x3)
        #x8_2 = self.skipblock2(x1)
        #x9 = Add()([x8_1, x8_2])
        x = self.block4(x)
        #x9_2 = self.skipblock4(x2)
        #x10 = Add()([x9_1, x9_2])
        x = self.block6(x)

        x8 = self.block8(x)
        #拼接------------------------------------------
        x_con = tf.keras.layers.concatenate([x7, x8])
        # x = tf.nn.leaky_relu(self.bn4(self.conv4(x_con), training=training))
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x_con), training=training))
        outputs = tf.nn.tanh(self.conv6(x))


        return outputs

# model =Generator_new4(128)
# model.build(input_shape=(None, 100, 100, 128)) # 可选步骤，构建模型的网络结构
# model.summary()




