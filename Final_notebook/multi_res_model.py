import tensorflow as tf
from keras.layers import Layer, Conv2D, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D, Activation, Multiply, GlobalMaxPooling2D, Input, Concatenate, Conv2DTranspose, Resizing
from keras.models import Model
from keras.utils import plot_model

class AttentionBlock(Layer):
  def __init__(self, ratio=8, **kwargs):
    super(AttentionBlock, self).__init__(**kwargs)
    self.ratio = ratio

  def build(self, input_shape):
    channel = input_shape[-1]
    #Shared Layers
    self.dense1 = Dense(channel//self.ratio, activation='relu', use_bias=False)
    self.dense2 = Dense(channel, use_bias=False)
    #Conv layer for spatial attention
    self.conv = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
    super(AttentionBlock, self).build(input_shape)

  def call(self, inputs):
    x = self.channel_attention_module(inputs)
    x = self.spatial_attention_module(x)
    return x

  def channel_attention_module(self, x):
    channel = x.shape[-1]
    #Global Average Pooling
    x1 = GlobalAveragePooling2D()(x)
    x1 = self.dense1(x1)
    x1 = self.dense2(x1)
    #Global Max Pooling
    x2 = GlobalMaxPooling2D()(x)
    x2 = self.dense1(x2)
    x2 = self.dense2(x2)
    #Combine and apply sigmoid activation
    combined = x1 + x2
    combined = Activation("sigmoid")(combined)
    combined = tf.reshape(combined, (-1, 1, 1, channel))
    #Multiply with the input feature map
    return Multiply()([x, combined])

  def spatial_attention_module(self, x):
    #Average Pooling
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    #Max Pooling
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    #Concatenate along the channel axis
    combined = Concatenate(axis=-1)([avg_pool, max_pool])
    #Conv Layer
    combined = self.conv(combined)
    #Multiply with the input feature map
    return Multiply()([x, combined])
  

class EncoderBlock(Layer):
  def __init__(self, num_filters, kernel_size=(3,3), padding='same', activation='relu', initial_block=False,**kwargs):
    super(EncoderBlock, self).__init__(**kwargs)
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.padding = padding
    self.activation = activation
    self.initial_block = initial_block

    self.conv1 = Conv2D(filters=num_filters, kernel_size=kernel_size, activation=self.activation, padding=padding)
    self.attention = AttentionBlock()
    self.conv2 = [Conv2D(filters=num_filters, kernel_size=kernel_size, activation=self.activation, padding=padding) for _ in range(2)]
    self.maxpool = MaxPooling2D((2, 2))
    # self.drop = Dropout(0.1)

  def call(self, inputs, features=None):
    x1 = self.conv1(inputs)
    x2 = self.attention(x1)

    if self.initial_block:
      x3 = Concatenate()([x1, x2])
    else:
      x3 = Concatenate()([x1, x2, features])

    for conv_layer in self.conv2:
      x3 = conv_layer(x3)

    x4 = self.maxpool(x3)

    return x4, x3
  

class DecoderBlock(Layer):
  def __init__(self, num_filters, kernel_size=(3,3), padding='same', activation='relu', last_block=False,**kwargs):
    super(DecoderBlock, self).__init__(**kwargs)
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.padding = padding
    self.activation = activation
    self.last_block = last_block

    self.convT = Conv2DTranspose(filters=self.num_filters, kernel_size=self.kernel_size, strides=(2, 2), padding=self.padding)
    self.conv = [Conv2D(filters=num_filters, kernel_size=kernel_size, activation=self.activation, padding=padding) for _ in range(2)]

  def call(self, inputs, skip):
    x = self.convT(inputs)
    x = Concatenate()([skip, x])
    for conv_layer in self.conv:
      x = conv_layer(x)
    return x
  

class OutputBlock(Layer):
  def __init__(self, activation='sigmoid',**kwargs):
    super(OutputBlock, self).__init__(**kwargs)
    self.activation = activation

    self.conv = Conv2D(1, (1,1), padding='same', activation=self.activation)

  def call(self, input):
    return self.conv(input)



image_resolution = 256
kernel_size = (5, 5)

input_layer_1 = Input(shape=(image_resolution, image_resolution, 1)) # input_image with one layer (2D image)
input_layer_2 = Resizing(128, 128)(input_layer_1)
input_layer_3 = Resizing(64, 64)(input_layer_1)

encoder_block_01 = EncoderBlock(num_filters=128, kernel_size=kernel_size, initial_block=True)
encoder_block_02 = EncoderBlock(num_filters=64, kernel_size=kernel_size)
encoder_block_03 = EncoderBlock(num_filters=32, kernel_size=kernel_size)

attention_block_01 = AttentionBlock()
attention_block_02 = AttentionBlock()

decoder_block_01 = DecoderBlock(num_filters=64, kernel_size=kernel_size)
decoder_block_02 = DecoderBlock(num_filters=128, kernel_size=kernel_size)

output_block = OutputBlock()


encoder_block_01_output, skip_01 = encoder_block_01(input_layer_1)
skip_01 = attention_block_01(skip_01)

encoder_block_02_output, skip_02 = encoder_block_02(input_layer_2, encoder_block_01_output)
skip_02 = attention_block_02(skip_02)

_, encoder_block_03_output = encoder_block_03(input_layer_3, encoder_block_02_output)

decoder_block_01_output = decoder_block_01(encoder_block_03_output, skip_02)
decoder_block_02_output = decoder_block_02(decoder_block_01_output, skip_01)
output = output_block(decoder_block_02_output)


model = Model(inputs=input_layer_1, outputs=output)
print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])