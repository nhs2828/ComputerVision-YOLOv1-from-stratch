import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Reshape


HIDDEN_LAYER_ACTIVATION = LeakyReLU(alpha = 0.1)

# output shape [(W-K+2P)/S]+1

class MyCustomLayer(tf.keras.layers.Layer):
    def __init__(self, nb_boxes, nb_classes):
        super().__init__()
        self.nb_boxes = nb_boxes
        self.nb_classes = nb_classes

    def call(self, inputs):
        """
            Apply softmax to classes, apply sigmoid to other information to scale to [0-1], also to prevent sqrt by negatives values when calculating loss
        """
        classes = tf.nn.softmax(inputs[..., 5*self.nb_boxes : 5*self.nb_boxes+self.nb_classes], axis=-1)
        p_coord = tf.sigmoid(inputs[..., 0 : 5*self.nb_boxes])
        output = tf.concat([p_coord, classes], axis = -1)
        return output

class YoloV1(tf.keras.Model):
  def __init__(self, img_size = 448, in_channels = 3, nb_cells = 7, nb_boxes = 2, nb_classes=2, **kwargs):
    super().__init__()
    self.nb_cells = nb_cells
    self.nb_boxes = nb_boxes
    self.nb_classes = nb_classes
    self.in_channels = in_channels
    self.img_size = img_size

    self.conv = self.create_conv_seq(self.img_size, self.in_channels)
    self.fc = self.create_fc_seq()

  def create_conv_seq(self, img_size, in_channels):
    model = tf.keras.models.Sequential()
    ########
    #Block 1
    model.add(Conv2D(filters=64, kernel_size=(7,7), strides=(2, 2), padding="valid", activation=HIDDEN_LAYER_ACTIVATION, input_shape=(img_size, img_size, in_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    ########
    #Block 2
    model.add(Conv2D(filters=192, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    #######
    #Block 3
    model.add(Conv2D(filters=128, kernel_size=(1,1), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    #######
    #Block 4
    model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(1,1), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    #######
    #Block 5
    model.add(Conv2D(filters=512, kernel_size=(1,1), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(1,1), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=(2, 2), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    #######
    #Block 6
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=(1, 1), padding=("same"), activation=HIDDEN_LAYER_ACTIVATION))
    model.add(BatchNormalization())
      
    return model

  def create_fc_seq(self):
    
    output_pre_reshape = self.nb_cells * self.nb_cells * (self.nb_boxes * 5 + self.nb_classes)
   
    fc_layers_seq = tf.keras.Sequential()

    fc_layers_seq.add(Flatten())
    fc_layers_seq.add(Dense(4096))
    fc_layers_seq.add(Dropout(0.5))
    fc_layers_seq.add(LeakyReLU(alpha= 0.1))
    fc_layers_seq.add(Dense(output_pre_reshape)) 
    fc_layers_seq.add(Reshape((self.nb_cells, self.nb_cells, self.nb_boxes * 5 + self.nb_classes))) # Reshape
    fc_layers_seq.add(MyCustomLayer(self.nb_boxes, self.nb_classes))
    
    return fc_layers_seq

  def call(self, x):
    x = self.conv(x)
    x = self.fc(x)
    return x


