import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides = stride, padding = 'same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides = 1     , padding = 'same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1, 1), strides = stride))
        else:
            self.downsample = lambda x : x

    def call(self, input, training = None):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(input)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output

class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, num_classes):
        super(ResNet, self).__init__()
        # 预处理层
        self.stem = Sequential([
            layers.Conv2D(64, (5, 5), strides = (1, 1)), #卷积核个数，卷积核大小，步长
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size = (2, 2), strides=(1, 1), padding = 'same')
        ])
        # resblock
        self.layer1 = self.build_resblock(64 , layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], 2)
        self.layer3 = self.build_resblock(256, layer_dims[2], 2)
        self.layer4 = self.build_resblock(512, layer_dims[3], 2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, input, training=None):
        x = self.stem(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride = 1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride = 1))
        return res_blocks

def create_class_weight(labels_dict, total):
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        class_weight[key] = total // labels_dict[key]
    return class_weight

def train_model(train_x, train_y, test_x, test_y, num_task, epoch, btch, width):
    model = ResNet([2, 2, 2, 2], num_task)
    model.build(input_shape = (None, width, width, 3))
    model.summary()

    optimizer = optimizers.Adam(learning_rate = 1e-3) #学习率
    model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])

    total = train_y.shape[0]
    labels_dict = dict(zip(range(num_task), [sum(train_y[:, i]) for i in range(num_task)]))
    cls_wght = create_class_weight(labels_dict, total)

    history = model.fit(train_x, train_y, epochs=epoch, batch_size=btch, validation_data=(train_x, train_y), class_weight=cls_wght)

    print ('testing the model')
    score = model.evaluate(train_x, train_y)
    print("train_score: ", score)
    score = model.evaluate(test_x , test_y )
    print("test_score: " , score)