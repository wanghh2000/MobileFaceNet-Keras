import os
# Set log level before import, 0-debug(default) 1-info 2-warnning 3-error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.callbacks import TensorBoard

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data preprocessing
# 以训练集为例，将样本数据的维度转换为四维，(60000,28,28)->(60000,28,28,1)，
# 其中60000代表样本的个数，（28,28）代表像素的长宽，1代表的是深度。
# 深度为1时，表示黑白，为3时代表彩色，可以近似理解为图片的颜色通道。然后是归一化处理。
x_train = x_train.reshape(-1, 28, 28, 1)/255.0
x_test = x_test.reshape(-1, 28, 28, 1)/255.0
# 标签转换独热编码
# 对标签数据转换为分类的 one-hot （独热）编码。
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Create model
# 第一层：第一层卷积层
# 第二层：第一层池化层
# 第三层：第二层卷积层
# 第四层：第二层池化层
# 第五层：扁平化
# 第六层：第一层全连接层
# 第七层：Dropout处理
# 第八层：第二层全连接层，输出

model = Sequential()
model.add(Convolution2D(  # 第一层卷积(28*28)
    input_shape=(28, 28, 1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))
model.add(MaxPooling2D(  # 第一层池化(14*14),相当于28除以2
    pool_size=2,
    strides=2,
    padding='same'
))
model.add(Convolution2D(  # 第二层卷积
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))
model.add(MaxPooling2D(  # 第二层池化
    pool_size=2,
    strides=2,
    padding='same'))
model.add(Flatten())  # 把第二个池化层的输出扁平化为一维数据
model.add(Dense(1024, activation='relu'))  # 第一层全连接层
model.add(Dropout(0.5))  # Dropout
model.add(Dense(10, activation='softmax'))  # 第二层全连接层

# nput_shape——输入平面
# filters——卷积核/滤波器个数
# kernel_size——卷积窗口大小（5*5）
# strides——步长
# activation——激活函数
# pool_size——池化的核大小
# padding——填充取值：same/valid，same：在输入周围尽可能均匀填充零；valid：不适用零填充。

# compile model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# tensorboard
tbCallBack = TensorBoard(log_dir="./logs", histogram_freq=1, write_grads=True, write_images=True)

# train
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2, validation_data=(x_test, y_test), callbacks=[tbCallBack])

# validate model
loss_train, accuracy_train = model.evaluate(x_train, y_train)
print('train loss:', loss_train, 'train accuracy:', accuracy_train)
loss_test, accuracy_test = model.evaluate(x_test, y_test)
print('test loss:', loss_test, 'test accuracy:', accuracy_test)

# tensorboard --logdir=logs