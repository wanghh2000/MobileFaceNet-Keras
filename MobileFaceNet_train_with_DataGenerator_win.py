# -*- coding: utf-8 -*-
import os
# Set log level before import, 0-debug(default) 1-info 2-warnning 3-error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Model_Structures.MobileFaceNet import mobile_face_net_train
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, CSVLogger, ReduceLROnPlateau
import sys
# from tensorflow.python.keras.utils.np_utils import to_categorical

sys.path.append('../')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#BATCH_SIZE = 128
BATCH_SIZE = 4
# NUM_LABELS = 67960
NUM_LABELS = 12
m = 15090270
DATA_SPLIT = 0.005
TOTAL_EPOCHS = 1000
TOTAL_EPOCHS = 10

'''Importing the data set'''
train_path = r'C:/bd_ai/dli/celeba/img_celeba_processed'

train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=DATA_SPLIT)

def mobilefacenet_input_generator(generator, directory, subset, loss='arcface'):
    gen = generator.flow_from_directory(
        directory,
        target_size=(112, 112),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset=subset)

    while True:
        X = gen.next()
        if loss == 'arcface':
            yield [X[0], X[1]], X[1]
        else:
            yield X[0], X[1]


train_generator = mobilefacenet_input_generator(train_datagen, train_path, 'training', 'softmax')

validate_generator = mobilefacenet_input_generator(train_datagen, train_path, 'validation', 'softmax')

'''Training the Model'''
# Train on multiple GPUs
# from tensorflow.keras.utils import multi_gpu_model
# model = multi_gpu_model(model, gpus = 2)

# change the loss to 'arcface' for fine-tuning
model = mobile_face_net_train(NUM_LABELS, loss='softmax')
model.summary()
model.layers

model.compile(optimizer=Adam(lr=0.001, epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model after every epoch
# set verbose=2 to remove warning on_batch_end() is slow
check_pointer = ModelCheckpoint(filepath='./Models/MobileFaceNet_train.h5', verbose=2, save_best_only=True)

# Interrupt the training when the validation loss is not decreasing
early_stopping = EarlyStopping(monitor='val_loss', patience=10000)

# Record the loss history
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


history = LossHistory()

# Stream each epoch results into a .csv file
csv_logger = CSVLogger('training.csv', separator=',', append=True)
# append = True append if file exists (useful for continuing training)
# append = False overwrite existing file

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=200, min_lr=0)

'''Importing the data & training the model'''
# Model.fit_generator is deprecated and will be removed in a future version,
# Please use Model.fit, which supports generators.
hist = model.fit(
    train_generator,
    steps_per_epoch=int(m * (1 - DATA_SPLIT) / BATCH_SIZE),
    epochs=TOTAL_EPOCHS,
    callbacks=[check_pointer, early_stopping, history, csv_logger, reduce_lr],
    validation_data=validate_generator,
    validation_steps=int(m * DATA_SPLIT / BATCH_SIZE),
    workers=1,
    use_multiprocessing=False)  
    # For TensorFlow 2, Multi-Processing here is not able to use. Use tf.data API instead.

print(hist.history)
