import os
# Set log level before import, 0-debug(default) 1-info 2-warnning 3-error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import TensorBoard


def Preprocessing(directory, input_size, BATCH_SIZE):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rescale=1/255, 
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        validation_split=0.023)
    traingen = datagen.flow_from_directory(
        directory,
        target_size=input_size,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training')
    
    valgen = datagen.flow_from_directory(
        directory,
        target_size=input_size,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation')
    print(traingen.samples, valgen.samples)
    return traingen, valgen

def CreateModel(nb_classes, input_size):
    #baseModel = EfficientNetB7(input_shape=input_size, include_top=False, weights=None)
    #baseModel = EfficientNetB0(input_shape=input_size, include_top=False, weights=None)
    baseModel = MobileNetV2(input_shape=input_size, include_top=False, weights=None)
    x = baseModel.output
    x = GlobalAveragePooling2D()(x)
    # new Full Connection layer with 1024 nodes
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2, name='Dropout')(x)
    # new softmax layer
    x = Dense(nb_classes, activation='softmax')(x)
    # Update model
    model = Model(inputs=baseModel.input, outputs=x)

    # compile model
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    #model.summary()
    return model

def train(model, traingen, valgen, BATCH_SIZE):
    # tensorboard --logdir=logs
    tbCallBack = TensorBoard(
                    log_dir="logs", 
                    histogram_freq=1, 
                    #write_grads=True, 
                    write_images=True)
    train_number = traingen.samples
    val_number = valgen.samples
    # train
    hist = model.fit(traingen,
                    epochs=20,
                    steps_per_epoch=int(train_number/BATCH_SIZE),
                    validation_data=valgen,
                    validation_steps=int(val_number/BATCH_SIZE),
                    callbacks=[tbCallBack])
    # the big steps_per_epoch is, the small memory size is used 
    return hist

if __name__ == '__main__':
    bSize = 4
    path = r'C:/bd_ai/dli/inceptionv4/flower_photos/train'
    traingen, valgen = Preprocessing(directory=path, input_size=(224, 224), BATCH_SIZE=bSize)
    
    model = CreateModel(nb_classes=5, input_size=(224, 224, 3))
    hist = train(model, traingen, valgen, BATCH_SIZE=bSize)
