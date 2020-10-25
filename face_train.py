import os
# Set log level before import, 0-debug(default) 1-info 2-warnning 3-error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Model_Structures.MobileFaceNet import mobile_face_net
from Model_Structures.MobileFaceNet import mobile_face_net_train
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard

def face_generator(directory, input_size, BATCH_SIZE, loss='arcface'):
    datagen = ImageDataGenerator(
            rescale=1/255,
            #rotation_range = 30,
            #width_shift_range = 0.2,
            #height_shift_range = 0.2,
            #shear_range = 0.2,
            #zoom_range = 0.2,
            #horizontal_flip = True,
            validation_split=0.1)

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
    
    print(traingen.samples, valgen.samples, traingen.num_classes, valgen.num_classes)
    return traingen, valgen

def CreateModel(nb_classes):
    model = mobile_face_net_train(nb_classes, loss='softmax')
    #model.compile(optimizer=Adam(lr=0.001, epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(model, traingen, valgen, TOTAL_EPOCHS, BATCH_SIZE):
    # tensorboard --logdir=logs
    tbCallBack = TensorBoard(
                    log_dir="logs", 
                    histogram_freq=1, 
                    #write_grads=True, 
                    write_images=True)
    train_number = traingen.samples
    val_number = valgen.samples
    hist = model.fit(
        traingen,
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=int(train_number/BATCH_SIZE),
        callbacks=[tbCallBack],
        validation_data=valgen,
        validation_steps=int(val_number/BATCH_SIZE),
        workers=1,
        use_multiprocessing=False)

def printModel():
    model = mobile_face_net()
    model.summary()

if __name__ == '__main__':
    bSize = 8
    path = r'C:/bd_ai/dli/celeba/img_celeba_processed'
    path = r'C:/bd_ai/dli/celeba/img_celeba_raw'
    # printModel()
    traingen, valgen = face_generator(directory=path, input_size=(112, 112), BATCH_SIZE=bSize, loss='softmax')
    num_classes = traingen.num_classes
    model = CreateModel(nb_classes=num_classes)
    #model.summary()
    hist = train(model, traingen, valgen, TOTAL_EPOCHS=10, BATCH_SIZE=bSize)
