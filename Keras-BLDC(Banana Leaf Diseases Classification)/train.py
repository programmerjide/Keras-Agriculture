import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as k

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# hyper parameters for model
nb_classes = 4  # number of classes
img_width, img_height = 60, 60
batch_size = 10
nb_epoch = 30
learn_rate = 0.001
momentum = .9
weight_decay = 0.005


def train(train_data_dir, validation_data_dir, model_path, is_color=True):
    if is_color:
        input_shape = (img_width, img_height, 3)
    else:
        input_shape = (img_width, img_height, 1)

    # build LeNet
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax'))
    print(model.summary())

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    sgd = SGD(lr=learn_rate, momentum=momentum, decay=weight_decay)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc
    final_acc_weights_path = os.path.join(os.path.abspath(model_path), 'model_acc_weights.h5')
    final_loss_weights_path = os.path.join(os.path.abspath(model_path), 'model_loss_weights.h5')

    callbacks_list = [
        ModelCheckpoint(final_acc_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        ModelCheckpoint(final_loss_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
        # EarlyStopping(monitor='val_loss', patience=15, verbose=0),
        TensorBoard(log_dir='graph/train', histogram_freq=0, write_graph=True)
    ]

    # fine-tune the model
    model.fit_generator(train_generator,
                        epochs=nb_epoch,
                        validation_data=validation_generator,
                        callbacks=callbacks_list)

    # save model
    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)


if __name__ == '__main__':
    data_dir = '../Dataset/BLDC/BLDC_color'
    train_dir = os.path.join(os.path.abspath(data_dir), 'train')  # Inside, each class should have it's own folder
    validation_dir = os.path.join(os.path.abspath(data_dir), 'test')  # each class should have it's own folder
    model_dir = 'model/'

    os.makedirs(model_dir, exist_ok=True)

    train(train_dir, validation_dir, model_dir)  # train model

    # release memory
    k.clear_session()