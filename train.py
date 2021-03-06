"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
#from data import DataSet
import os.path

global trainPath
trainPath = os.path.join('data', 'train')
os.makedirs(trainPath, exist_ok=True)
logPath = os.path.join('data', 'logs')
os.makedirs(logPath, exist_ok=True)
modelPath = os.path.join('data', 'checkpoints')
os.makedirs(modelPath, exist_ok=True)

global classes
classes = os.listdir(trainPath)

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join(
        modelPath, "inception.{epoch:03d}-{val_loss:.2f}.hdf5"),
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard

tensorboard = TensorBoard(log_dir=logPath)


def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        validation_split=0.2)

    #test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        trainPath,
        target_size=(299, 299),
        # batch_size=32,
        batch_size=20,
        classes=classes,
        class_mode='categorical',
        subset='training'
    )

    # validation_generator = test_datagen.flow_from_directory(
    validation_generator = train_datagen.flow_from_directory(
        trainPath,
        target_size=(299, 299),
        # batch_size=32,
        batch_size=20,
        classes=classes,
        class_mode='categorical',
        # validation split finds no images, but if I comment it out, it finds all the training data
        subset='validation'
    )

    return train_generator, validation_generator


def get_model(weights='imagenet'):
    # add an input layer
    i = Input([None, None, 3])
    x = preprocess_input(i)
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False)
    x = base_model(x)
    # add a global spatial average pooling layer
    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    #predictions = Dense(len(data.classes), activation='softmax')(x)

    predictions = Dense(len(classes), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model


def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=100,
    #     validation_data=validation_generator,
    #     validation_steps=10,
    #     epochs=nb_epoch,
    #     callbacks=callbacks)
    model.fit(
        x=train_generator,
        steps_per_epoch=20,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model


def main(weights_file):
    model = get_model()
    generators = get_generators()

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = freeze_all_but_top(model)
        #model = train_model(model, 10, generators)
        model = train_model(model, 10, generators)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    model = freeze_all_but_mid_and_top(model)
    # model = train_model(model, 1000, generators,
    #                    [checkpointer, early_stopper, tensorboard])
    model = train_model(model, 100, generators, [
                        checkpointer, early_stopper, tensorboard])


if __name__ == '__main__':
    weights_file = None
    main(weights_file)
