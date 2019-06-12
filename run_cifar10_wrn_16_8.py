import os
import argparse
import numpy as np
import sklearn.metrics as metrics

import wide_residual_network as wrn
from keras.datasets import cifar10
import keras.utils.np_utils as kutilsi
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

from keras import backend as K


def main(epoch=10,
         batch_size=128,
         weights_file_path=None,
         img_h=32,
         img_w=32,
         img_c=3)

    # Load cifar10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalization
    X_train = X_train.astype('float32')
    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0))
    X_test = X_test.astype('float32')
    X_test = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0))

    # Convert y to one-hot vector
    y_train = kutils.to_categorical(trainY)
    y_test = kutils.to_categorical(testY)

    # Data augumentation
    generator = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=5./32,
                                   height_shift_range=5./32,)

    if K.image_dim_ordering() == 'th':
        init_shape = (img_c, img_h, img_w) # shape=(3, 32, 32) for Theano
    else:
        init_shape = (img_h, img_w, img_c) # shape=(32, 32, 3) for Tensorflow

    # For WRN-16-8  put N = 2, k = 8
    # For WRN-28-10 put N = 4, k = 10
    # For WRN-40-4  put N = 6, k = 4
    model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=2, k=8, dropout=0.00)
    model.summary()
    # Plot model graph structure
    plot_model(model, "WRN-16-8.png", show_shapes=False)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    # Read pre-trained weights
    if weights_file_path:
        # weights_file_path="./weights/WRN-16-8_WEIGHTS.h5"
        model.load_weights(weights_file_path)

    callbacks=[]
    callbacks.append(ModelCheckpoint("./weight/WFN-16-8_weights.h5", monitor="val_acc", save_best_only=True, verbose=1))
    model.fit_generator(generator.flow(X_trainX, y_train,
                        batch_size=batch_size), 
                        steps_per_epoch=len(X_train) // batch_size, 
                        epochs=epoch,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test),
                        validation_steps=X_testX.shape[0] // batch_size,)

    # Validation
    y_preds = model.predict(X_test)
    y_pred = np.argmax(y_preds, axis=1)
    y_pred = kutils.to_categorical(y_pred)
    y_true = Y_test

    test_accuracy = metrics.accuracy_score(y_true, y_pred) * 100
    test_error = 100 - accuracy
    print("Test Accuracy (top 1%) : ", test_accuracy)
    print("Test Error    (top 1%) : ", test_error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', help='')
    parser.add_argument('--batch_size', help='')
    parser.add_argument('--weight_file_path', help='')
    parser.add_argument('--img_h', help='')
    parser.add_argument('--img_w', help='')
    parser.add_argument('--img_c', help='')
    args = parser.parse_args()
    main(args.epoch, args.batch_size, args.weight_file_path, args.igm_h, args.img_w, args.img_c)
