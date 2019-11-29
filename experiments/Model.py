from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.models import Model
from keras.layers import Input
from keras import optimizers



class Network:

    @staticmethod
    def build(input_shape, num_class, number_convolutional_layers, filters_size, filters_numbers, weight_decay=1e-4, dropout=0.75, lr=0.01, momentum=0.9):

        if number_convolutional_layers < 1:
            print("ERROR: Number of convolutional layers must be greater than 0")

        L2_norm = regularizers.l2(weight_decay)
        model_input = Input(shape=input_shape)
        output = Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), kernel_regularizer=L2_norm, \
                                        activation='relu', input_shape=input_shape, data_format='channels_first')(model_input)


        output = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(output)

        for i in range(1, number_convolutional_layers):
            output = Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), kernel_regularizer=L2_norm, \
                       activation='relu', data_format='channels_first')(output)

            if i != number_convolutional_layers - 1:
                output = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(output)

        output = Flatten()(output)

        output = Dense(2048)(output)
        output = Dropout(dropout, name = 'transfer')(output)

        output = Dense(num_class, activation='softmax', name='id')(output)
        output = Dropout(dropout)(output)


        model = Model(inputs=model_input, outputs=output)

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr, 0.9), metrics=['accuracy'])

        return model


    def fit(model, epochs, callbacks, images, labels, val_images, val_labels, actual_step, steps_per_epoch, validation_steps):


        lr = model.optimizer.lr

        model.layers.pop(0)
        newInput = Input(tensor=images)
        newOutput = model(newInput)
        model = Model(inputs=newInput, outputs=newOutput)


        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr, 0.9), target_tensors=[labels],
                      metrics=['accuracy'])
        model.fit(images, labels, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                  validation_data= (val_images, val_labels), validation_steps = validation_steps,
                  initial_epoch=actual_step)

        return model