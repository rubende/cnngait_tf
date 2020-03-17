from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.models import Model
from keras.layers import Input
from keras import optimizers
from keras.models import load_model


class Network:

    model = None
    lr_optimizer = None

    def __init__(self, model=None):
        self.model = model

    def build(self, input_shape, num_class, number_convolutional_layers, filters_size, filters_numbers, weight_decay=1e-4, dropout=0.75):
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
        model._layers.pop(0)
        model.summary()
        self.model = model

    def fit(self, epochs, callbacks, images, labels, val_images, val_labels, actual_step, steps_per_epoch,
            validation_steps, lr=None, momentum=0.9):

        if lr == None:
            lr = self.lr_optimizer

        newInput = Input(tensor=images)
        newOutput = self.model(newInput)
        model_input = Model(inputs=newInput, outputs=newOutput)


        model_input.summary()

        model_input.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr, momentum), target_tensors=[labels],
                      metrics=['accuracy'])
        model_input.fit(images, labels, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                  validation_data= (val_images, val_labels), validation_steps = validation_steps,
                  initial_epoch=actual_step)

        self.lr_optimizer = model_input.optimizer.lr

    def fit_val(self, epochs, callbacks, images, labels, actual_step, steps_per_epoch, lr=None, momentum=0.9):

        if lr is None:
            lr = self.lr_optimizer

        newInput = Input(tensor=images)
        newOutput = self.model(newInput)
        model_input = Model(inputs=newInput, outputs=newOutput)

        model_input.summary()

        model_input.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr, momentum),
                            target_tensors=[labels],
                            metrics=['accuracy'])
        model_input.fit(images, labels, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                        initial_epoch=actual_step)

        self.lr_optimizer = model_input.optimizer.lr

    def save(self, path):
        t_model = Model(self.model.layers[0].input, self.model.layers[-1].output)
        t_model.save(path)

    def load_to_ft(self, path, num_class, dropout=0.75):

        # Load Model
        self.model = load_model(path)

        # Freeze the weights
        for layer in self.model.layers:
            layer.trainable = False

        # Change FC
        self.model.layers.pop()
        self.model.layers.pop()
        x = Dense(num_class, activation='softmax', name='id')(self.model.layers[-1].output)
        x = Dropout(dropout)(x)
        self.model = Model(self.model.layers[0].input, x)
        self.model.layers.pop(0)

    def load_to_predict(self, path, images, steps):

        # Load Model
        self.model = load_model(path)

        newInput = Input(tensor=images)
        newOutput = self.model(newInput)
        model_input = Model(inputs=newInput, outputs=newOutput)

        model_input.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(),
                      metrics=['accuracy'])

        predictions = model_input.predict(images, steps=steps)

        return predictions