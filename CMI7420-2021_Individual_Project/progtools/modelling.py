from tensorflow.config import list_physical_devices
from tensorflow.keras import Sequential, layers, optimizers
from keras_tuner import HyperModel

def gpu_check():
    """short function to check GPU is set up to use for training and inference
    """
    if len(list_physical_devices("GPU")) > 0:
        print("GPU Ready and available")
    else:
        print("panic")

class PhysicalSensorHyperModel(HyperModel):
    """dense neural network hyper-model for physical sensor

    Args:
        HyperModel (hyper-model): inherited from keras_tuner base HyperModel class 
    """

    def build(self, hp):

        """creates a hyper-model with between 1 and 5 dense layers, all ReLU and up to 256 units

        Returns:
            model: hyper-model
        """
        model = Sequential()

        #input layer of 4 linaer units for 4 input predictors
        model.add(layers.Dense(units=4))

        #dense layer #1 with 2-256 units with Rectified Linear Units
        model.add(layers.Dense(units=hp.Int(name="dense_layer_1",
                                        min_value=2,
                                        max_value=256,
                                        step=2),
                                activation="relu"))

        #dense layer #2 with 2-256 units with Rectified Linear Units
        model.add(layers.Dense(units=hp.Int(name="dense_layer_2",
                                        min_value=2,
                                        max_value=256,
                                        step=2),
                                activation="relu"))

        #dense layer #3 with 2-256 units with Rectified Linear Units
        model.add(layers.Dense(units=hp.Int(name="dense_layer_3",
                                        min_value=2,
                                        max_value=256,
                                        step=2),
                                activation="relu"))

        #dense layer #4 with 2-256 units with Rectified Linear Units
        model.add(layers.Dense(units=hp.Int(name="dense_layer_4",
                                        min_value=2,
                                        max_value=256,
                                        step=2),
                                activation="relu"))

        #dense layer #5 with 2-256 units with Rectified Linear Units
        model.add(layers.Dense(units=hp.Int(name="dense_layer_5",
                                        min_value=2,
                                        max_value=256,
                                        step=2),
                                activation="relu"))       

        #dropout lyaer between 0.05 and 0.50
        model.add(layers.Dropout(rate=hp.Float(name="dropout",
                                        min_value=0.05,
                                        max_value=0.50,
                                        step=0.01
                                            )))
        #single linear output layer
        model.add(layers.Dense(1))

        #adam optimiser with variable learning rate
        model.compile(optimizer=optimizers.Adam(hp.Float(name="learning_rate",
                                                            min_value=1e-5,
                                                            max_value=1e-3,
                                                            step=5e-5)),
                    loss="mean_squared_error",
                    metrics=["mse"])

        return model

    def fit(self, hp, model, *args, **kwargs):
        """fit hyper-parameter tuning of epochs and batch_size in fit method

        Args:
            hp (hyper-parameter): hyper-parameter settings for build from tuner
            model (model): model to  be fitted

        Returns:
            model: trained model
        """
        return model.fit(
            *args,
            epochs=hp.Int(name = "epochs",
                          min_value = 5,
                          max_value = 50,
                          step = 5),
            batch_size=hp.Int(name = "batch_size",
                              min_value=256,
                              max_value=1024,
                              step=64),
            **kwargs)


class RemainingUsefulLifeHyperModel(HyperModel):
    def __init__(self, features):
        self.features = features

    def build(self, hp):
        model = Sequential()

        #first convolutional layer with range of filters and kernel sizes
        model.add(layers.Conv1D(filters=hp.Int(name="conv1d_filter_1",
                                            min_value=2,
                                            max_value=64,
                                            step=2
                                            ),
                            kernel_size=hp.Choice(name="conv2d_kernal_1",
                                                    values = [2, 4]),
                            activation="relu",
                            input_shape=(2000, self.features)
                            ))

        #second convolutional layer with range of filters and kernel sizes
        model.add(layers.Conv1D(filters=hp.Int(name="conv1d_filter_2",
                                            min_value=512,
                                            max_value=2048,
                                            step=8,
                                            ),
                            kernel_size=hp.Choice(name="conv2d_kernal_2",
                                                    values = [2, 4]),
                            activation="relu"
                            ))

        #max pooling reduction of feature map
        model.add(layers.MaxPooling1D((2)))

        #dropout between 0.05 and 0.50
        model.add(layers.Dropout(rate=hp.Float(name="dropout_1",
                                        min_value=0.05,
                                        max_value=0.50,
                                        step=0.01)))

        #third convolutional layer with range of filters and kernel sizes
        model.add(layers.Conv1D(filters=hp.Int(name="conv1d_filter_3",
                                            min_value=2,
                                            max_value=64,
                                            step=2),
                            kernel_size=hp.Choice(name="conv2d_kernal_3",
                                                    values = [2, 4]),
                            activation="relu"
                            ))

        #max pooling
        model.add(layers.MaxPooling1D((2)))

        #dropout between 0.05 and 0.50
        model.add(layers.Dropout(rate=hp.Float(name="dropout_2",
                                        min_value=0.05,
                                        max_value=0.50,
                                        step=0.01
                                            )))

        #flatten to vector for dense layer input
        model.add(layers.Flatten()),

        #single dense layer ahead of output with linear activation [-inf to inf]
        model.add(layers.Dense(units=hp.Int(name="dense_layer_1",
                                            min_value=2,
                                            max_value=64,
                                            step=2,
                                            )))

        #single output for regression with linear activation
        model.add(layers.Dense(1))

        #adam optimizer with variable learning rate
        model.compile(optimizer=optimizers.Adam(hp.Float(name="learning_rate",
                                                         min_value=1e-5,
                                                         max_value=1e-3,
                                                         step=5e-5)),
                      loss="mean_squared_error",
                      metrics=["mse"])

        return model

    def fit(self, hp, model, *args, **kwargs):
        """fit hyper-parameter tuning of epochs and batch_size in fit method

        Args:
            hp (hyper-parameter): hyper-parameter settings for build from tuner
            model (model): model to  be fitted

        Returns:
            model: trained model
        """
        return model.fit(
            *args,
            epochs=hp.Int(name = "epochs",
                          min_value = 50,
                          max_value = 1000,
                          step = 25),
            batch_size=hp.Int(name = "batch_size",
                              min_value=4,
                              max_value=16,
                              step=2),
            **kwargs)