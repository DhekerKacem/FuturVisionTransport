import os

import keras.src.saving


class ModelUtils:
    keras_model = None

    def __init__(self):
        if not self.keras_model:
            keras_model = self.__load_model()

    def __load_model(self):
        return keras.src.saving.load_model(os.environ['MODEL_PATH'])

    def get_loaded_model(self):
        return self.keras_model
