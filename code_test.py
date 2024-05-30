import json
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import keras.src.saving
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import keras

@keras.saving.register_keras_serializable(package="MyMetrics")
class CustomMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes=8, name='mean_iou', **kwargs):
        super(CustomMeanIoU, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count_classes = self.add_weight(name='count_classes', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) == 4:
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)

        mean_iou = 0.0
        class_count = 0.0

        for i in range(self.num_classes):
            true_class = tf.cast(tf.equal(y_true, i), dtype=tf.float32)
            pred_class = tf.cast(tf.equal(y_pred, i), dtype=tf.float32)

            intersection = tf.reduce_sum(true_class * pred_class)
            union = tf.reduce_sum(true_class) + tf.reduce_sum(pred_class) - intersection

            iou = intersection / (union + K.epsilon())
            condition = tf.equal(union, 0)
            mean_iou = tf.where(condition, mean_iou, mean_iou + iou)
            class_count = tf.where(condition, class_count, class_count + 1)

        self.total_iou.assign_add(mean_iou)
        self.count_classes.assign_add(class_count)

    def result(self):
        return self.total_iou / (self.count_classes + K.epsilon())

    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count_classes.assign(0.0)

    def get_config(self):
        config = super(CustomMeanIoU, self).get_config()
        config.update({
            'num_classes': self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

custom_object = CustomMeanIoU()

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512, 512))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def run(raw_data):
    image = preprocess_image(raw_data)
    model = tf.keras.models.load_model("./model/mini_unet_SA_complete.keras", custom_objects={"CustomMeanIoU": custom_object})
    prediction = model.predict(image)
    predicted_mask = np.argmax(prediction, axis=-1).squeeze().tolist()  # Remove batch dimension and convert to list
    print(predicted_mask)

with open("./model/test.png", "rb") as file:
        image_bytes = file.read()
run(image_bytes)