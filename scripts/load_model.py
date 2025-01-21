import pickle

import tensorflow as tf


class Models():
    def __init__(self):
        self.model = tf.keras.models.load_model("E:/Individual thesis development/recommendation_system/models/model.keras")
        with open("E:/Individual thesis development/recommendation_system/models/encoders_scalers.pkl", "rb") as f:
            encoders_scalers = pickle.load(f)

        # Extract individual encoders and scaler
        self.user_encoder = encoders_scalers["user_encoder"]
        self.product_encoder = encoders_scalers["product_encoder"]
        self.gender_encoder = encoders_scalers["gender_encoder"]
        self.location_encoder = encoders_scalers["location_encoder"]
        self.scaler = encoders_scalers["scaler"]