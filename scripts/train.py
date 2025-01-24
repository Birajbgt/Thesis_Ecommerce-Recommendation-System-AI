import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from Dataframes import TrainingDataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# from tensorflow.keras.layers import (BatchNormalization, Concatenate, Dense,
#                                      Dropout, Embedding, Flatten, Input)
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# ============================
# STEP 1: Load & Encode Data
# ============================

# Load Data

# Encode users and products
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

training_data_obj = TrainingDataFrame()
interactions_df = training_data_obj.interactions_df
user_df = training_data_obj.user_df

interactions_df["user_id"] = user_encoder.fit_transform(interactions_df["user_id"])
interactions_df["product_id"] = product_encoder.fit_transform(interactions_df["product_id"])
user_df["user_id"] = user_encoder.transform(user_df["user_id"])

# Encode categorical features (Gender & Location)
gender_encoder = LabelEncoder()
location_encoder = LabelEncoder()

user_df["gender"] = gender_encoder.fit_transform(user_df["gender"])
user_df["location"] = location_encoder.fit_transform(user_df["location"])

# Normalize Age (Feature Scaling)
scaler = StandardScaler()
user_df["age"] = scaler.fit_transform(user_df[["age"]])

# ============================
# STEP 2: Train-Test Split
# ============================

train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)

# Remove users/products in test that were not seen in training
train_users = train_df["user_id"].unique()
train_products = train_df["product_id"].unique()
test_df = test_df[test_df["user_id"].isin(train_users) & test_df["product_id"].isin(train_products)]

# Prepare training & test sets
def prepare_inputs(train_df, test_df, user_df):
    """Prepares model inputs and labels from train and test data."""

    X_train = {
        "users": train_df["user_id"].values,
        "products": train_df["product_id"].values,
        "gender": user_df.loc[train_df["user_id"], "gender"].values,
        "location": user_df.loc[train_df["user_id"], "location"].values,
        "age": user_df.loc[train_df["user_id"], "age"].values,
    }
    y_train = train_df["product_id"].values

    X_test = {
        "users": test_df["user_id"].values,
        "products": test_df["product_id"].values,
        "gender": user_df.loc[test_df["user_id"], "gender"].values,
        "location": user_df.loc[test_df["user_id"], "location"].values,
        "age": user_df.loc[test_df["user_id"], "age"].values,
    }
    y_test = test_df["product_id"].values

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = prepare_inputs(train_df, test_df, user_df)

# ============================
# STEP 3: Build Hybrid NCF Model
# ============================



def build_cf_model(num_users, num_products, embedding_size=50):
    """
    Builds the Collaborative Filtering (CF) component of the hybrid model.
    It learns latent user and product representations.
    """
    # **User Embeddings**
    user_input = Input(shape=(1,), name="users")
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
    user_vec = Flatten()(user_embedding)

    # **Product Embeddings**
    product_input = Input(shape=(1,), name="products")
    product_embedding = Embedding(input_dim=num_products, output_dim=embedding_size)(product_input)
    product_vec = Flatten()(product_embedding)

    return user_input, product_input, user_vec, product_vec


def build_cbf_model(num_genders, num_locations):
    """
    Builds the Content-Based Filtering (CBF) component of the hybrid model.
    It encodes user metadata like age, gender, and location.
    """
    # **Metadata Inputs**
    gender_input = Input(shape=(1,), name="gender")
    location_input = Input(shape=(1,), name="location")
    age_input = Input(shape=(1,), name="age")  # Direct numerical input

    # **Metadata Embeddings**
    gender_embedding = Embedding(input_dim=num_genders, output_dim=5)(gender_input)
    location_embedding = Embedding(input_dim=num_locations, output_dim=5)(location_input)

    gender_vec = Flatten()(gender_embedding)
    location_vec = Flatten()(location_embedding)

    return gender_input, location_input, age_input, gender_vec, location_vec


def build_hybrid_ncf_model(num_users, num_products, num_genders, num_locations, embedding_size=50):
    """
    Combines CF and CBF components to build the full Hybrid Neural Collaborative Filtering (NCF) model.
    """

    # Build Collaborative Filtering (CF) Model
    user_input, product_input, user_vec, product_vec = build_cf_model(num_users, num_products, embedding_size)

    # Build Content-Based Filtering (CBF) Model
    gender_input, location_input, age_input, gender_vec, location_vec = build_cbf_model(num_genders, num_locations)

    # **Concatenate Features**
    concat = Concatenate()([user_vec, product_vec, gender_vec, location_vec, age_input])

    # **Dense Layers for Final Prediction**
    dense1 = Dense(128, activation="relu")(concat)
    dense1 = Dropout(0.2)(dense1)
    dense1 = BatchNormalization()(dense1)

    dense2 = Dense(64, activation="relu")(dense1)
    dense2 = Dropout(0.2)(dense2)
    dense2 = BatchNormalization()(dense2)

    output = Dense(num_products, activation="softmax")(dense2)  # Predict product_id

    # **Build & Compile Model**
    model = Model([user_input, product_input, gender_input, location_input, age_input], output)
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def train_model():
    num_users = interactions_df["user_id"].nunique()
    num_products = interactions_df["product_id"].nunique()
    num_genders = user_df["gender"].nunique()
    num_locations = user_df["location"].nunique()

    model = build_hybrid_ncf_model(num_users, num_products, num_genders, num_locations)

    model.fit([X_train["users"], X_train["products"], X_train["gender"], X_train["location"], X_train["age"]], 
            y_train, 
            epochs=10, 
            batch_size=32, 
            validation_split=0.2)
    model.save('E:/Individual thesis development/recommendation_system/models/model.keras')

    with open("E:/Individual thesis development/recommendation_system/models/encoders_scalers.pkl", "wb") as f:
        pickle.dump({
            "user_encoder": user_encoder,
            "product_encoder": product_encoder,
            "gender_encoder": gender_encoder,
            "location_encoder": location_encoder,
            "scaler": scaler
        }, f)

    print(" Model and encoders saved successfully!")

train_model()
