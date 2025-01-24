# import numpy as np
# from scripts.Dataframes import TrainingDataFrame
# from scripts.load_model import Models


# def recommend_products( user_id = None, age=None, gender=None, location=None, top_n=5):
#     model_obj = Models()
#     model = model_obj.model
    
#     user_encoder = model_obj.user_encoder
#     product_encoder = model_obj.product_encoder
#     gender_encoder = model_obj.gender_encoder
#     location_encoder = model_obj.location_encoder
#     scaler = model_obj.scaler
    
#     training_data_obj = TrainingDataFrame()
#     interactions_df = training_data_obj.interactions_df
#     products_df = training_data_obj.prod_df
#     num_products = interactions_df["product_id"].nunique()
    
#     print(user_id)
    
#     # print("Age"+age )
#     print("gender"+gender)
#     print("location"+location)
#     if user_id is not None and user_id in user_encoder.classes_:
#         print("CASE 1")
#         encoded_user = user_encoder.transform([user_id])[0]
#         user_input = np.array([encoded_user] * num_products)
#         product_input = np.array(range(num_products))

#         predictions = model.predict([user_input, product_input, 
#                                      np.zeros_like(user_input), 
#                                      np.zeros_like(user_input), 
#                                      np.zeros_like(user_input)])

#         recommended_ids = np.argpartition(predictions[0], -top_n)[-top_n:]
#         recommendations = product_encoder.inverse_transform(recommended_ids)
#         products = products_df.set_index("product_id").loc[recommendations].reset_index().to_dict(orient="records")

#         print(products)
#         return products

#     if age and gender and location:
#         print("CASE 2")
#         gender_encoded = gender_encoder.transform([gender])[0]
#         location_encoded = location_encoder.transform([location])[0]
#         age_scaled = scaler.transform([[age]])[0][0]

#         # Generate predictions for all products
#         product_input = np.array(range(num_products))
#         user_placeholder = np.zeros_like(product_input)  # Placeholder for user ID

#         predictions = model.predict([user_placeholder, product_input, 
#                                      np.array([gender_encoded] * num_products), 
#                                      np.array([location_encoded] * num_products), 
#                                      np.array([age_scaled] * num_products)])

#         recommended_ids = np.argpartition(predictions[0], -top_n)[-top_n:]
#         recommendations =  product_encoder.inverse_transform(recommended_ids)
        
#         products = products_df.set_index("product_id").loc[recommendations].reset_index().to_dict(orient="records")
#         print(products)
#         return products
#     top_products = interactions_df.groupby("product_id").size().nlargest(top_n).index
#     recommendations = product_encoder.inverse_transform(top_products)
    
#     print("CASE 3")

#         # Get product details
#     products = products_df.set_index("product_id").loc[recommendations].reset_index().to_dict(orient="records")

#     print(products)
#     return products

import numpy as np
from scripts.Dataframes import TrainingDataFrame
from scripts.load_model import Models

# Dictionary to store past recommendations per user/session
previous_recommendations = {}

def recommend_products(user_id=None, age=None, gender=None, location=None, top_n=5):
    global previous_recommendations
    model_obj = Models()
    model = model_obj.model

    user_encoder = model_obj.user_encoder
    product_encoder = model_obj.product_encoder
    gender_encoder = model_obj.gender_encoder
    location_encoder = model_obj.location_encoder
    scaler = model_obj.scaler

    training_data_obj = TrainingDataFrame()
    interactions_df = training_data_obj.interactions_df
    products_df = training_data_obj.prod_df
    num_products = interactions_df["product_id"].nunique()

    print(f"User ID: {user_id}")
    print(f"Age: {age}")  
    print(f"Gender: {gender}")
    print(f"Location: {location}")

    # Get past recommendations for this user
    user_key = user_id if user_id else f"{age}-{gender}-{location}"
    prev_recs = previous_recommendations.get(user_key, set())

    if user_id is not None and user_id in user_encoder.classes_:
        print("CASE 1")
        encoded_user = user_encoder.transform([user_id])[0]
        user_input = np.array([encoded_user] * num_products)
        product_input = np.array(range(num_products))

        predictions = model.predict([
            user_input, product_input, 
            np.zeros_like(user_input), 
            np.zeros_like(user_input), 
            np.zeros_like(user_input)
        ])

        sorted_indices = np.argsort(predictions[0])[::-1]  # Sort by highest prediction score
        recommended_ids = [idx for idx in sorted_indices if idx not in prev_recs][:top_n]
        
        recommendations = product_encoder.inverse_transform(recommended_ids)
        products = products_df.set_index("product_id").loc[recommendations].reset_index().to_dict(orient="records")

        previous_recommendations[user_key] = prev_recs.union(set(recommended_ids))  # Save recommended items
        print(products)
        return products

    if age is not None and gender and location:
        print("CASE 2")
        gender_encoded = gender_encoder.transform([gender])[0]
        location_encoded = location_encoder.transform([location])[0]
        age_scaled = scaler.transform([[age]])[0][0]

        product_input = np.array(range(num_products))
        user_placeholder = np.zeros_like(product_input)  

        predictions = model.predict([
            user_placeholder, product_input, 
            np.array([gender_encoded] * num_products), 
            np.array([location_encoded] * num_products), 
            np.array([age_scaled] * num_products)
        ])

        sorted_indices = np.argsort(predictions[0])[::-1]
        recommended_ids = [idx for idx in sorted_indices if idx not in prev_recs][:top_n]
        
        recommendations = product_encoder.inverse_transform(recommended_ids)
        products = products_df.set_index("product_id").loc[recommendations].reset_index().to_dict(orient="records")

        previous_recommendations[user_key] = prev_recs.union(set(recommended_ids))
        print(products)
        return products

    print("CASE 3")
    top_products = interactions_df.groupby("product_id").size().nlargest(top_n).index
    recommendations = product_encoder.inverse_transform(top_products)

    products = products_df.set_index("product_id").loc[recommendations].reset_index().to_dict(orient="records")

    print(products)
    return products

# import numpy as np
# from scripts.Dataframes import TrainingDataFrame
# from scripts.load_model import Models


# def recommend_products(user_id=None, age=None, gender=None, location=None, top_n=5):
#     model_obj = Models()
#     model = model_obj.model

#     user_encoder = model_obj.user_encoder
#     product_encoder = model_obj.product_encoder
#     gender_encoder = model_obj.gender_encoder
#     location_encoder = model_obj.location_encoder
#     scaler = model_obj.scaler

#     training_data_obj = TrainingDataFrame()
#     interactions_df = training_data_obj.interactions_df
#     products_df = training_data_obj.prod_df
#     num_products = interactions_df["product_id"].nunique()

#     print(f"User ID: {user_id}")
#     print(f"Age: {age}")  # Convert to string if needed
#     print(f"Gender: {gender}")
#     print(f"Location: {location}")

#     if user_id is not None and user_id in user_encoder.classes_:
#         print("CASE 1")
#         encoded_user = user_encoder.transform([user_id])[0]
#         user_input = np.array([encoded_user] * num_products)
#         product_input = np.array(range(num_products))

#         predictions = model.predict([
#             user_input, product_input, 
#             np.zeros_like(user_input), 
#             np.zeros_like(user_input), 
#             np.zeros_like(user_input)
#         ])

#         recommended_ids = np.argpartition(predictions[0], -top_n)[-top_n:]
#         recommendations = product_encoder.inverse_transform(recommended_ids)
#         products = products_df.set_index("product_id").loc[recommendations].reset_index().to_dict(orient="records")

#         print(products)
#         return products

#     if age is not None and gender and location:
#         print("CASE 2")
#         gender_encoded = gender_encoder.transform([gender])[0]
#         location_encoded = location_encoder.transform([location])[0]
#         age_scaled = scaler.transform([[age]])[0][0]

#         product_input = np.array(range(num_products))
#         user_placeholder = np.zeros_like(product_input)  # Placeholder for user ID

#         predictions = model.predict([
#             user_placeholder, product_input, 
#             np.array([gender_encoded] * num_products), 
#             np.array([location_encoded] * num_products), 
#             np.array([age_scaled] * num_products)
#         ])

#         recommended_ids = np.argpartition(predictions[0], -top_n)[-top_n:]
#         recommendations = product_encoder.inverse_transform(recommended_ids)

#         products = products_df.set_index("product_id").loc[recommendations].reset_index().to_dict(orient="records")
#         print(products)
#         return products

#     print("CASE 3")
#     top_products = interactions_df.groupby("product_id").size().nlargest(top_n).index
#     recommendations = product_encoder.inverse_transform(top_products)

#     products = products_df.set_index("product_id").loc[recommendations].reset_index().to_dict(orient="records")

#     print(products)
#     return products


