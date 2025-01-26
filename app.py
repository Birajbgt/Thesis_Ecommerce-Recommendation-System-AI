import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_file
from scripts.data_visualizations import generate_plot
from scripts.recommend import \
    recommend_products  # Import recommendation function

PLOTS = {
    "age_distribution": "Age Distribution of Users",
    "gender_distribution": "Gender Distribution of Users",
    "location_distribution": "City-Wise User Distribution",
    "most_purchased_categories": "Most Purchased Product Categories",
    "interaction_frequency": "Interaction Frequency by Type",
    "interaction_comparison": "Comparison of Interaction Types",
    "price_distribution": "Price Distribution Across Categories"
}


# Flask App
app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         user_id = request.form.get("user_id", None)
#         if user_id:
#             user_id = int(user_id)
#             print(f"next user id {user_id}")
#         age = request.form.get("age", None)
#         print("age" + str(age))
#         if age:
#             age = float(age)
#         gender = request.form.get("gender", None)
#         print("gender" + str(gender))
#         location = request.form.get("location", None)
#         print("location" + str(location))

#         # Predict Recommended Products
#         recommendations = recommend_products(user_id, age, gender, location)

#         return jsonify(recommendations)  # Return JSON response

#     return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_id = request.form.get("user_id", None)
        if user_id:
            user_id = int(user_id)
            print(f"next user id {user_id}")
        age = request.form.get("age", None)
        if age:
            age = float(age)
        gender = request.form.get("gender", None)
        location = request.form.get("location", None)

        # Check if user wants new recommendations
        reset = request.form.get("reset", "false").lower() == "true"
        if reset:
            previous_recommendations.pop(user_id, None)  # Reset only for this user

        recommendations = recommend_products(user_id, age, gender, location)
        return jsonify(recommendations)

    return render_template("index.html")



@app.route("/admin/visualization", methods=["GET"])
def serve_visualization():
    """Serves the selected visualization based on user input."""
    plot_id = request.args.get("plot_id")
    if plot_id not in PLOTS:
        return jsonify({"error": "Invalid plot request"}), 400

    plot_path = generate_plot(plot_id)
    return send_file(plot_path, mimetype='image/png')

@app.route("/admin", methods=["GET"])
def admin_dashboard():
    """Admin dashboard to select and view different plots."""
    return render_template("admin.html", plots=PLOTS)


if __name__ == "__main__":
    app.run(debug=True)
