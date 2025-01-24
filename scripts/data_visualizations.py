import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scripts.Dataframes import TrainingDataFrame

dataframes = TrainingDataFrame()

interactions_df = dataframes.interactions_df
users_df = dataframes.user_df
products_df = dataframes.prod_df
# Convert timestamp column to datetime
interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])

# def age_distribution():
# # Plot 1: Age distribution of users
#     plt.figure(figsize=(8, 5))
#     sns.histplot(users_df["age"], bins=20, kde=True, color="blue")
#     plt.title("Age Distribution of Users")
#     plt.xlabel("Age")
#     plt.ylabel("Count")
#     plt.show()
    


# # Plot 2: Gender distribution in dataset
# plt.figure(figsize=(6, 5))
# sns.countplot(x="gender", data=users_df, palette="coolwarm")
# plt.title("Gender Distribution of Users")
# plt.xlabel("Gender")
# plt.ylabel("Count")
# plt.show()



# # Plot 3: City-wise online shopping participation
# plt.figure(figsize=(10, 6))
# sns.countplot(y="location", data=users_df, order=users_df["location"].value_counts().index, palette="viridis")
# plt.title("City-Wise User Distribution")
# plt.xlabel("Count")
# plt.ylabel("Location")
# plt.show()




# # Plot 4: Most purchased product categories
# most_purchased = interactions_df[interactions_df["interaction_type"] == "purchase"].merge(products_df, on="product_id")
# plt.figure(figsize=(8, 5))
# sns.countplot(y=most_purchased["category"], order=most_purchased["category"].value_counts().index, palette="rocket")
# plt.title("Most Purchased Product Categories")
# plt.xlabel("Count")
# plt.ylabel("Category")
# plt.show()


# # Plot 5: Interaction frequency for different interaction types
# plt.figure(figsize=(7, 5))
# sns.countplot(x="interaction_type", data=interactions_df, palette="pastel")
# plt.title("Interaction Frequency by Type")
# plt.xlabel("Interaction Type")
# plt.ylabel("Count")
# plt.show()


# # Plot 6: Purchase frequency vs. Views
# interaction_counts = interactions_df["interaction_type"].value_counts()
# plt.figure(figsize=(7, 5))
# interaction_counts.plot(kind="bar", color=["blue", "orange", "green", "red"])
# plt.title("Comparison of Interaction Types")
# plt.xlabel("Interaction Type")
# plt.ylabel("Frequency")
# plt.show()

# # Plot 7: Price distribution across categories
# plt.figure(figsize=(12, 5))
# sns.boxplot(x="category", y="price", data=products_df, palette="coolwarm")
# plt.xticks(rotation=45)
# plt.title("Price Distribution Across Categories")
# plt.xlabel("Category")
# plt.ylabel("Price (NPR)")
# plt.show()


# 📌 Function Mapping for Visualization
PLOTS = {
    "age_distribution": "Age Distribution of Users",
    "gender_distribution": "Gender Distribution of Users",
    "location_distribution": "City-Wise User Distribution",
    "most_purchased_categories": "Most Purchased Product Categories",
    "interaction_frequency": "Interaction Frequency by Type",
    "interaction_comparison": "Comparison of Interaction Types",
    "price_distribution": "Price Distribution Across Categories"
}


def generate_plot(plot_id):
    """Generates the requested plot and saves it as an image."""
    
        # Ensure 'static/plots' directory exists
    plot_dir = "static/plots"
    os.makedirs(plot_dir, exist_ok=True)  # Creates directory if it doesn't exist
    
    plot_path = os.path.join(plot_dir, f"{plot_id}.png")
    
    plt.figure(figsize=(8, 5))

    if plot_id == "age_distribution":
        sns.histplot(users_df["age"], bins=20, kde=True, color="blue")
        plt.xlabel("Age"), plt.ylabel("Count")

    elif plot_id == "gender_distribution":
        sns.countplot(x="gender", data=users_df, palette="coolwarm")
        plt.xlabel("Gender"), plt.ylabel("Count")

    elif plot_id == "location_distribution":
        plt.figure(figsize=(10, 6))
        sns.countplot(y="location", data=users_df, order=users_df["location"].value_counts().index, palette="viridis")
        plt.xlabel("Count"), plt.ylabel("Location")

    elif plot_id == "most_purchased_categories":
        most_purchased = interactions_df[interactions_df["interaction_type"] == "purchase"].merge(products_df, on="product_id")
        sns.countplot(y=most_purchased["category"], order=most_purchased["category"].value_counts().index, palette="rocket")
        plt.xlabel("Count"), plt.ylabel("Category")

    elif plot_id == "interaction_frequency":
        sns.countplot(x="interaction_type", data=interactions_df, palette="pastel")
        plt.xlabel("Interaction Type"), plt.ylabel("Count")

    elif plot_id == "interaction_comparison":
        interaction_counts = interactions_df["interaction_type"].value_counts()
        interaction_counts.plot(kind="bar", color=["blue", "orange", "green", "red"])
        plt.xlabel("Interaction Type"), plt.ylabel("Frequency")

    elif plot_id == "price_distribution":
        plt.figure(figsize=(12, 5))
        sns.boxplot(x="category", y="price", data=products_df, palette="coolwarm")
        plt.xticks(rotation=45)
        plt.xlabel("Category"), plt.ylabel("Price (NPR)")

    else:
        return None  # Invalid plot_id

    plt.title(PLOTS[plot_id])
    plt.savefig(plot_path)
    plt.close()
    return plot_path