from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import urllib.parse
import pickle

app = Flask(__name__)

# Load cluster-related scaler and data
cluster_scaler = joblib.load('model/cluster_scaler.joblib')
df = pd.read_csv('data/kickstarter_cluster_data.csv')
X_scaled = cluster_scaler.transform(df[['log_goal', 'duration', 'launch_month', 'launch_year', 
                                        'category_success_rate', 'country_success_rate']])

# Load success rate mappings
with open('model/category_success_rate.pkl', 'rb') as f:
    category_success_rate = pickle.load(f)
with open('model/country_success_rate.pkl', 'rb') as f:
    country_success_rate = pickle.load(f)

# Load Random Forest model, scaler, and feature names
rf_model = joblib.load('model/rf_model.pkl')
rf_scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl')

# Function to create Kickstarter search URL
def create_search_url(project_name):
    encoded_name = urllib.parse.quote(project_name)
    return f"https://www.kickstarter.com/discover/advanced?ref=nav_search&term={encoded_name}"

# Function to preprocess data for Random Forest
def preprocess_rf_input(data, feature_names):
    # Create a DataFrame from the input data
    new_project = {
        'log_goal': np.log1p(float(data['goal'])),
        'duration': int(data['duration']),
        'launch_month': int(data['launch_month']),
        'launch_year': int(data['launch_year']),
        'category': data['category'],
        'country': data['country']
    }

    # Initialize a DataFrame with all required feature columns set to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # Fill in numeric features
    input_df['log_goal'] = new_project['log_goal']
    input_df['duration'] = new_project['duration']
    input_df['launch_month'] = new_project['launch_month']
    input_df['launch_year'] = new_project['launch_year']

    # One-hot encode category
    category_col = f"category_{new_project['category']}"
    if category_col in feature_names:
        input_df[category_col] = 1

    # One-hot encode country
    country_col = f"country_{new_project['country']}"
    if country_col in feature_names:
        input_df[country_col] = 1

    # Note: Subcategory is not provided in the POST request; assuming it's not critical here
    # If subcategory is needed, you'll need to add it to the POST data and encode it similarly

    return input_df

# API endpoint for nearest projects and Random Forest prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from POST request
        data = request.get_json()
        new_project = {
            'goal': float(data['goal']),
            'duration': int(data['duration']),
            'launch_month': int(data['launch_month']),
            'launch_year': int(data['launch_year']),
            'category': data['category'],
            'subcategory': data['subcategory'],
            'country': data['country']
        }

        # --- Cluster-based nearest projects ---
        # Prepare the new project data for clustering
        new_project['log_goal'] = np.log1p(new_project['goal'])
        new_project['category_success_rate'] = category_success_rate.get(new_project['category'], 0.5)
        new_project['country_success_rate'] = country_success_rate.get(new_project['country'], 0.5)
        new_project_data = [new_project['log_goal'], new_project['duration'], new_project['launch_month'],
                            new_project['launch_year'], new_project['category_success_rate'], new_project['country_success_rate']]

        # Scale the new project data
        new_project_scaled = cluster_scaler.transform([new_project_data])

        # Calculate distances to all projects
        distances = pairwise_distances(new_project_scaled, X_scaled, metric='euclidean').flatten()
        df['distance_to_new'] = distances

        # Get 5 closest successful and failed projects
        closest_successful = df[df['state'] == 'Successful'].nsmallest(5, 'distance_to_new')
        closest_failed = df[df['state'] == 'Failed'].nsmallest(5, 'distance_to_new')

        # Format results with search URLs
        closest_successful_data = [
            {
                'name': row['name'],
                'category': row['category'],
                'country': row['country'],
                'goal': row['goal'],
                'pledged': row['pledged'],
                'state': row['state'],
                'distance': row['distance_to_new'],
                'search_url': create_search_url(row['name'])
            } for _, row in closest_successful.iterrows()
        ]
        closest_failed_data = [
            {
                'name': row['name'],
                'category': row['category'],
                'country': row['country'],
                'goal': row['goal'],
                'pledged': row['pledged'],
                'state': row['state'],
                'distance': row['distance_to_new'],
                'search_url': create_search_url(row['name'])
            } for _, row in closest_failed.iterrows()
        ]

        # --- Random Forest Prediction ---
        # Preprocess input for Random Forest
        rf_input_df = preprocess_rf_input(new_project, feature_names)

        # Scale the input
        rf_input_scaled = rf_scaler.transform(rf_input_df)

        # Predict success probability
        rf_proba = rf_model.predict_proba(rf_input_scaled)[0][1]  # Probability of "Successful" (class 1)

        # Return JSON response with both cluster and RF results
        response = {
            'success_prediction': float(rf_proba),  # Replace hardcoded 0.42 with RF prediction
            'closest_successful': closest_successful_data,
            'closest_failed': closest_failed_data
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5001)  # Changed port to avoid conflict
    app.run()