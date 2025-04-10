from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import pairwise_distances
import urllib.parse
import pickle

app = Flask(__name__)

# Load scaler and data
scaler = joblib.load('model/cluster_scaler.joblib')
df = pd.read_csv('data/kickstarter_cluster_data.csv')  # Load your dataset
X_scaled = scaler.transform(df[['log_goal', 'duration', 'launch_month', 'launch_year', 
                                'category_success_rate', 'country_success_rate']])  # Pre-scale dataset features

# Load success rate mappings
with open('model/category_success_rate.pkl', 'rb') as f:
    category_success_rate = pickle.load(f)
with open('model/country_success_rate.pkl', 'rb') as f:
    country_success_rate = pickle.load(f)

# Function to create Kickstarter search URL
def create_search_url(project_name):
    encoded_name = urllib.parse.quote(project_name)
    return f"https://www.kickstarter.com/discover/advanced?ref=nav_search&term={encoded_name}"

# API endpoint for nearest projects
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
            'country': data['country']
        }

        # Prepare the new project data
        new_project['log_goal'] = np.log1p(new_project['goal'])
        new_project['category_success_rate'] = category_success_rate.get(new_project['category'], 0.5)  # Default 0.5 if missing
        new_project['country_success_rate'] = country_success_rate.get(new_project['country'], 0.5)  # Default 0.5 if missing
        new_project_data = [new_project['log_goal'], new_project['duration'], new_project['launch_month'],
                            new_project['launch_year'], new_project['category_success_rate'], new_project['country_success_rate']]

        # Scale the new project data
        new_project_scaled = scaler.transform([new_project_data])

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

        # Return JSON response
        response = {
            'success_predition': 0.42,
            'closest_successful': closest_successful_data,
            'closest_failed': closest_failed_data
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5001)
    app.run()