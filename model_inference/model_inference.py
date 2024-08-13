from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Define the data path
data_path = r"C:/NYP_JiayiCourses/Y3S1/EGT309 - AI SOLUTION DEVELOPMENT PROJECT/App/volume"

# Load customer dataset and extract unique cities and states
customer_data = pd.read_csv(f"{data_path}/customer_dataset.csv")
unique_states = customer_data['customer_state'].unique()
unique_cities = customer_data['customer_city'].unique()

# Manually set the min and max values for each feature
min_vals = np.array([0.0, 0.0, 0.0, 0.85, 0.91, 1.0, 0.0, 7.0, 2.0, 6.0, 0.0, 0.0])
max_vals = np.array([4092.0, 26.0, 70.0, 277.3, 33.4, 6.0, 4050.0, 68.0, 38.0, 52.0, 3.0, 8.0])

# Create a mapping from city to state
city_to_state = customer_data.set_index('customer_city')['customer_state'].to_dict()

def manual_min_max_scaler(X, min_vals, max_vals):
    # Ensure all data is numeric for scaling
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Replace NaNs resulting from conversion with column means or other imputation strategy
    X.fillna(X.mean(), inplace=True)
    
    return (X - min_vals) / (max_vals - min_vals)

# Load the trained model
model = joblib.load(f"{data_path}/saved_models/rf.pkl")

@app.route('/')
def index():
    return render_template('index.html', states=unique_states, cities=unique_cities)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/city_state', methods=['GET'])
def get_state_for_city():
    city = request.args.get('city')
    if city in city_to_state:
        state = city_to_state[city]
        return jsonify({'state': state})
    else:
        return jsonify({'error': 'City not found'}), 404

@app.route('/categories', methods=['GET'])
def get_categories():
    unique_categories = customer_data['product_category_name_english'].dropna().unique().tolist()
    return jsonify(unique_categories)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate customer state
    if data['customer_state'] not in unique_states:
        return jsonify({'error': 'Invalid customer state'}), 400

    # Validate customer city
    if data['customer_city'] not in unique_cities:
        return jsonify({'error': 'Invalid customer city'}), 400

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])
    
    # Clean and preprocess the data
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].fillna(input_df[col].mode()[0])
        else:
            input_df[col] = input_df[col].fillna(input_df[col].median())

    # Label Encoding for categorical columns
    categorical_columns = ['customer_city', 'customer_state', 'product_category_name_english', 'payment_type']
    for col in categorical_columns:
        if col in input_df.columns:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col].astype(str))

    # Manual Scaling
    input_df_scaled = manual_min_max_scaler(input_df, min_vals, max_vals)
    
    # Make prediction
    prediction = model.predict(input_df_scaled)
    probability = model.predict_proba(input_df_scaled)[:, 1][0]  # Assuming binary classification

    if prediction[0] == 1:
        result_message = "Customer is <b>likely</b> to be a repeat buyer"
    else:
        result_message = "Customer is <b>not likely</b> to be a repeat buyer"

    return jsonify({'prediction': result_message, 'probability': probability})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
