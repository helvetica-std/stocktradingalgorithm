import sys
import subprocess
import json
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'IBM').upper()
        
        print(f"Received request for ticker: {ticker}")
        
        # Run the trading algorithm with the specified ticker
        result = subprocess.run(
            [sys.executable, 'trading_algorithm.py', ticker],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"Error running script: {result.stderr}")
            return jsonify({'error': f'Failed to run prediction: {result.stderr}'}), 500
        
        print(f"Script output: {result.stdout}")
        
        # Read the generated CSV files
        validation_csv_path = os.path.join(os.path.dirname(__file__), 'predicted_vs_actual_validation.csv')
        
        if not os.path.exists(validation_csv_path):
            return jsonify({'error': 'CSV file not generated'}), 500
        
        # Read CSV and convert to JSON
        df = pd.read_csv(validation_csv_path)
        csv_data = df.to_dict('records')
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'data': csv_data,
            'message': result.stdout
        })
    
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Prediction took too long (timeout)'}), 500
    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/csv', methods=['GET'])
def get_csv():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'predicted_vs_actual_validation.csv')
        if os.path.exists(csv_path):
            return send_file(csv_path, mimetype='text/csv', as_attachment=False)
        return jsonify({'error': 'CSV not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')
