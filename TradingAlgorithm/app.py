import sys
import subprocess
import json
import os
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd

# Set up Flask with static and template folders pointing to Frontend directory
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'Frontend')
app = Flask(__name__, static_folder=frontend_path, static_url_path='')

# Enable CORS with specific settings
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/')
def serve_index():
    return send_from_directory(frontend_path, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(frontend_path, filename)

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
            encoding='utf-8',
            timeout=300
        )

        if result.returncode != 0:
            print(f"Error running script: {result.stderr}")
            return jsonify({'error': f'Failed to run prediction: {result.stderr}'}), 500
        
        print(f"Script output: {result.stdout}")
        
        # Run sentiment analysis
        print(f"Running sentiment analysis for ticker: {ticker}")
        result_sentiment = subprocess.run(
            [sys.executable, 'onlysentiment.py', ticker],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300
        )

        if result_sentiment.returncode != 0:
            print(f"Error running sentiment script: {result_sentiment.stderr}")
            return jsonify({'error': f'Failed to run sentiment analysis: {result_sentiment.stderr}'}), 500
        
        print(f"Sentiment script output: {result_sentiment.stdout}")
        
        # Read the generated CSV files
        validation_csv_path = os.path.join(os.path.dirname(__file__), 'predicted_vs_actual_validation.csv')
        sentiment_csv_path = os.path.join(os.path.dirname(__file__), 'sentiment_results.csv')
        next_month_csv_path = os.path.join(os.path.dirname(__file__), 'next_month_prediction.csv')
        
        if not os.path.exists(validation_csv_path):
            return jsonify({'error': 'Prediction CSV file not generated'}), 500
        
        # Read trading algorithm CSV
        df = pd.read_csv(validation_csv_path)
        csv_data = df.to_dict('records')

        # Read next month prediction CSV if it exists
        chart_data = None
        if os.path.exists(next_month_csv_path):
            df_chart = pd.read_csv(next_month_csv_path)
            chart_data = df_chart.to_dict('records')

        # Read sentiment analysis CSV if it exists
        sentiment_data = None
        if os.path.exists(sentiment_csv_path):
            df_sentiment = pd.read_csv(sentiment_csv_path)
            sentiment_data = df_sentiment.to_dict('records')
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'data': csv_data,
            'chart_data': chart_data,
            'validation_data': csv_data,
            'sentiment_data': sentiment_data,
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
