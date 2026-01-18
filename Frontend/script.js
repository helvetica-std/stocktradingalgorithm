document.getElementById('processBtn').addEventListener('click', async function() {
    const ticker = document.getElementById('days').value.trim();
    
    if (!ticker) {
        alert('Please enter a stock ticker');
        return;
    }
    
    console.log('Submitting ticker:', ticker);
    
    try {
        document.getElementById('processBtn').disabled = true;
        document.getElementById('processBtn').textContent = 'Processing...';
        
        // Determine the API URL - use current domain if available, otherwise localhost
        const apiUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://localhost:5000/api/predict'
            : `http://${window.location.hostname}:5000/api/predict`;
        
        console.log('Fetching from:', apiUrl);
        
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ticker: ticker })
        });
        
        if (!response.ok) {
            try {
                const error = await response.json();
                throw new Error(error.error || `Failed to get prediction (Status: ${response.status})`);
            } catch (e) {
                throw new Error(`Failed to get prediction (Status: ${response.status})`);
            }
        }
        
        const result = await response.json();
        console.log('Prediction result:', result);
        
        // Update sentiment ticker
        document.getElementById('sentiment-ticker').textContent = `${ticker} - Market Sentiment`;
        
        // Wait a bit for chart functions to be available
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Draw prediction chart if data is available
        if (result.chart_data && result.chart_data.length > 0) {
            console.log('Drawing prediction chart with data:', result.chart_data);
            if (typeof window.drawPredictionChart === 'function') {
                window.drawPredictionChart(result.chart_data);
            } else {
                console.error('drawPredictionChart function not available');
            }
        }

        // Draw validation chart if data is available
        if (result.validation_data && result.validation_data.length > 0) {
            console.log('Drawing validation chart with data:', result.validation_data);
            if (typeof window.drawValidationChart === 'function') {
                window.drawValidationChart(result.validation_data);
            } else {
                console.error('drawValidationChart function not available');
            }
        }
        
        // Draw sentiment chart if data is available
        if (result.sentiment_data && result.sentiment_data.length > 0) {
            console.log('Drawing sentiment chart with data:', result.sentiment_data);
            if (typeof window.drawSentimentChart === 'function') {
                window.drawSentimentChart(result.sentiment_data);
            } else {
                console.error('drawSentimentChart function not available');
            }
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error.message + '\n\nMake sure the backend server is running on port 5000.');
    } finally {
        document.getElementById('processBtn').disabled = false;
        document.getElementById('processBtn').textContent = 'ANALYZE';
    }
});
