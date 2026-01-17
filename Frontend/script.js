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
        
        const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ticker: ticker })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to get prediction');
        }
        
        const result = await response.json();
        console.log('Prediction result:', result);
        
        // Reload the page to show new data
        location.reload();
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error.message);
    } finally {
        document.getElementById('processBtn').disabled = false;
        document.getElementById('processBtn').textContent = 'SUBMIT';
    }
});
