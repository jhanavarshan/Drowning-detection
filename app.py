from flask import Flask, request, jsonify, render_template
from model import DrowningDetectionModel

app = Flask(__name__)
model = DrowningDetectionModel()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            heart_rate = float(request.form['heart_rate'])
            swim_angle = float(request.form['swim_angle'])
            
            result = model.predict(temperature, humidity, heart_rate, swim_angle)
            return render_template('result.html', result=result)
        
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        temperature = data['temperature']
        humidity = data['humidity']
        heart_rate = data['heart_rate']
        swim_angle = data['swim_angle']
        
        result = model.predict(temperature, humidity, heart_rate, swim_angle)
        return jsonify({'status': 'success', 'result': result})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    try:
        model.train()
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)