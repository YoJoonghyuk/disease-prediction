from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import traceback 
import numpy as np 

app = Flask(__name__)

# 1. Загрузка токенизатора
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

# 2. Загрузка LabelEncoder
try:
    with open('label_encoder.pickle', 'rb') as handle:
        le = pickle.load(handle)
except Exception as e:
    print(f"Error loading label encoder: {e}")
    le = None

# 3. Загрузка модели
try:
    model = load_model("gru_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

MAX_SEQUENCE_LENGTH = 200

# Маршрут для отображения HTML-страницы
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if tokenizer is None or le is None or model is None:
        return jsonify({'error': 'Model or preprocessing files not loaded'}), 500

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        text_sequence = tokenizer.texts_to_sequences([text])
        text_padded = pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH)
        prediction = model.predict(text_padded)
        predicted_class = np.argmax(prediction)
        predicted_disease = le.inverse_transform([predicted_class])[0]
        result = {'disease': predicted_disease}
        return jsonify(result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()  
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
