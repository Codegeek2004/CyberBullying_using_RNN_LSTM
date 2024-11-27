# app.py
from flask import Flask, render_template, request
from text_prediction_rnn import predict_cyberbullying

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    result = predict_cyberbullying(user_input)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
