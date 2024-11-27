import webbrowser
from flask import Flask, render_template, request
from text_prediction_rnn import predict_cyberbullying

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', text_input='', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    result = predict_cyberbullying(user_input)
    return render_template('index.html', result=result, text_input=user_input)

if __name__ == '__main__':
    # Open the app in the browser automatically
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)
