import webbrowser
from flask import Flask, render_template, request
from text_prediction_rnn import predict_cyberbullying  # Make sure this module is available and works correctly

app = Flask(__name__)

@app.route('/')
def index():
    # Render the template with an empty initial input and no result
    return render_template('index.html', text_input='', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    result = predict_cyberbullying(user_input)
    # Pass both the user input and the result to the template
    return render_template('index.html', result=result, text_input=user_input)

if __name__ == '__main__':
    # Automatically open the web page when the server starts
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)
