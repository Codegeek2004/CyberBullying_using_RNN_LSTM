from flask import Flask, render_template, request
import webbrowser
from text_prediction_rnn import predict_cyberbullying  # Make sure this import matches your actual file

app = Flask(__name__)

@app.route('/')
def index():
    # Render the index.html file when the user visits the root URL
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    user_input = request.form['text']
    
    # Call the prediction function from the text_prediction_rnn module
    result = predict_cyberbullying(user_input)
    
    # Render the template and pass the result back to the user
    return render_template('index.html', result=result, text_input=user_input)

if __name__ == '__main__':
    # Open the app in the browser automatically when the server starts
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)
