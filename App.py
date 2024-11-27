import webbrowser
from flask import Flask, render_template, request
#from text_prediction_rnn import predict_cyberbullying

app = Flask(__name__)


if __name__ == '__main__':
    # Open the app in the browser automatically
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)
