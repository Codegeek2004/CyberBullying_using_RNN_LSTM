from flask import Flask, render_template, request
import webbrowser
from text_prediction_rnn import predict_cyberbullying  # Import your prediction function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_input = None

    if request.method == 'POST':
        # Get the user input from the form
        user_input = request.form['text']
        
        # Call the prediction function from the text_prediction_rnn module
        result = predict_cyberbullying(user_input)

    # Render the index page with or without the result
    return render_template('index.html', result=result, text_input=user_input)

if __name__ == '__main__':
    # Open the app in the browser automatically before starting the server
    webbrowser.open('http://127.0.0.1:5000/', new=2)  # 'new=2' opens in a new tab or window
    app.run(debug=True)
