import webbrowser  # Import webbrowser module
from flask import Flask, render_template, request
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
    # Delay the browser opening slightly to ensure the server is running
    app.run(debug=False)
