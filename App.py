from flask import Flask, render_template, request
from text_prediction import predict_cyberbullying

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle both GET (initial) and POST (form submission) requests.
    On POST, predict cyberbullying for the input text.
    """
    result = None
    if request.method == "POST":
        # Correctly access the form data using the input's `name` attribute
        text = request.form.get('text')  # Use 'text', which matches the input field name
        result = predict_cyberbullying(text)  # Predict cyberbullying

    return render_template("index.html", result=result)