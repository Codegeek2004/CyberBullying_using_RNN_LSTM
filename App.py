from flask import Flask, request, render_template
from text_prediction import predict_cyberbullying  # Assuming this function exists

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle both GET (initial) and POST (form submission) requests.
    On POST, predict cyberbullying for the input text.
    """
    result = None
    text_input = ""  # Default empty value for text input

    if request.method == "POST":
        text_input = request.form.get("text")  # Get the text input from the form
        if text_input:  # Ensure text is not empty
            result = predict_cyberbullying(text_input)  # Predict cyberbullying

    # Return the result and the text input to the template
    return render_template("index.html", result=result, text_input=text_input)

if __name__ == "__main__":
    app.run(debug=True)  # Run the app in debug mode
