from flask import Flask, render_template
import webbrowser

app = Flask(__name__)

@app.route('/')
def index():
    # Render the index.html file
    return render_template('index.html')

if __name__ == '__main__':
    # Open the app in the browser automatically when the server starts
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)
