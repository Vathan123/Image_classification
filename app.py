from flask import Flask, render_template, request
import requests

app = Flask(__name__)

api_url = "http://127.0.0.1:8000/predict"  # the backend link

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            files = {"file": (file.filename, file.stream, file.mimetype)}
            response = requests.post(api_url, files=files)
            print(response)
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("predicted_class")
                confidence = result.get("confidence")
            else:
                prediction = "Error from API"
    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
