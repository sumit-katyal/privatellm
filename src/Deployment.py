from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app and model pipeline
app = Flask(__name__)
model = pipeline("text-generation", model="gpt2")

# Define API endpoint
@app.route("/generate", methods=["POST"])
def generate():
    # Get input text from the request
    input_text = request.json["input"]
    # Generate text using the model
    output = model(input_text, max_length=50)
    # Return the generated text as a JSON response
    return jsonify(output)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)