from flask import Flask, request, jsonify
import torch
from train import CNN

# Load the trained model parameters
model = CNN(10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Initialize a Flask instance
app = Flask(__name__)

# Define a POST route for handling requests from the frontend
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Process the data into the format expected by the model
    input_data = torch.Tensor(data['input_data'])

    # Run the model on the processed data
    output_data = model(input_data)

    # Convert the output data to a list and return as JSON
    output_data = output_data.detach().numpy().tolist()
    return jsonify({'output_data': output_data})

# Run the Flask application if the script is run directly
if __name__ == "__main__":
    app.run()

