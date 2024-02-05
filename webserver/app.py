from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    # Ensure there is JSON data in the request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Parse the JSON data
    data = request.get_json()

    # Print json 
    print("Received data:", data)

    # Data was successfully received
    return jsonify({"message": "Data received successfully"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
