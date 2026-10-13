from flask import Flask, request, jsonify

app = Flask(__name__)

# Định nghĩa routes cơ bản
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Flask API running",
        "version": "1.0.0"
    })

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": "2023-10-27T12:00:00Z"
    })

if __name__ == '__main__':
    # Chạy server cho dev
    app.run(host='0.0.0.0', port=5000, debug=True)