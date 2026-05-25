from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "models/pca_mnist_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    print(f"✓ Modelo cargado desde {MODEL_PATH}")
else:
    print(f"✗ ERROR: Modelo no encontrado")
    model_data = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Servidor PCA MNIST",
        "model_loaded": model_data is not None
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/compress', methods=['POST'])
def compress():
    try:
        if model_data is None:
            return jsonify({"error": "Modelo no cargado"}), 500
        
        data = request.json
        image = np.array(data['image'], dtype=float)
        
        if image.shape[0] != 784:
            return jsonify({"error": "Imagen debe tener 784 valores"}), 400
        
        V_n = model_data['V_n']
        avg = model_data['avg']
        
        image_centered = image - avg
        compressed = np.dot(image_centered, V_n.T)
        
        return jsonify({
            "status": "success",
            "compressed_size": int(compressed.shape[0]),
            "compression_ratio": f"{(1 - compressed.shape[0]/784) * 100:.2f}%",
            "compressed_image": compressed.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    try:
        if model_data is None:
            return jsonify({"error": "Modelo no cargado"}), 500
        
        data = request.json
        V_n = model_data['V_n']
        avg = model_data['avg']
        
        compressed = np.array(data['compressed_image'], dtype=float)
        reconstructed = np.dot(compressed, V_n) + avg
        
        return jsonify({
            "status": "success",
            "reconstructed_image": reconstructed.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("=" * 60)
    print("INICIANDO SERVIDOR PCA MNIST")
    print("=" * 60)
    print("Servidor: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
