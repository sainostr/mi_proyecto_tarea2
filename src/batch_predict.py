import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import os

MODEL_PATH = "models/pca_mnist_model.pkl"

def batch_predict(n_samples=20):
    print("=" * 60)
    print("PREDICCIONES EN LOTE (BATCH)")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"✗ ERROR: Modelo no encontrado")
        return
    
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"\n✓ Modelo cargado")
    print(f"[1/3] Descargando MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    data_mnist = mnist.data.astype(float)
    y_data = mnist.target
    
    print(f"[2/3] Seleccionando {n_samples} muestras...")
    np.random.seed(42)
    indices = np.random.choice(len(data_mnist), n_samples, replace=False)
    x_batch = data_mnist[indices]
    y_batch = y_data[indices]
    
    V_n = model_data['V_n']
    avg = model_data['avg']
    
    print(f"[3/3] Realizando predicciones...")
    
    results = []
    for i, (image, label) in enumerate(zip(x_batch, y_batch)):
        image_centered = image - avg
        compressed = np.dot(image_centered, V_n.T)
        reconstructed = np.dot(compressed, V_n) + avg
        mse = np.mean((image - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        
        results.append({
            'sample_id': i,
            'label': int(label),
            'mse': float(mse),
            'rmse': float(rmse)
        })
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(df_results.to_string(index=False))
    
    output_file = "predicciones_batch.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Guardado en: {output_file}")

if __name__ == "__main__":
    batch_predict(n_samples=20)
