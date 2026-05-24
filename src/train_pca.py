import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
import time

# Configurar MLflow LOCAL (sin servidor remoto)
mlflow.set_experiment("PCA_MNIST")

def main():
    """Script de entrenamiento para PCA en MNIST"""
    
    n_samples = 5000
    n_components = 392
    random_seed = 100
    
    with mlflow.start_run():
        print("=" * 60)
        print("INICIANDO ENTRENAMIENTO PCA - MNIST")
        print("=" * 60)
        
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("random_seed", random_seed)
        
        start_time = time.time()
        
        try:
            print("\n[1/6] Descargando datos MNIST...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            data_mnist = mnist.data.astype(float)
            print(f"✓ Datos cargados: {data_mnist.shape}")
            
            print(f"\n[2/6] Seleccionando {n_samples} muestras...")
            np.random.seed(random_seed)
            indices = np.random.choice(len(data_mnist), n_samples, replace=False)
            x_data = data_mnist[indices]
            y_data = mnist.target[indices]
            print(f"✓ Muestra seleccionada: {x_data.shape}")
            
            print("\n[3/6] Normalizando datos...")
            avg = x_data.mean(axis=0)
            centrado = x_data - avg
            print(f"✓ Media calculada: {avg.shape}")
            
            print("\n[4/6] Calculando SVD...")
            U, sigma, V = np.linalg.svd(centrado)
            
            n1 = x_data.shape[0]
            n2 = x_data.shape[1]
            Sigma = np.zeros((n1, n2))
            np.fill_diagonal(Sigma, sigma)
            
            print(f"✓ SVD completado: U{U.shape}, Sigma{Sigma.shape}, V{V.shape}")
            
            print(f"\n[5/6] Reduciendo a {n_components} componentes...")
            V_n = V[:n_components, :]
            X_new = np.dot(centrado, V_n.T)
            print(f"✓ Original: {centrado.shape} → Reducida: {X_new.shape}")
            
            print("\n[6/6] Reconstruyendo imágenes...")
            X_reconstructed = np.dot(X_new, V_n) + avg
            print(f"✓ Imágenes reconstruidas: {X_reconstructed.shape}")
            
            print("\n" + "=" * 60)
            print("CALCULANDO MÉTRICAS")
            print("=" * 60)
            
            var_total = np.sum(Sigma**2)
            var_n = np.sum(Sigma[:n_components]**2)
            varianza_explicada = (var_n / var_total) * 100
            
            print(f"\nVarianza explicada: {varianza_explicada:.2f}%")
            mlflow.log_metric("varianza_explicada", varianza_explicada)
            
            mse = mean_squared_error(x_data, X_reconstructed)
            rmse = np.sqrt(mse)
            
            print(f"MSE: {mse:.6f}")
            print(f"RMSE: {rmse:.6f}")
            
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            
            tamaño_original = n_samples * n2
            tamaño_comprimido = n_samples * n_components + n_components * n2
            ratio_compresion = (1 - tamaño_comprimido / tamaño_original) * 100
            
            print(f"Ratio de compresión: {ratio_compresion:.2f}%")
            mlflow.log_metric("ratio_compresion", ratio_compresion)
            
            elapsed_time = time.time() - start_time
            print(f"Tiempo total: {elapsed_time:.2f} segundos")
            mlflow.log_metric("tiempo_ejecucion", elapsed_time)
            
            print("\n" + "=" * 60)
            print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            print("=" * 60)
            print(f"\nRun ID: {mlflow.active_run().info.run_id}")
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            raise

if __name__ == "__main__":
    main()
