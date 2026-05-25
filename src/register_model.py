import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
import pickle
import os

# Configurar MLflow
mlflow.set_experiment("PCA_MNIST_Model_Registry")

def train_and_register_best_model():
    """
    Entrena el modelo PCA con los mejores parámetros
    y lo registra en MLflow Model Registry
    """
    
    n_components = 392  # Mejor balance de nuestros experimentos
    n_samples = 5000
    random_seed = 100
    
    with mlflow.start_run():
        print("=" * 60)
        print("REGISTRANDO MODELO - PCA MNIST")
        print("=" * 60)
        
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("model_type", "PCA")
        
        try:
            print("\n[1/5] Descargando MNIST...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            data_mnist = mnist.data.astype(float)
            
            print("[2/5] Preparando datos...")
            np.random.seed(random_seed)
            indices = np.random.choice(len(data_mnist), n_samples, replace=False)
            x_data = data_mnist[indices]
            y_data = mnist.target[indices]
            
            print("[3/5] Normalizando y calculando SVD...")
            avg = x_data.mean(axis=0)
            centrado = x_data - avg
            
            U, sigma, V = np.linalg.svd(centrado)
            
            n1 = x_data.shape[0]
            n2 = x_data.shape[1]
            Sigma = np.zeros((n1, n2))
            np.fill_diagonal(Sigma, sigma)
            
            print("[4/5] Reduciendo dimensionalidad...")
            V_n = V[:n_components, :]
            X_new = np.dot(centrado, V_n.T)
            X_reconstructed = np.dot(X_new, V_n) + avg
            
            print("[5/5] Calculando métricas...")
            var_total = np.sum(Sigma**2)
            var_n = np.sum(Sigma[:n_components]**2)
            varianza_explicada = (var_n / var_total) * 100
            
            mse = mean_squared_error(x_data, X_reconstructed)
            rmse = np.sqrt(mse)
            
            tamaño_original = n_samples * n2
            tamaño_comprimido = n_samples * n_components + n_components * n2
            ratio_compresion = (1 - tamaño_comprimido / tamaño_original) * 100
            
            # Log métricas
            mlflow.log_metric("varianza_explicada", varianza_explicada)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("ratio_compresion", ratio_compresion)
            
            print(f"\n✓ Varianza explicada: {varianza_explicada:.2f}%")
            print(f"✓ MSE: {mse:.6f}")
            print(f"✓ RMSE: {rmse:.6f}")
            print(f"✓ Ratio de compresión: {ratio_compresion:.2f}%")
            
            # Crear diccionario con componentes del modelo
            model_data = {
                'V_n': V_n,
                'avg': avg,
                'n_components': n_components,
                'original_features': n2
            }
            
            # Guardar modelo localmente
            print("\n[+] Guardando modelo...")
            os.makedirs("models", exist_ok=True)
            model_path = "models/pca_mnist_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Log modelo en MLflow
            mlflow.log_artifact(model_path, artifact_path="pca_model")
            
            # Registrar modelo en Model Registry
            print("[+] Registrando en Model Registry...")
            mlflow.sklearn.log_model(
                sk_model=None,  # Usamos artifacts en su lugar
                artifact_path="pca_model_registry"
            )
            
            run_id = mlflow.active_run().info.run_id
            
            print("\n" + "=" * 60)
            print("✓ MODELO REGISTRADO EXITOSAMENTE")
            print("=" * 60)
            print(f"Run ID: {run_id}")
            print(f"Modelo guardado en: {model_path}")
            print(f"Disponible en MLflow UI para deployment")
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            raise

if __name__ == "__main__":
    train_and_register_best_model()
