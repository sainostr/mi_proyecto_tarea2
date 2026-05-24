import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
import time

# Configurar MLflow
mlflow.set_experiment("PCA_MNIST_Experiments")

def train_pca(n_components, n_samples=5000, random_seed=100):
    """
    Entrena PCA con n_components específico y retorna métricas
    """
    
    with mlflow.start_run():
        print(f"\n{'='*60}")
        print(f"EXPERIMENTO: n_components = {n_components}")
        print(f"{'='*60}")
        
        # Log parámetros
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("random_seed", random_seed)
        
        start_time = time.time()
        
        try:
            # Cargar datos (solo primera vez)
            print(f"Descargando MNIST...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            data_mnist = mnist.data.astype(float)
            
            # Seleccionar muestra
            np.random.seed(random_seed)
            indices = np.random.choice(len(data_mnist), n_samples, replace=False)
            x_data = data_mnist[indices]
            y_data = mnist.target[indices]
            
            # Normalizar
            avg = x_data.mean(axis=0)
            centrado = x_data - avg
            
            # SVD
            U, sigma, V = np.linalg.svd(centrado)
            
            n1 = x_data.shape[0]
            n2 = x_data.shape[1]
            Sigma = np.zeros((n1, n2))
            np.fill_diagonal(Sigma, sigma)
            
            # Reducir dimensionalidad
            V_n = V[:n_components, :]
            X_new = np.dot(centrado, V_n.T)
            
            # Reconstruir
            X_reconstructed = np.dot(X_new, V_n) + avg
            
            # Calcular métricas
            var_total = np.sum(Sigma**2)
            var_n = np.sum(Sigma[:n_components]**2)
            varianza_explicada = (var_n / var_total) * 100
            
            mse = mean_squared_error(x_data, X_reconstructed)
            rmse = np.sqrt(mse)
            
            tamaño_original = n_samples * n2
            tamaño_comprimido = n_samples * n_components + n_components * n2
            ratio_compresion = (1 - tamaño_comprimido / tamaño_original) * 100
            
            elapsed_time = time.time() - start_time
            
            # Log métricas
            mlflow.log_metric("varianza_explicada", varianza_explicada)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("ratio_compresion", ratio_compresion)
            mlflow.log_metric("tiempo_ejecucion", elapsed_time)
            
            # Imprimir resultados
            print(f"✓ Varianza explicada: {varianza_explicada:.2f}%")
            print(f"✓ MSE: {mse:.6f}")
            print(f"✓ RMSE: {rmse:.6f}")
            print(f"✓ Ratio de compresión: {ratio_compresion:.2f}%")
            print(f"✓ Tiempo: {elapsed_time:.2f} segundos")
            
            return varianza_explicada, mse, rmse, ratio_compresion, elapsed_time
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            mlflow.log_param("error", str(e))
            raise

def main():
    """Ejecuta múltiples experimentos con diferentes n_components"""
    
    print("\n" + "="*60)
    print("INICIANDO EXPERIMENTOS - PCA MNIST")
    print("="*60)
    
    # Parámetros a probar
    components_to_test = [50, 100, 200, 392, 500, 600]
    
    results = {}
    
    for n_comp in components_to_test:
        try:
            var, mse, rmse, ratio, tiempo = train_pca(n_comp)
            results[n_comp] = {
                'varianza': var,
                'mse': mse,
                'rmse': rmse,
                'ratio': ratio,
                'tiempo': tiempo
            }
        except Exception as e:
            print(f"Fallo con n_components={n_comp}: {e}")
            continue
    
    # Resumen de resultados
    print("\n" + "="*60)
    print("RESUMEN DE EXPERIMENTOS")
    print("="*60)
    print(f"\n{'n_components':<15} {'Varianza':<15} {'MSE':<15} {'Compresión':<15} {'Tiempo':<10}")
    print("-" * 70)
    
    for n_comp, metrics in sorted(results.items()):
        print(f"{n_comp:<15} {metrics['varianza']:<15.2f} {metrics['mse']:<15.6f} {metrics['ratio']:<15.2f} {metrics['tiempo']:<10.2f}s")
    
    print("\n✓ TODOS LOS EXPERIMENTOS COMPLETADOS")
    print("  Ve a MLflow UI (http://localhost:5000) para ver comparativa")

if __name__ == "__main__":
    main()
