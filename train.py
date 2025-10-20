import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
# Se quita 'from pathlib import Path'

print("🚀 Iniciando entrenamiento...")

# Configurar MLflow para usar una ruta relativa simple, compatible con CI/CD (Linux)
mlruns_dir = "./mlruns"
os.makedirs(mlruns_dir, exist_ok=True)

# Establecer la URI de tracking. MLflow manejará la ruta absoluta internamente.
mlflow.set_tracking_uri(mlruns_dir)

print(f"📂 Directorio de trabajo: {os.getcwd()}")
print(f"📊 MLflow tracking URI: {mlruns_dir}")

# Crear o usar experimento
experiment_name = "CI-CD-Lab2"
try:
    # Intenta crear el experimento (esto fallará si ya existe)
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"✨ Experimento creado: {experiment_name}")
except:
    # Si el experimento existe, obtén su ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"📌 Usando experimento existente: {experiment_name}")

# Cargar datos
print("📥 Cargando dataset diabetes...")
X, y = load_diabetes(return_X_y=True)
# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
print("🏋️ Entrenando modelo...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"📊 MSE: {mse:.4f}")

# Guardar con MLflow
with mlflow.start_run(experiment_id=experiment_id) as run:
    # Loguear parámetros y métricas
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("mse", mse)
    
    # Guardar modelo en MLflow (como artefacto)
    mlflow.sklearn.log_model(model, "model")
    
    run_id = run.info.run_id
    print(f"✅ Run ID: {run_id}")
    
    # Guardar run_id para el script de validación
    with open("run_id.txt", "w") as f:
        f.write(run_id)

# Guardar con joblib (para validación local y el artefacto)
joblib.dump(model, "model.pkl")
print("💾 Modelo guardado como model.pkl")
print("✅ Entrenamiento completado exitosamente")