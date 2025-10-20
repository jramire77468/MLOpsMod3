import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
# Se quita 'from pathlib import Path'
import pandas as pd
from mlflow.models.signature import infer_signature

print("🚀 Iniciando entrenamiento...")

# Configurar MLflow para usar una ruta relativa simple, compatible con CI/CD (Linux)
mlruns_dir = "./mlruns"
# Quitamos os.makedirs aquí, dejando que MLflow lo haga
mlflow.set_tracking_uri(mlruns_dir)

print(f"📂 Directorio de trabajo: {os.getcwd()}")
print(f"📊 MLflow tracking URI: {mlruns_dir}")

# Crear o usar experimento
experiment_name = "CI-CD-Lab2"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"✨ Experimento creado: {experiment_name}")
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"📌 Usando experimento existente: {experiment_name}")

# Cargar datos
print("📥 Cargando dataset diabetes...")
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
print("🏋️ Entrenando modelo...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"📊 MSE: {mse:.4f}")

# Obtener la firma del modelo
signature = infer_signature(X_train, y_train)

# Guardar con MLflow
with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("mse", mse)

    # Guardar modelo en MLflow con signature e input_example
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train[:5] # Usar las primeras 5 filas como ejemplo de entrada
    )

    run_id = run.info.run_id
    print(f"✅ Run ID: {run_id}")

    # Guardar run_id para validación
    with open("run_id.txt", "w") as f:
        f.write(run_id)

# TAMBIÉN guardar con joblib para compatibilidad
joblib.dump(model, "model.pkl")
print("💾 Modelo guardado como model.pkl")
print("✅ Entrenamiento completado exitosamente")