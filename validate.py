import os
import sys
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("🔍 Iniciando validación...")

# Cargar modelo desde joblib
print("📥 Cargando modelo...")
try:
    model = joblib.load("model.pkl")
    print("✅ Modelo cargado exitosamente")
except FileNotFoundError:
    print("❌ ERROR: No se encontró model.pkl")
    print("💡 Asegúrate de ejecutar 'python train.py' primero")
    sys.exit(1)

# Cargar datos de prueba (mismo split que en train)
print("📊 Cargando datos de prueba...")
X, y = load_diabetes(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Validar
print("🧪 Ejecutando validación...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Umbral de validación
THRESHOLD = 5000.0

print(f"\n📊 Resultados:")
print(f"   MSE: {mse:.4f}")
print(f"   Umbral: {THRESHOLD}")

if mse <= THRESHOLD:
    print("✅ VALIDACIÓN EXITOSA - Modelo aprobado")
    sys.exit(0)
else:
    print("❌ VALIDACIÓN FALLIDA - MSE supera el umbral")
    sys.exit(1)