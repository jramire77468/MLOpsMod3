.PHONY: install train validate all clean

install:
	pip install -r requirements.txt

train:
	python train.py

validate:
	python validate.py

all: install train validate

clean:
	rmdir /s /q mlruns 2>nul || echo "mlruns no existe"
	del /f run_id.txt model.pkl 2>nul || echo "Archivos ya limpios"