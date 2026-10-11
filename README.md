# Guia para ejecutar el script:

## 1. Instalar el ambiente de Python 3.14 del proyecto:

Primero, se debe crear un entorno de Python 3.14, ya sea un entorno virtual o un entorno de Conda. Después, se deben instalar las dependencias definidas en el archivo requirements.txt. Una vez completada la instalación, se debe activar el entorno para ejecutar el script del proyecto.

## 2. Iniciar servidor de ML Flow:

Antes de ejecutar el script `train.py`, es necesario iniciar una instancia local de MLflow Tracking Server. Este servicio se utiliza para registrar los parámetros de ejecución, métricas, artefactos y demás información generada durante el proceso de entrenamiento. Una vez iniciado el servidor, el script podrá almacenar automáticamente los resultados de cada ejecución para su posterior consulta y comparación. El siguiente comando permite iniciar un servidor local de MLflow:

```powershell
mlflow ui
```

## 3. Ejecutar el script de train.py:

El script `train.py` se utiliza para entrenar un modelo de regresión logística a partir de los datos correspondientes a campañas de telemarketing de un banco portugués. Su ejecución se realiza mediante una interfaz de línea de comandos (CLI) que permite configurar la partición de los datos y los hiperparámetros del modelo. A continuación, se presentan ejemplos de ejecución para PowerShell en Windows y Bash en Linux, así como una descripción detallada de los argumentos soportados por el script.

### Ejemplo para Windows Power Shell:

```powershell
python train.py `
    --split-seed 123 `
    --test-size 0.25 `
    --C 0.5 `
    --solver saga `
    --max-iter 1000 `
    --tol 1e-5 `
    --class-weight balanced `
    --l1-ratio 0.2 `
    --model-seed 123
```

### Ejemplo para Linux Bash:

```bash
python train.py \
    --split-seed 123 \
    --test-size 0.25 \
    --C 0.5 \
    --solver saga \
    --max-iter 1000 \
    --tol 1e-5 \
    --class-weight balanced \
    --l1-ratio 0.2 \
    --model-seed 123
```

### Parámetros del script

| Parámetro | Tipo | Valor predeterminado | Descripción | Opciones |
|-----------|------|----------------------|-------------|----------|
| --split-seed | int | 42 | Semilla aleatoria para la división entre entrenamiento y prueba. | Cualquier entero |
| --test-size | float | 0.2 | Proporción de los datos utilizada para pruebas. | Valor entre `0` y `1` |
| --C | float | 1.0 | Inverso de la fuerza de regularización. | Valor positivo |
| --solver | str | "lbfgs" | Algoritmo de optimización utilizado para entrenar el modelo. | `lbfgs`, `liblinear`, `newton-cg`, `newton-cholesky`, `sag`, `saga` |
| --max-iter | int | 100 | Número máximo de iteraciones. | Entero positivo |
| --tol | float | 1e-4 | Tolerancia para el criterio de convergencia. | Valor positivo |
| --fit-intercept | flag | True | Incluye el término independiente (intercepto) en el modelo. | Presencia del argumento |
| --no-fit-intercept | flag | False | Excluye el término independiente (intercepto) del modelo. | Presencia del argumento |
| --class-weight | str \| None | None | Estrategia de ponderación de clases. | `balanced`, `None` |
| --l1-ratio | float | 0.0 | Tipo de regularización. `0` = L2, `1` = L1, valores intermedios = Elastic Net. | Entre `0` y `1` |
| --model-seed | int | 42 | Semilla aleatoria utilizada por la Regresión Logística. | Cualquier entero |