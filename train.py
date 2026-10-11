import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
import argparse

import mlflow


def parse_args():
    """Permite al usuario pasar parámetros al script desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de Regresión Logística"
    )

    # Hiperparámetros para la división de datos.
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Semilla aleatoria para la división entre entrenamiento y prueba (predeterminado: 42)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción de los datos utilizada para pruebas (predeterminado: 0.2)",
    )

    # Hiperparámetros de Regresión Logística.
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        help="Algoritmo de optimización utilizado para entrenar el modelo",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Número máximo de iteraciones",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolerancia para el criterio de convergencia",
    )

    parser.add_argument(
        "--fit-intercept",
        action="store_true",
        help="Incluir término independiente (intercepto) en el modelo",
    )

    parser.add_argument(
        "--no-fit-intercept",
        dest="fit_intercept",
        action="store_false",
        help="No incluir término independiente (intercepto) en el modelo",
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        default=None,
        choices=["balanced", None],
        help="Estrategia de ponderación de clases",
    )

    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0,
        help="Tipo de regularización: L2=0, L1=1, Elastic-Net=valor entre 0 y 1",
    )

    parser.add_argument(
        "--model-seed",
        type=int,
        default=42,
        help="Semilla aleatoria para la Regresión Logística",
    )

    parser.set_defaults(fit_intercept=True)

    return parser.parse_args()


def load_data():
    """Carga los datos de las campañas publicitarias del banco portugués."""
    bank_data_df = pd.read_csv("bank-additional-full.csv", sep=";")

    return bank_data_df


def _get_column_transformer(bank_data_df: pd.DataFrame, skewed_vars: list[str]):
    """
    Crea un ColumnTransformer para preprocesar variables numéricas y categóricas.

    Aplica estandarización a las variables numéricas, transformación
    Yeo-Johnson seguida de estandarización a las variables numéricas con sesgo y
    One-Hot Encoding a las variables categóricas.

    Parameters
    ----------
    bank_data_df : pd.DataFrame
        Conjunto de datos de entrada.
    skewed_vars : list[str]
        Variables numéricas con sesgo.

    Returns
    -------
    ColumnTransformer
        Transformador configurado para el preprocesamiento de los datos.
    """
    # Se obtienen las variables categóricas del set de datos.
    cat_vars = bank_data_df.select_dtypes(include=["str"])
    # Se obtienen las variables numéricas del set de datos.
    num_vars = bank_data_df.select_dtypes(include=["int", "float"])

    # Transformaciones para las variables numéricas sin sesgo.
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])

    # Transformaciones para las variables numéricas con sesgo.
    numeric_skewed_pipeline = Pipeline(
        [
            ("power", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler()),
        ]
    )

    # Transformaciones para las variables categóricas.
    cat_pipeline = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(drop="first", sparse_output=False)),
        ]
    )

    # Transformador para aplicarle las transformaciones al set de datos de entrenamiento y prueba.
    columnTransformer = ColumnTransformer(
        transformers=[
            (
                "numeric_pipeline",
                numeric_pipeline,
                num_vars.columns.drop(skewed_vars).to_list(),
            ),
            ("numeric_skewed_pipeline", numeric_skewed_pipeline, skewed_vars),
            ("cat_pipeline", cat_pipeline, cat_vars.columns.drop("y").to_list()),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return columnTransformer


def preprocessing(bank_data_df: pd.DataFrame, test_size: float, split_seed: int):
    """
    Preprocesa los datos para el entrenamiento del modelo.

    Elimina variables no utilizadas, codifica la variable objetivo,
    divide los datos en conjuntos de entrenamiento y prueba y aplica
    las transformaciones necesarias a las variables numéricas y
    categóricas.

    Parameters
    ----------
    bank_data_df : pd.DataFrame
        Conjunto de datos original.
    test_size : float
        Proporción de observaciones destinadas al conjunto de prueba.
    split_seed : int
        Semilla aleatoria utilizada para la división de los datos.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train y y_test preprocesados.
    """
    # Se elimina la variable "duration" del set de datos y las variables numéricas.
    bank_data_df = bank_data_df.drop(columns=["duration"])

    # Se divide el set de datos en variables dependientes e independientes.
    X = bank_data_df.drop(columns=["y"])
    # Adicionalmente se transforman los valores de la variable dependiente en valores numéricos.
    y = bank_data_df["y"].map({"no": 0, "yes": 1})

    # Se dividen los datos en conjuntos de prueba y entrenamiento.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed, stratify=y
    )

    columnTransformer = _get_column_transformer(bank_data_df, ["campaign", "previous"])

    # Se le aplican las transformaciones al set de entrenamiento.
    X_train_index = X_train.index
    X_train = columnTransformer.fit_transform(X_train)

    X_train = pd.DataFrame(
        data=X_train,
        index=X_train_index,
        columns=columnTransformer.get_feature_names_out(),
    )

    # Se le aplican las transformaciones al set de pruebas.
    X_test_index = X_test.index
    X_test = columnTransformer.transform(X_test)

    X_test = pd.DataFrame(
        data=X_test,
        index=X_test_index,
        columns=columnTransformer.get_feature_names_out(),
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, args: argparse.Namespace):
    """
    Entrena un modelo de Regresión Logística utilizando los
    hiperparámetros proporcionados.

    Parameters
    ----------
    X_train : pd.DataFrame
        Variables predictoras de entrenamiento.
    y_train : pd.Series
        Variable objetivo de entrenamiento.
    args : argparse.Namespace
        Argumentos de configuración del modelo.

    Returns
    -------
    LogisticRegression
        Modelo entrenado.
    """
    model = LogisticRegression(
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        tol=args.tol,
        fit_intercept=args.fit_intercept,
        class_weight=args.class_weight,
        l1_ratio=args.l1_ratio,
        random_state=args.model_seed,
    )

    model.fit(X_train, y_train)

    return model


def _get_roc_auc_metrics(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
):
    """
    Calcula el ROC AUC del modelo y genera la gráfica de la curva ROC.

    Parameters
    ----------
    model : LogisticRegression
        Modelo entrenado.
    X_test : pd.DataFrame
        Variables predictoras de prueba.
    y_test : pd.Series
        Variable objetivo de prueba.
    y_prob : np.ndarray
        Probabilidades predichas para la clase positiva.

    Returns
    -------
    tuple[float, plt.Figure]
        Valor de ROC AUC y figura de la curva ROC.
    """
    auc = roc_auc_score(y_test, y_prob)

    fig_roc_curve, ax = plt.subplots(figsize=(7, 7))

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)

    return auc, fig_roc_curve


def _get_cumulative_lift_metrics(y_test: pd.Series, y_prob: np.ndarray):
    """
    Calcula métricas de ganancia acumulada (Lift) y genera su gráfica.

    Parameters
    ----------
    y_test : pd.Series
        Valores reales de la variable objetivo.
    y_prob : np.ndarray
        Probabilidades predichas para la clase positiva.

    Returns
    -------
    tuple[dict[str, float], Figure]
        Diccionario con las tasas de captura para distintos porcentajes
        de clientes contactados y figura de la curva de ganancia acumulada.
    """
    # Crear DataFrame con valores reales y probabilidades.
    lift_df = pd.DataFrame({"actual": y_test, "prob": y_prob})

    # Ordenar de mayor a menor probabilidad.
    lift_df = lift_df.sort_values("prob", ascending=False)

    # Calcular positivos acumulados.
    lift_df["cum_positive"] = lift_df["actual"].cumsum()

    # Obtener total de positivos.
    total_positive = lift_df["actual"].sum()

    # Calcular ganancia acumulada.
    lift_df["gain"] = lift_df["cum_positive"] / total_positive

    # Calcular porcentaje de clientes contactados
    lift_df["customers"] = np.arange(1, len(lift_df) + 1) / len(lift_df)

    # Crear figura
    fig_cum_lift_curve, ax = plt.subplots(figsize=(8, 6))

    # Graficar curva del modelo
    ax.plot(lift_df["customers"], lift_df["gain"], label="Modelo")

    # Graficar línea de selección aleatoria
    ax.plot([0, 1], [0, 1], "--", label="Aleatorio")

    # Etiquetas de ejes
    ax.set_xlabel("% de clientes contactados")
    ax.set_ylabel("% de suscriptores capturados")

    # Título y leyenda
    ax.set_title("Curva LIFT de Ganancias Acumuladas")
    ax.legend()

    # Tasas de captura para distintos porcentajes de clientes.
    capture_rates = {}

    # Porcentajes de clientes evaluados.
    sample_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for size in sample_sizes:
        # Ganancia acumulada al contactar el top N% de clientes.
        gain_size = lift_df.loc[lift_df["customers"] <= size, "gain"].iloc[-1]

        # Registrar la tasa de captura correspondiente.
        capture_rates[f"capture_rate_top_{int(size * 100)}pct"] = gain_size

    return capture_rates, fig_cum_lift_curve


def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evalúa el desempeño del modelo utilizando métricas ROC AUC y Lift.

    Parameters
    ----------
    model : LogisticRegression
        Modelo entrenado.
    X_test : pd.DataFrame
        Variables predictoras de prueba.
    y_test : pd.Series
        Variable objetivo de prueba.

    Returns
    -------
    tuple[float, Figure, dict[str, float], Figure]
        Valor de ROC AUC, figura de la curva ROC, tasas de captura
        (Lift) y figura de la curva de ganancia acumulada.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    auc, fig_roc_curve = _get_roc_auc_metrics(model, X_test, y_test, y_prob)

    capture_rates, fig_cum_lift_curve = _get_cumulative_lift_metrics(y_test, y_prob)

    return auc, fig_roc_curve, capture_rates, fig_cum_lift_curve


def main():
    """
    Ejecuta el flujo completo de entrenamiento, evaluación y registro
    de un modelo de Regresión Logística utilizando MLflow.

    El proceso incluye la carga y preprocesamiento de los datos,
    el entrenamiento del modelo, la evaluación mediante métricas
    ROC AUC y Lift, y el registro de parámetros, métricas,
    visualizaciones y artefactos en MLflow.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Bank long-term deposit telemarketing campaign")

    with mlflow.start_run():
        # Se obtienen los parámetros del usuario.
        args = parse_args()

        # Se cargan los datos.
        bank_data_df = load_data()

        # Se preprocesan los datos.
        X_train, X_test, y_train, y_test = preprocessing(
            bank_data_df, args.test_size, args.split_seed
        )

        # Se entrena un modelo.
        model = train_model(X_train, y_train, args)

        # Se guarda el modelo y sus hiperparametros.
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(model, name="logistic_regression")

        # Se evalúa el modelo.
        auc, fig_roc_curve, capture_rates, fig_cum_lift_curve = evaluate_model(
            model, X_test, y_test
        )

        # Se guarda la métrica de ROC-AUC y la imagen de su curva.
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_figure(fig_roc_curve, "fig_roc_curve.png")

        # Se guarda la imagen de la curva LIFT cumulativa,
        # y varias métricas de subscriptores capturados
        # dependiendo del porcentaje de clientes contactados.
        for key, value in capture_rates.items():
            mlflow.log_metric(key, value)

        mlflow.log_figure(fig_cum_lift_curve, "fig_cum_lift_curve.png")


if __name__ == "__main__":
    main()

# Ejemplo de uso de script en Power Shell.
# python train.py `
#     --split-seed 123 `
#     --test-size 0.25 `
#     --C 0.5 `
#     --solver saga `
#     --max-iter 1000 `
#     --tol 1e-5 `
#     --fit-intercept `
#     --class-weight balanced `
#     --l1-ratio 0.5 `
#     --model-seed 456

# python train.py --max-iter 1000
