"""
collaborative_filter.py
-----------------------
Modelo 2: Collaborative Filtering con ALS (Alternating Least Squares).

Responsabilidades:
    - Entrenar el modelo ALS sobre la matriz usuario-producto (feedback implícito)
    - Generar recomendaciones personalizadas por usuario
    - Generar productos similares (item-to-item)
    - Guardar y cargar el modelo entrenado

Justificación de ALS sobre SVD:
    Instacart no provee ratings explícitos. La señal de preferencia es
    la frecuencia de compra (feedback implícito). En este escenario:

    - SVD (scikit-surprise) está diseñado para ratings 1-5 explícitos.
      Forzarlo con frecuencias produce estimaciones sesgadas porque
      no modela la incertidumbre de los ceros (¿no compró porque no
      le gusta o porque no lo vio?).

    - ALS (librería `implicit`) fue diseñado específicamente para este
      problema. Modela cada interacción como (confianza, preferencia):
      * Preferencia = 1 si compró alguna vez, 0 si no
      * Confianza   = 1 + alpha × frecuencia (alpha=40 por defecto)

    Con sparsity > 99.9% y feedback puramente implícito, ALS supera
    a SVD según la literatura (Hu et al., 2008 — Collaborative Filtering
    for Implicit Feedback Datasets).

PEP 8 | Proyecto Final Henry — Cross-Selling Recommender
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "trained"


def build_implicit_matrix(
    df: pd.DataFrame,
    alpha: float = 40.0,
) -> Tuple[csr_matrix, List, List]:
    """
    Construye la matriz de confianza para ALS desde el DataFrame de compras.

    Fórmula de confianza (Hu et al., 2008):
        C_ui = 1 + alpha * frecuencia_ui

    Donde frecuencia_ui = número de veces que el usuario u compró producto i.
    alpha=40 es el valor canónico del paper original.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas user_id y product_id.
    alpha : float
        Factor de escala de confianza. Default: 40.0.

    Returns
    -------
    Tuple[csr_matrix, List, List]
        - item_user_matrix : Matriz CSR de forma (n_items, n_users)
          Nota: implicit requiere item × user (transpuesta de user × item).
        - user_ids         : Lista de user_id en el orden de las columnas.
        - product_ids      : Lista de product_id en el orden de las filas.
    """
    logger.info("Construyendo matriz de confianza para ALS (alpha=%.1f)...", alpha)

    # Calcular frecuencia de compra por par usuario-producto
    freq = (
        df.groupby(["user_id", "product_id"])
        .size()
        .reset_index(name="frequency")
    )

    # Aplicar fórmula de confianza
    freq["confidence"] = 1.0 + alpha * freq["frequency"]

    user_ids = sorted(freq["user_id"].unique().tolist())
    product_ids = sorted(freq["product_id"].unique().tolist())

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    product_to_idx = {p: i for i, p in enumerate(product_ids)}

    rows = freq["product_id"].map(product_to_idx).values
    cols = freq["user_id"].map(user_to_idx).values
    data = freq["confidence"].values.astype("float32")

    # ALS de `implicit` espera matriz item × user
    item_user_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(product_ids), len(user_ids)),
    )

    sparsity = 1 - item_user_matrix.nnz / (item_user_matrix.shape[0] * item_user_matrix.shape[1])
    logger.info(
        "Matriz ALS: %s ítems × %s usuarios | Sparsity: %.2f%%",
        f"{len(product_ids):,}",
        f"{len(user_ids):,}",
        sparsity * 100,
    )
    return item_user_matrix, user_ids, product_ids


def train_als(
    item_user_matrix: csr_matrix,
    factors: int = 50,
    iterations: int = 20,
    regularization: float = 0.01,
    random_state: int = 42,
) -> object:
    """
    Entrena el modelo ALS con feedback implícito.

    ALS alterna entre fijar los factores de usuario y resolver para los
    factores de ítem (y viceversa) en cada iteración. Converge en
    O(k²·nnz) por iteración, donde k=factors y nnz=entradas no cero.

    Justificación de hiperparámetros:
    - factors=50      : Balance entre expresividad y overfitting.
                        Más de 100 factores rara vez mejora en datasets
                        de tamaño medio. Elegido por grid search estándar.
    - iterations=20   : Suficiente para convergencia. La pérdida típica
                        estabiliza antes de la iteración 15.
    - regularization=0.01 : L2 estándar para prevenir overfitting.

    Parameters
    ----------
    item_user_matrix : csr_matrix
        Matriz de confianza (n_items × n_users) de build_implicit_matrix().
    factors : int
        Dimensiones del espacio latente. Default: 50.
    iterations : int
        Épocas de entrenamiento. Default: 20.
    regularization : float
        Parámetro de regularización L2. Default: 0.01.
    random_state : int
        Semilla para reproducibilidad. Default: 42.

    Returns
    -------
    implicit.als.AlternatingLeastSquares
        Modelo ALS entrenado.
    """
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        raise ImportError(
            "Librería 'implicit' no instalada. Ejecuta: pip install implicit"
        )

    logger.info(
        "Entrenando ALS | factors=%d | iterations=%d | reg=%.4f",
        factors, iterations, regularization,
    )

    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=random_state,
        use_gpu=False,
    )
    model.fit(item_user_matrix)

    logger.info("✅ Modelo ALS entrenado.")
    return model


def get_recommendations_for_user(
    user_id: int,
    model,
    user_ids: List,
    product_ids: List,
    item_user_matrix: csr_matrix,
    product_names: Optional[Dict[int, str]] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Genera top-K recomendaciones personalizadas para un usuario.

    Excluye automáticamente los productos que el usuario ya compró
    (cross-selling puro: solo recomienda lo que no conoce).

    Parameters
    ----------
    user_id : int
        ID del usuario.
    model : AlternatingLeastSquares
        Modelo ALS entrenado.
    user_ids : list
        Lista de user_ids en el orden de la matriz (de build_implicit_matrix).
    product_ids : list
        Lista de product_ids en el orden de la matriz.
    item_user_matrix : csr_matrix
        Matriz de confianza (n_items × n_users).
    product_names : dict, optional
        Mapeo {product_id: product_name}.
    top_k : int
        Número de recomendaciones. Default: 10.

    Returns
    -------
    pd.DataFrame
        Recomendaciones con columnas: product_id, score [, product_name].
        Vacío si el usuario no está en el modelo.
    """
    if user_id not in user_ids:
        logger.warning("user_id=%s no está en el modelo.", user_id)
        return pd.DataFrame()

    user_idx = user_ids.index(user_id)

    # La matriz para `recommend` debe ser user × item
    user_items = item_user_matrix.T.tocsr()
    item_ids_arr, scores = model.recommend(
        user_idx,
        user_items[user_idx],
        N=top_k,
        filter_already_liked_items=True,
    )

    recs = pd.DataFrame({
        "product_id": [product_ids[i] for i in item_ids_arr],
        "score": scores.round(4),
    })

    if product_names:
        recs["product_name"] = recs["product_id"].map(product_names)

    return recs


def get_similar_products(
    product_id: int,
    model,
    product_ids: List,
    product_names: Optional[Dict[int, str]] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Encuentra los productos más similares a un producto dado (item-to-item).

    Calcula similitud coseno en el espacio de factores latentes.
    Útil para recomendación en páginas de producto ("también te puede gustar").

    Parameters
    ----------
    product_id : int
        ID del producto de referencia.
    model : AlternatingLeastSquares
        Modelo ALS entrenado.
    product_ids : list
        Lista de product_ids en el orden de la matriz.
    product_names : dict, optional
        Mapeo {product_id: product_name}.
    top_k : int
        Número de productos similares. Default: 10.

    Returns
    -------
    pd.DataFrame
        Productos similares con columnas: product_id, similarity_score
        [, product_name]. Vacío si el producto no está en el modelo.
    """
    if product_id not in product_ids:
        logger.warning("product_id=%s no está en el modelo.", product_id)
        return pd.DataFrame()

    item_idx = product_ids.index(product_id)
    similar_items, scores = model.similar_items(item_idx, N=top_k + 1)

    similar = pd.DataFrame({
        "product_id": [product_ids[i] for i in similar_items],
        "similarity_score": scores.round(4),
    })

    # Excluir el propio producto
    similar = similar[similar["product_id"] != product_id].head(top_k)

    if product_names:
        similar["product_name"] = similar["product_id"].map(product_names)

    return similar


def save_als_model(
    model,
    user_ids: List,
    product_ids: List,
    filename: str = "als_model.pkl",
) -> Path:
    """
    Guarda el modelo ALS, user_ids y product_ids en disco.

    Parameters
    ----------
    model : AlternatingLeastSquares
        Modelo entrenado.
    user_ids : list
        Mapeo de índices a user_ids.
    product_ids : list
        Mapeo de índices a product_ids.
    filename : str
        Nombre del archivo. Default: 'als_model.pkl'.

    Returns
    -------
    Path
        Ruta donde se guardó el modelo.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    payload = {
        "model": model,
        "user_ids": user_ids,
        "product_ids": product_ids,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Modelo ALS guardado en: %s", path)
    return path


def load_als_model(filename: str = "als_model.pkl") -> Tuple[object, List, List]:
    """
    Carga el modelo ALS desde disco.

    Parameters
    ----------
    filename : str
        Nombre del archivo pickle. Default: 'als_model.pkl'.

    Returns
    -------
    Tuple[model, List, List]
        (modelo, user_ids, product_ids)

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {path}\n"
            "Entrena el modelo primero con train_als()."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)

    logger.info("Modelo ALS cargado desde: %s", path)
    return payload["model"], payload["user_ids"], payload["product_ids"]
