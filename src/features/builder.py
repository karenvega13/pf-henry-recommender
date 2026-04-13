"""
builder.py
----------
Ingeniería de características para el sistema de recomendación cross-selling.

Genera las estructuras de datos necesarias para cada tipo de modelo:
    - Matriz de cestas (basket matrix)   : Para Apriori / FP-Growth
    - Matriz usuario-ítem (user-item)    : Para Collaborative Filtering (ALS)
    - Matriz de co-ocurrencias           : Para EDA y baseline

Dataset: Instacart Market Basket Analysis
    Columnas clave: order_id, user_id, product_id, reordered

Justificación estadística:
    La sparsity (densidad de ceros) de la matriz usuario-ítem determina
    qué modelo de CF es más apropiado. Con 200K usuarios y 50K productos,
    la sparsity esperada es > 99.9% (cada usuario compra ~30 productos
    distintos sobre un catálogo de 50K). En ese régimen, ALS sobre
    feedback implícito supera a SVD estándar porque está diseñado
    específicamente para matrices ultra-dispersas sin ratings explícitos.
    Se calcula y reporta la sparsity real para documentar esta elección.

PEP 8 | Proyecto Final Henry — Cross-Selling Recommender
"""

import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz binaria de cestas para Apriori / FP-Growth.

    Formato: filas = order_id, columnas = product_id.
    Valor True si el producto fue comprado en esa orden, False si no.

    Esta es la entrada directa de mlxtend.frequent_patterns.fpgrowth().
    Se usa el subset muestreado (df_apriori) para mantener la RAM manejable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame muestreado con columnas order_id y product_id.
        Tipicamente el output de preprocessor.sample_orders_for_apriori().

    Returns
    -------
    pd.DataFrame
        Matriz binaria de cestas (bool dtype para eficiencia con mlxtend).

    Notes
    -----
    Se usa groupby + unstack en lugar de pivot_table para mejor rendimiento
    con DataFrames grandes. aggfunc='max' implícito vía nunique > 0.
    La conversión a bool es requerida por mlxtend.
    """
    logger.info("Construyendo basket matrix...")

    # Agregar por (order_id, product_id) y crear columna de presencia binaria
    presence = (
        df.groupby(["order_id", "product_id"])["product_id"]
        .count()
        .unstack(fill_value=0)
    )

    # Convertir a bool (requerido por mlxtend)
    basket_binary = presence.astype(bool)

    n_items = basket_binary.values.sum()
    sparsity = 1 - n_items / basket_binary.size

    logger.info(
        "Basket matrix: %s órdenes × %s productos | Sparsity: %.2f%%",
        f"{basket_binary.shape[0]:,}",
        f"{basket_binary.shape[1]:,}",
        sparsity * 100,
    )
    logger.info("  Ítems por orden (promedio): %.1f", n_items / basket_binary.shape[0])
    return basket_binary


def build_user_item_matrix(
    df: pd.DataFrame,
    value_col: str = "purchase_count",
) -> Tuple[pd.DataFrame, float]:
    """
    Construye la matriz usuario-ítem para Collaborative Filtering (ALS).

    Formato: filas = user_id, columnas = product_id.
    Valores = frecuencia de compra del usuario para ese producto.

    La frecuencia de compra es el feedback implícito que ALS interpreta
    como 'nivel de confianza' en la preferencia del usuario.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas user_id, product_id y value_col.
        Si value_col='purchase_count', usar el output de
        preprocessor.compute_purchase_frequency().
        Si se usa el df maestro directamente, value_col='reordered'
        y la función agrega automáticamente por count.
    value_col : str
        Columna a usar como valor de interacción.
        'purchase_count' (recomendado) o 'reordered'.
        Default: 'purchase_count'.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        - user_item_matrix : DataFrame usuarios × productos con frecuencias.
        - sparsity         : Porcentaje de ceros en la matriz (0-100).

    Notes
    -----
    Si el DataFrame no tiene la columna value_col (ej. viene del df maestro),
    se calcula automáticamente usando count de compras por par usuario-producto.
    """
    logger.info("Construyendo user-item matrix...")

    if value_col not in df.columns:
        logger.info(
            "Columna '%s' no encontrada. Calculando frecuencia de compra...",
            value_col,
        )
        df = (
            df.groupby(["user_id", "product_id"])
            .size()
            .reset_index(name="purchase_count")
        )
        value_col = "purchase_count"

    user_item = (
        df.groupby(["user_id", "product_id"])[value_col]
        .sum()
        .unstack(fill_value=0)
    )

    total_cells = user_item.shape[0] * user_item.shape[1]
    non_zero = (user_item > 0).values.sum()
    sparsity = (1 - non_zero / total_cells) * 100

    logger.info(
        "User-item matrix: %s usuarios × %s productos | Sparsity: %.2f%%",
        f"{user_item.shape[0]:,}",
        f"{user_item.shape[1]:,}",
        sparsity,
    )

    if sparsity > 99.5:
        logger.info(
            "ℹ️  Sparsity %.2f%% → se usará ALS (implicit) optimizado "
            "para matrices ultra-dispersas con feedback implícito.",
            sparsity,
        )

    return user_item, sparsity


def build_user_item_sparse(
    df: pd.DataFrame,
) -> Tuple[csr_matrix, list, list]:
    """
    Construye la matriz usuario-ítem en formato CSR (scipy sparse).

    Recomendado para datasets grandes donde la versión densa en pandas
    no cabe en memoria. La librería 'implicit' acepta CSR directamente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas user_id, product_id y purchase_count.
        Típicamente el output de preprocessor.compute_purchase_frequency().

    Returns
    -------
    Tuple[csr_matrix, list, list]
        - sparse_matrix : Matriz CSR de shape (n_users, n_items).
        - user_ids      : Lista de user_id en el orden de las filas.
        - product_ids   : Lista de product_id en el orden de las columnas.
    """
    logger.info("Construyendo matriz CSR sparse para ALS...")

    if "purchase_count" not in df.columns:
        df = (
            df.groupby(["user_id", "product_id"])
            .size()
            .reset_index(name="purchase_count")
        )

    # Crear índices enteros para usuarios y productos
    user_ids = sorted(df["user_id"].unique().tolist())
    product_ids = sorted(df["product_id"].unique().tolist())

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    product_to_idx = {p: i for i, p in enumerate(product_ids)}

    rows = df["user_id"].map(user_to_idx).values
    cols = df["product_id"].map(product_to_idx).values
    data = df["purchase_count"].values.astype("float32")

    sparse_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(product_ids)),
    )

    logger.info(
        "Matriz CSR: %s usuarios × %s productos | %.1f MB",
        f"{sparse_matrix.shape[0]:,}",
        f"{sparse_matrix.shape[1]:,}",
        sparse_matrix.data.nbytes / 1e6,
    )
    return sparse_matrix, user_ids, product_ids


def build_cooccurrence_matrix(
    df: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Construye la matriz de co-ocurrencias entre los top N productos.

    La co-ocurrencia cuenta cuántas veces dos productos aparecen
    en la misma orden. Es la base conceptual de las reglas de asociación
    y una herramienta clave para el EDA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas order_id y product_id.
    top_n : int
        Número de productos más frecuentes a incluir. Default: 50.

    Returns
    -------
    pd.DataFrame
        Matriz simétrica de co-ocurrencias (top_n × top_n).
        Diagonal = 0 (auto-co-ocurrencia eliminada).
    """
    logger.info("Construyendo matriz de co-ocurrencias (top %d productos)...", top_n)

    top_products = df["product_id"].value_counts().head(top_n).index.tolist()
    df_top = df[df["product_id"].isin(top_products)]

    # Pivot binario: 1 si el producto está en la orden
    basket = (
        df_top.groupby(["order_id", "product_id"])["product_id"]
        .count()
        .unstack(fill_value=0)
        .astype(bool)
        .astype(int)
    )

    # Co-ocurrencia = producto matricial B^T × B
    cooc = basket.T.dot(basket)
    np.fill_diagonal(cooc.values, 0)

    logger.info(
        "Matriz de co-ocurrencias: %d × %d | Par más frecuente: (%s, %s) = %d",
        cooc.shape[0],
        cooc.shape[1],
        str(cooc.stack().idxmax()[0]),
        str(cooc.stack().idxmax()[1]),
        cooc.values.max(),
    )
    return cooc


def build_product_features(
    df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Genera features agregadas a nivel de producto para el análisis.

    Features calculadas:
        - n_orders      : Número de órdenes distintas que contienen el producto
        - n_users       : Usuarios únicos que compraron el producto
        - reorder_rate  : Proporción de compras que son recompras
        - avg_cart_pos  : Posición promedio en el carrito (add_to_cart_order)
        - product_name  : Nombre del producto (si products_df disponible)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame maestro limpio con columnas product_id, order_id,
        user_id, reordered, add_to_cart_order.
    products_df : pd.DataFrame, optional
        Tabla products.csv con product_id y product_name.
        Si se provee, se añade el nombre al resultado.

    Returns
    -------
    pd.DataFrame
        DataFrame indexado por product_id con features de producto,
        ordenado por popularidad (n_orders descendente).
    """
    logger.info("Generando features de producto...")

    agg_dict = {
        "n_orders": ("order_id", "nunique"),
        "n_users": ("user_id", "nunique"),
        "reorder_rate": ("reordered", "mean"),
    }

    if "add_to_cart_order" in df.columns:
        agg_dict["avg_cart_position"] = ("add_to_cart_order", "mean")

    agg = df.groupby("product_id").agg(**agg_dict)

    # Enriquecer con nombres de productos si se provee la tabla
    if products_df is not None and "product_name" in products_df.columns:
        product_names = products_df.set_index("product_id")["product_name"]
        agg["product_name"] = agg.index.map(product_names)

        if "aisle" in products_df.columns:
            agg["aisle"] = agg.index.map(
                products_df.set_index("product_id").get("aisle", pd.Series())
            )
        if "department" in products_df.columns:
            agg["department"] = agg.index.map(
                products_df.set_index("product_id").get("department", pd.Series())
            )

    agg = agg.sort_values("n_orders", ascending=False)
    logger.info("Features generadas para %s productos.", f"{len(agg):,}")
    return agg
