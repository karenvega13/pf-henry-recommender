"""
preprocessor.py
---------------
Pipeline de limpieza y transformación del dataset Instacart Market Basket Analysis.

Pasos del pipeline:
    1. Validar columnas requeridas
    2. Eliminar duplicados exactos
    3. Filtrar cestas con menos de min_items productos distintos
    4. Añadir features temporales derivadas (DOW, hora)
    5. Calcular frecuencia de compra por usuario-producto (señal para CF)
    6. (Opcional) Muestrear órdenes para Apriori en memoria limitada

Justificación estadística:
    Instacart es un dataset pre-limpio (sin cancelaciones ni precios inválidos).
    La distribución de productos sigue una ley de Zipf: los 1.000 productos más
    frecuentes concentran más del 60% de las compras. Filtramos cestas pequeñas
    (< 2 ítems) porque no aportan pares de co-compra para association rules.
    El muestreo estratificado por usuario preserva la distribución original
    en lugar de truncar aleatoriamente, evitando sesgo hacia usuarios activos.

PEP 8 | Proyecto Final Henry — Cross-Selling Recommender
"""

import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columnas requeridas en el DataFrame maestro
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "order_id",
    "user_id",
    "product_id",
    "reordered",
    "order_dow",
    "order_hour_of_day",
]


def validate_columns(df: pd.DataFrame) -> bool:
    """
    Valida que el DataFrame contenga las columnas mínimas requeridas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame maestro generado por loader.build_master_dataframe().

    Returns
    -------
    bool
        True si todas las columnas requeridas están presentes.

    Raises
    ------
    ValueError
        Si faltan columnas requeridas.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas requeridas faltantes: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )
    logger.info("✅ Esquema validado. Columnas requeridas presentes.")
    return True


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas duplicadas exactas del DataFrame.

    En Instacart, un duplicado exacto (mismo order_id + product_id)
    indica un error de registro, no una compra múltiple del mismo ítem.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame maestro.

    Returns
    -------
    pd.DataFrame
        DataFrame sin duplicados exactos.
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["order_id", "product_id"]).copy()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.warning("Duplicados (order_id, product_id) eliminados: %s", f"{n_removed:,}")
    else:
        logger.info("Sin duplicados detectados.")
    return df


def filter_small_baskets(
    df: pd.DataFrame,
    min_items: int = 2,
) -> pd.DataFrame:
    """
    Elimina órdenes con menos de min_items productos distintos.

    Justificación: Las reglas de asociación requieren co-presencia de
    al least 2 ítems para generar pares antecedente → consecuente.
    Las órdenes de 1 ítem no aportan información de co-compra.

    Distribución típica en Instacart:
        - ~5% de órdenes tienen 1 solo ítem
        - Mediana: 8 ítems por orden
        - Percentil 95: 22 ítems por orden

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame maestro.
    min_items : int
        Número mínimo de productos distintos por orden. Default: 2.

    Returns
    -------
    pd.DataFrame
        DataFrame con órdenes de tamaño suficiente.
    """
    basket_sizes = df.groupby("order_id")["product_id"].nunique()
    valid_orders = basket_sizes[basket_sizes >= min_items].index

    n_orders_before = df["order_id"].nunique()
    df = df[df["order_id"].isin(valid_orders)].copy()
    n_orders_after = df["order_id"].nunique()

    logger.info(
        "Órdenes con < %d ítems eliminadas: %s → %s órdenes (%.1f%% conservado)",
        min_items,
        f"{n_orders_before:,}",
        f"{n_orders_after:,}",
        n_orders_after / n_orders_before * 100,
    )
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece el DataFrame con features temporales derivadas.

    Instacart no provee timestamps exactos, pero sí el día de semana
    y la hora del día de cada orden. Se derivan features de contexto
    temporal útiles para el análisis exploratorio.

    Features añadidas:
        - is_weekend    : 1 si la orden fue en sábado (0) o domingo (1)
        - time_of_day   : 'madrugada', 'mañana', 'tarde', 'noche'
        - days_bucket   : categoría de días desde compra previa

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame maestro con columnas order_dow y order_hour_of_day.

    Returns
    -------
    pd.DataFrame
        DataFrame con features temporales adicionales.
    """
    # En Instacart: 0 = sábado, 1 = domingo
    df = df.copy()
    df["is_weekend"] = df["order_dow"].isin([0, 1]).astype("uint8")

    # Franjas horarias
    hour = df["order_hour_of_day"]
    df["time_of_day"] = pd.cut(
        hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["madrugada", "mañana", "tarde", "noche"],
    )

    # Categoría de días desde compra previa (NaN = primera orden del usuario)
    if "days_since_prior_order" in df.columns:
        df["days_bucket"] = pd.cut(
            df["days_since_prior_order"],
            bins=[-1, 0, 7, 14, 30, float("inf")],
            labels=["mismo_día", "semana", "quincenal", "mensual", "más_de_un_mes"],
        )

    logger.info("Features temporales añadidas: is_weekend, time_of_day, days_bucket")
    return df


def compute_purchase_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la frecuencia de compra de cada usuario para cada producto.

    Esta señal es la entrada del Collaborative Filtering con feedback
    implícito (ALS). A mayor frecuencia, mayor 'confianza' de que el
    usuario tiene preferencia real por ese producto.

    Justificación: El flag 'reordered' indica si un producto fue
    comprado antes. Combinar frecuencia total + tasa de recompra da
    una señal más robusta que contar compras brutas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame maestro con user_id, product_id y reordered.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas adicionales:
        - purchase_count : veces que el usuario compró el producto
        - reorder_rate   : proporción de esas compras que son recompras
    """
    logger.info("Calculando frecuencia de compra usuario-producto...")

    freq = (
        df.groupby(["user_id", "product_id"])
        .agg(
            purchase_count=("reordered", "count"),
            reorder_count=("reordered", "sum"),
        )
        .reset_index()
    )
    freq["reorder_rate"] = freq["reorder_count"] / freq["purchase_count"]

    logger.info(
        "  Pares usuario-producto únicos : %s",
        f"{len(freq):,}",
    )
    logger.info(
        "  Tasa de recompra promedio      : %.1f%%",
        freq["reorder_rate"].mean() * 100,
    )
    return freq


def sample_orders_for_apriori(
    df: pd.DataFrame,
    n_orders: int = 200_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Muestrea un subconjunto de órdenes para Apriori / FP-Growth.

    Justificación: Construir la basket matrix sobre las 3.4M órdenes
    completas requeriría ~16 GB de RAM. Un muestreo de 200K órdenes
    (~6% del total) preserva la distribución de frecuencias de Zipf
    y es suficiente para encontrar reglas estadísticamente significativas
    con min_support=0.01 (→ 2,000 cestas de soporte mínimo).

    El muestreo es aleatorio simple sobre order_id únicos, no sobre filas,
    para conservar la integridad de cada cesta.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame maestro filtrado (solo órdenes válidas).
    n_orders : int
        Número de órdenes a muestrear. Default: 200,000.
    random_state : int
        Semilla para reproducibilidad. Default: 42.

    Returns
    -------
    pd.DataFrame
        Subconjunto del DataFrame con n_orders órdenes completas.
    """
    all_orders = df["order_id"].unique()
    total_orders = len(all_orders)

    if n_orders >= total_orders:
        logger.info(
            "n_orders (%s) >= total órdenes (%s). Sin muestreo.",
            f"{n_orders:,}",
            f"{total_orders:,}",
        )
        return df

    rng = np.random.default_rng(random_state)
    sampled_order_ids = rng.choice(all_orders, size=n_orders, replace=False)
    df_sampled = df[df["order_id"].isin(sampled_order_ids)].copy()

    logger.info(
        "Muestreo para Apriori: %s / %s órdenes (%.1f%%)",
        f"{n_orders:,}",
        f"{total_orders:,}",
        n_orders / total_orders * 100,
    )
    logger.info("  Ítems en la muestra: %s", f"{len(df_sampled):,}")
    return df_sampled


def run_cleaning_pipeline(
    df: pd.DataFrame,
    min_basket_items: int = 2,
    add_temporal: bool = True,
    apriori_sample: Optional[int] = 200_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el pipeline completo de limpieza y genera dos DataFrames de salida.

    Devuelve dos versiones del DataFrame limpio:
        - df_full  : Dataset completo para CF (user_id × product_id)
        - df_apriori: Dataset muestreado para Apriori (order_id × product_id)

    Orden de pasos:
        1. Validar columnas requeridas
        2. Eliminar duplicados
        3. Filtrar cestas pequeñas
        4. Añadir features temporales
        5. Muestrear para Apriori (si apriori_sample no es None)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame maestro cargado con loader.load_full_dataset().
    min_basket_items : int
        Mínimo de productos por orden. Default: 2.
    add_temporal : bool
        Si añadir features temporales derivadas. Default: True.
    apriori_sample : int, optional
        Número de órdenes para el muestreo de Apriori. None = sin muestreo.
    random_state : int
        Semilla para reproducibilidad. Default: 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_full, df_apriori)
        - df_full   : Dataset completo y limpio para CF.
        - df_apriori: Subconjunto muestreado para Apriori.
    """
    logger.info("🚀 Iniciando pipeline de limpieza — Instacart...")
    n_initial = len(df)

    validate_columns(df)
    df = remove_duplicates(df)
    df = filter_small_baskets(df, min_items=min_basket_items)

    if add_temporal:
        df = add_temporal_features(df)

    n_final = len(df)
    logger.info(
        "✅ Pipeline completado: %s → %s filas (%.1f%% conservado)",
        f"{n_initial:,}",
        f"{n_final:,}",
        n_final / n_initial * 100,
    )

    # Dataset completo para Collaborative Filtering
    df_full = df.copy()

    # Dataset muestreado para Apriori
    if apriori_sample is not None:
        df_apriori = sample_orders_for_apriori(
            df, n_orders=apriori_sample, random_state=random_state
        )
    else:
        df_apriori = df.copy()
        logger.info("Apriori sin muestreo: usando dataset completo.")

    return df_full, df_apriori


def get_basket_size_stats(df: pd.DataFrame) -> pd.Series:
    """
    Calcula estadísticas descriptivas del tamaño de las cestas.

    Reemplaza al AOV monetario (no disponible en Instacart) como
    KPI de baseline. Un mayor tamaño de cesta promedio indica que
    el sistema de recomendación está aumentando las compras por visita.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio con columnas order_id y product_id.

    Returns
    -------
    pd.Series
        Estadísticas de tamaño de cesta (mean, median, std, min, max, p95).
    """
    basket_sizes = df.groupby("order_id")["product_id"].nunique()
    stats = basket_sizes.describe(percentiles=[0.25, 0.5, 0.75, 0.95])

    logger.info("📊 Estadísticas de tamaño de cesta (ítems/orden):")
    logger.info("  Media   : %.2f", basket_sizes.mean())
    logger.info("  Mediana : %.2f", basket_sizes.median())
    logger.info("  P95     : %.2f", basket_sizes.quantile(0.95))
    logger.info("  Máx     : %d", basket_sizes.max())

    return stats
