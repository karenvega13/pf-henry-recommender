"""
metrics.py
----------
Métricas de evaluación para el sistema de recomendación cross-selling.

Métricas implementadas:
    - Precision@K : Proporción de recomendaciones relevantes en top-K
    - Recall@K    : Proporción de ítems relevantes recuperados en top-K
    - NDCG@K      : Normalized Discounted Cumulative Gain (penaliza por posición)
    - Hit Rate    : % de usuarios con al menos 1 recomendación relevante
    - Coverage    : % del catálogo que el sistema recomienda
    - AOV Lift    : Incremento simulado en Average Order Value

Justificación de métricas:
    Precision@K y Recall@K son las métricas estándar en sistemas de
    recomendación top-K. NDCG@K agrega sensibilidad al ranking —
    importante porque el usuario ve primero el ítem en posición 1.
    El AOV Lift conecta las métricas técnicas con el KPI de negocio.

PEP 8 | Proyecto Final Henry — Cross-Selling Recommender
"""

import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def precision_at_k(
    recommended: List,
    relevant: List,
    k: int,
) -> float:
    """
    Calcula Precision@K para una sola consulta.

    Precision@K = |recomendados ∩ relevantes| / K

    Parameters
    ----------
    recommended : list
        Lista ordenada de ítems recomendados.
    relevant : list
        Lista de ítems realmente relevantes (ground truth).
    k : int
        Número de recomendaciones a considerar.

    Returns
    -------
    float
        Precision@K en rango [0, 1].
    """
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(
    recommended: List,
    relevant: List,
    k: int,
) -> float:
    """
    Calcula Recall@K para una sola consulta.

    Recall@K = |recomendados ∩ relevantes| / |relevantes|

    Parameters
    ----------
    recommended : list
        Lista ordenada de ítems recomendados.
    relevant : list
        Lista de ítems realmente relevantes (ground truth).
    k : int
        Número de recomendaciones a considerar.

    Returns
    -------
    float
        Recall@K en rango [0, 1]. Retorna 0.0 si no hay ítems relevantes.
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(
    recommended: List,
    relevant: List,
    k: int,
) -> float:
    """
    Calcula NDCG@K (Normalized Discounted Cumulative Gain).

    Penaliza recomendaciones relevantes que aparecen en posiciones bajas.
    Un ítem relevante en posición 1 vale más que en posición 5.

    NDCG@K = DCG@K / IDCG@K
    donde DCG@K = Σ rel_i / log2(i+1) para i=1..K

    Parameters
    ----------
    recommended : list
        Lista ordenada de ítems recomendados (posición 0 = más relevante).
    relevant : list
        Lista de ítems realmente relevantes (ground truth).
    k : int
        Número de posiciones a considerar.

    Returns
    -------
    float
        NDCG@K en rango [0, 1].
    """
    relevant_set = set(relevant)
    top_k = recommended[:k]

    # DCG real
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant_set
    )

    # IDCG (DCG ideal: todos los relevantes en las primeras posiciones)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(
    recommendations: Dict[str, List],
    ground_truth: Dict[str, List],
    k: int,
) -> float:
    """
    Calcula el Hit Rate global del sistema.

    Hit Rate = Nº usuarios con al menos 1 hit en top-K / Nº usuarios total

    Parameters
    ----------
    recommendations : dict
        {user_id: [lista de ítems recomendados]}.
    ground_truth : dict
        {user_id: [lista de ítems relevantes en test]}.
    k : int
        Número de recomendaciones a considerar.

    Returns
    -------
    float
        Hit Rate en rango [0, 1].
    """
    users = [u for u in recommendations if u in ground_truth]
    if not users:
        return 0.0

    hits = sum(
        1
        for u in users
        if any(
            item in set(ground_truth[u])
            for item in recommendations[u][:k]
        )
    )
    return hits / len(users)


def coverage(
    recommendations: Dict[str, List],
    catalog: List,
) -> float:
    """
    Calcula la cobertura del catálogo.

    Coverage = Nº ítems únicos recomendados / Nº ítems totales en catálogo

    Una cobertura baja indica que el sistema siempre recomienda los
    mismos productos populares (problema de burbuja de popularidad).

    Parameters
    ----------
    recommendations : dict
        {user_id o product_id: [lista de ítems recomendados]}.
    catalog : list
        Lista completa de ítems del catálogo.

    Returns
    -------
    float
        Coverage en rango [0, 1].
    """
    all_recommended = set(
        item
        for recs in recommendations.values()
        for item in recs
    )
    return len(all_recommended) / len(catalog) if catalog else 0.0


def evaluate_model(
    recommendations: Dict[str, List],
    ground_truth: Dict[str, List],
    catalog: List,
    k_values: List[int] = [5, 10],
) -> pd.DataFrame:
    """
    Evalúa un modelo de recomendación con todas las métricas estándar.

    Parameters
    ----------
    recommendations : dict
        {user_id: [lista ordenada de ítems recomendados]}.
    ground_truth : dict
        {user_id: [lista de ítems relevantes en test]}.
    catalog : list
        Lista completa de ítems del catálogo (para coverage).
    k_values : list of int
        Valores de K a evaluar. Default: [5, 10].

    Returns
    -------
    pd.DataFrame
        Tabla de métricas con columnas [K, Precision, Recall, NDCG,
        HitRate, Coverage].
    """
    results = []
    users = [u for u in recommendations if u in ground_truth]

    if not users:
        logger.warning("No hay usuarios en común entre recomendaciones y ground truth.")
        return pd.DataFrame()

    for k in k_values:
        prec_list, rec_list, ndcg_list = [], [], []

        for user in users:
            recs = recommendations[user]
            relevant = ground_truth[user]
            prec_list.append(precision_at_k(recs, relevant, k))
            rec_list.append(recall_at_k(recs, relevant, k))
            ndcg_list.append(ndcg_at_k(recs, relevant, k))

        hr = hit_rate(recommendations, ground_truth, k)
        cov = coverage(recommendations, catalog)

        results.append({
            "K": k,
            "Precision@K": np.mean(prec_list),
            "Recall@K": np.mean(rec_list),
            "NDCG@K": np.mean(ndcg_list),
            "HitRate@K": hr,
            "Coverage": cov,
        })

        logger.info(
            "K=%d | Prec=%.4f | Rec=%.4f | NDCG=%.4f | HR=%.4f | Cov=%.4f",
            k,
            np.mean(prec_list),
            np.mean(rec_list),
            np.mean(ndcg_list),
            hr,
            cov,
        )

    return pd.DataFrame(results).set_index("K")


def compute_aov_lift(
    df_orders: pd.DataFrame,
    recommended_products: Dict[str, List],
    avg_product_price: float,
) -> Dict[str, float]:
    """
    Simula el incremento en AOV si el usuario acepta 1 recomendación.

    Esta función conecta las métricas técnicas con el KPI de negocio:
    AOV Lift = (AOV baseline + precio promedio del producto recomendado)
               / AOV baseline - 1

    Asunción conservadora: se asume una tasa de aceptación del 10%,
    que es un benchmark estándar en la industria de e-commerce.

    Parameters
    ----------
    df_orders : pd.DataFrame
        DataFrame con columnas invoice_id y revenue (para calcular AOV base).
    recommended_products : dict
        {invoice_id: [lista de productos recomendados]}.
    avg_product_price : float
        Precio promedio de los productos recomendados.

    Returns
    -------
    dict
        Diccionario con métricas de AOV: baseline, projected, lift_pct.
    """
    aov_baseline = df_orders.groupby("invoice_id")["revenue"].sum().mean()
    acceptance_rate = 0.10  # 10% tasa de aceptación conservadora

    aov_projected = aov_baseline + (acceptance_rate * avg_product_price)
    lift_pct = (aov_projected / aov_baseline - 1) * 100

    result = {
        "aov_baseline": round(aov_baseline, 2),
        "aov_projected": round(aov_projected, 2),
        "lift_pct": round(lift_pct, 2),
        "acceptance_rate": acceptance_rate,
    }

    logger.info("📈 AOV Baseline  : £%.2f", aov_baseline)
    logger.info("📈 AOV Proyectado: £%.2f", aov_projected)
    logger.info("📈 AOV Lift      : %.2f%%", lift_pct)

    return result
