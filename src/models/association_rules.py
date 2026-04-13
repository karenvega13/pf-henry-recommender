"""
association_rules.py
--------------------
Modelo 1: Reglas de Asociación con FP-Growth para cross-selling.

Responsabilidades:
    - Entrenar el modelo FP-Growth sobre la basket matrix
    - Generar reglas de asociación filtradas por soporte, confianza y lift
    - Proveer recomendaciones de cross-selling para un producto dado
    - Guardar y cargar el modelo entrenado

Justificación de FP-Growth sobre Apriori:
    FP-Growth construye un árbol comprimido (FP-tree) y extrae
    frecuentes sin generar candidatos explícitamente. Su complejidad
    es O(n) en número de transacciones vs O(2^k) de Apriori. Con
    200K órdenes de Instacart y ~15K productos únicos en la muestra,
    FP-Growth es entre 10x y 50x más rápido que Apriori clásico.

    La elección de umbrales:
    - min_support=0.01: un par aparece en ≥2000 cestas → estadístico
    - min_confidence=0.30: 30% de veces que aparece A, aparece B → útil
    - min_lift=1.2: la asociación es 20% más frecuente que azar

PEP 8 | Proyecto Final Henry — Cross-Selling Recommender
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "trained"


def build_transactions_list(df: pd.DataFrame) -> List[List[str]]:
    """
    Convierte el DataFrame de órdenes a lista de transacciones.

    Formato requerido por TransactionEncoder de mlxtend:
    cada elemento es una lista de product_ids en una orden.

    Se convierte product_id a string para evitar problemas con
    el TransactionEncoder al detectar tipos numéricos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas order_id y product_id.

    Returns
    -------
    List[List[str]]
        Lista de transacciones, cada una es lista de product_ids como str.
    """
    transactions = (
        df.groupby("order_id")["product_id"]
        .apply(lambda x: [str(p) for p in x.tolist()])
        .tolist()
    )
    logger.info(
        "Transacciones preparadas: %s órdenes | %s ítems totales",
        f"{len(transactions):,}",
        f"{sum(len(t) for t in transactions):,}",
    )
    return transactions


def encode_transactions(
    transactions: List[List[str]],
) -> pd.DataFrame:
    """
    Codifica la lista de transacciones en una matriz binaria (one-hot).

    Usa TransactionEncoder de mlxtend que es más eficiente en memoria
    que un pivot_table directo para listas de transacciones.

    Parameters
    ----------
    transactions : List[List[str]]
        Lista de transacciones como product_ids strings.

    Returns
    -------
    pd.DataFrame
        Matriz booleana de forma (n_transacciones × n_productos).
    """
    logger.info("Codificando transacciones con TransactionEncoder...")
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    logger.info(
        "Matriz encoded: %s × %s | %.1f MB",
        f"{df_encoded.shape[0]:,}",
        f"{df_encoded.shape[1]:,}",
        df_encoded.memory_usage(deep=True).sum() / 1e6,
    )
    return df_encoded


def train_fpgrowth(
    basket_matrix: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.30,
    min_lift: float = 1.2,
    max_len: Optional[int] = 3,
) -> pd.DataFrame:
    """
    Entrena el modelo FP-Growth y genera las reglas de asociación.

    Proceso:
        1. Extraer itemsets frecuentes con FP-Growth
        2. Generar reglas de asociación desde los itemsets
        3. Filtrar por confidence y lift mínimos

    Justificación de max_len=3:
        Las reglas de longitud > 3 tienen soporte muy bajo en la práctica
        y son difíciles de interpretar. Limitamos a pares y tripletes para
        asegurar reglas estadísticamente robustas y accionables.

    Parameters
    ----------
    basket_matrix : pd.DataFrame
        Matriz booleana de cestas (order_id × product_id).
        Output de build_basket_matrix() o encode_transactions().
    min_support : float
        Soporte mínimo. Default: 0.01 (1% de cestas).
    min_confidence : float
        Confianza mínima. Default: 0.30.
    min_lift : float
        Lift mínimo. Default: 1.2.
    max_len : int, optional
        Longitud máxima del itemset. Default: 3. None = sin límite.

    Returns
    -------
    pd.DataFrame
        DataFrame de reglas con columnas: antecedents, consequents,
        support, confidence, lift, conviction, leverage.

    Raises
    ------
    ValueError
        Si no se encuentran itemsets frecuentes con los umbrales dados.
    """
    logger.info(
        "Entrenando FP-Growth | support=%.3f | confidence=%.2f | lift=%.2f",
        min_support, min_confidence, min_lift,
    )

    # Paso 1: Itemsets frecuentes
    frequent_itemsets = fpgrowth(
        basket_matrix,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if frequent_itemsets.empty:
        raise ValueError(
            f"No se encontraron itemsets frecuentes con min_support={min_support}. "
            "Considera reducir el umbral o aumentar el tamaño de la muestra."
        )

    logger.info(
        "Itemsets frecuentes encontrados: %s", f"{len(frequent_itemsets):,}"
    )

    # Paso 2: Generar reglas
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    # Paso 3: Filtrar por lift
    rules = rules[rules["lift"] >= min_lift].copy()
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    logger.info(
        "Reglas generadas: %s (tras filtro lift >= %.2f)",
        f"{len(rules):,}", min_lift,
    )

    # Añadir columna de conteo de antecedentes/consecuentes
    rules["n_antecedents"] = rules["antecedents"].apply(len)
    rules["n_consequents"] = rules["consequents"].apply(len)

    return rules


def get_recommendations_for_product(
    product_id: int,
    rules: pd.DataFrame,
    product_names: Optional[Dict[int, str]] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Retorna los top-K productos recomendados para un producto dado.

    Busca las reglas donde el producto es parte del antecedente y
    ordena los consecuentes por lift (mejor señal de asociación real).

    Parameters
    ----------
    product_id : int
        ID del producto para el que se generan recomendaciones.
    rules : pd.DataFrame
        DataFrame de reglas de asociación generado por train_fpgrowth().
    product_names : dict, optional
        Mapeo {product_id: product_name} para enriquecer el resultado.
    top_k : int
        Número máximo de recomendaciones. Default: 10.

    Returns
    -------
    pd.DataFrame
        Tabla de recomendaciones con columnas: product_id, product_name
        (si disponible), confidence, lift, support.
        Vacío si no hay reglas para el producto.
    """
    pid_str = str(product_id)

    # Filtrar reglas donde el producto aparece en el antecedente
    mask = rules["antecedents"].apply(lambda x: pid_str in {str(i) for i in x})
    relevant_rules = rules[mask].copy()

    if relevant_rules.empty:
        logger.warning(
            "Sin reglas para product_id=%s. Verifica que esté en la muestra.",
            product_id,
        )
        return pd.DataFrame()

    # Explotar consecuentes (puede haber múltiples por regla)
    recs = []
    for _, row in relevant_rules.iterrows():
        for consequent in row["consequents"]:
            c_id = int(consequent)
            if c_id != product_id:
                recs.append({
                    "product_id": c_id,
                    "confidence": round(row["confidence"], 4),
                    "lift": round(row["lift"], 4),
                    "support": round(row["support"], 4),
                })

    if not recs:
        return pd.DataFrame()

    df_recs = (
        pd.DataFrame(recs)
        .sort_values("lift", ascending=False)
        .drop_duplicates(subset="product_id")
        .head(top_k)
        .reset_index(drop=True)
    )

    if product_names:
        df_recs["product_name"] = df_recs["product_id"].map(product_names)

    return df_recs


def save_rules(rules: pd.DataFrame, filename: str = "fpgrowth_rules.pkl") -> Path:
    """
    Guarda las reglas de asociación en disco como pickle.

    Parameters
    ----------
    rules : pd.DataFrame
        DataFrame de reglas generado por train_fpgrowth().
    filename : str
        Nombre del archivo. Default: 'fpgrowth_rules.pkl'.

    Returns
    -------
    Path
        Ruta donde se guardó el archivo.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    rules.to_pickle(path)
    logger.info("Reglas guardadas en: %s", path)
    return path


def load_rules(filename: str = "fpgrowth_rules.pkl") -> pd.DataFrame:
    """
    Carga las reglas de asociación desde disco.

    Parameters
    ----------
    filename : str
        Nombre del archivo pickle. Default: 'fpgrowth_rules.pkl'.

    Returns
    -------
    pd.DataFrame
        DataFrame de reglas de asociación.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en models/trained/.
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {path}\n"
            "Entrena el modelo primero con train_fpgrowth()."
        )
    rules = pd.read_pickle(path)
    logger.info("Reglas cargadas desde: %s (%s reglas)", path, f"{len(rules):,}")
    return rules


def summarize_rules(rules: pd.DataFrame) -> None:
    """
    Imprime un resumen estadístico de las reglas generadas.

    Parameters
    ----------
    rules : pd.DataFrame
        DataFrame de reglas de asociación.
    """
    logger.info("=" * 55)
    logger.info("RESUMEN — REGLAS DE ASOCIACIÓN (FP-Growth)")
    logger.info("  Total reglas          : %s", f"{len(rules):,}")
    logger.info("  Lift promedio         : %.4f", rules["lift"].mean())
    logger.info("  Lift máximo           : %.4f", rules["lift"].max())
    logger.info("  Confianza promedio    : %.4f", rules["confidence"].mean())
    logger.info("  Soporte promedio      : %.4f", rules["support"].mean())
    logger.info(
        "  Reglas lift > 2.0     : %s",
        f"{(rules['lift'] > 2.0).sum():,}",
    )
    logger.info(
        "  Reglas lift > 5.0     : %s",
        f"{(rules['lift'] > 5.0).sum():,}",
    )
    logger.info("=" * 55)
