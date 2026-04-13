"""
loader.py
---------
Módulo de carga y validación del dataset Instacart Market Basket Analysis.

Responsabilidades:
    - Cargar los 6 CSV desde data/raw/ con dtypes optimizados (RAM)
    - Unificar en un DataFrame de trabajo maestro con JOIN progresivo
    - Separar split prior / train según eval_set
    - Reportar estadísticas básicas de carga

Dataset: Instacart Market Basket Analysis
    URL: https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis

Archivos:
    orders.csv               3 421 083 órdenes  |  7 columnas
    order_products__prior.csv  32 434 489 ítems  |  4 columnas  (histórico)
    order_products__train.csv   1 384 617 ítems  |  4 columnas  (validación)
    products.csv               49 688 productos  |  4 columnas
    aisles.csv                    134 pasillos   |  2 columnas
    departments.csv                21 deptos     |  2 columnas

Justificación estadística:
    Los dtypes se eligen para minimizar uso de RAM sin pérdida de información.
    int32 es suficiente para IDs (máx ~50K productos, ~200K usuarios).
    uint8 es suficiente para reordered (0 o 1) y order_dow (0-6).
    Esto reduce el DataFrame de ~4 GB a ~1.8 GB en memoria.

PEP 8 | Proyecto Final Henry — Cross-Selling Recommender
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Dtypes optimizados para minimizar uso de RAM
# Justificación: int32 reduce a la mitad el espacio vs int64 por defecto
DTYPES_ORDERS = {
    "order_id": "int32",
    "user_id": "int32",
    "eval_set": "category",
    "order_number": "int8",
    "order_dow": "uint8",
    "order_hour_of_day": "uint8",
    "days_since_prior_order": "float32",
}

DTYPES_ORDER_PRODUCTS = {
    "order_id": "int32",
    "product_id": "int32",
    "add_to_cart_order": "int8",
    "reordered": "uint8",
}

DTYPES_PRODUCTS = {
    "product_id": "int32",
    "product_name": "str",
    "aisle_id": "int16",
    "department_id": "int8",
}

DTYPES_AISLES = {
    "aisle_id": "int16",
    "aisle": "str",
}

DTYPES_DEPARTMENTS = {
    "department_id": "int8",
    "department": "str",
}


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Carga el archivo de configuración YAML del proyecto.

    Parameters
    ----------
    config_path : Path, optional
        Ruta al archivo config.yaml. Si es None, usa la ruta por defecto.

    Returns
    -------
    dict
        Diccionario con todos los parámetros del proyecto.

    Raises
    ------
    FileNotFoundError
        Si el archivo de configuración no existe.
    """
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo de configuración no encontrado: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Configuración cargada desde: %s", path)
    return config


def load_instacart_tables(
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga las 5 tablas principales del dataset Instacart con dtypes optimizados.

    No carga order_products__train aquí porque es el split de evaluación
    y se carga por separado en load_train_split().

    Parameters
    ----------
    config : dict, optional
        Configuración del proyecto. Si es None, se carga automáticamente.

    Returns
    -------
    Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]
        (orders, order_products_prior, products, aisles, departments)

    Raises
    ------
    FileNotFoundError
        Si algún archivo CSV no existe en data/raw/.
    """
    cfg = config or load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data"]
    files = cfg["data"]["files"]

    tables = {}
    specs = [
        ("orders", files["orders"], DTYPES_ORDERS),
        ("order_products_prior", files["order_products_prior"], DTYPES_ORDER_PRODUCTS),
        ("products", files["products"], DTYPES_PRODUCTS),
        ("aisles", files["aisles"], DTYPES_AISLES),
        ("departments", files["departments"], DTYPES_DEPARTMENTS),
    ]

    for name, filename, dtypes in specs:
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {path}\n"
                f"Asegúrate de colocar los CSV de Instacart en data/raw/"
            )
        logger.info("Cargando %s...", filename)
        tables[name] = pd.read_csv(path, dtype=dtypes)
        logger.info(
            "  %-35s %s filas | %.1f MB",
            filename,
            f"{len(tables[name]):,}",
            tables[name].memory_usage(deep=True).sum() / 1e6,
        )

    return (
        tables["orders"],
        tables["order_products_prior"],
        tables["products"],
        tables["aisles"],
        tables["departments"],
    )


def load_train_split(config: Optional[dict] = None) -> pd.DataFrame:
    """
    Carga el split de evaluación (order_products__train.csv).

    Este split contiene la última orden de cada usuario y se usa
    para evaluar las recomendaciones generadas con el histórico (prior).

    Parameters
    ----------
    config : dict, optional
        Configuración del proyecto.

    Returns
    -------
    pd.DataFrame
        DataFrame con las órdenes de evaluación.
    """
    cfg = config or load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data"]
    path = raw_dir / cfg["data"]["files"]["order_products_train"]

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info("Cargando split de evaluación: %s", path.name)
    df_train = pd.read_csv(path, dtype=DTYPES_ORDER_PRODUCTS)
    logger.info(
        "  Split train: %s ítems | %.1f MB",
        f"{len(df_train):,}",
        df_train.memory_usage(deep=True).sum() / 1e6,
    )
    return df_train


def build_master_dataframe(
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
    products: pd.DataFrame,
    aisles: pd.DataFrame,
    departments: pd.DataFrame,
    eval_set: str = "prior",
) -> pd.DataFrame:
    """
    Construye el DataFrame maestro unificando las 5 tablas mediante JOINs.

    Proceso:
        1. Filtrar orders por eval_set ('prior' para entrenamiento)
        2. JOIN order_products + orders (por order_id)
        3. JOIN + products (por product_id)
        4. JOIN + aisles (por aisle_id)
        5. JOIN + departments (por department_id)

    El resultado contiene toda la información necesaria para los modelos
    en un único DataFrame plano.

    Parameters
    ----------
    orders : pd.DataFrame
        Tabla de órdenes (orders.csv).
    order_products : pd.DataFrame
        Tabla de ítems por orden (order_products__prior.csv).
    products : pd.DataFrame
        Tabla de productos (products.csv).
    aisles : pd.DataFrame
        Tabla de pasillos (aisles.csv).
    departments : pd.DataFrame
        Tabla de departamentos (departments.csv).
    eval_set : str
        Split a usar: 'prior' (entrenamiento) o 'train' (evaluación).
        Default: 'prior'.

    Returns
    -------
    pd.DataFrame
        DataFrame maestro con columnas:
        order_id, user_id, eval_set, order_number, order_dow,
        order_hour_of_day, days_since_prior_order, product_id,
        add_to_cart_order, reordered, product_name, aisle_id,
        aisle, department_id, department
    """
    logger.info("Construyendo DataFrame maestro (eval_set='%s')...", eval_set)

    # Filtrar órdenes por el split deseado
    orders_filtered = orders[orders["eval_set"] == eval_set].copy()
    logger.info(
        "  Órdenes en split '%s': %s", eval_set, f"{len(orders_filtered):,}"
    )

    # JOIN progresivo
    df = order_products.merge(orders_filtered, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(aisles, on="aisle_id", how="left")
    df = df.merge(departments, on="department_id", how="left")

    _log_master_summary(df)
    return df


def _log_master_summary(df: pd.DataFrame) -> None:
    """
    Imprime un resumen del DataFrame maestro construido.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame maestro unificado.
    """
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info("=" * 55)
    logger.info("RESUMEN DEL DATAFRAME MAESTRO")
    logger.info("  Filas            : %s", f"{len(df):,}")
    logger.info("  Columnas         : %s", len(df.columns))
    logger.info("  Órdenes únicas   : %s", f"{df['order_id'].nunique():,}")
    logger.info("  Usuarios únicos  : %s", f"{df['user_id'].nunique():,}")
    logger.info("  Productos únicos : %s", f"{df['product_id'].nunique():,}")
    logger.info("  Uso de memoria   : %.1f MB", mem_mb)
    logger.info("  Tasa de recompra : %.1f%%", df["reordered"].mean() * 100)
    logger.info("=" * 55)


def load_full_dataset(
    config: Optional[dict] = None,
    eval_set: str = "prior",
) -> pd.DataFrame:
    """
    Función de conveniencia: carga y une todo el dataset en un solo paso.

    Es el punto de entrada principal para los notebooks. Equivale a
    llamar load_instacart_tables() + build_master_dataframe().

    Parameters
    ----------
    config : dict, optional
        Configuración del proyecto.
    eval_set : str
        Split a cargar: 'prior' o 'train'. Default: 'prior'.

    Returns
    -------
    pd.DataFrame
        DataFrame maestro listo para preprocessing.

    Example
    -------
    >>> from src.data.loader import load_full_dataset
    >>> df = load_full_dataset()
    >>> print(df.shape)  # (~32M, 15)
    """
    cfg = config or load_config()
    orders, op_prior, products, aisles, departments = load_instacart_tables(cfg)
    df = build_master_dataframe(
        orders, op_prior, products, aisles, departments, eval_set=eval_set
    )
    return df
