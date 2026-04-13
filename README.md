# 🛒 Cross-Selling Recommender System

> **Proyecto Final | Data Science — Henry**
> Sistema de recomendación de productos complementarios para marketplace de tecnología.

---

## Problema de Negocio

Un marketplace de tecnología pierde oportunidades de cross-selling en productos de alta rotación. No existe un sistema automatizado que sugiera accesorios complementarios al usuario en el momento de la compra (ej. funda para laptop recién vista, auriculares al comprar un smartphone).

**KPI Principal:** Average Order Value (AOV)

---

## Solución

Sistema de recomendación de productos complementarios basado en dos enfoques:

| Modelo | Algoritmo | Librería | Cuándo usarlo |
|---|---|---|---|
| **Modelo 1** | Apriori / FP-Growth (Association Rules) | `mlxtend` | Dataset con estructura de cesta de compra explícita |
| **Modelo 2** | SVD Collaborative Filtering | `scikit-surprise` | Dataset con historial de compras por cliente |
| **Baseline** | Most Popular (Top-N global) | `pandas` | Referencia sin personalización |

---

## Dataset

**Online Retail II** (UCI / Kaggle)
- ~500K transacciones | 4.300+ productos | 2 años de historial
- Descarga: [Kaggle Mirror](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
- Colocar en: `data/raw/online_retail_II.xlsx`

---

## Estructura del Proyecto

```
cross-selling-recommender/
├── data/
│   ├── raw/                   # Dataset original (no versionar)
│   ├── processed/             # Artefactos generados por el pipeline
│   └── external/              # Datos auxiliares
├── notebooks/
│   ├── 01_eda/                # Análisis exploratorio
│   ├── 02_preprocessing/      # Pipeline ETL
│   ├── 03_modeling/           # Entrenamiento de modelos
│   └── 04_evaluation/         # Métricas y comparación
├── src/
│   ├── data/
│   │   ├── loader.py          # Carga y validación del dataset
│   │   └── preprocessor.py    # Pipeline de limpieza
│   ├── features/
│   │   └── builder.py         # Basket matrix, user-item matrix
│   ├── models/
│   │   ├── baseline.py        # Most-popular baseline
│   │   ├── association_rules.py
│   │   └── collaborative_filter.py
│   └── evaluation/
│       └── metrics.py         # Precision@K, Recall@K, NDCG@K, AOV Lift
├── models/trained/            # Modelos serializados
├── outputs/
│   ├── figures/               # Todos los plots del proyecto
│   └── reports/               # Reportes y métricas
├── app/
│   ├── api/                   # FastAPI (Sprint 2)
│   └── streamlit/             # Demo interactivo (Sprint 2)
├── configs/config.yaml        # Parámetros globales
├── requirements.txt
└── .gitignore
```

---

## Setup del Entorno

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPO>
cd cross-selling-recommender

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset
# Descargar online_retail_II.xlsx desde Kaggle y colocarlo en:
# data/raw/online_retail_II.xlsx
```

---

## Ejecución del Sprint 1

```bash
# Orden recomendado:
# 1. EDA
jupyter notebook notebooks/01_eda/01_exploratory_analysis.ipynb

# 2. Preprocessing Pipeline
jupyter notebook notebooks/02_preprocessing/02_preprocessing_pipeline.ipynb

# 3. Modelado (próximo notebook)
jupyter notebook notebooks/03_modeling/03_association_rules.ipynb
```

---

## Métricas de Evaluación

| Métrica | Descripción | Meta Sprint 1 |
|---|---|---|
| **AOV Baseline** | Valor promedio de orden sin recomendaciones | Calculado en EDA |
| **Precision@5** | % de recomendaciones relevantes en top-5 | > 0.15 |
| **Recall@5** | % de ítems relevantes recuperados | > 0.10 |
| **NDCG@5** | Calidad del ranking de recomendaciones | > 0.15 |
| **Hit Rate@5** | % de usuarios con al menos 1 hit | > 0.30 |

---

## Metodología

- **CRISP-DM** (IBM): Entendimiento del negocio → Datos → Modelado → Evaluación → Despliegue
- **Scrum**: Sprints semanales con Daily Stand-ups, Sprint Review y Retrospective

