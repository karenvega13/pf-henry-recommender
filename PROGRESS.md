# PROGRESS.md — Cross-Selling Recommender System
> **Proyecto Final | Data Science — Henry**
> Última actualización: 2026-04-12

---

## Estado General del Proyecto

| Sprint | Estado | Avance |
|---|---|---|
| Propuesta | ✅ Confirmada | Dataset: Instacart Market Basket Analysis |
| Sprint 1 (EDA + Modelado) | 🚀 En progreso | Infraestructura completa · Dataset listo |
| Sprint 2 (Despliegue + Demo) | 🔜 Pendiente | — |

---

## Visión del Proyecto

**Problema de negocio:** Un marketplace pierde oportunidades de cross-selling en productos de alta rotación. No existe un sistema automatizado que sugiera productos complementarios al momento de la compra.

**Solución propuesta:** Sistema de recomendación de productos complementarios basado en:
- **Association Rules (FP-Growth)** — para reglas de co-compra en cestas
- **Collaborative Filtering (ALS — implicit feedback)** — para preferencias latentes por usuario

**KPI Principal:** Tamaño promedio de cesta (basket size) como proxy del AOV
**KPIs Técnicos:** Precision@K, Recall@K, NDCG@K, Coverage, Hit Rate

---

## Dataset ✅

| Campo | Valor |
|---|---|
| **Dataset** | Instacart Market Basket Analysis |
| **URL Kaggle** | https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis |
| **Ubicación** | `data/raw/` (6 archivos CSV) |
| **Estado** | ✅ DESCARGADO Y UBICADO EN data/raw/ |

### Archivos en data/raw/

| Archivo | Filas | Descripción |
|---|---|---|
| `orders.csv` | 3 421 083 | Órdenes con user_id, día, hora, días desde compra previa |
| `order_products__prior.csv` | **32 434 489** | Histórico de ítems por orden (entrenamiento) |
| `order_products__train.csv` | 1 384 617 | Última orden por usuario (evaluación) |
| `products.csv` | 49 688 | Catálogo de productos con nombre, pasillo, departamento |
| `aisles.csv` | 134 | Pasillos del supermercado |
| `departments.csv` | 21 | Departamentos del supermercado |

### Decisiones técnicas clave

| Decisión | Justificación estadística |
|---|---|
| FP-Growth en lugar de Apriori | Más eficiente en memoria para 32M filas. Complejidad O(n) vs O(2^k) |
| ALS (implicit) en lugar de SVD | Instacart no tiene ratings explícitos. ALS está diseñado para feedback implícito (frecuencia de compra). Con sparsity > 99.9%, ALS supera a SVD |
| Muestreo de 200K órdenes para Apriori | 200K órdenes = ~6% del total. Con min_support=0.01 → 2000 cestas de soporte mínimo. Estadísticamente robusto y manejable en RAM |
| dtypes int32/uint8 en carga | Reduce uso de RAM de ~4 GB a ~1.8 GB sin pérdida de información |
| Split temporal por order_number | Evita data leakage. order_products__train = última orden real de cada usuario |

---

## Archivos del Proyecto

```
cross-selling-recommender/
├── .github/workflows/ci.yml              ✅ GitHub Actions (lint + tests)
├── .gitignore                             ✅
├── configs/config.yaml                    ✅ Parámetros Instacart (v2.0)
├── data/
│   ├── raw/                               ✅ 6 CSV de Instacart
│   │   ├── orders.csv
│   │   ├── order_products__prior.csv
│   │   ├── order_products__train.csv
│   │   ├── products.csv
│   │   ├── aisles.csv
│   │   └── departments.csv
│   ├── processed/                         🔜 Artefactos ETL (próxima sesión)
│   └── external/
├── notebooks/
│   ├── 01_eda/01_exploratory_analysis.ipynb     🔜 Actualizar para Instacart
│   └── 02_preprocessing/02_preprocessing_pipeline.ipynb  🔜 Actualizar
├── outputs/figures/                       ✅ Carpeta lista
├── README.md                              ✅
├── requirements.txt                       ✅ implicit, mlxtend, scipy
├── src/
│   ├── data/
│   │   ├── loader.py                      ✅ Carga 6 CSV con dtypes optimizados
│   │   └── preprocessor.py               ✅ Pipeline ETL Instacart (7 pasos)
│   ├── evaluation/metrics.py             ✅ Precision@K, Recall@K, NDCG@K
│   └── features/builder.py              ✅ Basket matrix, CSR sparse, co-occ.
└── tests/test_preprocessor.py            ✅ 28 tests — 28/28 PASSED ✅
```

---

## Próximos Pasos (Sesión 4)

### 🔴 PASO 1 — Actualizar notebooks para Instacart
```
1. notebooks/01_eda/01_exploratory_analysis.ipynb
   → Adaptar celdas para 6 CSV de Instacart
   → Plots: distribución de cestas, top productos, heatmap por día/hora
2. notebooks/02_preprocessing/02_preprocessing_pipeline.ipynb
   → Usar loader.load_full_dataset() + preprocessor.run_cleaning_pipeline()
   → Guardar df_processed a data/processed/
```

### 🟡 PASO 2 — Crear notebooks de modelado
```
notebooks/03_modeling/03_association_rules.ipynb     ← PENDIENTE
notebooks/03_modeling/04_collaborative_filter.ipynb  ← PENDIENTE
src/models/association_rules.py                      ← PENDIENTE
src/models/collaborative_filter.py                   ← PENDIENTE
```

### 🟢 PASO 3 — GitHub (primer commit)
```bash
git init
git add .
git commit -m "feat: Sprint 1 — Infraestructura completa + dataset Instacart"
git remote add origin <URL_DEL_REPO>
git push -u origin main
```

---

## Stack Tecnológico

```
Python 3.10+
pandas, numpy, scipy          # Data + matrices sparse
mlxtend                       # FP-Growth / Association Rules
implicit                      # ALS (Collaborative Filtering — feedback implícito)
scikit-learn                  # Preprocessing, métricas auxiliares
matplotlib, seaborn           # Visualización → outputs/figures/
mlflow                        # Experiment tracking (Sprint 2)
fastapi, uvicorn              # API REST (Sprint 2)
streamlit                     # Demo interactivo (Sprint 2)
pytest                        # 28 tests unitarios ✅
```

---

## Sesiones de Trabajo

### Sesión 1 — 2026-04-10
- Análisis de los 4 documentos del PF
- Propuesta de arquitectura y roadmap

### Sesión 2 — 2026-04-12
- Estructura completa de carpetas creada
- `src/` completo: loader, preprocessor, features, metrics
- Notebooks EDA y ETL (basados en UCI Online Retail II — descartado)
- 12 tests unitarios

### Sesión 3 — 2026-04-12
**Dataset cambiado: UCI Online Retail II → Instacart Market Basket Analysis**

Motivo: Instacart tiene 32M filas (vs ~1M de UCI), cestas de 10 ítems en promedio
(vs 1-2 en Olist/UCI), y es el dataset canónico de basket analysis para portfolios.

**Realizado:**
- Evaluación técnica y comparación UCI vs Instacart vs Olist
- Dataset Instacart descargado y ubicado en `data/raw/`
- `configs/config.yaml` — reescrito para schema Instacart (v2.0)
- `src/data/loader.py` — reescrito: carga 6 CSV, dtypes optimizados, JOIN progresivo
- `src/data/preprocessor.py` — reescrito: pipeline ETL para Instacart (7 pasos)
- `src/features/builder.py` — actualizado: basket matrix, CSR sparse para ALS
- `tests/test_preprocessor.py` — reescrito: 28 tests unitarios — **28/28 PASSED**
- `requirements.txt` — actualizado: scipy explícito, implicit priorizado
