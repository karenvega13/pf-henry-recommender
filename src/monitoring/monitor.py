"""
src/monitoring/monitor.py
=========================
Módulo de monitoreo para el sistema de recomendación de cross-selling.

Cubre cuatro dimensiones de monitoreo:
1. Calidad de datos — validación de schema, nulos, rangos
2. Artefactos del modelo — existencia, tamaño, antigüedad
3. Deriva de distribución — basket size, frecuencia de productos
4. Métricas de recomendación — evaluación sobre muestra de usuarios

Uso básico::

    from src.monitoring.monitor import RecommenderMonitor
    monitor = RecommenderMonitor(project_root=Path(".")))
    report = monitor.run_full_report()
    monitor.save_report(report)
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Constantes ──────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "orders": ["order_id", "user_id", "eval_set", "order_number",
               "order_dow", "order_hour_of_day", "days_since_prior_order"],
    "order_products__prior": ["order_id", "product_id", "add_to_cart_order", "reordered"],
    "products": ["product_id", "product_name", "aisle_id", "department_id"],
    "aisles": ["aisle_id", "aisle"],
    "departments": ["department_id", "department"],
}

MODEL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "fpgrowth_rules.pkl": {
        "description": "Reglas FP-Growth",
        "max_age_days": 90,
        "min_size_kb": 1,
    },
    "als_model.pkl": {
        "description": "Modelo ALS (implicit feedback)",
        "max_age_days": 90,
        "min_size_kb": 50,
    },
}


# ── Clase principal ──────────────────────────────────────────────────────────

class RecommenderMonitor:
    """Monitoreo integral del sistema de recomendación.

    Parameters
    ----------
    project_root : Path
        Ruta raíz del proyecto. Se asume la estructura estándar:
        ``data/raw/``, ``models/trained/``, ``outputs/reports/``.
    baseline_path : Path, optional
        Ruta al archivo parquet de referencia (baseline de distribución).
        Por defecto: ``data/processed/df_full.parquet``.
    """

    def __init__(
        self,
        project_root: Path,
        baseline_path: Path | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data" / "raw"
        self.models_dir = self.project_root / "models" / "trained"
        self.reports_dir = self.project_root / "outputs" / "reports"
        self.baseline_path = baseline_path or (
            self.project_root / "data" / "processed" / "df_full.parquet"
        )
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Calidad de datos ──────────────────────────────────────────────────

    def check_data_quality(self) -> dict[str, Any]:
        """Valida schema, nulos y rangos en los CSV de raw.

        Returns
        -------
        dict
            ``status`` (ok/warning/error), ``tables`` con resultado por tabla,
            ``issues`` lista de problemas encontrados.
        """
        issues: list[str] = []
        tables_report: dict[str, Any] = {}

        for table_name, required_cols in REQUIRED_COLUMNS.items():
            csv_path = self.data_dir / f"{table_name}.csv"
            if not csv_path.exists():
                issues.append(f"MISSING FILE: {csv_path.name}")
                tables_report[table_name] = {"status": "error", "message": "Archivo no encontrado"}
                continue

            try:
                # Leer sólo las primeras 10 000 filas para validar schema
                df_sample = pd.read_csv(csv_path, nrows=10_000)
            except Exception as exc:  # noqa: BLE001
                issues.append(f"READ ERROR {table_name}: {exc}")
                tables_report[table_name] = {"status": "error", "message": str(exc)}
                continue

            tbl_issues: list[str] = []

            # Validar columnas requeridas
            missing_cols = [c for c in required_cols if c not in df_sample.columns]
            if missing_cols:
                tbl_issues.append(f"Columnas faltantes: {missing_cols}")

            # Tasa de nulos por columna
            null_rates = (df_sample.isnull().mean() * 100).round(2)
            high_null = null_rates[null_rates > 50].to_dict()
            if high_null:
                tbl_issues.append(f"Columnas con >50% nulos: {high_null}")

            # Validaciones específicas por tabla
            if table_name == "orders" and "order_hour_of_day" in df_sample.columns:
                invalid_hours = df_sample["order_hour_of_day"].dropna()
                if ((invalid_hours < 0) | (invalid_hours > 23)).any():
                    tbl_issues.append("order_hour_of_day fuera de rango [0, 23]")

            if table_name == "orders" and "order_dow" in df_sample.columns:
                invalid_dow = df_sample["order_dow"].dropna()
                if ((invalid_dow < 0) | (invalid_dow > 6)).any():
                    tbl_issues.append("order_dow fuera de rango [0, 6]")

            if table_name == "order_products__prior" and "reordered" in df_sample.columns:
                invalid_reord = df_sample["reordered"].dropna()
                if not set(invalid_reord.unique()).issubset({0, 1}):
                    tbl_issues.append("reordered contiene valores distintos de {0, 1}")

            tables_report[table_name] = {
                "status": "error" if tbl_issues else "ok",
                "rows_sample": len(df_sample),
                "columns_found": list(df_sample.columns),
                "null_rates_pct": null_rates.to_dict(),
                "issues": tbl_issues,
            }
            issues.extend([f"[{table_name}] {i}" for i in tbl_issues])

        overall = "ok" if not issues else ("error" if any("MISSING" in i for i in issues) else "warning")
        return {"status": overall, "tables": tables_report, "issues": issues}

    # ── 2. Artefactos del modelo ─────────────────────────────────────────────

    def check_model_artifacts(self) -> dict[str, Any]:
        """Verifica existencia, tamaño y antigüedad de los modelos entrenados.

        Returns
        -------
        dict
            ``status``, ``artifacts`` con detalle por archivo, ``issues``.
        """
        issues: list[str] = []
        artifacts_report: dict[str, Any] = {}
        now = datetime.now(tz=timezone.utc)

        for filename, spec in MODEL_ARTIFACTS.items():
            path = self.models_dir / filename
            art: dict[str, Any] = {"description": spec["description"]}

            if not path.exists():
                art["status"] = "error"
                art["message"] = "Archivo no encontrado"
                issues.append(f"MISSING ARTIFACT: {filename}")
                artifacts_report[filename] = art
                continue

            size_kb = path.stat().st_size / 1024
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age_days = (now - mtime).days

            art["size_kb"] = round(size_kb, 1)
            art["age_days"] = age_days
            art["last_modified"] = mtime.isoformat()

            art_issues: list[str] = []
            if size_kb < spec["min_size_kb"]:
                art_issues.append(f"Tamaño {size_kb:.1f} KB < mínimo {spec['min_size_kb']} KB (posible corrupción)")
            if age_days > spec["max_age_days"]:
                art_issues.append(f"Antigüedad {age_days}d > máximo {spec['max_age_days']}d (reentrenar)")

            art["status"] = "warning" if art_issues else "ok"
            art["issues"] = art_issues
            issues.extend([f"[{filename}] {i}" for i in art_issues])
            artifacts_report[filename] = art

        overall = "ok" if not issues else ("error" if any("MISSING" in i for i in issues) else "warning")
        return {"status": overall, "artifacts": artifacts_report, "issues": issues}

    # ── 3. Deriva de distribución ────────────────────────────────────────────

    def check_distribution_drift(
        self,
        current_df: pd.DataFrame | None = None,
        sample_size: int = 5_000,
        ks_alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Compara distribución actual vs baseline mediante KS-test.

        El test de Kolmogorov-Smirnov es no paramétrico → no asume normalidad,
        adecuado para distribuciones de basket size (right-skewed, discreta).

        Parameters
        ----------
        current_df : pd.DataFrame, optional
            DataFrame actual con columnas ``order_id`` y ``product_id``.
            Si es None, intenta cargar desde ``baseline_path``.
        sample_size : int
            Máximo de órdenes a muestrear para el test.
        ks_alpha : float
            Nivel de significancia para rechazar H0 (sin deriva).

        Returns
        -------
        dict
            Resultado KS-test para basket_size y product_frequency.
        """
        issues: list[str] = []
        drift_report: dict[str, Any] = {}

        # Cargar baseline
        if not self.baseline_path.exists():
            return {
                "status": "warning",
                "message": "Baseline no disponible — ejecutar notebooks 02-04 primero",
                "issues": ["Baseline parquet no encontrado"],
            }

        try:
            df_baseline = pd.read_parquet(self.baseline_path)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": str(exc), "issues": [str(exc)]}

        # Usar baseline como "current" si no se pasa DataFrame externo
        if current_df is None:
            logger.warning("No se pasó current_df — usando baseline como referencia cruzada (split 50/50)")
            n = min(sample_size * 2, len(df_baseline))
            df_shuffled = df_baseline.sample(n=n, random_state=0)
            df_ref = df_shuffled.iloc[: n // 2]
            df_cur = df_shuffled.iloc[n // 2 :]
        else:
            df_ref = df_baseline
            df_cur = current_df

        # ── Basket size ──────────────────────────────────────────────────────
        def basket_sizes(df: pd.DataFrame) -> np.ndarray:
            """Calcula tamaño de cesta por orden."""
            return df.groupby("order_id")["product_id"].count().values

        ref_bs = basket_sizes(df_ref.sample(min(sample_size * 5, len(df_ref)), random_state=1))
        cur_bs = basket_sizes(df_cur.sample(min(sample_size * 5, len(df_cur)), random_state=2))

        ks_stat_bs, ks_p_bs = stats.ks_2samp(ref_bs, cur_bs)
        drift_bs = ks_p_bs < ks_alpha
        drift_report["basket_size"] = {
            "ks_statistic": round(float(ks_stat_bs), 4),
            "p_value": round(float(ks_p_bs), 4),
            "drift_detected": drift_bs,
            "baseline_mean": round(float(np.mean(ref_bs)), 3),
            "current_mean": round(float(np.mean(cur_bs)), 3),
            "delta_mean": round(float(np.mean(cur_bs) - np.mean(ref_bs)), 3),
        }
        if drift_bs:
            issues.append(f"DRIFT basket_size: KS={ks_stat_bs:.4f}, p={ks_p_bs:.4f}")

        # ── Frecuencia de productos ──────────────────────────────────────────
        ref_freq = df_ref["product_id"].value_counts(normalize=True).head(100)
        cur_freq = df_cur["product_id"].value_counts(normalize=True).head(100)
        common = ref_freq.index.intersection(cur_freq.index)
        if len(common) >= 10:
            ks_stat_pf, ks_p_pf = stats.ks_2samp(
                ref_freq.loc[common].values, cur_freq.loc[common].values
            )
            drift_pf = ks_p_pf < ks_alpha
            drift_report["product_frequency_top100"] = {
                "ks_statistic": round(float(ks_stat_pf), 4),
                "p_value": round(float(ks_p_pf), 4),
                "drift_detected": drift_pf,
                "common_products": len(common),
            }
            if drift_pf:
                issues.append(f"DRIFT product_frequency: KS={ks_stat_pf:.4f}, p={ks_p_pf:.4f}")
        else:
            drift_report["product_frequency_top100"] = {
                "status": "warning",
                "message": f"Solo {len(common)} productos en común (mínimo 10)",
            }

        overall = "ok" if not issues else "warning"
        return {"status": overall, "tests": drift_report, "issues": issues}

    # ── 4. Resumen de métricas offline ───────────────────────────────────────

    def check_metrics_report(self) -> dict[str, Any]:
        """Carga el último reporte de evaluación y lo resume.

        Returns
        -------
        dict
            Métricas @ K=10 para ALS y FP-Growth, más estado semafórico.
        """
        report_path = self.reports_dir / "evaluation_results.csv"
        if not report_path.exists():
            return {
                "status": "warning",
                "message": "evaluation_results.csv no encontrado — ejecutar notebook 05",
                "issues": ["Reporte de evaluación faltante"],
            }

        try:
            df_eval = pd.read_csv(report_path)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": str(exc), "issues": [str(exc)]}

        issues: list[str] = []
        metrics_report: dict[str, Any] = {}

        # Umbrales mínimos aceptables (benchmarks e-commerce, conservadores)
        thresholds = {
            "Precision@K": 0.01,
            "Recall@K": 0.01,
            "NDCG@K": 0.01,
            "HitRate@K": 0.05,
        }

        for model_name in ["ALS", "FP-Growth"]:
            row = df_eval[(df_eval["Modelo"] == model_name) & (df_eval["K"] == 10)]
            if row.empty:
                issues.append(f"No se encontraron métricas @ K=10 para {model_name}")
                continue

            model_metrics: dict[str, Any] = {}
            for metric, threshold in thresholds.items():
                if metric not in row.columns:
                    continue
                value = float(row[metric].iloc[0])
                below = value < threshold
                model_metrics[metric] = {
                    "value": round(value, 4),
                    "threshold": threshold,
                    "below_threshold": below,
                }
                if below:
                    issues.append(f"{model_name} {metric}={value:.4f} < umbral {threshold}")

            metrics_report[model_name] = model_metrics

        overall = "ok" if not issues else "warning"
        return {"status": overall, "metrics_at_k10": metrics_report, "issues": issues}

    # ── Reporte completo ─────────────────────────────────────────────────────

    def run_full_report(
        self,
        current_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Ejecuta todos los checks y consolida el reporte.

        Parameters
        ----------
        current_df : pd.DataFrame, optional
            DataFrame de producción para el check de deriva.
            Si es None, usa el baseline para auto-validación.

        Returns
        -------
        dict
            Reporte completo con timestamp, estado global y secciones.
        """
        logger.info("Iniciando monitoreo completo del sistema de recomendación...")

        sections = {
            "data_quality": self.check_data_quality,
            "model_artifacts": self.check_model_artifacts,
            "distribution_drift": lambda: self.check_distribution_drift(current_df),
            "metrics": self.check_metrics_report,
        }

        results: dict[str, Any] = {}
        all_issues: list[str] = []

        for section_name, check_fn in sections.items():
            logger.info("  Ejecutando: %s", section_name)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = check_fn()
            except Exception as exc:  # noqa: BLE001
                result = {"status": "error", "message": str(exc), "issues": [str(exc)]}
                logger.error("  Error en %s: %s", section_name, exc)

            results[section_name] = result
            all_issues.extend(result.get("issues", []))

        # Estado global: el peor estado entre todas las secciones
        statuses = [r.get("status", "ok") for r in results.values()]
        if "error" in statuses:
            global_status = "error"
        elif "warning" in statuses:
            global_status = "warning"
        else:
            global_status = "ok"

        report = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "global_status": global_status,
            "total_issues": len(all_issues),
            "all_issues": all_issues,
            "sections": results,
        }

        logger.info("Monitoreo completo. Estado: %s | Issues: %d", global_status, len(all_issues))
        return report

    # ── Persistencia ─────────────────────────────────────────────────────────

    def save_report(self, report: dict[str, Any], filename: str = "monitor_report.json") -> Path:
        """Guarda el reporte en ``outputs/reports/`` en formato JSON.

        Parameters
        ----------
        report : dict
            Reporte generado por :meth:`run_full_report`.
        filename : str
            Nombre del archivo de salida.

        Returns
        -------
        Path
            Ruta absoluta del archivo guardado.
        """
        output_path = self.reports_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Reporte guardado: %s", output_path)
        return output_path

    # ── Resumen en consola ────────────────────────────────────────────────────

    @staticmethod
    def print_summary(report: dict[str, Any]) -> None:
        """Imprime un resumen legible del reporte en consola.

        Parameters
        ----------
        report : dict
            Reporte generado por :meth:`run_full_report`.
        """
        status_icon = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(
            report["global_status"], "❓"
        )
        print(f"\n{'='*60}")
        print(f"  REPORTE DE MONITOREO  {status_icon} {report['global_status'].upper()}")
        print(f"  {report['timestamp']}")
        print(f"{'='*60}")

        for section, data in report["sections"].items():
            sec_icon = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(
                data.get("status", "ok"), "❓"
            )
            print(f"\n{sec_icon} {section.replace('_', ' ').title()}")
            for issue in data.get("issues", []):
                print(f"   • {issue}")
            if not data.get("issues"):
                print("   Sin problemas detectados.")

        print(f"\n{'─'*60}")
        print(f"  Total issues: {report['total_issues']}")
        print(f"{'='*60}\n")
