"""
test_preprocessor.py
--------------------
Tests unitarios para src/data/preprocessor.py — dataset Instacart.

Cubre:
    - validate_columns         : detección de columnas faltantes
    - remove_duplicates        : eliminación de pares (order_id, product_id) duplicados
    - filter_small_baskets     : filtrado de órdenes con < N productos
    - add_temporal_features    : generación de features de tiempo
    - sample_orders_for_apriori: muestreo reproducible por order_id
    - compute_purchase_frequency: cálculo de frecuencia usuario-producto
    - get_basket_size_stats    : estadísticas de tamaño de cesta

PEP 8 | Proyecto Final Henry — Cross-Selling Recommender
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import (
    validate_columns,
    remove_duplicates,
    filter_small_baskets,
    add_temporal_features,
    sample_orders_for_apriori,
    compute_purchase_frequency,
    get_basket_size_stats,
    run_cleaning_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """DataFrame con estructura del DataFrame maestro de Instacart."""
    return pd.DataFrame({
        "order_id": [1, 1, 1, 2, 2, 3, 4, 4, 4],
        "user_id":  [10, 10, 10, 20, 20, 30, 10, 10, 10],
        "product_id": [101, 102, 103, 101, 104, 101, 105, 106, 107],
        "reordered": [1, 0, 1, 1, 0, 0, 1, 1, 0],
        "order_dow": [1, 1, 1, 3, 3, 5, 0, 0, 0],
        "order_hour_of_day": [10, 10, 10, 14, 14, 8, 20, 20, 20],
        "add_to_cart_order": [1, 2, 3, 1, 2, 1, 1, 2, 3],
        "days_since_prior_order": [7.0, 7.0, 7.0, 14.0, 14.0, float("nan"),
                                   3.0, 3.0, 3.0],
        "order_number": [2, 2, 2, 5, 5, 1, 3, 3, 3],
    })


@pytest.fixture
def df_with_duplicates(sample_df):
    """DataFrame con un par (order_id, product_id) duplicado."""
    dup_row = sample_df.iloc[[0]].copy()  # duplica la primera fila
    return pd.concat([sample_df, dup_row], ignore_index=True)


@pytest.fixture
def df_missing_columns():
    """DataFrame que le falta la columna 'reordered'."""
    return pd.DataFrame({
        "order_id": [1, 2],
        "user_id": [10, 20],
        "product_id": [101, 102],
        # falta reordered, order_dow, order_hour_of_day
    })


# ---------------------------------------------------------------------------
# Tests — validate_columns
# ---------------------------------------------------------------------------

class TestValidateColumns:
    def test_passes_with_all_required_columns(self, sample_df):
        """Debe retornar True con todas las columnas requeridas presentes."""
        result = validate_columns(sample_df)
        assert result is True

    def test_raises_on_missing_columns(self, df_missing_columns):
        """Debe lanzar ValueError si faltan columnas requeridas."""
        with pytest.raises(ValueError, match="Columnas requeridas faltantes"):
            validate_columns(df_missing_columns)


# ---------------------------------------------------------------------------
# Tests — remove_duplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:
    def test_removes_exact_pair_duplicates(self, df_with_duplicates, sample_df):
        """Debe eliminar filas con (order_id, product_id) idéntico."""
        df_clean = remove_duplicates(df_with_duplicates)
        assert len(df_clean) == len(sample_df)

    def test_no_change_when_no_duplicates(self, sample_df):
        """No debe modificar un DataFrame sin duplicados."""
        df_clean = remove_duplicates(sample_df)
        assert len(df_clean) == len(sample_df)

    def test_preserves_columns(self, sample_df):
        """Debe conservar todas las columnas originales."""
        df_clean = remove_duplicates(sample_df)
        assert set(df_clean.columns) == set(sample_df.columns)


# ---------------------------------------------------------------------------
# Tests — filter_small_baskets
# ---------------------------------------------------------------------------

class TestFilterSmallBaskets:
    def test_removes_single_item_orders(self, sample_df):
        """Órdenes con 1 solo producto (order_id=3) deben eliminarse."""
        df_filtered = filter_small_baskets(sample_df, min_items=2)
        assert 3 not in df_filtered["order_id"].values

    def test_keeps_multi_item_orders(self, sample_df):
        """Órdenes con >= 2 productos deben conservarse."""
        df_filtered = filter_small_baskets(sample_df, min_items=2)
        # order_id 1 tiene 3 ítems, order_id 2 tiene 2, order_id 4 tiene 3
        for order_id in [1, 2, 4]:
            assert order_id in df_filtered["order_id"].values

    def test_row_count_decreases(self, sample_df):
        """El número de filas debe disminuir al filtrar."""
        df_filtered = filter_small_baskets(sample_df, min_items=2)
        assert len(df_filtered) < len(sample_df)

    def test_min_items_3_filters_more(self, sample_df):
        """Con min_items=3 debe filtrar más órdenes que con min_items=2."""
        df_2 = filter_small_baskets(sample_df, min_items=2)
        df_3 = filter_small_baskets(sample_df, min_items=3)
        assert len(df_3) <= len(df_2)


# ---------------------------------------------------------------------------
# Tests — add_temporal_features
# ---------------------------------------------------------------------------

class TestAddTemporalFeatures:
    def test_adds_is_weekend_column(self, sample_df):
        """Debe añadir la columna 'is_weekend'."""
        df_out = add_temporal_features(sample_df)
        assert "is_weekend" in df_out.columns

    def test_is_weekend_binary(self, sample_df):
        """is_weekend solo debe tener valores 0 o 1."""
        df_out = add_temporal_features(sample_df)
        assert set(df_out["is_weekend"].unique()).issubset({0, 1})

    def test_adds_time_of_day_column(self, sample_df):
        """Debe añadir la columna 'time_of_day'."""
        df_out = add_temporal_features(sample_df)
        assert "time_of_day" in df_out.columns

    def test_dow_0_is_weekend(self, sample_df):
        """En Instacart, order_dow=0 es sábado → is_weekend=1."""
        df_out = add_temporal_features(sample_df)
        mask = df_out["order_dow"] == 0
        assert (df_out.loc[mask, "is_weekend"] == 1).all()

    def test_does_not_drop_rows(self, sample_df):
        """No debe eliminar filas."""
        df_out = add_temporal_features(sample_df)
        assert len(df_out) == len(sample_df)


# ---------------------------------------------------------------------------
# Tests — sample_orders_for_apriori
# ---------------------------------------------------------------------------

class TestSampleOrdersForApriori:
    def test_returns_correct_number_of_orders(self, sample_df):
        """Debe retornar exactamente n_orders órdenes únicas."""
        n = 2
        df_sampled = sample_orders_for_apriori(sample_df, n_orders=n, random_state=0)
        assert df_sampled["order_id"].nunique() == n

    def test_reproducible_with_same_seed(self, sample_df):
        """Dos llamadas con el mismo random_state deben devolver el mismo resultado."""
        df1 = sample_orders_for_apriori(sample_df, n_orders=2, random_state=99)
        df2 = sample_orders_for_apriori(sample_df, n_orders=2, random_state=99)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                       df2.reset_index(drop=True))

    def test_no_sampling_when_n_exceeds_total(self, sample_df):
        """Si n_orders >= total órdenes, retorna el DataFrame completo."""
        total_orders = sample_df["order_id"].nunique()
        df_sampled = sample_orders_for_apriori(sample_df, n_orders=total_orders + 100)
        assert len(df_sampled) == len(sample_df)

    def test_basket_integrity_preserved(self, sample_df):
        """Cada order_id muestreado debe tener todos sus ítems."""
        df_sampled = sample_orders_for_apriori(sample_df, n_orders=2, random_state=0)
        for order_id in df_sampled["order_id"].unique():
            original_items = set(sample_df[sample_df["order_id"] == order_id]["product_id"])
            sampled_items = set(df_sampled[df_sampled["order_id"] == order_id]["product_id"])
            assert original_items == sampled_items


# ---------------------------------------------------------------------------
# Tests — compute_purchase_frequency
# ---------------------------------------------------------------------------

class TestComputePurchaseFrequency:
    def test_returns_dataframe(self, sample_df):
        """Debe retornar un DataFrame."""
        result = compute_purchase_frequency(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_df):
        """El resultado debe tener user_id, product_id, purchase_count, reorder_rate."""
        result = compute_purchase_frequency(sample_df)
        for col in ["user_id", "product_id", "purchase_count", "reorder_rate"]:
            assert col in result.columns

    def test_reorder_rate_between_0_and_1(self, sample_df):
        """reorder_rate debe estar siempre entre 0 y 1."""
        result = compute_purchase_frequency(sample_df)
        assert (result["reorder_rate"] >= 0).all()
        assert (result["reorder_rate"] <= 1).all()

    def test_purchase_count_positive(self, sample_df):
        """purchase_count debe ser siempre > 0."""
        result = compute_purchase_frequency(sample_df)
        assert (result["purchase_count"] > 0).all()


# ---------------------------------------------------------------------------
# Tests — get_basket_size_stats
# ---------------------------------------------------------------------------

class TestGetBasketSizeStats:
    def test_returns_series(self, sample_df):
        """Debe retornar un objeto pd.Series."""
        result = get_basket_size_stats(sample_df)
        assert isinstance(result, pd.Series)

    def test_mean_greater_than_zero(self, sample_df):
        """La media del tamaño de cesta debe ser positiva."""
        stats = get_basket_size_stats(sample_df)
        assert stats["mean"] > 0

    def test_max_in_sample(self, sample_df):
        """El máximo debe coincidir con la cesta más grande."""
        stats = get_basket_size_stats(sample_df)
        expected_max = sample_df.groupby("order_id")["product_id"].nunique().max()
        assert stats["max"] == expected_max


# ---------------------------------------------------------------------------
# Tests — run_cleaning_pipeline (integración)
# ---------------------------------------------------------------------------

class TestRunCleaningPipeline:
    def test_returns_two_dataframes(self, sample_df):
        """Debe retornar una tupla con dos DataFrames."""
        df_full, df_apriori = run_cleaning_pipeline(
            sample_df, min_basket_items=2, apriori_sample=2
        )
        assert isinstance(df_full, pd.DataFrame)
        assert isinstance(df_apriori, pd.DataFrame)

    def test_apriori_subset_smaller_or_equal(self, sample_df):
        """df_apriori debe tener <= órdenes que df_full."""
        df_full, df_apriori = run_cleaning_pipeline(
            sample_df, min_basket_items=2, apriori_sample=2
        )
        assert df_apriori["order_id"].nunique() <= df_full["order_id"].nunique()

    def test_full_df_has_no_single_item_orders(self, sample_df):
        """df_full no debe tener órdenes con 1 solo producto."""
        df_full, _ = run_cleaning_pipeline(sample_df, min_basket_items=2)
        basket_sizes = df_full.groupby("order_id")["product_id"].nunique()
        assert (basket_sizes >= 2).all()
