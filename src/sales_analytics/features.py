import numpy as np
import pandas as pd

from .utils import mode_or_unknown


def build_order_level_features(tables: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    order_ids = set(cohort["order_id"].tolist())

    category_translation = tables["category_translation"].rename(
        columns={"product_category_name": "product_category_name_pt", "product_category_name_english": "product_category_name_en"}
    )
    products = tables["products"].merge(
        category_translation,
        left_on="product_category_name",
        right_on="product_category_name_pt",
        how="left",
    )
    sellers = tables["sellers"][["seller_id", "seller_state"]].copy()

    items = tables["order_items"].copy()
    items = items.loc[items["order_id"].isin(order_ids)].copy()
    items = items.merge(products, on="product_id", how="left")
    items = items.merge(sellers, on="seller_id", how="left")
    items["package_volume_cm3"] = (
        items["product_length_cm"].fillna(0) * items["product_height_cm"].fillna(0) * items["product_width_cm"].fillna(0)
    )

    item_agg = items.groupby("order_id").agg(
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        item_count=("order_item_id", "count"),
        seller_count=("seller_id", pd.Series.nunique),
        product_count=("product_id", pd.Series.nunique),
        product_weight_g_mean=("product_weight_g", "mean"),
        package_volume_cm3_mean=("package_volume_cm3", "mean"),
        product_photos_qty_mean=("product_photos_qty", "mean"),
        product_description_lenght_mean=("product_description_lenght", "mean"),
    ).reset_index()

    dominant_category_df = items.groupby("order_id")["product_category_name_en"].agg(mode_or_unknown).reset_index(name="product_category_main")
    seller_state_df = items.groupby("order_id")["seller_state"].agg(mode_or_unknown).reset_index(name="seller_state_mode")

    payments = tables["order_payments"].copy()
    payments = payments.loc[payments["order_id"].isin(order_ids)].copy()
    payment_agg = payments.groupby("order_id").agg(
        payment_value_total=("payment_value", "sum"),
        payment_installments_max=("payment_installments", "max"),
        payment_records=("payment_sequential", "count"),
        payment_type_nunique=("payment_type", pd.Series.nunique),
    ).reset_index()
    payment_type_df = payments.groupby("order_id")["payment_type"].agg(mode_or_unknown).reset_index(name="payment_type_mode")

    out = cohort.copy()
    out = out.merge(item_agg, on="order_id", how="left")
    out = out.merge(dominant_category_df, on="order_id", how="left")
    out = out.merge(seller_state_df, on="order_id", how="left")
    out = out.merge(payment_agg, on="order_id", how="left")
    out = out.merge(payment_type_df, on="order_id", how="left")

    out["review_comment_message"] = out["review_comment_message"].fillna("")
    out["review_text"] = out["review_comment_message"].replace("", np.nan).fillna("__NO_REVIEW_TEXT__")
    out["text_present"] = (out["review_text"] != "__NO_REVIEW_TEXT__").astype(int)
    clean_text = out["review_text"].replace("__NO_REVIEW_TEXT__", "")
    out["text_char_len"] = clean_text.str.len().fillna(0).astype(int)
    out["text_word_count"] = clean_text.str.split().str.len().fillna(0).astype(int)
    out["exclamation_count"] = clean_text.str.count("!").fillna(0).astype(int)
    out["question_count"] = clean_text.str.count(r"\?").fillna(0).astype(int)

    out["approval_lag_hours"] = (out["order_approved_at"] - out["order_purchase_timestamp"]).dt.total_seconds() / 3600
    out["delivery_days"] = (out["order_delivered_customer_date"] - out["order_purchase_timestamp"]).dt.total_seconds() / 86400
    out["delivery_delay_days"] = (out["order_delivered_customer_date"] - out["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    out["delivery_delay_days_clipped"] = out["delivery_delay_days"].clip(lower=-30, upper=30)
    out["late_delivery_flag"] = (out["delivery_delay_days"] > 0).astype(int)
    out["freight_ratio"] = (out["total_freight"] / out["total_price"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["payment_gap"] = out["payment_value_total"] - (out["total_price"] + out["total_freight"])
    out["same_state_seller_customer"] = (out["seller_state_mode"] == out["customer_state"]).astype(int)
    out["purchase_month"] = out["order_purchase_timestamp"].dt.month
    out["purchase_quarter"] = out["order_purchase_timestamp"].dt.quarter
    out["weekend_purchase_flag"] = out["order_purchase_timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    for col in ["total_price", "total_freight", "payment_value_total", "product_weight_g_mean", "package_volume_cm3_mean", "approval_lag_hours", "delivery_days"]:
        out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))

    keep_cols = [
        "customer_unique_id", "order_id", "score_time", "target_repeat_within_180d", "review_score", "review_text",
        "text_present", "text_char_len", "text_word_count", "exclamation_count", "question_count",
        "total_price", "total_freight", "payment_value_total", "payment_installments_max", "payment_records",
        "payment_type_nunique", "payment_type_mode", "item_count", "seller_count", "product_count",
        "product_category_main", "seller_state_mode", "customer_state", "same_state_seller_customer",
        "approval_lag_hours", "delivery_days", "delivery_delay_days", "delivery_delay_days_clipped", "late_delivery_flag",
        "freight_ratio", "payment_gap", "product_weight_g_mean", "package_volume_cm3_mean", "product_photos_qty_mean",
        "product_description_lenght_mean", "purchase_month", "purchase_quarter", "weekend_purchase_flag",
        "log1p_total_price", "log1p_total_freight", "log1p_payment_value_total", "log1p_product_weight_g_mean",
        "log1p_package_volume_cm3_mean", "log1p_approval_lag_hours", "log1p_delivery_days",
    ]
    return out[keep_cols].copy()
