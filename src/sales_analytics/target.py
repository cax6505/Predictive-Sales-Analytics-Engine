import pandas as pd

REVIEW_COLUMNS = [
    "review_id",
    "order_id",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
]


def select_one_review_per_order(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews[REVIEW_COLUMNS].copy()
    reviews = reviews.sort_values(["order_id", "review_creation_date", "review_answer_timestamp"])
    return reviews.groupby("order_id", as_index=False).first()


def build_customer_first_order_cohort(tables: dict, repeat_window_days: int = 180) -> pd.DataFrame:
    customers = tables["customers"].copy()
    orders = tables["orders"].copy()
    reviews = select_one_review_per_order(tables["order_reviews"])

    delivered = orders.loc[
        orders["order_status"].eq("delivered")
        & orders["order_purchase_timestamp"].notna()
        & orders["order_delivered_customer_date"].notna()
    ].copy()

    delivered = delivered.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
        how="left",
    )
    delivered = delivered.merge(reviews, on="order_id", how="left")
    delivered["score_time"] = delivered[["order_delivered_customer_date", "review_creation_date"]].max(axis=1)
    delivered["score_time"] = delivered["score_time"].fillna(delivered["order_delivered_customer_date"])
    delivered = delivered.sort_values(["customer_unique_id", "order_purchase_timestamp", "order_id"])

    first_orders = delivered.groupby("customer_unique_id", as_index=False).first()
    dataset_end = delivered["order_purchase_timestamp"].max()
    first_orders["eligible_end_time"] = first_orders["score_time"] + pd.to_timedelta(repeat_window_days, unit="D")
    first_orders = first_orders.loc[first_orders["eligible_end_time"] <= dataset_end].copy()

    purchase_map = delivered.groupby("customer_unique_id")["order_purchase_timestamp"].apply(list).to_dict()

    def first_purchase_after_score(row: pd.Series):
        purchases = purchase_map.get(row["customer_unique_id"], [])
        later = [ts for ts in purchases if pd.notna(ts) and ts > row["score_time"]]
        if not later:
            return pd.NaT
        return min(later)

    first_orders["next_purchase_after_score"] = first_orders.apply(first_purchase_after_score, axis=1)
    limit = first_orders["score_time"] + pd.to_timedelta(repeat_window_days, unit="D")
    first_orders["target_repeat_within_180d"] = (
        first_orders["next_purchase_after_score"].notna()
        & (first_orders["next_purchase_after_score"] <= limit)
    ).astype(int)
    return first_orders
