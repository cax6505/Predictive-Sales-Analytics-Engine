from pathlib import Path

import pandas as pd


DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
    "order_items": ["shipping_limit_date"],
}


def load_raw_tables(config: dict, project_root: Path) -> dict:
    raw_dir = project_root / config["data"]["raw_dir"]
    file_map = config["data"]["files"]
    tables = {}

    for table_name, file_name in file_map.items():
        path = raw_dir / file_name
        parse_dates = DATE_COLUMNS.get(table_name, None)
        tables[table_name] = pd.read_csv(
            path,
            parse_dates=parse_dates,
            encoding="utf-8-sig",
        )

    return tables
