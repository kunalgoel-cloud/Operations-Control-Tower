"""
sanity.py — Data quality / sanity checks for OCT.

Two checks:
  1. Missing critical fields — AWBs where key columns are blank
  2. Cross-file consistency — AWBs from WMS with no Courier MIS status after N days
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


# ── 1. Missing critical fields ────────────────────────────────────────────

CRITICAL_FIELDS = {
    "awb":                     "AWB",
    "status":                  "Status",
    "customer_name":           "Customer Name",
    "drop_city":               "Drop City",
    "drop_state":              "Drop State",
    "order_date":              "Order Date",
    "estimated_delivery_date": "EDD",
    "transporter":             "Transporter",
}


def check_missing_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame listing AWBs with missing critical fields.
    Columns: AWB, SO#, Customer, Missing Fields, Days Old
    """
    if df.empty:
        return pd.DataFrame(
            columns=["AWB", "SO #", "Customer", "Missing Fields", "Days Old"]
        )

    issues: list[dict] = []
    today = pd.Timestamp(date.today())

    for _, row in df.iterrows():
        missing = []
        for col, label in CRITICAL_FIELDS.items():
            val = row.get(col)
            if val is None or (isinstance(val, float) and val != val):
                missing.append(label)
            elif str(val).strip() in ("", "nan", "NaT", "None"):
                missing.append(label)
            elif col in ("order_date", "estimated_delivery_date") and pd.isna(
                pd.to_datetime(val, errors="coerce")
            ):
                missing.append(label)

        if not missing:
            continue

        # Compute days old
        od = row.get("order_date")
        days_old = 0
        if od is not None and not (isinstance(od, float) and od != od):
            try:
                days_old = max(0, (today - pd.Timestamp(od)).days)
            except Exception:
                pass

        issues.append({
            "AWB":            row.get("awb", ""),
            "SO #":           row.get("so_number") or "–",
            "Customer":       row.get("customer_name") or "–",
            "Missing Fields": ", ".join(missing),
            "Days Old":       days_old,
        })

    if not issues:
        return pd.DataFrame(
            columns=["AWB", "SO #", "Customer", "Missing Fields", "Days Old"]
        )

    result = pd.DataFrame(issues).sort_values("Days Old", ascending=False)
    return result


# ── 2. Cross-file consistency ──────────────────────────────────────────────

def check_cross_file_consistency(
    df: pd.DataFrame,
    min_dispatch_age_days: int = 3,
) -> pd.DataFrame:
    """
    Flag AWBs that:
      - Have a dispatch_date older than min_dispatch_age_days days
      - But have no Status (meaning Courier MIS has never reported on them)
      - And are NOT Porter/Self-pickup (those don't get Courier MIS entries)

    These suggest the AWB was dispatched but Courier MIS has never been uploaded
    for it, or the AWB is wrong.

    Returns DataFrame: AWB, SO#, Customer, Transporter, Dispatch Date, Days Since Dispatch
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "AWB", "SO #", "Customer", "Transporter",
                "Dispatch Date", "Days Since Dispatch",
            ]
        )

    today = pd.Timestamp(date.today())
    cutoff = today - pd.Timedelta(days=min_dispatch_age_days)

    df = df.copy()
    df["dispatch_date"] = pd.to_datetime(df["dispatch_date"], errors="coerce")

    # AWBs with no status (or only WMS-sourced status) and dispatch older than cutoff
    no_courier_status = df[
        df["status"].isna()
        | df["status"].str.strip().eq("")
        | df["status"].str.upper().isin(["MANIFESTED"])   # only WMS-class statuses
    ]

    old_dispatch = no_courier_status[
        no_courier_status["dispatch_date"].notna()
        & (no_courier_status["dispatch_date"] <= cutoff)
    ]

    # Exclude Porter/self-pickup (their Transporter contains porter/self/pickup)
    def is_self_pickup(tp: str) -> bool:
        t = (tp or "").lower()
        return any(kw in t for kw in ("porter", "poter", "pickup", "self"))

    mask = (~old_dispatch["transporter"].apply(
        lambda t: is_self_pickup(str(t) if t is not None else "")
    )).astype(bool)
    # Also exclude ORD/ AWBs (Porter format)
    # Use astype(bool) + non-inplace & to avoid Arrow/numpy dtype conflicts
    ord_mask = old_dispatch["awb"].astype(str).str.upper().str.startswith("ORD/", na=False).astype(bool)
    mask = mask & ~ord_mask

    flagged = old_dispatch[mask].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "AWB", "SO #", "Customer", "Transporter",
                "Dispatch Date", "Days Since Dispatch",
            ]
        )

    flagged["days_since_dispatch"] = (today - flagged["dispatch_date"]).dt.days.astype(int)

    result = flagged[[
        "awb", "so_number", "customer_name", "transporter",
        "dispatch_date", "days_since_dispatch",
    ]].copy()
    result.columns = [
        "AWB", "SO #", "Customer", "Transporter",
        "Dispatch Date", "Days Since Dispatch",
    ]
    result["Dispatch Date"] = result["Dispatch Date"].dt.strftime("%d %b %y")
    return result.sort_values("Days Since Dispatch", ascending=False).reset_index(drop=True)


# ── Sanity summary ─────────────────────────────────────────────────────────

def sanity_summary(df: pd.DataFrame) -> dict:
    """Quick totals for sidebar / header badges."""
    missing = check_missing_fields(df)
    cross   = check_cross_file_consistency(df)
    return {
        "missing_fields_count":    len(missing),
        "cross_file_count":        len(cross),
    }
