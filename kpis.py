"""
kpis.py — KPI calculations and display-row building for OCT.

Mirrors GAS v28 _buildDispRow / _kpisFromDispRows / _varianceByState logic.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd


# ── AfterShip carrier resolver ─────────────────────────────────────────────

_CARRIER_MAP = [
    # Safexpress — check FIRST; many name variants in the wild
    (["safex", "safe-x", "safe x", "safe express", "safexpress"], "safexpress"),
    # Bluedart / Bluedart Surface / Bluedart Air
    (["bluedart", "blue dart"],  "bluedart"),
    # Delhivery / Delhivery B2B / Delhivery Surface
    (["delhivery"],              "delhivery"),
    # Gati / Gati-KWE
    (["gati"],                   "gati-kwe"),
    # Ecom Express
    (["ecom"],                   "ecom-express"),
    # XpressBees
    (["xpressbees"],             "xpressbees"),
    # DTDC
    (["dtdc"],                   "dtdc"),
    # Ekart
    (["ekart"],                  "ekart"),
    # Shadowfax
    (["shadowfax"],              "shadowfax"),
    # Rivigo / Rivigo Surface
    (["rivigo"],                 "rivigo"),
    # Smartr / SmartR Logistics
    (["smartr"],                 "smartr-logistics"),
]
_DEFAULT_CARRIER = "safexpress"


def track_url(awb: str, transporter: str) -> str:
    t = (transporter or "").lower()
    # safex must be checked BEFORE xpressbees to avoid false match
    for keywords, carrier in _CARRIER_MAP:
        if any(kw in t for kw in keywords):
            return f"https://www.aftership.com/track?t={awb}&c={carrier}"
    return f"https://www.aftership.com/track?t={awb}&c={_DEFAULT_CARRIER}"


# ── Appointment resolution ─────────────────────────────────────────────────

def resolve_appt_status(row: pd.Series, appt_config: dict[str, bool]) -> str:
    """
    Returns one of: 'N' | 'pending' | 'scheduled' | 'delivered'
    Uses appt_config (from DB) with fallback to Appointment_Required column.
    """
    status = (row.get("status") or "").upper().strip()
    if status == "DELIVERED":
        return "delivered"

    cust_key = (row.get("customer_name") or "").lower().strip()
    if cust_key in appt_config:
        needs_appt = appt_config[cust_key]
    else:
        raw = str(row.get("appointment_required") or "").lower().strip()
        needs_appt = raw in ("true", "1", "yes")

    if not needs_appt:
        return "N"

    appt_date = row.get("appointment_date")
    try:
        has_date = appt_date is not None and not pd.isna(appt_date)
    except (TypeError, ValueError):
        has_date = appt_date is not None
    if has_date:
        return "scheduled"

    return "pending"


# ── Display row builder ────────────────────────────────────────────────────

def build_display_rows(df: pd.DataFrame, appt_config: dict[str, bool]) -> pd.DataFrame:
    """
    Augment the raw DB DataFrame with computed display columns:
      days_old, var_days, var_str, appt_status, is_deliv_today, track_url
    Returns the enriched DataFrame.
    """
    if df.empty:
        return df

    today = pd.Timestamp(date.today())

    def _is_null(val) -> bool:
        """Safely check for null / NaT / NaN across Python, numpy, and Arrow types."""
        if val is None:
            return True
        try:
            return bool(pd.isna(val))
        except (TypeError, ValueError):
            return False

    def _to_ts(val) -> pd.Timestamp | None:
        """Convert a value to Timestamp, returning None for nulls/errors."""
        if _is_null(val):
            return None
        try:
            ts = pd.Timestamp(val)
            return None if pd.isna(ts) else ts
        except Exception:
            return None

    # ── days_old ──────────────────────────────────────────────────────────
    def calc_days_old(row: pd.Series) -> int:
        ts = _to_ts(row.get("order_date"))
        if ts is None:
            return 0
        try:
            return max(0, (today - ts).days)
        except Exception:
            return 0

    df = df.copy()
    df["days_old"] = df.apply(calc_days_old, axis=1)

    # ── variance (EDD delta) ───────────────────────────────────────────────
    def calc_var(row: pd.Series):
        edd_ts = _to_ts(row.get("estimated_delivery_date"))
        if edd_ts is None:
            return 0, ""
        is_deliv = str(row.get("status") or "").upper() == "DELIVERED"
        base_ts  = _to_ts(row.get("delivery_date")) if is_deliv else None
        base     = base_ts if base_ts is not None else today
        try:
            var = int((base - edd_ts).days)
        except Exception:
            return 0, ""
        var_str = (f"+{var}d" if var > 0 else f"{var}d") if var != 0 else "0d"
        return var, var_str

    var_data = df.apply(lambda r: calc_var(r), axis=1)
    df["var_days"] = var_data.apply(lambda x: x[0])
    df["var_str"]  = var_data.apply(lambda x: x[1])

    # ── appt_status ───────────────────────────────────────────────────────
    df["appt_status"] = df.apply(
        lambda r: resolve_appt_status(r, appt_config), axis=1
    )

    # ── is_deliv_today ────────────────────────────────────────────────────
    def is_today(row: pd.Series) -> bool:
        if str(row.get("status") or "").upper() != "DELIVERED":
            return False
        ts = _to_ts(row.get("delivery_date"))
        if ts is None:
            return False
        try:
            return ts.date() == date.today()
        except Exception:
            return False

    df["is_deliv_today"] = df.apply(is_today, axis=1)
    # Arrow-safe is_delivered: avoid .str.upper().eq() which can fail on Arrow dtypes
    df["is_delivered"] = df.apply(
        lambda r: str(r.get("status") or "").upper() == "DELIVERED", axis=1
    )

    # ── track_url ─────────────────────────────────────────────────────────
    df["track_url"] = df.apply(
        lambda r: track_url(r.get("awb", ""), r.get("transporter", "")), axis=1
    )

    return df


# ── KPI summary ────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict[str, int]:
    """
    Compute the 5 KPI card values.
    Input df should already have display columns from build_display_rows.
    """
    if df.empty:
        return {
            "total_active": 0,
            "stuck": 0,
            "pending_appt": 0,
            "delivered_today": 0,
            "edd_breached": 0,
        }

    active = df[~df["is_delivered"]]
    return {
        "total_active":    int(len(active)),
        "stuck":           int((active["days_old"] > 14).sum()),
        "pending_appt":    int((active["appt_status"] == "pending").sum()),
        "delivered_today": int(df["is_deliv_today"].sum()),
        "edd_breached":    int(((~df["is_delivered"]) & (df["var_days"] > 0)).sum()),
    }


# ── Cohort DataFrames ──────────────────────────────────────────────────────

def cohort_stuck(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    active = df[~df["is_delivered"]]
    return active[active["days_old"] > 14].sort_values("days_old", ascending=False)


def cohort_pending_appt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["appt_status"] == "pending"]


def cohort_delivered_today(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["is_deliv_today"]]


def cohort_edd_breached(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(~df["is_delivered"]) & (df["var_days"] > 0)].sort_values(
        "var_days", ascending=False
    )


# ── Variance by state ──────────────────────────────────────────────────────

def variance_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      State, Total, Delivered, On Time, Late, Avg Variance (d)
    """
    if df.empty:
        return pd.DataFrame(
            columns=["State", "Total", "Delivered", "On Time", "Late", "Avg Var (d)"]
        )

    today = pd.Timestamp(date.today())
    rows: list[dict] = []

    for state, grp in df.groupby("drop_state", dropna=False):
        state_label = str(state) if state and str(state) != "nan" else "Unknown"
        total = len(grp)
        delivered = grp[grp["status"].str.upper() == "DELIVERED"]
        d_count = len(delivered)

        on_time = 0
        late = 0
        var_sum = 0
        var_n = 0

        for _, row in delivered.iterrows():
            edd = row.get("estimated_delivery_date")
            dd  = row.get("delivery_date")
            if pd.notna(edd) and pd.notna(dd):
                diff = (pd.Timestamp(dd) - pd.Timestamp(edd)).days
                if diff <= 0:
                    on_time += 1
                else:
                    late += 1
                var_sum += diff
                var_n += 1

        # Count in-transit EDD breaches
        in_transit = grp[grp["status"].str.upper() != "DELIVERED"]
        for _, row in in_transit.iterrows():
            edd = row.get("estimated_delivery_date")
            if pd.notna(edd) and today > pd.Timestamp(edd):
                late += 1

        avg_var = round(var_sum / var_n, 1) if var_n else 0.0

        rows.append({
            "State":        state_label,
            "Total":        total,
            "Delivered":    d_count,
            "On Time":      on_time,
            "Late":         late,
            "Avg Var (d)":  avg_var,
        })

    result = pd.DataFrame(rows).sort_values("State")
    return result
