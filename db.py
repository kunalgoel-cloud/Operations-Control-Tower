"""
db.py — Supabase client and all database operations for OCT.

Sticky upsert rules (mirrors GAS v28):
  CANCELLED > DELIVERED > UNDELIVERED/RTO > everything else
  Drop_City / Drop_State: first-write wins
  Customer_Name: longest value wins
  SO_Number: NSO variant preferred; else first-write wins
  Delivery_Date: only valid when status == DELIVERED; cleared otherwise
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st
from supabase import create_client, Client

# ── Client ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_client() -> Client:
    """Return a cached Supabase client using Streamlit secrets."""
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


# ── Helpers ────────────────────────────────────────────────────────────────

_STATUS_RANK = {
    "CANCELLED":        100,
    "DELIVERED":         80,
    "UNDELIVERED":       70,
    "RTO":               70,
    "OUT FOR DELIVERY":  40,
    "IN_TRANSIT":        30,
    "MANIFESTED":        20,
    "AWB_REGISTERED":     5,
    "":                   0,
}


def _status_rank(s: str) -> int:
    return _STATUS_RANK.get((s or "").upper().strip(), 30)


def _merge_status(existing: str, incoming: str) -> str:
    """
    Sticky status merge (highest rank wins, with special cases):
      - CANCELLED is permanent — nothing can overwrite it
      - DELIVERED is sticky against lower-rank statuses
      - UNDELIVERED / RTO CAN overwrite DELIVERED (delivery failure is authoritative)
    """
    ex = (existing or "").upper().strip()
    inc = (incoming or "").upper().strip()

    if not inc:
        return existing

    if ex == "CANCELLED":
        return existing                    # CANCELLED is permanent
    if inc == "CANCELLED":
        return "CANCELLED"                 # incoming CANCELLED always wins

    # AWB_REGISTERED only suppresses MANIFESTED/unknown — never a real courier status
    if inc == "AWB_REGISTERED":
        return incoming if _status_rank(ex) < _STATUS_RANK["IN_TRANSIT"] else existing

    if inc in ("UNDELIVERED", "RTO"):
        return incoming                    # failure can correct a wrong DELIVERED

    if ex == "DELIVERED":
        return existing                    # protect genuine DELIVERED

    if inc == "DELIVERED":
        return incoming                    # incoming DELIVERED wins over lower statuses

    # last-write wins for everything else
    return incoming if inc else existing


def _safe_date(val: Any) -> date | None:
    """Parse a date from various input types, return None on failure."""
    if val is None or val == "" or val != val:   # NaN check
        return None
    if isinstance(val, (date, datetime)):
        return val.date() if isinstance(val, datetime) else val
    try:
        d = pd.to_datetime(val, dayfirst=True, errors="coerce")
        return None if pd.isna(d) else d.date()
    except Exception:
        return None


def _fmt_date(val: Any) -> str | None:
    """Return ISO string for Supabase or None."""
    d = _safe_date(val)
    return d.isoformat() if d else None


def _coerce_float(val: Any) -> float | None:
    if val is None or val == "" or val != val:
        return None
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return None


# ── Read ───────────────────────────────────────────────────────────────────

def load_awb_view(
    status_filter: list[str] | None = None,
    state_filter: str | None = None,
    transporter_filter: list[str] | None = None,
    search_field: str | None = None,
    search_query: str | None = None,
    limit: int = 5000,
) -> pd.DataFrame:
    """Load awb_view with optional server-side filters. Returns a DataFrame."""
    client = get_client()
    query = client.table("awb_view").select("*")

    if status_filter:
        query = query.in_("status", status_filter)
    if state_filter:
        query = query.eq("drop_state", state_filter)
    if transporter_filter:
        query = query.in_("transporter", transporter_filter)
    if search_query and search_field:
        col_map = {
            "awb": "awb",
            "invoice": "invoice_number",
            "customer": "customer_name",
            "so": "so_number",
        }
        col = col_map.get(search_field, "awb")
        query = query.ilike(col, f"%{search_query}%")

    # Exclude CANCELLED and AWB_REGISTERED from main view
    query = (
        query
        .not_.in_("status", ["CANCELLED", "AWB_REGISTERED"])
        .limit(limit)
        .order("order_date", desc=True)
    )

    resp = query.execute()
    if not resp.data:
        return pd.DataFrame()

    df = pd.DataFrame(resp.data)
    date_cols = [
        "order_date", "dispatch_date", "delivery_date",
        "estimated_delivery_date", "expected_ship_date", "appointment_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_all_awb_raw() -> pd.DataFrame:
    """Load all awb_view rows (including CANCELLED) for sanity checks / analytics."""
    client = get_client()
    resp = client.table("awb_view").select("*").limit(10000).execute()
    if not resp.data:
        return pd.DataFrame()
    df = pd.DataFrame(resp.data)
    date_cols = [
        "order_date", "dispatch_date", "delivery_date",
        "estimated_delivery_date", "expected_ship_date", "appointment_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_appt_config() -> dict[str, bool]:
    """Return {customer_name_lower: bool} appointment requirement map."""
    client = get_client()
    resp = client.table("appt_config").select("customer_name,appointment_required").execute()
    if not resp.data:
        return {}
    return {
        row["customer_name"].lower().strip(): bool(row["appointment_required"])
        for row in resp.data
        if row.get("customer_name")
    }


def load_upload_log(limit: int = 50) -> pd.DataFrame:
    client = get_client()
    resp = (
        client.table("upload_log")
        .select("*")
        .order("uploaded_at", desc=True)
        .limit(limit)
        .execute()
    )
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()


def get_distinct_values(column: str) -> list[str]:
    """Get distinct non-null values of a column from awb_view."""
    client = get_client()
    resp = (
        client.table("awb_view")
        .select(column)
        .not_.is_(column, "null")
        .execute()
    )
    if not resp.data:
        return []
    vals = sorted({str(row[column]).strip() for row in resp.data if row.get(column)})
    return vals


# ── Upsert ─────────────────────────────────────────────────────────────────

def upsert_awb_records(records: list[dict]) -> tuple[int, int]:
    """
    Upsert a list of awb records applying sticky merge rules.

    Returns (inserted, updated) counts.
    """
    if not records:
        return 0, 0

    client = get_client()

    # Fetch existing rows for the AWBs we're about to touch
    awbs = [r["awb"] for r in records if r.get("awb")]
    if not awbs:
        return 0, 0

    existing_resp = (
        client.table("awb_view")
        .select("*")
        .in_("awb", awbs)
        .execute()
    )
    existing_map: dict[str, dict] = {
        row["awb"]: row for row in (existing_resp.data or [])
    }

    to_insert: list[dict] = []
    to_update: list[dict] = []

    for rec in records:
        awb = rec.get("awb", "").strip()
        if not awb:
            continue

        if awb not in existing_map:
            # New record — clean up None date fields
            cleaned = {k: v for k, v in rec.items() if v is not None and v != ""}
            to_insert.append(cleaned)
        else:
            ex = dict(existing_map[awb])
            merged = _merge_record(ex, rec)
            to_update.append(merged)

    inserted = 0
    updated = 0

    # Batch insert
    if to_insert:
        client.table("awb_view").insert(to_insert).execute()
        inserted = len(to_insert)

    # Batch update (upsert with on_conflict=awb)
    if to_update:
        client.table("awb_view").upsert(
            to_update, on_conflict="awb"
        ).execute()
        updated = len(to_update)

    return inserted, updated


def _merge_record(existing: dict, incoming: dict) -> dict:
    """Apply all GAS v28 sticky merge rules to produce the final merged record."""
    merged = dict(existing)

    inc_status = (incoming.get("status") or "").upper().strip()
    ex_status = (existing.get("status") or "").upper().strip()

    # ── Status merge ──────────────────────────────────────────────────────
    new_status = _merge_status(ex_status, inc_status)
    merged["status"] = new_status

    # ── Delivery_Date: only valid when DELIVERED ──────────────────────────
    if new_status != "DELIVERED":
        merged["delivery_date"] = None
    elif incoming.get("delivery_date"):
        merged["delivery_date"] = incoming["delivery_date"]

    # ── Drop_City / Drop_State: first-write wins ──────────────────────────
    for col in ("drop_city", "drop_state"):
        if not existing.get(col) and incoming.get(col):
            merged[col] = incoming[col]

    # ── Customer_Name: longest value wins ────────────────────────────────
    inc_name = incoming.get("customer_name") or ""
    ex_name = existing.get("customer_name") or ""
    if len(inc_name) > len(ex_name):
        merged["customer_name"] = inc_name

    # ── SO_Number: NSO variant preferred; else first-write ────────────────
    inc_so = incoming.get("so_number") or ""
    ex_so = existing.get("so_number") or ""
    if inc_so:
        if "NSO" in inc_so.upper() and "NSO" not in ex_so.upper():
            merged["so_number"] = inc_so
        elif not ex_so:
            merged["so_number"] = inc_so

    # ── All other fields: incoming non-empty overwrites ───────────────────
    skip = {
        "awb", "status", "delivery_date",
        "drop_city", "drop_state", "customer_name", "so_number",
        "created_at", "updated_at",
    }
    for col, val in incoming.items():
        if col in skip:
            continue
        if val is not None and val != "":
            merged[col] = val

    return merged


# ── Appt config upsert ─────────────────────────────────────────────────────

def upsert_appt_config(records: list[dict]) -> int:
    """Upsert appt_config records. Returns count written."""
    if not records:
        return 0
    client = get_client()
    client.table("appt_config").upsert(
        records, on_conflict="customer_name"
    ).execute()
    return len(records)


def delete_appt_config_entry(customer_name: str) -> None:
    client = get_client()
    client.table("appt_config").delete().eq("customer_name", customer_name).execute()


# ── Upload log ─────────────────────────────────────────────────────────────

def log_upload(
    filename: str,
    file_type: str,
    rows_processed: int,
    rows_inserted: int,
    rows_updated: int,
    status: str = "ok",
    error_message: str | None = None,
) -> None:
    client = get_client()
    client.table("upload_log").insert({
        "filename": filename,
        "file_type": file_type,
        "rows_processed": rows_processed,
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
        "upload_status": status,
        "error_message": error_message,
    }).execute()


# ── Reset ──────────────────────────────────────────────────────────────────

def reset_awb_view() -> None:
    """Truncate awb_view — use with caution."""
    client = get_client()
    # Supabase doesn't expose TRUNCATE via REST; delete all rows instead
    client.table("awb_view").delete().neq("awb", "__never_matches__zzz").execute()


def reset_upload_log() -> None:
    client = get_client()
    client.table("upload_log").delete().neq("id", -1).execute()
