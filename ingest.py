"""
ingest.py — File parsing and ingestion for OCT.

Supports CSV and Excel (.xlsx/.xls).
Ports all GAS v28 business rules faithfully:
  - File-type auto-detection by column headers
  - AWB validation (must contain a digit, or start with ORD/)
  - Status owned exclusively by Courier MIS
  - Porter/self-pickup DELIVERED detection from WMS
  - Status normalisation with correct precedence order
  - All column alias mappings
"""

from __future__ import annotations

import io
import re
from typing import Any

import chardet
import pandas as pd

import db

# ── AWB helpers ────────────────────────────────────────────────────────────

_AWB_TRAILING_ZEROS = re.compile(r"\.0+$")


def norm_awb(v: Any) -> str:
    s = str(v or "").strip()
    return _AWB_TRAILING_ZEROS.sub("", s)


def is_valid_awb(v: Any) -> bool:
    """Must contain at least one digit, OR start with ORD/ (Porter)."""
    s = str(v or "").strip()
    if not s:
        return False
    return bool(re.search(r"\d", s)) or s.upper().startswith("ORD/")


# ── Date parser ────────────────────────────────────────────────────────────

def parse_date(val: Any) -> str | None:
    """Return ISO date string or None."""
    if val is None or (isinstance(val, float) and val != val):
        return None
    s = str(val).strip()
    if not s or s in ("Invalid Date", "-", "nan", "NaT", "NaN"):
        return None
    # DD/MM/YYYY  or  D/M/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            from datetime import date
            return date(y, mo, d).isoformat()
        except ValueError:
            pass
    try:
        # ISO format (YYYY-MM-DD…) — skip dayfirst to avoid pandas UserWarning
        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            ts = pd.to_datetime(s, errors="coerce")
        else:
            ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    except Exception:
        return None


def _parse_series_date(series: pd.Series) -> pd.Series:
    return series.apply(parse_date)


# ── File type detection ─────────────────────────────────────────────────────

def detect_file_type(headers: list[str]) -> str:
    """
    Returns one of:
      wms_dispatch | courier_tracking | stuck_orders | appt_config | unknown
    """
    lc = {h.lower().strip() for h in headers}

    def has(*cols: str) -> bool:
        return all(c.lower() in lc for c in cols)

    def has_any(*cols: str) -> bool:
        return any(c.lower() in lc for c in cols)

    # Appt config: has customer name + appointment column, NO awb column
    if (
        "awb" not in lc
        and has_any("customer name", "customer_name")
        and has_any(
            "appointment required", "appointment_required",
            "appt required", "appointment needed",
        )
    ):
        return "appt_config"

    # WMS dispatch: channel order id/code OR (order created date + shipper)
    if has_any("channel order id", "channel order code"):
        return "wms_dispatch"
    if has("order created date", "shipper"):
        return "wms_dispatch"

    # Courier tracking: courier partner + awb  OR  reference number + awb + status
    if has("courier partner", "awb"):
        return "courier_tracking"
    if has("reference number", "awb", "status"):
        return "courier_tracking"
    if has("awb", "drop city", "drop name"):
        return "courier_tracking"

    # Stuck orders: so number + days/stage info
    if has_any("so number", "so_number") and has_any(
        "days since order", "aging bucket", "current stage", "age (days)",
    ):
        return "stuck_orders"

    return "unknown"


# ── Column getter (case-insensitive alias lookup) ──────────────────────────

def make_getter(df: pd.DataFrame):
    """
    Returns a function get(row, aliases) → first non-empty match.
    Lookup is case-insensitive.
    """
    col_map: dict[str, str] = {c.lower().strip(): c for c in df.columns}

    def get(row: pd.Series, aliases: list[str]) -> Any:
        for alias in aliases:
            mapped = col_map.get(alias.lower().strip())
            if mapped is not None:
                val = row.get(mapped)
                if val is not None and str(val).strip() not in ("", "nan", "NaT", "NaN"):
                    return val
        return None

    return get


# ── Status normalisation (Courier MIS only) ───────────────────────────────

def normalise_status(raw: str) -> str | None:
    """
    Returns normalised status or None (for AWB_REGISTERED which is skipped).
    Order of checks matters — specific patterns must precede broad ones.
    """
    s = (raw or "").upper().strip()
    if not s:
        return "IN_TRANSIT"
    if "CANCEL" in s:
        return "CANCELLED"
    if "UNDELIVER" in s or "FAIL" in s:
        return "UNDELIVERED"
    if "OUT FOR" in s or "OUT_FOR" in s:
        return "OUT FOR DELIVERY"
    if "DELIVER" in s:
        return "DELIVERED"
    if "RTO" in s or "RETURN" in s:
        return "RTO"
    if "MANIFEST" in s or "CLOSED" in s:
        return "MANIFESTED"
    if "AWB" in s or "REGISTER" in s:
        return "AWB_REGISTERED"   # tag in DB so load filter suppresses it
    return "IN_TRANSIT"


# ── File reader ────────────────────────────────────────────────────────────

def read_file(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded Streamlit file object (CSV or Excel) into a DataFrame.
    Handles encoding detection for CSV files.
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw_bytes), dtype=str, keep_default_na=False)

    # CSV — detect encoding
    detected = chardet.detect(raw_bytes[:50_000])
    encoding = detected.get("encoding") or "utf-8"
    try:
        return pd.read_csv(
            io.BytesIO(raw_bytes),
            dtype=str,
            keep_default_na=False,
            encoding=encoding,
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            io.BytesIO(raw_bytes),
            dtype=str,
            keep_default_na=False,
            encoding="latin-1",
        )


# ── Transporters to exclude from B2B ingest ──────────────────────────────
# AWBs for these carriers are excluded at ingest time and never written to DB.
_SKIP_TRANSPORTERS = {"delhivery"}   # lowercase; add more as needed


def _is_skipped_transporter(name: str) -> bool:
    n = (name or "").lower().strip()
    return any(skip in n for skip in _SKIP_TRANSPORTERS)


# ── Ingest: WMS Dispatch ───────────────────────────────────────────────────

def ingest_wms(df: pd.DataFrame) -> list[dict]:
    """Parse WMS dispatch report into awb_view records."""
    records: list[dict] = []
    get = make_getter(df)

    for _, row in df.iterrows():
        awb = norm_awb(
            get(row, ["AWB", "AWB Number", "Tracking Number", "LR Number"])
        )
        if not awb or not is_valid_awb(awb):
            continue

        wms_raw_st = str(get(row, ["Status"]) or "").upper()
        wms_tp = str(get(row, ["Shipper"]) or "").upper()

        if _is_skipped_transporter(wms_tp):
            continue

        is_self_pickup = (
            "PORTER" in wms_tp
            or "POTER" in wms_tp      # WMS typo variant
            or "PICKUP" in wms_tp
            or "SELF" in wms_tp
            or awb.upper().startswith("ORD/")
        )
        is_closed = any(
            kw in wms_raw_st
            for kw in ("CLOSED", "MANIFEST", "DELIVER", "COMPLET")
        ) or wms_raw_st == "DONE"

        rec: dict[str, Any] = {"awb": awb}

        # Status: only Porter/Self pickup gets a status from WMS
        if is_self_pickup and is_closed:
            rec["status"] = "DELIVERED"

        # SO / Order Number: WMS may name this field in various ways.
        # IMPORTANT: "Order Id" / "Order ID" are WMS internal numeric PKs (e.g. "3443")
        # — do NOT read them as so_number.  Only business-meaningful order refs here.
        rec["so_number"] = str(
            get(row, [
                "SO Number", "SO_Number", "NSO Number",
                "Order Number", "Order No", "Order #", "Order_Number",
            ]) or ""
        ).strip() or None

        # Invoice number: Channel Order Code/Id carries the customer invoice ref
        rec["invoice_number"] = str(
            get(row, [
                "Channel Order Code", "Channel Order Id",
                "Invoice Number", "Invoice No", "Inv Number",
                "Channel Inv Number",
            ]) or ""
        ).strip() or None

        rec["customer_name"] = str(
            get(row, [
                "Customer Name", "Customer", "Customer_Name",
                "Consignee Name", "Consignee",
            ]) or ""
        ).strip() or None

        rec["drop_city"] = str(
            get(row, [
                "Drop City", "Destination City", "Delivery City", "Consignee City",
            ]) or ""
        ).strip() or None

        rec["drop_state"] = str(
            get(row, ["Drop State", "Destination State", "Delivery State"]) or ""
        ).strip() or None

        rec["transporter"] = str(
            get(row, [
                "Shipper", "Carrier", "Courier", "Logistics Partner",
                "Shipping Partner", "Shipping Method",
            ]) or ""
        ).strip() or None

        # order_date is owned exclusively by RPT Stuck Orders — not read from WMS.

        rec["dispatch_date"] = parse_date(
            get(row, ["Dispatch Date", "Dispatch_Date", "Shipped Date", "Pickup Date"])
        )

        # expected_ship_date is intentionally NOT read from WMS —
        # it is owned exclusively by the Stuck Orders report (SLA/promise date).

        amt = get(row, [
            "Invoice Amount", "Invoice Value", "Invoice_Amount",
            "Order Value", "COD Amount",
        ])
        try:
            rec["invoice_amount"] = float(str(amt).replace(",", "")) if amt else None
        except (ValueError, TypeError):
            rec["invoice_amount"] = None

        records.append(rec)

    return records


# ── Ingest: Courier MIS ────────────────────────────────────────────────────

def ingest_courier_mis(df: pd.DataFrame) -> list[dict]:
    """Parse B2B Courier MIS into awb_view records."""
    records: list[dict] = []
    get = make_getter(df)

    for _, row in df.iterrows():
        awb = norm_awb(
            get(row, [
                "AWB", "AWB Number", "Consignment Number",
                "Docket Number", "LR No", "LR Number",
            ])
        )
        if not awb or not is_valid_awb(awb):
            continue

        # Skip excluded transporters (e.g. Delhivery B2B)
        mis_tp = str(
            get(row, [
                "Courier Partner", "Carrier", "Logistics Partner",
                "Courier Name", "Service Provider",
            ]) or ""
        )
        if _is_skipped_transporter(mis_tp):
            continue

        raw_st = str(
            get(row, [
                "Status", "Current Status", "Shipment Status", "Tracking Status",
            ]) or ""
        )
        norm_st = normalise_status(raw_st)
        rec: dict[str, Any] = {"awb": awb, "status": norm_st}

        if norm_st == "AWB_REGISTERED":
            # Persist the status so DB load-filter suppresses it;
            # do NOT overwrite other fields — the WMS record is still valid.
            records.append(rec)
            continue

        if norm_st == "CANCELLED":
            records.append(rec)
            continue

        # Reference number → SO_Number (strip REF prefix)
        ref = str(get(row, [
            "Reference Number", "Ref Number", "Reference No", "Order Reference",
        ]) or "").strip()
        ref = re.sub(r"^REF", "", ref, flags=re.IGNORECASE).strip()
        # Only store as so_number if it looks like a business SO reference:
        # - not empty / not equal to the AWB
        # - does NOT start with '_' (carrier-internal refs like _100035069731)
        # - is NOT purely numeric (bare AWB stored as reference)
        if (
            ref
            and ref != awb
            and not ref.startswith("_")
            and not re.match(r"^\d+$", ref)
        ):
            rec["so_number"] = ref

        rec["invoice_number"] = str(
            get(row, ["Invoice Number", "Invoice No", "invoice_number"]) or ""
        ).strip() or None

        rec["customer_name"] = str(
            get(row, [
                "Drop Name", "Consignee Name", "Receiver Name",
                "Recipient Name", "Customer Name", "Consignee",
            ]) or ""
        ).strip() or None

        rec["drop_city"] = str(
            get(row, ["Drop City", "Destination City", "Delivery City"]) or ""
        ).strip() or None

        rec["drop_state"] = str(
            get(row, [
                "Drop State", "Destination State", "Delivery State", "Consignee State",
            ]) or ""
        ).strip() or None

        rec["transporter"] = str(
            get(row, [
                "Courier Partner", "Carrier", "Logistics Partner",
                "Courier Name", "Service Provider",
            ]) or ""
        ).strip() or None

        # Order_Date intentionally NOT read from Courier MIS
        rec["dispatch_date"] = parse_date(
            get(row, [
                "Dispatch Date", "Pickup Date", "Dispatched Date", "Shipped Date",
            ])
        )

        rec["estimated_delivery_date"] = parse_date(
            get(row, [
                "Estimated Delivery Date", "EDD", "Expected Delivery Date",
                "Committed Delivery Date", "Promised Delivery Date",
                "Scheduled Delivery Date", "Target Delivery Date",
                "Estimated Delivery", "Exp Del Date", "Expected Del Date",
                "ETA", "Estimated Date", "Promised Date",
            ])
        )

        # Delivery_Date only when DELIVERED
        if norm_st == "DELIVERED":
            rec["delivery_date"] = parse_date(
                get(row, [
                    "Delivery Date", "Delivered Date",
                    "Actual Delivery Date", "POD Date",
                ])
            )

        amt = get(row, [
            "Invoice Value", "Invoice Amount", "Declared Value", "COD Amount",
        ])
        try:
            rec["invoice_amount"] = float(str(amt).replace(",", "")) if amt else None
        except (ValueError, TypeError):
            rec["invoice_amount"] = None

        rec["pod_url"] = str(
            get(row, [
                "POD_URL", "POD URL", "pod_url", "POD Link",
                "Proof of Delivery URL", "POD Image",
            ]) or ""
        ).strip() or None

        rec["latest_remark"] = str(
            get(row, [
                "Latest Remark", "Last Remark", "Remarks", "Remark",
                "Last Status Remark", "Current Remark",
                "Activity", "Last Activity", "Latest Activity", "Tracking Remark",
            ]) or ""
        ).strip() or None

        # Appointment info
        appt_raw = str(
            get(row, [
                "Appointment Required", "Appointment_Required", "Appt Required",
            ]) or ""
        ).lower().strip()
        needs_appt = appt_raw in ("true", "1", "yes")
        rec["appointment_required"] = needs_appt
        if needs_appt:
            rec["appointment_date"] = parse_date(
                get(row, [
                    "Appointment Delivery Date", "Appointment Date",
                    "appointment_delivery_date",
                ])
            )

        records.append(rec)

    return records


# ── Ingest: RPT Stuck Orders ───────────────────────────────────────────────

def ingest_stuck_orders(df: pd.DataFrame) -> list[dict]:
    """
    Stuck orders join on SO_Number.
    Returns lightweight dicts with only the fields this source owns:
    so_number (key), invoice_number, customer_po_ref, expected_ship_date.
    The db layer handles the SO → AWB join via upsert_stuck_records.
    """
    records: list[dict] = []
    get = make_getter(df)

    for _, row in df.iterrows():
        so = str(
            get(row, [
                "SO Number", "SO_Number", "Order Number", "SO #",
                "NSO Number", "Order No", "Order_Number",
            ]) or ""
        ).strip() or None

        inv = str(
            get(row, [
                "inv_agg.Invoice Numbers", "Invoice Numbers", "Invoice Number",
                "Invoice No", "Invoice #", "Inv Number", "inv_number",
                "Invoice Num", "Invoices", "Invoice",
            ]) or ""
        ).strip() or None

        # Need at least one identifier to be able to join to an AWB
        if not so and not inv:
            continue

        po_ref = str(
            get(row, [
                "Customer PO Ref", "Customer_PO_Ref", "PO Ref",
                "PO Reference", "Customer PO", "PO Number",
            ]) or ""
        ).strip() or None

        exp_ship = parse_date(
            get(row, [
                "Expected Ship Date", "Expected_Ship_Date",
                "Promise Date", "SLA Date",
            ])
        )

        customer = str(
            get(row, ["Customer Name", "Customer_Name", "customer"]) or ""
        ).strip() or None

        dispatch = parse_date(
            get(row, ["Dispatch Date", "Dispatch_Date", "Dispatched Date"])
        )

        order_dt = parse_date(
            get(row, ["Order Date", "Order_Date"])
        )

        records.append({
            "so_number":          so,
            "invoice_number":     inv,
            "customer_po_ref":    po_ref,
            "expected_ship_date": exp_ship,
            "customer_name":      customer,
            "dispatch_date":      dispatch,
            "order_date":         order_dt,
        })

    return records


# ── Ingest: Appt Config ────────────────────────────────────────────────────

def ingest_appt_config(df: pd.DataFrame) -> list[dict]:
    """Parse appointment config CSV/Excel."""
    records: list[dict] = []
    get = make_getter(df)

    for _, row in df.iterrows():
        name = str(
            get(row, ["Customer Name", "Customer_Name", "customer"]) or ""
        ).strip()
        if not name:
            continue
        req_raw = str(
            get(row, [
                "Appointment Required", "Appointment_Required",
                "Appt Required", "appointment needed", "appointment",
            ]) or ""
        ).lower().strip()
        req = req_raw in ("yes", "true", "1")
        records.append({
            "customer_name": name,
            "appointment_required": req,
        })

    return records


# ── Fuzzy fallback: WMS-AWB → MIS-AWB bridge ──────────────────────────────

# Generic tokens that appear in Indian company names but don't disambiguate
_COMPANY_SKIP = {"pvt", "ltd", "private", "limited", "llp", "inc", "corp",
                 "co", "the", "and", "&", "of", "for"}


def _try_fuzzy_update(
    client: Any,
    wms_row: dict,
    stuck: dict,
    updates: list[dict],
) -> None:
    """
    When Path A found a WMS record by invoice_number, the WMS may have a
    wrong/old AWB.  Use customer_name + drop_state + dispatch_date ± 5 days
    from the WMS record to find the correct B2B MIS AWB and also update it.

    Only proceeds when:
    - customer_name, drop_state and expected_ship_date are all available
    - Exactly ONE unmatched (invoice_number IS NULL) record matches the criteria
    - That record's AWB is not already in the updates batch
    """
    cust  = (wms_row.get("customer_name") or "").strip()
    state = (wms_row.get("drop_state") or "").strip()
    exp_date = stuck.get("expected_ship_date")
    wms_awb  = wms_row.get("awb") or ""

    if not cust or not state or not exp_date:
        return

    # Pick first substantive word from customer name
    first_word = ""
    for token in re.split(r"[\s\-_/]+", cust):
        if token and token.lower() not in _COMPANY_SKIP and len(token) >= 3:
            first_word = token.upper()
            break
    if not first_word:
        return

    try:
        exp_ts   = pd.Timestamp(exp_date)
        date_lo  = (exp_ts - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        date_hi  = (exp_ts + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    except Exception:
        return

    try:
        resp = (
            client.table("awb_view")
            .select("awb")
            .ilike("customer_name", f"%{first_word}%")
            .eq("drop_state", state)
            .gte("dispatch_date", date_lo)
            .lte("dispatch_date", date_hi)
            .is_("invoice_number", "null")   # only un-matched MIS records
            .neq("awb", wms_awb)             # exclude the WMS record itself
            .execute()
        )
    except Exception:
        return

    candidates = resp.data or []
    if len(candidates) != 1:
        return  # ambiguous or no match — skip

    mis_awb = candidates[0]["awb"]
    # Skip if this AWB already has an update queued in this batch
    if any(u.get("awb") == mis_awb for u in updates):
        return

    patch: dict[str, Any] = {"awb": mis_awb}
    if stuck.get("invoice_number"):
        patch["invoice_number"] = stuck["invoice_number"]
    if stuck.get("so_number"):
        patch["so_number"] = stuck["so_number"]
    if stuck.get("customer_po_ref"):
        patch["customer_po_ref"] = stuck["customer_po_ref"]
    if stuck.get("expected_ship_date"):
        patch["expected_ship_date"] = stuck["expected_ship_date"]
    if len(patch) > 1:
        updates.append(patch)


# ── Stuck orders DB join ───────────────────────────────────────────────────

def apply_stuck_orders_to_db(stuck_records: list[dict]) -> tuple[int, int]:
    """
    Join stuck_orders onto awb_view via SO_Number.
    Updates invoice_number, customer_po_ref, expected_ship_date.
    Returns (rows_matched, rows_updated).
    """
    if not stuck_records:
        return 0, 0

    client = db.get_client()

    matched = 0
    updated = 0
    chunk = 200

    # Track which stuck records were resolved so Path C skips them
    matched_inv_set: set[str] = set()
    matched_so_set:  set[str] = set()

    # ── Path A: join via invoice_number ───────────────────────────────────────
    # WMS stores Channel Order Code ("MH/26-27/XXXX") as invoice_number.
    # Stuck Orders has that same value in inv_agg.Invoice Numbers.
    # Use ORIGINAL case (no .lower()) so PostgREST case-sensitive match works.
    inv_records = [r for r in stuck_records if r.get("invoice_number")]
    if inv_records:
        # Split comma-separated invoice numbers so multi-invoice rows
        # (e.g. "MH/26-27/0121, MH/26-27/0143") match individual AWBs.
        inv_map: dict[str, dict] = {}
        for r in inv_records:
            for inv_num in r["invoice_number"].split(","):
                inv_num = inv_num.strip()
                if inv_num:
                    inv_map[inv_num] = r
        inv_list = list(inv_map.keys())

        for i in range(0, len(inv_list), chunk):
            batch = inv_list[i : i + chunk]
            resp = (
                client.table("awb_view")
                # Fetch extra fields needed for fuzzy fallback
                .select("awb, invoice_number, customer_name, drop_state, dispatch_date")
                .in_("invoice_number", batch)
                .execute()
            )
            rows = resp.data or []
            matched += len(rows)

            updates: list[dict] = []
            for row in rows:
                inv_key = row.get("invoice_number") or ""
                stuck = inv_map.get(inv_key)
                if not stuck:
                    continue
                patch: dict[str, Any] = {"awb": row["awb"]}
                # Write SO number, PO ref, expected ship date from stuck orders
                if stuck.get("so_number"):
                    patch["so_number"] = stuck["so_number"]
                if stuck.get("customer_po_ref"):
                    patch["customer_po_ref"] = stuck["customer_po_ref"]
                if stuck.get("expected_ship_date"):
                    patch["expected_ship_date"] = stuck["expected_ship_date"]
                if stuck.get("order_date"):
                    patch["order_date"] = stuck["order_date"]
                if len(patch) > 1:
                    updates.append(patch)

                # Fuzzy fallback: also try to find the correct B2B MIS AWB
                # for cases where WMS stored a wrong/old AWB for this invoice
                _try_fuzzy_update(client, row, stuck, updates)

            if updates:
                client.table("awb_view").upsert(updates, on_conflict="awb").execute()
                updated += len(updates)

            # Track which invoices were resolved
            for row in rows:
                matched_inv_set.add((row.get("invoice_number") or "").strip())

    # ── Path B: join via so_number ────────────────────────────────────────────
    # Covers AWBs where Courier MIS Reference Number = "NSO-MH/…" (SO format).
    # BUG FIX: use ORIGINAL case in the .in_() query (not .lower()) —
    # PostgREST IN filter is case-sensitive.
    so_records = [r for r in stuck_records if r.get("so_number")]
    if so_records:
        # Keep original case for DB query; lowercase for dict lookup
        so_map_lc: dict[str, dict] = {
            r["so_number"].lower(): r for r in so_records
        }
        so_list_orig = list({r["so_number"] for r in so_records})

        for i in range(0, len(so_list_orig), chunk):
            batch_sos = so_list_orig[i : i + chunk]
            resp = (
                client.table("awb_view")
                .select("awb, so_number, customer_po_ref, expected_ship_date")
                .in_("so_number", batch_sos)   # ← original case matches DB
                .execute()
            )
            rows = resp.data or []
            matched += len(rows)

            updates = []
            for row in rows:
                so_key = (row.get("so_number") or "").lower()
                stuck = so_map_lc.get(so_key)
                if not stuck:
                    continue
                patch = {"awb": row["awb"]}
                if stuck.get("invoice_number"):
                    patch["invoice_number"] = stuck["invoice_number"]
                if stuck.get("customer_po_ref"):
                    patch["customer_po_ref"] = stuck["customer_po_ref"]
                if stuck.get("expected_ship_date"):
                    patch["expected_ship_date"] = stuck["expected_ship_date"]
                if stuck.get("order_date"):
                    patch["order_date"] = stuck["order_date"]
                if len(patch) > 1:
                    updates.append(patch)

            if updates:
                client.table("awb_view").upsert(updates, on_conflict="awb").execute()
                updated += len(updates)

            # Track which SOs were resolved
            for row in rows:
                matched_so_set.add((row.get("so_number") or "").lower().strip())

    # ── Path B2: stuck.so_number → awb_view.invoice_number ──────────────────
    # Covers: WMS stored the order ref as Channel Order Code (→ invoice_number)
    # while stuck orders has the same value as its SO Number.
    # e.g. WMS Channel Order Code = "NSO-MH/2026/0049" → awb_view.invoice_number
    #      stuck orders SO Number  = "NSO-MH/2026/0049"
    so_via_inv_records = [
        r for r in stuck_records
        if r.get("so_number")
        and r["so_number"].lower() not in matched_so_set
    ]
    if so_via_inv_records:
        so_to_inv_map: dict[str, dict] = {
            r["so_number"]: r for r in so_via_inv_records
        }
        so_to_inv_list = list(so_to_inv_map.keys())

        for i in range(0, len(so_to_inv_list), chunk):
            batch = so_to_inv_list[i : i + chunk]
            resp = (
                client.table("awb_view")
                .select("awb, invoice_number")
                .in_("invoice_number", batch)
                .execute()
            )
            rows = resp.data or []
            matched += len(rows)

            updates = []
            for row in rows:
                inv_key = (row.get("invoice_number") or "").strip()
                stuck = so_to_inv_map.get(inv_key)
                if not stuck:
                    continue
                patch: dict[str, Any] = {"awb": row["awb"]}
                if stuck.get("so_number"):
                    patch["so_number"] = stuck["so_number"]
                if stuck.get("invoice_number"):
                    patch["invoice_number"] = stuck["invoice_number"]
                if stuck.get("customer_po_ref"):
                    patch["customer_po_ref"] = stuck["customer_po_ref"]
                if stuck.get("expected_ship_date"):
                    patch["expected_ship_date"] = stuck["expected_ship_date"]
                if stuck.get("order_date"):
                    patch["order_date"] = stuck["order_date"]
                if len(patch) > 1:
                    updates.append(patch)

            if updates:
                client.table("awb_view").upsert(updates, on_conflict="awb").execute()
                updated += len(updates)

            # Track resolved SO numbers so Path C skips them
            for row in rows:
                matched_so_set.add((row.get("invoice_number") or "").lower().strip())

    # ── Path C: join via customer_name + dispatch_date (fallback) ─────────────
    # For stuck records not matched by invoice or SO, try customer name +
    # dispatch date proximity.  Uses customer_name / dispatch_date parsed
    # from the stuck_orders file itself — safer than using WMS as an
    # intermediary.  Only writes when exactly 1 awb_view record matches.
    unmatched = [
        r for r in stuck_records
        if r.get("customer_name")
        and (r.get("dispatch_date") or r.get("order_date"))
        and (not r.get("invoice_number") or r["invoice_number"] not in matched_inv_set)
        and (not r.get("so_number") or r["so_number"].lower() not in matched_so_set)
    ]
    if unmatched:
        # Fetch all awb_view rows that still have no expected_ship_date in one shot
        try:
            mis_resp = (
                client.table("awb_view")
                .select("awb, customer_name, drop_state, dispatch_date")
                .is_("expected_ship_date", "null")
                .limit(5000)
                .execute()
            )
            mis_pool = mis_resp.data or []
        except Exception:
            mis_pool = []

        if mis_pool:
            used_awbs: set[str] = set()
            c_updates: list[dict] = []

            for stuck in unmatched:
                cust     = (stuck.get("customer_name") or "").strip()
                ref_date = stuck.get("dispatch_date") or stuck.get("order_date")

                first_word = ""
                for token in re.split(r"[\s\-_/]+", cust):
                    if token and token.lower() not in _COMPANY_SKIP and len(token) >= 3:
                        first_word = token.upper()
                        break
                if not first_word or not ref_date:
                    continue

                try:
                    ref_ts = pd.Timestamp(ref_date)
                except Exception:
                    continue

                candidates = []
                for mis in mis_pool:
                    mis_awb = mis.get("awb") or ""
                    if mis_awb in used_awbs:
                        continue
                    mis_cust = (mis.get("customer_name") or "").upper()
                    if first_word not in mis_cust:
                        continue
                    mis_disp = mis.get("dispatch_date")
                    if not mis_disp:
                        continue
                    try:
                        if abs((pd.Timestamp(mis_disp) - ref_ts).days) > 5:
                            continue
                    except Exception:
                        continue
                    candidates.append(mis)

                if len(candidates) != 1:
                    continue  # ambiguous or no match — skip

                mis_awb = candidates[0]["awb"]
                used_awbs.add(mis_awb)
                patch: dict[str, Any] = {"awb": mis_awb}
                if stuck.get("invoice_number"):
                    patch["invoice_number"] = stuck["invoice_number"]
                if stuck.get("so_number"):
                    patch["so_number"] = stuck["so_number"]
                if stuck.get("customer_po_ref"):
                    patch["customer_po_ref"] = stuck["customer_po_ref"]
                if stuck.get("expected_ship_date"):
                    patch["expected_ship_date"] = stuck["expected_ship_date"]
                if stuck.get("order_date"):
                    patch["order_date"] = stuck["order_date"]
                if len(patch) > 1:
                    c_updates.append(patch)

            if c_updates:
                for i in range(0, len(c_updates), chunk):
                    client.table("awb_view").upsert(
                        c_updates[i : i + chunk], on_conflict="awb"
                    ).execute()
                matched += len(c_updates)
                updated += len(c_updates)

    return matched, updated


# ── Main entry point ───────────────────────────────────────────────────────

class IngestResult:
    def __init__(
        self,
        filename: str,
        file_type: str,
        rows_processed: int,
        rows_inserted: int,
        rows_updated: int,
        ok: bool = True,
        message: str = "",
    ):
        self.filename = filename
        self.file_type = file_type
        self.rows_processed = rows_processed
        self.rows_inserted = rows_inserted
        self.rows_updated = rows_updated
        self.ok = ok
        self.message = message


def process_uploaded_file(uploaded_file) -> IngestResult:
    """
    Full pipeline: read → detect → parse → upsert → log.
    Returns an IngestResult describing what happened.
    """
    filename = uploaded_file.name

    try:
        df = read_file(uploaded_file)
    except Exception as e:
        return IngestResult(filename, "unknown", 0, 0, 0, ok=False,
                            message=f"Could not read file: {e}")

    if df.empty or len(df) < 1:
        return IngestResult(filename, "unknown", 0, 0, 0, ok=False,
                            message="File is empty or has no data rows.")

    headers = list(df.columns)
    file_type = detect_file_type(headers)

    if file_type == "unknown":
        sample = " | ".join(headers[:10])
        return IngestResult(
            filename, "unknown", 0, 0, 0, ok=False,
            message=(
                f"File type not recognised. Headers found: {sample}\n"
                "Expected one of: WMS Dispatch, Courier MIS, RPT Stuck Orders, Appt Config."
            ),
        )

    rows_processed = len(df)
    inserted = updated = 0

    try:
        if file_type == "wms_dispatch":
            records = ingest_wms(df)
            inserted, updated = db.upsert_awb_records(records)
            # Backfill SO#/PO Ref/Invoice#/Exp Ship from a previous Stuck
            # Orders upload, so these AWBs don't sit blank until Stuck
            # Orders happens to be re-uploaded.
            touched_awbs = [r["awb"] for r in records if r.get("awb")]
            db.backfill_awb_from_stuck_cache(touched_awbs)

        elif file_type == "courier_tracking":
            records = ingest_courier_mis(df)
            inserted, updated = db.upsert_awb_records(records)
            touched_awbs = [r["awb"] for r in records if r.get("awb")]
            db.backfill_awb_from_stuck_cache(touched_awbs)

        elif file_type == "stuck_orders":
            stuck = ingest_stuck_orders(df)
            # Persist SO# data to cache so manual mappings can look up
            # expected_ship_date later without requiring a re-upload.
            db.upsert_stuck_orders_cache(stuck)
            matched, updated = apply_stuck_orders_to_db(stuck)
            inserted = 0   # stuck orders never insert new AWBs

        elif file_type == "appt_config":
            appt_records = ingest_appt_config(df)
            count = db.upsert_appt_config(appt_records)
            inserted = count

    except Exception as e:
        db.log_upload(filename, file_type, rows_processed, 0, 0,
                      status="error", error_message=str(e))
        return IngestResult(filename, file_type, rows_processed, 0, 0,
                            ok=False, message=f"Ingest error: {e}")

    db.log_upload(filename, file_type, rows_processed, inserted, updated)
    return IngestResult(filename, file_type, rows_processed, inserted, updated,
                        ok=True,
                        message=f"{inserted} inserted, {updated} updated")
