"""
app.py — Operations Control Tower (Streamlit)

Tabs:
  📦 Dashboard      — KPI filter pills + shipment tracker (sidebar filters)
  ⬆️ Upload         — CSV/Excel ingest with per-file result
  🔍 Data Sanity    — Missing fields, cross-file consistency
  ⚙️ Settings       — Appointment config management, reset, upload log
  🔗 Manual Mapping — Manual AWB→SO# for direct-booked shipments
"""

import time
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import db
import ingest
import kpis
import sanity


# ── Cached-file wrapper ─────────────────────────────────────────────────────

class _CachedFile:
    """Minimal file-like object that re-presents stored bytes."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def read(self) -> bytes:
        return self._data


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Operations Control Tower",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #F8F9FA;
    border-right: 1px solid #E8EAED;
    padding-top: 8px;
}
[data-testid="stSidebar"] .block-container { padding-top: 12px; }

/* ── Header ──────────────────────────────────────────────────────────────── */
.oct-header {
    background: linear-gradient(135deg, #1A73E8, #0D47A1);
    color: white;
    padding: 14px 24px;
    border-radius: 10px;
    margin-bottom: 14px;
}
.oct-header h1 { font-size: 20px; margin: 0; font-weight: 700; }
.oct-header .sub { font-size: 11px; opacity: .75; margin-top: 2px; }

/* ── KPI pill buttons ────────────────────────────────────────────────────── */
/* Target buttons inside the horizontal block that holds the pills */
div[data-testid="stHorizontalBlock"] > div > div > div > .stButton > button {
    border-radius: 22px !important;
    font-size: 13px !important;
    padding: 7px 10px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all .15s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.07) !important;
    line-height: 1.4 !important;
}
div[data-testid="stHorizontalBlock"] > div > div > div > .stButton > button[kind="secondary"] {
    background: white !important;
    border: 1.5px solid #E0E0E0 !important;
    color: #3C4043 !important;
}
div[data-testid="stHorizontalBlock"] > div > div > div > .stButton > button[kind="secondary"]:hover {
    border-color: #1A73E8 !important;
    color: #1A73E8 !important;
    background: #F4F8FF !important;
}
div[data-testid="stHorizontalBlock"] > div > div > div > .stButton > button[kind="primary"] {
    background: #E8F0FE !important;
    border: 2px solid #1A73E8 !important;
    color: #1558D6 !important;
}

/* ── Status badges ───────────────────────────────────────────────────────── */
.badge { display:inline-block; padding:2px 8px; border-radius:10px;
         font-size:10px; font-weight:700; white-space:nowrap; }
.badge-green  { background:#E6F4EA; color:#137333; }
.badge-blue   { background:#E8F0FE; color:#1558D6; }
.badge-red    { background:#FCE8E6; color:#C5221F; }
.badge-orange { background:#FFF0E0; color:#B06000; }
.badge-yellow { background:#FEF7E0; color:#AA8000; }
.badge-grey   { background:#F1F3F4; color:#5F6368; }
.badge-purple { background:#F3E8FD; color:#7B1FA2; }

/* ── Table ───────────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] table { font-size:11px !important; }
div[data-testid="stDataFrame"] th {
    background:#F8F9FA !important;
    font-weight:600 !important;
    color:#5F6368 !important;
}

/* ── Metric ──────────────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: white;
    border-radius: 8px;
    padding: 10px 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}

/* ── Divider spacing ─────────────────────────────────────────────────────── */
hr { margin: 10px 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_date(val) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "–"
    try:
        ts = pd.Timestamp(val)
        if pd.isna(ts):
            return "–"
        return ts.strftime("%d %b %y")
    except Exception:
        return "–"


@st.cache_data(ttl=120, show_spinner=False)
def _cached_load():
    raw = db.load_awb_view()
    appt_config = db.load_appt_config()
    try:
        groups = db.load_customer_groups()
    except AttributeError:
        groups = {}  # db.py not yet updated — degrade gracefully
    return raw, appt_config, groups


def load_data(force: bool = False):
    if force:
        _cached_load.clear()
    raw, appt_config, groups = _cached_load()
    if raw.empty:
        return pd.DataFrame(), appt_config, {}, groups
    df = kpis.build_display_rows(raw, appt_config)
    kpi_vals = kpis.compute_kpis(df)
    return df, appt_config, kpi_vals, groups


# ── Time filter ──────────────────────────────────────────────────────────────

_TIME_OPTIONS = {
    "This Month":     0,
    "Last 3 Months":  3,
    "Last 6 Months":  6,
    "Last 12 Months": 12,
    "All Time":       None,
    "Custom Range":   "custom",
}


def _apply_time_filter(
    df: pd.DataFrame,
    label: str,
    date_from: date | None = None,
    date_to: date | None = None,
) -> pd.DataFrame:
    if df.empty or "dispatch_date" not in df.columns:
        return df
    mask_no_date = df["dispatch_date"].isna()

    if label == "Custom Range":
        if date_from is None and date_to is None:
            return df
        lo = pd.Timestamp(date_from) if date_from else pd.Timestamp.min
        hi = pd.Timestamp(date_to)   if date_to   else pd.Timestamp.max
        return df[mask_no_date | ((df["dispatch_date"] >= lo) & (df["dispatch_date"] <= hi))]

    months = _TIME_OPTIONS.get(label)
    if months is None:
        return df

    now = pd.Timestamp.now().normalize()
    cutoff = now.replace(day=1) if months == 0 else now.replace(day=1) - pd.DateOffset(months=months)
    return df[mask_no_date | (df["dispatch_date"] >= cutoff)]


# ── Render AWB table ─────────────────────────────────────────────────────────

def render_awb_table(df: pd.DataFrame, key_suffix: str = "main") -> None:
    if df.empty:
        st.info("No shipments match the current filters.")
        return

    active    = df[~df["is_delivered"]].sort_values("days_old", ascending=False)
    delivered = df[df["is_delivered"]].sort_values("days_old", ascending=False)
    combined  = pd.concat([active, delivered], ignore_index=True)

    if not active.empty and not delivered.empty:
        st.caption(f"{len(active)} active · {len(delivered)} delivered")
    else:
        st.caption(f"{len(combined)} shipments")

    display_rows = []
    for _, row in combined.iterrows():
        awb    = row.get("awb", "")
        url    = row.get("track_url", "#")
        is_del = row.get("is_delivered", False)
        d_old  = int(row.get("days_old", 0) or 0)

        # Time to deliver: delivery_date − order_date (fallback: dispatch_date)
        ttd = None
        if is_del:
            try:
                def _safe_ts_local(v):
                    if v is None: return None
                    ts = pd.Timestamp(v)
                    return None if pd.isna(ts) else ts
                _od = _safe_ts_local(row.get("order_date")) or _safe_ts_local(row.get("dispatch_date"))
                _dd = _safe_ts_local(row.get("delivery_date"))
                if _od is not None and _dd is not None:
                    ttd = int((_dd - _od).days)
            except Exception:
                pass

        remark = str(row.get("latest_remark") or "")
        remark_short = remark[:40] + "…" if len(remark) > 40 else remark

        _pod_raw = row.get("pod_url")
        pod_cell = str(_pod_raw).strip() if _pod_raw and str(_pod_raw).startswith("http") else ""
        awb_cell = url if (awb and url and url.startswith("http")) else ""

        display_rows.append({
            "SO #":          row.get("so_number") or "–",
            "PO Ref":        row.get("customer_po_ref") or "–",
            "Invoice #":     row.get("invoice_number") or "–",
            "AWB":           awb_cell,
            "Customer":      row.get("customer_name") or "–",
            "City":          row.get("drop_city") or "–",
            "State":         row.get("drop_state") or "–",
            "Transporter":   row.get("transporter") or "–",
            "Order Date":    fmt_date(row.get("order_date")),
            "Dispatch":      fmt_date(row.get("dispatch_date")),
            "Days Old":      d_old,
            "Exp Ship":      fmt_date(row.get("expected_ship_date")),
            "Status":        row.get("status") or "–",
            "Del Date":      fmt_date(row.get("delivery_date")),
            "EDD":           fmt_date(row.get("estimated_delivery_date")),
            "Variance":      row.get("var_str") or "–",
            "Appt":          row.get("appt_status") or "–",
            "Appt Date":     fmt_date(row.get("appointment_date")),
            "POD":              pod_cell,
            "Time to Deliver":  ttd,
            "Latest Remark":    remark_short,
        })

    disp_df = pd.DataFrame(display_rows)

    # ── Row colours by status + stuck-days highlight ──────────────────────
    _STATUS_BG = {
        "delivered":   "#EAFAF1",   # soft green
        "undelivered": "#FEF9E7",   # soft yellow
        "rto":         "#FEF5E7",   # soft orange
        "out_for":     "#EBF5FB",   # soft blue
        "cancelled":   "#FDEDEC",   # soft red
        "manifested":  "#F8F9FA",   # light grey
        "default":     "#FFFFFF",
    }

    def _classify(status: str) -> str:
        s = (status or "").upper().strip()
        if "CANCEL" in s:                        return "cancelled"
        if "UNDELIVER" in s or "FAIL" in s:      return "undelivered"
        if "RTO" in s or "RETURN" in s:          return "rto"
        if "OUT FOR" in s or "OUT_FOR" in s:     return "out_for"
        if "DELIVER" in s:                       return "delivered"
        if "MANIFEST" in s:                      return "manifested"
        return "default"

    col_idx = {col: i for i, col in enumerate(disp_df.columns)}

    def _row_style(row: pd.Series) -> list[str]:
        cls = _classify(row.get("Status", ""))
        bg  = _STATUS_BG[cls]
        styles = [f"background-color: {bg}"] * len(row)
        # Bold red for Days Old > 14 on non-delivered rows
        if cls not in ("delivered", "cancelled"):
            d = row.get("Days Old", 0)
            if isinstance(d, (int, float)) and int(d) > 14:
                i = col_idx.get("Days Old", -1)
                if i >= 0:
                    styles[i] = f"background-color: {bg}; color: #C5221F; font-weight: 700"
        return styles

    styled = disp_df.style.apply(_row_style, axis=1)

    # ── Column widths ─────────────────────────────────────────────────────
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=560,
        column_config={
            "SO #":          st.column_config.TextColumn("SO #",          width=165),
            "PO Ref":        st.column_config.TextColumn("PO Ref",        width=120),
            "Invoice #":     st.column_config.TextColumn("Invoice #",     width=155),
            "AWB":           st.column_config.LinkColumn("AWB",           display_text=r"[?&]t=([^&]+)", width=140),
            "Customer":      st.column_config.TextColumn("Customer",      width=195),
            "City":          st.column_config.TextColumn("City",          width=105),
            "State":         st.column_config.TextColumn("State",         width=105),
            "Transporter":   st.column_config.TextColumn("Transporter",   width=120),
            "Order Date":    st.column_config.TextColumn("Order Date",    width=90),
            "Dispatch":      st.column_config.TextColumn("Dispatch",      width=90),
            "Days Old":      st.column_config.NumberColumn("Days Old",    width=78,  format="%d d"),
            "Exp Ship":      st.column_config.TextColumn("Exp Ship",      width=90),
            "Status":        st.column_config.TextColumn("Status",        width=135),
            "Del Date":      st.column_config.TextColumn("Del Date",      width=90),
            "EDD":           st.column_config.TextColumn("EDD",           width=90),
            "Variance":      st.column_config.TextColumn("Variance",      width=80),
            "Appt":          st.column_config.TextColumn("Appt",          width=105),
            "Appt Date":     st.column_config.TextColumn("Appt Date",     width=100),
            "POD":              st.column_config.LinkColumn("POD",              display_text="📄", width=55),
            "Time to Deliver":  st.column_config.NumberColumn("Time to Deliver",  width=115, format="%d d"),
            "Latest Remark":    st.column_config.TextColumn("Latest Remark",      width=210),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def tab_dashboard(filtered: pd.DataFrame, kpi_vals: dict) -> None:
    """KPI filter pills → shipment tracker. No expanders, no variance table."""

    # ── KPI state ─────────────────────────────────────────────────────────
    if "active_kpi" not in st.session_state:
        st.session_state["active_kpi"] = "all"

    _del7_fn = getattr(kpis, "cohort_delivered_last_7", None)
    _edd7_fn = getattr(kpis, "cohort_edd_next_7", None)

    kpi_defs = [
        ("all",     "All",              None,                        kpi_vals.get("total_active",    0)),
        ("stuck",   "🚨 Stuck 14d+",    kpis.cohort_stuck,           kpi_vals.get("stuck",           0)),
        ("appt",    "📋 Pending Appt",  kpis.cohort_pending_appt,    kpi_vals.get("pending_appt",    0)),
        ("deltoday","✅ Del'd Today",   kpis.cohort_delivered_today, kpi_vals.get("delivered_today", 0)),
        ("edd",     "⏰ EDD Breached",  kpis.cohort_edd_breached,    kpi_vals.get("edd_breached",    0)),
        ("del7",    "📦 Del'd Last 7d", _del7_fn,                    kpi_vals.get("del_last_7",      0)),
        ("edd7",    "📅 EDD Next 7d",   _edd7_fn,                    kpi_vals.get("edd_next_7",      0)),
    ]

    active_key = st.session_state.get("active_kpi", "all")

    # ── Render pill row ───────────────────────────────────────────────────
    pill_cols = st.columns(len(kpi_defs))
    for col, (key, label, _, count) in zip(pill_cols, kpi_defs):
        with col:
            if st.button(
                f"{label}  ·  {count}",
                key=f"kpi_{key}",
                use_container_width=True,
                type="primary" if active_key == key else "secondary",
            ):
                st.session_state["active_kpi"] = key
                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Avg TTD + Avg OTD metric cards ───────────────────────────────────
    avg_ttd_val = kpi_vals.get("avg_ttd")
    del_count   = kpi_vals.get("delivered_count", 0)
    ttd_display = f"{avg_ttd_val} d" if avg_ttd_val is not None else "–"

    avg_otd_val = kpi_vals.get("avg_otd")
    otd_count   = kpi_vals.get("otd_count", 0)
    otd_display = f"{avg_otd_val} d" if avg_otd_val is not None else "–"

    _m1, _m2, _mspc = st.columns([1, 1, 4])
    with _m1:
        st.metric(
            label="⏱️ Avg Time to Deliver",
            value=ttd_display,
            delta=f"{del_count} delivered total",
            delta_color="off",
        )
    with _m2:
        st.metric(
            label="📦 Avg Order → Dispatch",
            value=otd_display,
            delta=f"{otd_count} shipments",
            delta_color="off",
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Apply KPI cohort filter ───────────────────────────────────────────
    fn_map = {
        "all":      None,
        "stuck":    kpis.cohort_stuck,
        "appt":     kpis.cohort_pending_appt,
        "deltoday": kpis.cohort_delivered_today,
        "edd":      kpis.cohort_edd_breached,
        "del7":     _del7_fn,
        "edd7":     _edd7_fn,
    }
    fn = fn_map.get(active_key)
    table_df = fn(filtered) if (fn is not None and not filtered.empty) else filtered

    render_awb_table(table_df, key_suffix="dashboard")

    # ── TTD Charts ───────────────────────────────────────────────────────
    try:
        state_df = kpis.ttd_by_dimension(filtered, "drop_state",    "State")
        cust_df  = kpis.ttd_by_dimension(filtered, "customer_name", "Customer")
    except AttributeError:
        state_df = cust_df = pd.DataFrame()  # kpis.py not yet updated

    st.markdown("---")
    st.markdown("#### ⏱️ Avg Time to Deliver Breakdown")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.caption("By State")
        if state_df.empty:
            st.info("No delivered shipments with date data for the current filter.")
        else:
            chart_s = (
                alt.Chart(state_df)
                .mark_bar(color="#1A73E8", cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Avg TTD (d):Q", title="Avg TTD (days)", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("State:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("State:N"),
                        alt.Tooltip("Avg TTD (d):Q", title="Avg TTD", format=".1f"),
                    ],
                )
                .properties(height=max(160, len(state_df) * 30))
            )
            st.altair_chart(chart_s, use_container_width=True)

    with ch2:
        st.caption("By Customer")
        if cust_df.empty:
            st.info("No delivered shipments with date data for the current filter.")
        else:
            chart_c = (
                alt.Chart(cust_df)
                .mark_bar(color="#34A853", cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Avg TTD (d):Q", title="Avg TTD (days)", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("Customer:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("Customer:N"),
                        alt.Tooltip("Avg TTD (d):Q", title="Avg TTD", format=".1f"),
                    ],
                )
                .properties(height=max(160, len(cust_df) * 30))
            )
            st.altair_chart(chart_c, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

def _run_ingest(file_objects: list) -> None:
    results = []
    progress = st.progress(0)
    for i, uf in enumerate(file_objects):
        with st.spinner(f"Processing {uf.name}…"):
            result = ingest.process_uploaded_file(uf)
        results.append(result)
        progress.progress((i + 1) / len(file_objects))

    progress.empty()
    st.markdown("#### Results")
    all_ok = True
    for r in results:
        if r.ok:
            st.success(
                f"✅ **{r.filename}** ({r.file_type}) — "
                f"{r.rows_processed} rows · {r.rows_inserted} inserted · {r.rows_updated} updated"
            )
        else:
            all_ok = False
            st.error(f"❌ **{r.filename}** — {r.message}")

    if all_ok:
        st.info("All files ingested. Dashboard data will refresh automatically.")
        _cached_load.clear()
        time.sleep(1)
        st.rerun()


def tab_upload() -> None:
    st.markdown("### ⬆️ Upload Files")
    st.markdown(
        "Upload one or more files. File type is auto-detected from column headers.\n\n"
        "**Supported:** WMS Dispatch · B2B Courier MIS · RPT Stuck Orders · "
        "Appointment Config  |  **Formats:** CSV, Excel (.xlsx / .xls)"
    )

    cached = st.session_state.get("last_upload_cache")
    if cached:
        cached_names = ", ".join(c["name"] for c in cached)
        st.info(f"**Last ingested:** {cached_names}")
        if st.button("🔄 Re-run Last Ingest", key="rerun_btn"):
            file_objects = [_CachedFile(c["name"], c["data"]) for c in cached]
            _run_ingest(file_objects)
            return

    st.markdown("---")

    uploaded = st.file_uploader(
        "Drop files here or click to browse",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="uploader",
    )

    if not uploaded:
        st.caption("No files selected yet.")
        return

    st.markdown(f"**{len(uploaded)} file(s) ready**")
    if st.button("🚀 Ingest All Files", type="primary"):
        file_cache = []
        file_objects = []
        for uf in uploaded:
            raw = uf.read()
            file_cache.append({"name": uf.name, "data": raw})
            file_objects.append(_CachedFile(uf.name, raw))
        st.session_state["last_upload_cache"] = file_cache
        _run_ingest(file_objects)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: DATA SANITY
# ══════════════════════════════════════════════════════════════════════════════

def tab_sanity(df: pd.DataFrame) -> None:
    st.markdown("### 🔍 Data Sanity Checks")
    if df.empty:
        st.info("No data loaded. Upload files first.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Missing Critical Fields")
        st.caption(
            "AWBs where one or more of: AWB, Status, Customer Name, Drop City, "
            "Drop State, Order Date, EDD, Transporter are blank."
        )
        missing_df = sanity.check_missing_fields(df)
        if missing_df.empty:
            st.success("✅ No missing critical fields found.")
        else:
            st.warning(f"⚠️ {len(missing_df)} AWBs have missing critical fields")
            st.dataframe(missing_df, use_container_width=True, hide_index=True, height=350)

    with col_right:
        st.markdown("#### Cross-File Consistency")
        age_threshold = st.number_input(
            "Flag AWBs dispatched >N days ago with no Courier MIS status",
            min_value=1, max_value=30, value=3, step=1, key="cross_age",
        )
        st.caption(
            "AWBs in WMS Dispatch but Courier MIS has never reported a status."
        )
        cross_df = sanity.check_cross_file_consistency(df, min_dispatch_age_days=int(age_threshold))
        if cross_df.empty:
            st.success("✅ No cross-file consistency issues found.")
        else:
            st.warning(f"⚠️ {len(cross_df)} AWBs dispatched but missing from Courier MIS")
            st.dataframe(cross_df, use_container_width=True, hide_index=True, height=350)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def tab_settings(df: pd.DataFrame, groups: dict) -> None:
    st.markdown("### ⚙️ Settings")

    # ── Customer Groups ───────────────────────────────────────────────────
    with st.expander("👥 Customer Groups", expanded=True):
        st.markdown(
            "Group customer names so you can filter the dashboard by group via the sidebar."
        )

        if groups:
            group_rows = [
                {"Group": g, "Customers": ", ".join(members)}
                for g, members in groups.items()
            ]
            st.dataframe(
                pd.DataFrame(group_rows),
                use_container_width=True, hide_index=True,
                height=min(220, len(groups) * 36 + 40),
            )
        else:
            st.caption("No groups defined yet.")

        st.markdown("**Add customer to group:**")
        ag1, ag2, ag3 = st.columns([2, 3, 1])
        with ag1:
            grp_name = st.text_input(
                "Group name (new or existing)",
                placeholder="e.g. Key Accounts",
                key="grp_name_input",
            )
        with ag2:
            cust_opts = []
            if not df.empty and "customer_name" in df.columns:
                cust_opts = sorted(set(
                    str(v).strip() for v in df["customer_name"].dropna().unique()
                    if str(v).strip() and str(v).strip() != "–"
                ))
            grp_cust = st.selectbox(
                "Customer", ["— select —"] + cust_opts, key="grp_cust_sel"
            )
        with ag3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add", key="grp_add_btn", type="primary"):
                if grp_name.strip() and grp_cust != "— select —":
                    db.add_customer_to_group(grp_name.strip(), grp_cust)
                    st.success(f"Added **{grp_cust}** to group '{grp_name.strip()}'")
                    _cached_load.clear()
                    st.rerun()
                else:
                    st.error("Enter a group name and select a customer.")

        if groups:
            st.markdown("---")
            st.markdown("**Remove customer from group:**")
            rg1, rg2, rg3 = st.columns([2, 3, 1])
            with rg1:
                del_group = st.selectbox(
                    "Group", ["— select —"] + list(groups.keys()), key="grp_del_group"
                )
            with rg2:
                del_cust_opts = groups.get(del_group, []) if del_group != "— select —" else []
                del_cust = st.selectbox(
                    "Customer", ["— select —"] + del_cust_opts, key="grp_del_cust"
                )
            with rg3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Remove", key="grp_rem_btn"):
                    if del_group != "— select —" and del_cust != "— select —":
                        db.remove_customer_from_group(del_group, del_cust)
                        st.success(f"Removed **{del_cust}** from '{del_group}'")
                        _cached_load.clear()
                        st.rerun()

            st.markdown("**Delete entire group:**")
            dg1, dg2 = st.columns([4, 1])
            with dg1:
                del_full_grp = st.selectbox(
                    "Group to delete", ["— select —"] + list(groups.keys()),
                    key="grp_del_full"
                )
            with dg2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Delete", key="grp_del_full_btn"):
                    if del_full_grp != "— select —":
                        db.delete_customer_group(del_full_grp)
                        st.success(f"Deleted group '{del_full_grp}'")
                        _cached_load.clear()
                        st.rerun()

    with st.expander("📋 Appointment Config", expanded=True):
        st.markdown("Define which customers require a delivery appointment.")
        appt_cfg = db.load_appt_config()
        if appt_cfg:
            appt_display = pd.DataFrame([
                {"Customer Name": k, "Appointment Required": "Yes" if v else "No"}
                for k, v in sorted(appt_cfg.items())
            ])
            st.dataframe(appt_display, use_container_width=True, hide_index=True, height=200)
        else:
            st.caption("No appointment config entries yet.")

        st.markdown("**Add / update entry:**")
        ac1, ac2, ac3 = st.columns([3, 1, 1])
        with ac1:
            new_cust = st.text_input("Customer Name", key="appt_cust")
        with ac2:
            new_req = st.selectbox("Required?", ["Yes", "No"], key="appt_req")
        with ac3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Save", key="appt_save") and new_cust.strip():
                db.upsert_appt_config([{
                    "customer_name": new_cust.strip().lower(),
                    "appointment_required": new_req == "Yes",
                }])
                st.success(f"Saved: {new_cust.strip()}")
                st.rerun()

        if appt_cfg:
            del_cust = st.selectbox(
                "Delete entry", ["— select —"] + list(sorted(appt_cfg.keys())),
                key="appt_del_sel",
            )
            if st.button("🗑️ Delete", key="appt_del_btn") and del_cust != "— select —":
                db.delete_appt_config_entry(del_cust)
                st.success(f"Deleted: {del_cust}")
                st.rerun()

    with st.expander("📜 Upload History", expanded=False):
        log_df = db.load_upload_log(50)
        if log_df.empty:
            st.caption("No uploads recorded yet.")
        else:
            if "uploaded_at" in log_df.columns:
                log_df["uploaded_at"] = (
                    pd.to_datetime(log_df["uploaded_at"], errors="coerce")
                    .dt.strftime("%d %b %y %H:%M")
                )
            st.dataframe(
                log_df[[
                    "uploaded_at", "filename", "file_type",
                    "rows_processed", "rows_inserted", "rows_updated",
                    "upload_status", "error_message",
                ]].rename(columns={
                    "uploaded_at":    "Time",
                    "filename":       "File",
                    "file_type":      "Type",
                    "rows_processed": "Rows",
                    "rows_inserted":  "Inserted",
                    "rows_updated":   "Updated",
                    "upload_status":  "Status",
                    "error_message":  "Error",
                }),
                use_container_width=True, hide_index=True, height=300,
            )

    with st.expander("🔄 Reset / Danger Zone", expanded=False):
        st.warning("**Reset AWB View** deletes ALL shipment records. Re-upload your files afterwards.")
        confirm = st.text_input('Type "RESET" to confirm', placeholder="RESET", key="reset_confirm")
        if st.button("🔄 Reset AWB View", type="secondary", key="reset_btn"):
            if confirm.strip().upper() == "RESET":
                db.reset_awb_view()
                _cached_load.clear()
                st.success("✅ AWB View cleared. Please re-upload your files.")
                time.sleep(1)
                st.rerun()
            else:
                st.error('Please type "RESET" to confirm.')


# ══════════════════════════════════════════════════════════════════════════════
# TAB: MANUAL SO MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def tab_manual_mapping(df: pd.DataFrame) -> None:
    st.markdown("### 🔗 Manual SO Mapping")
    st.markdown(
        "For AWBs booked **directly on the courier portal** (no WMS row), "
        "the automated SO# join can't work. Enter the mapping here manually. "
        "Click **Apply** to push the values into the main AWB tracker."
    )

    if not db._mapping_table_exists():
        st.error(
            "**Setup required:** the `awb_so_mapping` table does not exist yet in Supabase.\n\n"
            "Run the SQL migration once in **Supabase → SQL Editor**, then refresh this page:"
        )
        st.code(
            """CREATE TABLE IF NOT EXISTS awb_so_mapping (
    awb             TEXT PRIMARY KEY,
    so_number       TEXT,
    invoice_number  TEXT,
    customer_po_ref TEXT,
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);""",
            language="sql",
        )
        return

    with st.expander("📋 AWBs with no SO# (candidates)", expanded=True):
        if df.empty:
            st.info("No data loaded yet.")
        else:
            no_so = df[
                df["so_number"].isna()
                | (df["so_number"].str.strip() == "–")
                | (df["so_number"].str.strip() == "")
            ][["awb", "customer_name", "drop_city", "drop_state",
               "transporter", "dispatch_date", "status"]].copy()
            no_so["dispatch_date"] = no_so["dispatch_date"].apply(fmt_date)
            no_so = no_so.rename(columns={
                "awb": "AWB", "customer_name": "Customer",
                "drop_city": "City", "drop_state": "State",
                "transporter": "Transporter", "dispatch_date": "Dispatch",
                "status": "Status",
            })
            if no_so.empty:
                st.success("✅ All AWBs have a SO# — nothing to map.")
            else:
                st.caption(f"{len(no_so)} AWBs without SO#")
                st.dataframe(no_so, use_container_width=True, hide_index=True, height=260)

    st.markdown("---")
    st.markdown("#### ➕ Add / Update Mapping")
    m1, m2, m3, m4, m5 = st.columns([2, 2, 2, 2, 1])
    with m1:
        map_awb = st.text_input("AWB *", placeholder="e.g. 100036186384", key="map_awb")
    with m2:
        map_so = st.text_input("SO # *", placeholder="e.g. NSO-MH/2026/0224", key="map_so")
    with m3:
        map_inv = st.text_input("Invoice # (optional)", placeholder="e.g. MH/26-27/0210", key="map_inv")
    with m4:
        map_po = st.text_input("PO Ref (optional)", placeholder="e.g. PO-12345", key="map_po")
    with m5:
        st.markdown("<br>", unsafe_allow_html=True)
        save_clicked = st.button("💾 Save", key="map_save_btn", type="primary")

    if save_clicked:
        awb_val = map_awb.strip()
        so_val  = map_so.strip()
        if not awb_val:
            st.error("AWB is required.")
        elif not so_val:
            st.error("SO # is required.")
        else:
            record = {"awb": awb_val, "so_number": so_val}
            if map_inv.strip():
                record["invoice_number"] = map_inv.strip()
            if map_po.strip():
                record["customer_po_ref"] = map_po.strip()
            db.upsert_awb_so_mapping([record])
            st.success(f"✅ Saved mapping: {awb_val} → {so_val}")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📝 Saved Mappings")
    mapping_df = db.load_awb_so_mapping()

    if mapping_df.empty:
        st.caption("No mappings saved yet.")
    else:
        display_cols = ["awb", "so_number", "invoice_number", "customer_po_ref"]
        show_df = mapping_df[[c for c in display_cols if c in mapping_df.columns]].rename(columns={
            "awb": "AWB", "so_number": "SO #",
            "invoice_number": "Invoice #", "customer_po_ref": "PO Ref",
        })
        st.dataframe(show_df, use_container_width=True, hide_index=True, height=220)

        del_awb = st.selectbox(
            "Delete mapping",
            ["— select AWB —"] + list(mapping_df["awb"].tolist()),
            key="map_del_sel",
        )
        if st.button("🗑️ Delete", key="map_del_btn") and del_awb != "— select AWB —":
            db.delete_awb_so_mapping(del_awb)
            st.success(f"Deleted mapping for {del_awb}")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🚀 Apply Mappings")
    st.caption(
        "Pushes all saved mappings into the main AWB tracker, including Exp Ship "
        "from the last Stuck Orders upload. Safe to run multiple times."
    )
    if st.button("▶️ Apply All Mappings to AWB Tracker", type="primary", key="map_apply_btn"):
        with st.spinner("Applying…"):
            n = db.apply_manual_so_mapping()
        if n == 0:
            st.warning("No mappings to apply (table is empty or all patches are blank).")
        else:
            st.success(f"✅ Applied {n} mapping(s) to AWB tracker.")
            _cached_load.clear()
            time.sleep(1)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    now_str = pd.Timestamp.now().strftime("%d %b %Y %H:%M")

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="oct-header">
          <div>
            <h1>🏗️ Operations Control Tower</h1>
            <div class="sub">Last refresh: {now_str} IST</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        df, appt_config, kpi_vals, groups = load_data()

    # ── Sidebar — Part 1: time window only ───────────────────────────────
    with st.sidebar:
        st.markdown("## 🏗️ OCT")
        st.caption(f"Refreshed {now_str}")

        if st.button("🔄 Refresh", use_container_width=True, key="sidebar_refresh"):
            load_data(force=True)
            st.rerun()

        st.divider()

        st.markdown("**📅 Time Window**")
        time_sel = st.selectbox(
            "_tw",
            options=list(_TIME_OPTIONS.keys()),
            index=1,
            key="time_window",
            label_visibility="collapsed",
        )
        date_from = date_to = None
        if time_sel == "Custom Range":
            today = date.today()
            date_from = st.date_input("From", value=today.replace(day=1), key="custom_from")
            date_to   = st.date_input("To",   value=today,                key="custom_to")

    # ── Apply time filter ─────────────────────────────────────────────────
    df = _apply_time_filter(df, time_sel, date_from, date_to)
    kpi_vals = kpis.compute_kpis(df) if not df.empty else {}

    # ── Build unique option lists from time-filtered data ─────────────────
    all_states       = sorted(df["drop_state"].dropna().unique().tolist())  if not df.empty else []
    all_transporters = sorted(df["transporter"].dropna().unique().tolist()) if not df.empty else []
    all_statuses     = sorted(df["status"].dropna().unique().tolist())      if not df.empty else []

    _search_col_map = {
        "customer": "customer_name",
        "awb":      "awb",
        "so":       "so_number",
        "invoice":  "invoice_number",
    }
    _search_labels = {
        "customer": "Customer",
        "awb":      "AWB",
        "so":       "SO #",
        "invoice":  "Invoice #",
    }

    # ── Sidebar — Part 2: group + search + dimension filters ─────────────
    with st.sidebar:
        st.divider()

        # ── Customer Group filter ─────────────────────────────────────────
        st.markdown("**👥 Customer Group**")
        if groups:
            group_sel = st.multiselect(
                "_grp",
                options=list(groups.keys()),
                default=[],
                key="group_sel",
                label_visibility="collapsed",
                placeholder="All groups",
            )
        else:
            st.caption("No groups — add in ⚙️ Settings.")
            group_sel = []

        st.divider()

        st.markdown("**🔍 Search**")
        search_field = st.selectbox(
            "_sf",
            options=list(_search_col_map.keys()),
            format_func=lambda x: _search_labels[x],
            key="search_field",
            label_visibility="collapsed",
        )

        # Options for the selected search field
        _scol = _search_col_map[search_field]
        if not df.empty:
            _raw_opts = df[_scol].dropna().unique().tolist()
            search_options = sorted(
                [str(v).strip() for v in _raw_opts if str(v).strip() and str(v).strip() != "–"],
            )
        else:
            search_options = []

        search_vals = st.multiselect(
            f"Select {_search_labels[search_field]}",
            options=search_options,
            default=[],
            key=f"search_vals_{search_field}",
            placeholder=f"All {_search_labels[search_field]}s",
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("**📍 State**")
        state_sel = st.multiselect(
            "_s", all_states, default=all_states, key="state_sel",
            label_visibility="collapsed",
        )

        st.markdown("**🚛 Transporter**")
        tp_sel = st.multiselect(
            "_t", all_transporters, default=all_transporters, key="tp_sel",
            label_visibility="collapsed",
        )

        st.markdown("**📊 Status**")
        status_sel = st.multiselect(
            "_st", all_statuses, default=all_statuses, key="status_sel",
            label_visibility="collapsed",
        )

    # ── Apply dimension + search filters ─────────────────────────────────
    filtered = df.copy() if not df.empty else pd.DataFrame()
    if not filtered.empty:
        # Group filter (expands to customer names in the selected groups)
        if group_sel:
            group_customers: set[str] = set()
            for g in group_sel:
                group_customers.update(groups.get(g, []))
            if group_customers:
                filtered = filtered[filtered["customer_name"].isin(group_customers)]

        if state_sel and set(state_sel) != set(all_states):
            filtered = filtered[filtered["drop_state"].isin(state_sel)]
        if tp_sel and set(tp_sel) != set(all_transporters):
            filtered = filtered[filtered["transporter"].isin(tp_sel)]
        if status_sel and set(status_sel) != set(all_statuses):
            filtered = filtered[filtered["status"].isin(status_sel)]
        if search_vals:
            col = _search_col_map[search_field]
            filtered = filtered[filtered[col].fillna("").astype(str).isin(search_vals)]

    # Recompute KPIs from the fully-filtered data so every metric card
    # (including avg TTD, counts in pills) reflects the active filter set.
    kpi_vals = kpis.compute_kpis(filtered) if not filtered.empty else {}

    # ── Tab labels ────────────────────────────────────────────────────────
    sanity_counts = sanity.sanity_summary(df) if not df.empty else {
        "missing_fields_count": 0, "cross_file_count": 0,
    }
    total_issues = (
        sanity_counts["missing_fields_count"] + sanity_counts["cross_file_count"]
    )
    sanity_label = f"🔍 Data Sanity ({total_issues})" if total_issues else "🔍 Data Sanity"

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 Dashboard",
        "⬆️ Upload",
        sanity_label,
        "⚙️ Settings",
        "🔗 Manual Mapping",
    ])

    with tab1:
        tab_dashboard(filtered, kpi_vals)
    with tab2:
        tab_upload()
    with tab3:
        tab_sanity(df)
    with tab4:
        tab_settings(df, groups)
    with tab5:
        tab_manual_mapping(df)


if __name__ == "__main__":
    main()
