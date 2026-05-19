"""
app.py — Operations Control Tower (Streamlit)

Tabs:
  📦 Dashboard   — KPI cards, cohort panels, AWB tracker, variance by state
  ⬆️ Upload      — CSV/Excel ingest with per-file result
  📊 Analytics   — Trend charts, transporter scorecard, state heatmap
  🔍 Data Sanity — Missing fields, cross-file consistency
  ⚙️ Settings    — Appointment config management, reset, upload log
"""

import time
from datetime import date

import pandas as pd
import streamlit as st

import charts
import db
import ingest
import kpis
import sanity

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Operations Control Tower",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Header */
.oct-header {
    background: linear-gradient(135deg, #1A73E8, #0D47A1);
    color: white;
    padding: 14px 24px;
    border-radius: 10px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.oct-header h1 { font-size: 20px; margin: 0; font-weight: 700; }
.oct-header .sub { font-size: 11px; opacity: .75; margin-top: 2px; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 12px; margin-bottom: 16px; }
.kpi-card {
    background: white;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    border-left: 4px solid var(--kpi-color);
}
.kpi-value { font-size: 28px; font-weight: 700; color: var(--kpi-color); line-height: 1.1; }
.kpi-label { font-size: 11px; color: #5F6368; font-weight: 500; margin-top: 2px; }

/* Status badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 700;
    white-space: nowrap;
}
.badge-green  { background: #E6F4EA; color: #137333; }
.badge-blue   { background: #E8F0FE; color: #1558D6; }
.badge-red    { background: #FCE8E6; color: #C5221F; }
.badge-orange { background: #FFF0E0; color: #B06000; }
.badge-yellow { background: #FEF7E0; color: #AA8000; }
.badge-grey   { background: #F1F3F4; color: #5F6368; }
.badge-purple { background: #F3E8FD; color: #7B1FA2; }

/* Section cards */
.section-card {
    background: white;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    padding: 16px;
    margin-bottom: 14px;
}
.section-title { font-size: 14px; font-weight: 600; margin-bottom: 10px; }

/* Table tweaks */
div[data-testid="stDataFrame"] table { font-size: 11px !important; }
div[data-testid="stDataFrame"] th {
    background: #F8F9FA !important;
    font-weight: 600 !important;
    color: #5F6368 !important;
}

/* Compact metric */
div[data-testid="metric-container"] {
    background: white;
    border-radius: 8px;
    padding: 10px 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def status_badge(status: str) -> str:
    s = (status or "").upper().strip()
    if "DELIVER" in s and "UNDELIVER" not in s:
        return '<span class="badge badge-green">Delivered</span>'
    if "CANCEL" in s:
        return '<span class="badge badge-red">Cancelled</span>'
    if "UNDELIVER" in s or "FAIL" in s:
        return '<span class="badge badge-yellow">Undelivered</span>'
    if "OUT FOR" in s:
        return '<span class="badge badge-blue">Out for Del</span>'
    if "RTO" in s or "RETURN" in s:
        return '<span class="badge badge-orange">RTO</span>'
    if "MANIFEST" in s:
        return '<span class="badge badge-grey">Manifested</span>'
    if s in ("IN_TRANSIT", "IN TRANSIT", ""):
        return '<span class="badge badge-blue">In Transit</span>'
    return f'<span class="badge badge-grey">{status}</span>'


def appt_badge(appt_status: str, appt_date: str = "") -> str:
    if appt_status == "N" or appt_status == "delivered":
        return "N"
    if appt_status == "pending":
        return '<span class="badge badge-red">Appt Pending</span>'
    if appt_status == "scheduled":
        return '<span class="badge badge-purple">✓ Scheduled</span>'
    return "–"


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


def var_cell(var_days: int, var_str: str) -> str:
    if not var_str or var_str == "0d":
        return "–"
    color = "#C5221F" if var_days > 0 else "#137333"
    return f'<span style="color:{color};font-weight:600">{var_str}</span>'


@st.cache_data(ttl=120, show_spinner=False)
def _cached_load():
    """Load all data from Supabase with a 2-minute cache."""
    raw = db.load_awb_view()
    appt_config = db.load_appt_config()
    return raw, appt_config


def load_data(force: bool = False):
    if force:
        _cached_load.clear()
    raw, appt_config = _cached_load()
    if raw.empty:
        return pd.DataFrame(), appt_config, {}
    df = kpis.build_display_rows(raw, appt_config)
    kpi_vals = kpis.compute_kpis(df)
    return df, appt_config, kpi_vals


# ── Render AWB table ───────────────────────────────────────────────────────

_TABLE_COLS = [
    ("so_number",               "SO #"),
    ("customer_po_ref",         "PO Ref"),
    ("invoice_number",          "Invoice #"),
    ("awb",                     "AWB"),
    ("customer_name",           "Customer"),
    ("drop_city",               "City"),
    ("drop_state",              "State"),
    ("transporter",             "Transporter"),
    ("order_date",              "Order Date"),
    ("dispatch_date",           "Dispatch"),
    ("days_old",                "Days Old"),
    ("expected_ship_date",      "Exp Ship"),
    ("status",                  "Status"),
    ("delivery_date",           "Del Date"),
    ("estimated_delivery_date", "EDD"),
    ("var_str",                 "Variance"),
    ("appt_status",             "Appt"),
    ("appointment_date",        "Appt Date"),
    ("pod_url",                 "POD"),
    ("latest_remark",           "Latest Remark"),
]


def render_awb_table(df: pd.DataFrame, key_suffix: str = "main") -> None:
    if df.empty:
        st.info("No records to display.")
        return

    # Separate active and delivered
    active   = df[~df["is_delivered"]].sort_values("days_old", ascending=False)
    delivered = df[df["is_delivered"]].sort_values("days_old", ascending=False)

    display_parts = [active, delivered]
    combined = pd.concat(display_parts, ignore_index=True)

    # Build display DF
    display_rows = []
    for _, row in combined.iterrows():
        awb = row.get("awb", "")
        url = row.get("track_url", "#")
        is_del = row.get("is_delivered", False)

        d_old = row.get("days_old", 0)
        days_cell = f"**{d_old}**" if d_old > 14 and not is_del else str(d_old)

        remark = str(row.get("latest_remark") or "")
        remark_short = remark[:30] + "…" if len(remark) > 30 else remark

        _pod_raw = row.get("pod_url")
        _pod_str = "" if _pod_raw is None else str(_pod_raw).strip()
        pod_cell = _pod_str if _pod_str.startswith("http") else ""

        # AWB cell: plain URL so LinkColumn works; regex extracts AWB for display
        # URL format: https://www.aftership.com/track?t=<AWB>&c=<carrier>
        awb_cell = url if (awb and url and url.startswith("http")) else ""

        display_rows.append({
            "SO #":        row.get("so_number") or "–",
            "PO Ref":      row.get("customer_po_ref") or "–",
            "Invoice #":   row.get("invoice_number") or "–",
            "AWB":         awb_cell,
            "Customer":    row.get("customer_name") or "–",
            "City":        row.get("drop_city") or "–",
            "State":       row.get("drop_state") or "–",
            "Transporter": row.get("transporter") or "–",
            "Order Date":  fmt_date(row.get("order_date")),
            "Dispatch":    fmt_date(row.get("dispatch_date")),
            "Days Old":    days_cell,
            "Exp Ship":    fmt_date(row.get("expected_ship_date")),
            "Status":      row.get("status") or "–",
            "Del Date":    fmt_date(row.get("delivery_date")),
            "EDD":         fmt_date(row.get("estimated_delivery_date")),
            "Variance":    row.get("var_str") or "–",
            "Appt":        row.get("appt_status") or "–",
            "Appt Date":   fmt_date(row.get("appointment_date")),
            "POD":         pod_cell,
            "Latest Remark": remark_short,
        })

    disp_df = pd.DataFrame(display_rows)

    # Separator row between active and delivered
    if not active.empty and not delivered.empty:
        sep_idx = len(active)
        # Streamlit doesn't allow true separator rows, so we show counts above
        st.caption(
            f"**{len(active)} active** shipments · "
            f"**{len(delivered)} delivered** shipments (shown below separator)"
        )

    st.dataframe(
        disp_df,
        use_container_width=True,
        hide_index=True,
        height=480,
        column_config={
            # Regex extracts AWB number from ?t=<AWB>&c=<carrier>
            "AWB": st.column_config.LinkColumn("AWB", display_text=r"[?&]t=([^&]+)"),
            "POD": st.column_config.LinkColumn("POD", display_text="📄"),
            "Days Old": st.column_config.TextColumn("Days Old"),
        },
    )

    # CSV export
    col1, col2 = st.columns([6, 1])
    with col2:
        csv_data = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ CSV",
            data=csv_data,
            file_name=f"OCT_{date.today().isoformat()}.csv",
            mime="text/csv",
            key=f"csv_{key_suffix}",
        )


# ── KPI card HTML ──────────────────────────────────────────────────────────

def kpi_cards(kpi_vals: dict) -> None:
    cards = [
        ("total_active",    "📊", "Total Active",         "#1A73E8"),
        ("stuck",           "🚨", "Stuck (14d+)",         "#EA4335"),
        ("pending_appt",    "📋", "Pending Appointment",  "#F9AB00"),
        ("delivered_today", "✅", "Delivered Today",      "#34A853"),
        ("edd_breached",    "⏰", "EDD Breached",         "#FA7B17"),
    ]
    cols = st.columns(5)
    for col, (key, icon, label, color) in zip(cols, cards):
        val = kpi_vals.get(key, 0)
        col.markdown(
            f"""
            <div class="kpi-card" style="--kpi-color:{color}">
              <div style="font-size:20px">{icon}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def tab_dashboard(df: pd.DataFrame, kpi_vals: dict) -> None:
    # ── KPI row ───────────────────────────────────────────────────────────
    kpi_cards(kpi_vals)

    # ── Cohort expanders ──────────────────────────────────────────────────
    cohort_defs = [
        ("🚨 Stuck Shipments (>14 days)",     "stuck",          kpis.cohort_stuck),
        ("📋 Pending Appointments",            "appt",           kpis.cohort_pending_appt),
        ("✅ Delivered Today",                  "delivered_today",kpis.cohort_delivered_today),
        ("⏰ EDD Breached",                     "edd",            kpis.cohort_edd_breached),
    ]
    for title, key_sfx, cohort_fn in cohort_defs:
        cohort_df = cohort_fn(df) if not df.empty else pd.DataFrame()
        count = len(cohort_df)
        with st.expander(f"{title}  ({count})", expanded=False):
            render_awb_table(cohort_df, key_suffix=key_sfx)

    # ── Filters ───────────────────────────────────────────────────────────
    st.markdown("### 📦 Shipment Tracker")
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([2, 2, 2, 2, 2])

    with f_col1:
        search_field = st.selectbox(
            "Search by",
            options=["awb", "so", "invoice", "customer"],
            format_func=lambda x: {
                "awb": "AWB", "so": "SO #",
                "invoice": "Invoice #", "customer": "Customer",
            }[x],
            key="search_field",
        )
    with f_col2:
        search_q = st.text_input("Search", placeholder="Type to filter…", key="search_q")

    all_states      = sorted(df["drop_state"].dropna().unique()) if not df.empty else []
    all_transporters = sorted(df["transporter"].dropna().unique()) if not df.empty else []
    all_statuses    = sorted(df["status"].dropna().unique()) if not df.empty else []

    with f_col3:
        state_sel = st.selectbox("State", ["All"] + all_states, key="state_sel")
    with f_col4:
        tp_sel = st.multiselect("Transporter", all_transporters,
                                default=all_transporters, key="tp_sel")
    with f_col5:
        status_sel = st.selectbox("Status", ["All"] + all_statuses, key="status_sel")

    # Apply filters
    filtered = df.copy() if not df.empty else pd.DataFrame()
    if not filtered.empty:
        if state_sel != "All":
            filtered = filtered[filtered["drop_state"] == state_sel]
        if tp_sel:
            filtered = filtered[filtered["transporter"].isin(tp_sel)]
        if status_sel != "All":
            filtered = filtered[filtered["status"] == status_sel]
        if search_q.strip():
            q = search_q.strip().lower()
            col_map = {
                "awb": "awb", "so": "so_number",
                "invoice": "invoice_number", "customer": "customer_name",
            }
            col = col_map.get(search_field, "awb")
            filtered = filtered[
                filtered[col].fillna("").str.lower().str.contains(q, regex=False)
            ]

    render_awb_table(filtered, key_suffix="main")

    # ── Variance by state ─────────────────────────────────────────────────
    st.markdown("### 📍 Delivery Variance by State")
    var_df = kpis.variance_by_state(
        df[df["drop_state"] == state_sel] if state_sel != "All" else df
    ) if not df.empty else pd.DataFrame()

    if not var_df.empty:
        # Colour Avg Var column
        def colour_var(val):
            try:
                v = float(val)
                if v > 0:
                    return "color: #C5221F; font-weight: 600"
                if v < 0:
                    return "color: #137333; font-weight: 600"
            except Exception:
                pass
            return "color: #9AA0A6"

        styled = var_df.style.map(colour_var, subset=["Avg Var (d)"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=300)
    else:
        st.info("No variance data yet — upload a Courier MIS file.")


# ══════════════════════════════════════════════════════════════════════════
# TAB: UPLOAD
# ══════════════════════════════════════════════════════════════════════════

def tab_upload() -> None:
    st.markdown("### ⬆️ Upload Files")
    st.markdown(
        "Upload one or more files. File type is auto-detected from column headers.\n\n"
        "**Supported:** WMS Dispatch · B2B Courier MIS · RPT Stuck Orders · "
        "Appointment Config  |  **Formats:** CSV, Excel (.xlsx / .xls)"
    )

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
        results = []
        progress = st.progress(0)
        for i, uf in enumerate(uploaded):
            with st.spinner(f"Processing {uf.name}…"):
                result = ingest.process_uploaded_file(uf)
            results.append(result)
            progress.progress((i + 1) / len(uploaded))

        progress.empty()

        # Show results
        st.markdown("#### Results")
        all_ok = True
        for r in results:
            if r.ok:
                st.success(
                    f"✅ **{r.filename}** ({r.file_type}) — "
                    f"{r.rows_processed} rows processed · "
                    f"{r.rows_inserted} inserted · {r.rows_updated} updated"
                )
            else:
                all_ok = False
                st.error(f"❌ **{r.filename}** — {r.message}")

        if all_ok:
            st.info("All files ingested. Dashboard data will refresh automatically.")
            _cached_load.clear()
            time.sleep(1)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# TAB: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════

def tab_analytics(df: pd.DataFrame) -> None:
    st.markdown("### 📊 Analytics")

    if df.empty:
        st.info("Upload data to see analytics.")
        return

    # Load full dataset including CANCELLED for complete analytics
    @st.cache_data(ttl=120, show_spinner=False)
    def _load_full():
        return db.load_all_awb_raw()

    full_df = _load_full()

    days = st.slider("Time window (days)", 7, 90, 30, step=7, key="analytics_days")

    # Row 1: Daily delivered + On-time trend
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            charts.chart_daily_delivered(full_df, days=days),
            use_container_width=True,
        )
    with col2:
        weeks = max(4, days // 7)
        st.plotly_chart(
            charts.chart_ontime_trend(full_df, weeks=weeks),
            use_container_width=True,
        )

    # Row 2: Status mix + Transporter scorecard
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(charts.chart_status_mix(full_df), use_container_width=True)
    with col4:
        st.plotly_chart(
            charts.chart_transporter_scorecard(full_df), use_container_width=True
        )

    # Row 3: State heatmap (full width)
    st.plotly_chart(charts.chart_state_heatmap(full_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB: DATA SANITY
# ══════════════════════════════════════════════════════════════════════════

def tab_sanity(df: pd.DataFrame) -> None:
    st.markdown("### 🔍 Data Sanity Checks")

    if df.empty:
        st.info("No data loaded. Upload files first.")
        return

    col_left, col_right = st.columns(2)

    # ── Missing critical fields ───────────────────────────────────────────
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
            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True,
                height=350,
            )
            csv = missing_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Missing Fields",
                data=csv,
                file_name=f"OCT_missing_fields_{date.today().isoformat()}.csv",
                mime="text/csv",
                key="dl_missing",
            )

    # ── Cross-file consistency ────────────────────────────────────────────
    with col_right:
        st.markdown("#### Cross-File Consistency")
        age_threshold = st.number_input(
            "Flag AWBs dispatched >N days ago with no Courier MIS status",
            min_value=1, max_value=30, value=3, step=1,
            key="cross_age",
        )
        st.caption(
            "These AWBs appear in WMS Dispatch but Courier MIS has never "
            "reported a status for them. Possible causes: AWB not found by courier, "
            "Courier MIS not yet uploaded, or wrong AWB in WMS."
        )
        cross_df = sanity.check_cross_file_consistency(
            df, min_dispatch_age_days=int(age_threshold)
        )
        if cross_df.empty:
            st.success("✅ No cross-file consistency issues found.")
        else:
            st.warning(f"⚠️ {len(cross_df)} AWBs dispatched but missing from Courier MIS")
            st.dataframe(
                cross_df,
                use_container_width=True,
                hide_index=True,
                height=350,
            )
            csv = cross_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Consistency Issues",
                data=csv,
                file_name=f"OCT_cross_file_{date.today().isoformat()}.csv",
                mime="text/csv",
                key="dl_cross",
            )


# ══════════════════════════════════════════════════════════════════════════
# TAB: SETTINGS
# ══════════════════════════════════════════════════════════════════════════

def tab_settings() -> None:
    st.markdown("### ⚙️ Settings")

    # ── Appointment Config ────────────────────────────────────────────────
    with st.expander("📋 Appointment Config", expanded=True):
        st.markdown(
            "Define which customers require a delivery appointment. "
            "This overrides the Appointment_Required column from Courier MIS."
        )
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

    # ── Upload Log ────────────────────────────────────────────────────────
    with st.expander("📜 Upload History", expanded=False):
        log_df = db.load_upload_log(50)
        if log_df.empty:
            st.caption("No uploads recorded yet.")
        else:
            if "uploaded_at" in log_df.columns:
                log_df["uploaded_at"] = pd.to_datetime(
                    log_df["uploaded_at"], errors="coerce"
                ).dt.strftime("%d %b %y %H:%M")
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
                use_container_width=True,
                hide_index=True,
                height=300,
            )

    # ── Reset / Danger Zone ───────────────────────────────────────────────
    with st.expander("🔄 Reset / Danger Zone", expanded=False):
        st.warning(
            "**Reset AWB View** deletes ALL shipment records. "
            "Re-upload your CSV/Excel files afterwards."
        )
        confirm = st.text_input(
            'Type "RESET" to confirm',
            placeholder="RESET",
            key="reset_confirm",
        )
        if st.button("🔄 Reset AWB View", type="secondary", key="reset_btn"):
            if confirm.strip().upper() == "RESET":
                db.reset_awb_view()
                _cached_load.clear()
                st.success("✅ AWB View cleared. Please re-upload your files.")
                time.sleep(1)
                st.rerun()
            else:
                st.error('Please type "RESET" to confirm.')


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────
    now_str = pd.Timestamp.now().strftime("%d %b %Y %H:%M")
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
        df, appt_config, kpi_vals = load_data()

    # Sanity badge for tab label
    sanity_counts = sanity.sanity_summary(df) if not df.empty else {
        "missing_fields_count": 0, "cross_file_count": 0
    }
    total_issues = (
        sanity_counts["missing_fields_count"] + sanity_counts["cross_file_count"]
    )
    sanity_label = f"🔍 Data Sanity ({total_issues})" if total_issues else "🔍 Data Sanity"

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 Dashboard",
        "⬆️ Upload",
        "📊 Analytics",
        sanity_label,
        "⚙️ Settings",
    ])

    with tab1:
        tab_dashboard(df, kpi_vals)
    with tab2:
        tab_upload()
    with tab3:
        tab_analytics(df)
    with tab4:
        tab_sanity(df)
    with tab5:
        tab_settings()


if __name__ == "__main__":
    main()
