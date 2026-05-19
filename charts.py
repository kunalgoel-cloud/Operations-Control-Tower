"""
charts.py — Plotly analytics charts for OCT Analytics tab.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BLUE   = "#1A73E8"
_GREEN  = "#34A853"
_RED    = "#EA4335"
_ORANGE = "#FA7B17"
_YELLOW = "#F9AB00"
_GREY   = "#9AA0A6"
_BG     = "#F0F4F9"


def _empty_fig(msg: str = "No data available"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=_GREY),
    )
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=300,
    )
    return fig


# ── 1. Daily delivered count (last N days) ────────────────────────────────

def chart_daily_delivered(df: pd.DataFrame, days: int = 30) -> go.Figure:
    """Bar chart: delivered shipments per day over the last `days` days."""
    if df.empty:
        return _empty_fig("No delivery data yet")

    delivered = df[df["status"].str.upper() == "DELIVERED"].copy()
    if delivered.empty:
        return _empty_fig("No delivered shipments found")

    delivered["delivery_date"] = pd.to_datetime(
        delivered["delivery_date"], errors="coerce"
    )
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=days)
    delivered = delivered[delivered["delivery_date"] >= cutoff]

    if delivered.empty:
        return _empty_fig(f"No deliveries in last {days} days")

    daily = (
        delivered.groupby(delivered["delivery_date"].dt.date)
        .size()
        .reset_index(name="count")
        .rename(columns={"delivery_date": "Date"})
    )
    daily["Date"] = pd.to_datetime(daily["Date"])

    fig = px.bar(
        daily, x="Date", y="count",
        labels={"count": "Shipments Delivered", "Date": ""},
        color_discrete_sequence=[_GREEN],
        title=f"Daily Deliveries — Last {days} Days",
    )
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", size=12),
        title_font_size=14,
        hovermode="x unified",
        margin=dict(t=50, b=30, l=40, r=20),
        height=320,
    )
    fig.update_traces(hovertemplate="%{y} deliveries<extra></extra>")
    return fig


# ── 2. On-time % trend by week ─────────────────────────────────────────────

def chart_ontime_trend(df: pd.DataFrame, weeks: int = 12) -> go.Figure:
    """Line chart: weekly on-time delivery percentage."""
    if df.empty:
        return _empty_fig("No data available")

    delivered = df[
        (df["status"].str.upper() == "DELIVERED")
        & df["estimated_delivery_date"].notna()
        & df["delivery_date"].notna()
    ].copy()

    if delivered.empty:
        return _empty_fig("Need both EDD and Delivery Date to compute on-time %")

    delivered["delivery_date"] = pd.to_datetime(
        delivered["delivery_date"], errors="coerce"
    )
    delivered["estimated_delivery_date"] = pd.to_datetime(
        delivered["estimated_delivery_date"], errors="coerce"
    )

    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(weeks=weeks, days=0)
    delivered = delivered[delivered["delivery_date"] >= cutoff].copy()
    if delivered.empty:
        return _empty_fig(f"No delivery data in last {weeks} weeks")

    delivered["week"] = delivered["delivery_date"].dt.to_period("W").apply(
        lambda p: p.start_time
    )
    delivered["on_time"] = (
        delivered["delivery_date"] <= delivered["estimated_delivery_date"]
    )

    weekly = (
        delivered.groupby("week")
        .agg(total=("on_time", "count"), on_time_count=("on_time", "sum"))
        .reset_index()
    )
    weekly["on_time_pct"] = (weekly["on_time_count"] / weekly["total"] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["on_time_pct"],
        mode="lines+markers",
        name="On-Time %",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=6),
        hovertemplate="%{x|%d %b}: %{y:.1f}%<br>(%{customdata} deliveries)<extra></extra>",
        customdata=weekly["total"],
    ))
    fig.add_hline(
        y=80, line_dash="dot", line_color=_ORANGE,
        annotation_text="80% target", annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"On-Time Delivery % — Last {weeks} Weeks",
        yaxis=dict(range=[0, 105], ticksuffix="%", title=""),
        xaxis=dict(title=""),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", size=12),
        title_font_size=14,
        hovermode="x unified",
        margin=dict(t=50, b=30, l=50, r=20),
        height=320,
    )
    return fig


# ── 3. Transporter scorecard ───────────────────────────────────────────────

def chart_transporter_scorecard(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: on-time % and avg variance by transporter."""
    if df.empty:
        return _empty_fig("No data available")

    delivered = df[
        (df["status"].str.upper() == "DELIVERED")
        & df["estimated_delivery_date"].notna()
        & df["delivery_date"].notna()
    ].copy()

    if delivered.empty:
        return _empty_fig("No delivered shipments with EDD data")

    delivered["delivery_date"] = pd.to_datetime(
        delivered["delivery_date"], errors="coerce"
    )
    delivered["estimated_delivery_date"] = pd.to_datetime(
        delivered["estimated_delivery_date"], errors="coerce"
    )
    delivered["on_time"] = (
        delivered["delivery_date"] <= delivered["estimated_delivery_date"]
    )
    delivered["var_days_calc"] = (
        delivered["delivery_date"] - delivered["estimated_delivery_date"]
    ).dt.days

    scorecard = (
        delivered.groupby("transporter")
        .agg(
            total=("on_time", "count"),
            on_time_count=("on_time", "sum"),
            avg_var=("var_days_calc", "mean"),
        )
        .reset_index()
    )
    scorecard["on_time_pct"] = (
        scorecard["on_time_count"] / scorecard["total"] * 100
    ).round(1)
    scorecard["avg_var"] = scorecard["avg_var"].round(1)
    scorecard = scorecard.sort_values("on_time_pct", ascending=True)

    colors = [
        _GREEN if p >= 80 else _ORANGE if p >= 60 else _RED
        for p in scorecard["on_time_pct"]
    ]

    fig = go.Figure(go.Bar(
        x=scorecard["on_time_pct"],
        y=scorecard["transporter"],
        orientation="h",
        marker_color=colors,
        customdata=scorecard[["total", "avg_var"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "On-Time: %{x:.1f}%<br>"
            "Shipments: %{customdata[0]}<br>"
            "Avg variance: %{customdata[1]:+.1f}d<extra></extra>"
        ),
    ))
    fig.add_vline(x=80, line_dash="dot", line_color=_ORANGE)
    fig.update_layout(
        title="Transporter Scorecard — On-Time %",
        xaxis=dict(range=[0, 105], ticksuffix="%", title=""),
        yaxis=dict(title=""),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", size=12),
        title_font_size=14,
        margin=dict(t=50, b=30, l=140, r=20),
        height=max(280, 60 + 40 * len(scorecard)),
    )
    return fig


# ── 4. State heatmap (volume + on-time %) ─────────────────────────────────

def chart_state_heatmap(df: pd.DataFrame) -> go.Figure:
    """Bubble / scatter chart of state performance (volume vs on-time %)."""
    if df.empty:
        return _empty_fig("No data available")

    delivered = df[df["status"].str.upper() == "DELIVERED"].copy()
    all_df = df.copy()

    all_df["estimated_delivery_date"] = pd.to_datetime(
        all_df["estimated_delivery_date"], errors="coerce"
    )
    all_df["delivery_date"] = pd.to_datetime(
        all_df["delivery_date"], errors="coerce"
    )

    state_total = (
        all_df.groupby("drop_state", dropna=False)
        .size()
        .reset_index(name="total")
    )
    del_grp = delivered.copy()
    del_grp["delivery_date"] = pd.to_datetime(
        del_grp["delivery_date"], errors="coerce"
    )
    del_grp["estimated_delivery_date"] = pd.to_datetime(
        del_grp["estimated_delivery_date"], errors="coerce"
    )
    del_grp["on_time"] = (
        del_grp["delivery_date"] <= del_grp["estimated_delivery_date"]
    )
    state_ot = (
        del_grp.groupby("drop_state", dropna=False)
        .agg(delivered=("on_time", "count"), on_time_count=("on_time", "sum"))
        .reset_index()
    )
    state_ot["on_time_pct"] = (
        state_ot["on_time_count"] / state_ot["delivered"] * 100
    ).round(1)

    merged = state_total.merge(state_ot, on="drop_state", how="left")
    merged["on_time_pct"] = merged["on_time_pct"].fillna(0)
    merged["delivered"] = merged["delivered"].fillna(0).astype(int)
    merged["drop_state"] = merged["drop_state"].fillna("Unknown").astype(str)
    merged = merged[merged["drop_state"] != "nan"]

    fig = px.scatter(
        merged,
        x="total",
        y="on_time_pct",
        size="total",
        color="on_time_pct",
        text="drop_state",
        color_continuous_scale=["#EA4335", "#F9AB00", "#34A853"],
        range_color=[0, 100],
        labels={
            "total": "Total Shipments",
            "on_time_pct": "On-Time %",
            "drop_state": "State",
        },
        title="State Performance: Volume vs On-Time %",
        hover_data={"delivered": True},
    )
    fig.update_traces(
        textposition="top center",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Total: %{x}<br>"
            "On-Time: %{y:.1f}%<extra></extra>"
        ),
    )
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", size=11),
        title_font_size=14,
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(title="On-Time %", ticksuffix="%"),
        margin=dict(t=50, b=40, l=60, r=20),
        height=420,
    )
    return fig


# ── 5. Status mix donut ────────────────────────────────────────────────────

def chart_status_mix(df: pd.DataFrame) -> go.Figure:
    """Donut chart: breakdown by status."""
    if df.empty:
        return _empty_fig("No data")

    STATUS_COLORS = {
        "DELIVERED":        _GREEN,
        "IN_TRANSIT":       _BLUE,
        "OUT FOR DELIVERY": "#1558D6",
        "MANIFESTED":       "#4285F4",
        "UNDELIVERED":      _YELLOW,
        "RTO":              _ORANGE,
        "CANCELLED":        _RED,
    }

    counts = df["status"].str.upper().value_counts().reset_index()
    counts.columns = ["status", "count"]
    counts = counts[~counts["status"].isin(["AWB_REGISTERED", ""])]

    colors = [STATUS_COLORS.get(s, _GREY) for s in counts["status"]]

    label_map = {
        "IN_TRANSIT": "In Transit",
        "OUT FOR DELIVERY": "Out for Delivery",
        "AWB_REGISTERED": "AWB Reg",
    }
    labels = [label_map.get(s, s.title()) for s in counts["status"]]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts["count"],
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} shipments (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title="Status Mix",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", size=11),
        title_font_size=14,
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
        height=320,
    )
    return fig
