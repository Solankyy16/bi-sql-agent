"""
charts/auto_chart.py — Plotly-powered smart chart generation
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

_PURPLE  = "#7c6af5"
_TEAL    = "#1D9E75"
_CORAL   = "#D85A30"
_AMBER   = "#BA7517"
_BLUE    = "#378ADD"
_PALETTE = [_PURPLE, _TEAL, _CORAL, _AMBER, _BLUE, "#D4537E", "#639922"]

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, sans-serif", size=13, color="#888780"),
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(bgcolor="rgba(30,30,40,0.92)", font_size=13,
                    font_color="white", bordercolor="rgba(255,255,255,0.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                font=dict(size=12)),
    colorway=_PALETTE,
)


def auto_chart(df: pd.DataFrame) -> Optional[object]:
    if df is None or df.empty or len(df) < 2:
        return None
    try:
        import plotly.graph_objects as go
        df = df.copy()
        cat_cols = df.select_dtypes(include=["object","string","category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        dt_cols  = df.select_dtypes(include=["datetime","datetimetz"]).columns.tolist()

        # Try parsing string cols as dates
        for col in cat_cols[:]:
            sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
            if len(sample) >= 7 and (sample[4:5] == "-" or "/" in sample[:5]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                    dt_cols.append(col); cat_cols.remove(col)
                except Exception:
                    pass

        # Remove ID-like numeric cols
        num_cols = [c for c in num_cols
                    if not c.lower().endswith(("id","_id","key","code"))]

        fig = None
        if dt_cols and num_cols:
            fig = _time_series(df, dt_cols[0], num_cols[0])
        elif cat_cols and len(num_cols) == 1:
            n_unique = df[cat_cols[0]].nunique()
            if 2 <= n_unique <= 6:
                fig = _pie_chart(df, cat_cols[0], num_cols[0])
            else:
                fig = _bar_chart(df, cat_cols[0], num_cols[0])
        elif cat_cols and len(num_cols) >= 2:
            fig = _grouped_bar(df, cat_cols[0], num_cols[:3])
        elif len(num_cols) == 2 and not cat_cols:
            fig = _scatter(df, num_cols[0], num_cols[1])
        elif len(num_cols) == 1 and not cat_cols:
            fig = _histogram(df, num_cols[0])
        elif len(cat_cols) == 1 and not num_cols:
            counts = df[cat_cols[0]].value_counts().reset_index()
            counts.columns = [cat_cols[0], "count"]
            if len(counts) >= 2:
                fig = _bar_chart(counts, cat_cols[0], "count")
        return fig
    except Exception as e:
        print(f"[Chart] {e}")
        return None


def _time_series(df, x_col, y_col):
    import plotly.graph_objects as go
    df_s = df.sort_values(x_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_s[x_col], y=df_s[y_col], mode="lines+markers",
        name=_clean(y_col),
        line=dict(color=_TEAL, width=2.5),
        marker=dict(size=5, color=_TEAL),
        fill="tozeroy", fillcolor="rgba(29,158,117,0.08)",
        hovertemplate=f"<b>%{{x}}</b><br>{_clean(y_col)}: %{{y:,.2f}}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT,
        title=dict(text=f"{_clean(y_col)} over time", font=dict(size=15,color="#444441"), x=0.02),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,0.12)",
                   zeroline=False, tickfont=dict(size=11), tickformat=",.0f"),
        height=340)
    return fig


def _bar_chart(df, cat_col, num_col):
    import plotly.graph_objects as go
    df_p = df.nlargest(20, num_col).sort_values(num_col, ascending=True)
    n = len(df_p)
    colors = [f"rgba(124,106,245,{0.35+0.65*(i/max(n-1,1)):.2f})" for i in range(n)]
    labels = [str(v)[:28]+"…" if len(str(v))>28 else str(v) for v in df_p[cat_col]]
    fig = go.Figure(go.Bar(
        x=df_p[num_col], y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate=f"<b>%{{y}}</b><br>{_clean(num_col)}: %{{x:,.2f}}<extra></extra>",
        text=[_fmt(v) for v in df_p[num_col]],
        textposition="outside", textfont=dict(size=11, color="#5F5E5A"),
        cliponaxis=False,
    ))
    fig.update_layout(**_LAYOUT,
        title=dict(text=f"{_clean(num_col)} by {_clean(cat_col)}",
                   font=dict(size=15,color="#444441"), x=0.02),
        xaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,0.12)",
                   zeroline=False, tickformat=",.0f", tickfont=dict(size=11)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        height=max(280, 38*n+80), bargap=0.28)
    return fig


def _pie_chart(df, cat_col, num_col):
    import plotly.graph_objects as go
    df_p = df.nlargest(6, num_col)
    labels = [str(v)[:24] for v in df_p[cat_col]]
    fig = go.Figure(go.Pie(
        labels=labels, values=df_p[num_col],
        marker=dict(colors=_PALETTE[:len(df_p)],
                    line=dict(color="rgba(255,255,255,0.5)", width=2)),
        textinfo="label+percent", textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.2f}<br>%{percent}<extra></extra>",
        hole=0.38, pull=[0.04]+[0]*(len(df_p)-1),
    ))
    fig.update_layout(**_LAYOUT,
        title=dict(text=f"{_clean(num_col)} by {_clean(cat_col)}",
                   font=dict(size=15,color="#444441"), x=0.02),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11)),
        height=340)
    return fig


def _grouped_bar(df, cat_col, num_cols):
    import plotly.graph_objects as go
    df_p = df.head(15)
    labels = [str(v)[:20] for v in df_p[cat_col]]
    fig = go.Figure()
    for i, col in enumerate(num_cols):
        fig.add_trace(go.Bar(
            name=_clean(col), x=labels, y=df_p[col],
            marker_color=_PALETTE[i % len(_PALETTE)],
            hovertemplate=f"<b>%{{x}}</b><br>{_clean(col)}: %{{y:,.2f}}<extra></extra>",
        ))
    fig.update_layout(**_LAYOUT,
        title=dict(text=f"Comparison by {_clean(cat_col)}",
                   font=dict(size=15,color="#444441"), x=0.02),
        barmode="group",
        xaxis=dict(showgrid=False, zeroline=False, tickangle=-30,
                   tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,0.12)",
                   zeroline=False, tickformat=",.0f"),
        height=360, bargap=0.2, bargroupgap=0.08)
    return fig


def _scatter(df, x_col, y_col):
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        x=df[x_col], y=df[y_col], mode="markers",
        marker=dict(color=_PURPLE, size=8, opacity=0.7,
                    line=dict(color="white", width=1)),
        hovertemplate=(f"{_clean(x_col)}: %{{x:,.2f}}<br>"
                       f"{_clean(y_col)}: %{{y:,.2f}}<extra></extra>"),
    ))
    fig.update_layout(**_LAYOUT,
        title=dict(text=f"{_clean(y_col)} vs {_clean(x_col)}",
                   font=dict(size=15,color="#444441"), x=0.02),
        xaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,0.12)",
                   zeroline=False, title=_clean(x_col), tickformat=",.0f"),
        yaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,0.12)",
                   zeroline=False, title=_clean(y_col), tickformat=",.0f"),
        height=340)
    return fig


def _histogram(df, col):
    import plotly.graph_objects as go
    fig = go.Figure(go.Histogram(
        x=df[col], nbinsx=min(30, max(10, len(df)//5)),
        marker=dict(color=_BLUE, opacity=0.82,
                    line=dict(color="rgba(255,255,255,0.4)", width=0.8)),
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT,
        title=dict(text=f"Distribution of {_clean(col)}",
                   font=dict(size=15,color="#444441"), x=0.02),
        xaxis=dict(showgrid=False, zeroline=False, title=_clean(col),
                   tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,0.12)",
                   zeroline=False, title="Count"),
        bargap=0.05, height=320)
    return fig


def _clean(col_name: str) -> str:
    import re
    s = re.sub(r"([A-Z])", r" \1", str(col_name)).strip()
    s = re.sub(r"[_\-]+", " ", s)
    return s.title().strip()


def _fmt(v) -> str:
    try:
        v = float(v)
        if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
        if v >= 1_000:     return f"{v/1_000:.1f}K"
        if v == int(v):    return f"{int(v):,}"
        return f"{v:,.2f}"
    except Exception:
        return str(v)
