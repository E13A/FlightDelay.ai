# dashboard.py
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
from datetime import timedelta
import json



# Local timezone for displaying timestamps in the UI
LOCAL_TIMEZONE = "Asia/Baku"

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])



app.layout = dbc.Container([
    html.H1("SIEM Dashboard"),
    
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="filter-source", placeholder="Filter by Source", multi=True)),
        dbc.Col(dcc.Dropdown(id="filter-type", placeholder="Filter by Event Type", multi=True)),
        dbc.Col(dcc.Dropdown(id="filter-level", placeholder="Filter by Level", multi=True)),
    ], className="mb-4"),
    
    html.Div(id="alert-div", style={"color": "red", "font-weight": "bold"}),
    dcc.Interval(id="interval", interval=5000, n_intervals=0),  # refresh every 5s
    
    dcc.Tabs(id="tabs", value="events", children=[
        dcc.Tab(label="Events", value="events"),
        dcc.Tab(label="Alerts", value="alerts"),
        dcc.Tab(label="Dashboards", value="dashboards"),
        dcc.Tab(label="Rate Limited IPs", value="rate-limited"),
    ]),
    
    html.Div(id="tab-content")
])



# Callback to update filters and alert banner
@app.callback(
    Output("filter-source", "options"),
    Output("filter-type", "options"),
    Output("filter-level", "options"),
    Output("alert-div", "children"),
    Input("interval", "n_intervals")
)
def update_filters(n):
    try:
        resp = requests.get("http://127.0.0.1:8000/events")
        data = resp.json()
        if not data:
            return [], [], [], ""
        
        df = pd.DataFrame(data)
        # Parse timestamps - API returns naive timestamps
        # WAF sends UTC timestamps, log ingester sends local timestamps
        # We need to detect: if event_type is WAF alert type, it's UTC; otherwise it's local
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        
        # Ensure timestamp column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        
        # Check if timestamps are naive (no timezone)
        try:
            has_tz = df["timestamp"].dt.tz is not None
        except AttributeError:
            # If .dt accessor fails, ensure column is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
            has_tz = df["timestamp"].dt.tz is not None
        
        if not has_tz:
            # Detect WAF alerts - these come with UTC timestamps
            waf_types = ["SQLi", "XSS", "CmdInjection", "PathTraversal", "NoSQLInjection", "Blocked Request", "RateLimit"]
            is_waf = df["event_type"].isin(waf_types) | df["message"].str.contains("detected", case=False, na=False)
            
            # Process timestamps separately for WAF and non-WAF events
            # WAF events: treat as UTC and convert to local
            # Other events: treat as already local and just localize
            if is_waf.any():
                waf_ts = df.loc[is_waf, "timestamp"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TIMEZONE)
            else:
                waf_ts = pd.Series(dtype=f"datetime64[ns, {LOCAL_TIMEZONE}]")
            
            if (~is_waf).any():
                non_waf_ts = df.loc[~is_waf, "timestamp"].dt.tz_localize(LOCAL_TIMEZONE)
            else:
                non_waf_ts = pd.Series(dtype=f"datetime64[ns, {LOCAL_TIMEZONE}]")
            
            # Combine both series maintaining index order
            combined = pd.concat([waf_ts, non_waf_ts]).sort_index()
            df["timestamp"] = combined.reindex(df.index)
        else:
            # If they have timezone, convert to local
            df["timestamp"] = df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
        
        # Populate filter dropdowns
        source_opts = [{"label": s, "value": s} for s in sorted(df["source"].dropna().unique())]
        type_opts = [{"label": t, "value": t} for t in sorted(df["event_type"].dropna().unique())]
        level_opts = [{"label": l, "value": l} for l in sorted(df["level"].dropna().unique())]
        
        # Simple alert: more than 5 ERROR events in last 5 minutes
        now = pd.Timestamp.now(tz=LOCAL_TIMEZONE)
        recent_errors = df[
            (df["level"] == "ERROR")
            & (df["timestamp"] > now - timedelta(minutes=5))
        ]
        alert_text = ""
        if len(recent_errors) > 5:
            alert_text = f"⚠️ ALERT: {len(recent_errors)} ERROR events in the last 5 minutes!"
        
        return source_opts, type_opts, level_opts, alert_text
    
    except Exception as e:
        return [], [], [], f"Error fetching events: {e}"

# Callback to update tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("interval", "n_intervals"),
    Input("filter-source", "value"),
    Input("filter-type", "value"),
    Input("filter-level", "value")
)
def update_tab_content(active_tab, n, sources, types, levels):
    try:
        resp = requests.get("http://127.0.0.1:8000/events")
        data = resp.json()
        if not data:
            return html.Div("No events yet.")
        
        df = pd.DataFrame(data)
        # Parse timestamps - API returns naive timestamps
        # WAF sends UTC timestamps, log ingester sends local timestamps
        # We need to detect: if event_type is WAF alert type, it's UTC; otherwise it's local
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        
        # Ensure timestamp column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        
        # Check if timestamps are naive (no timezone)
        try:
            has_tz = df["timestamp"].dt.tz is not None
        except AttributeError:
            # If .dt accessor fails, ensure column is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
            has_tz = df["timestamp"].dt.tz is not None
        
        if not has_tz:
            # Detect WAF alerts - these come with UTC timestamps
            waf_types = ["SQLi", "XSS", "CmdInjection", "PathTraversal", "NoSQLInjection", "Blocked Request", "RateLimit"]
            is_waf = df["event_type"].isin(waf_types) | df["message"].str.contains("detected", case=False, na=False)
            
            # Process timestamps separately for WAF and non-WAF events
            # WAF events: treat as UTC and convert to local
            # Other events: treat as already local and just localize
            if is_waf.any():
                waf_ts = df.loc[is_waf, "timestamp"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TIMEZONE)
            else:
                waf_ts = pd.Series(dtype=f"datetime64[ns, {LOCAL_TIMEZONE}]")
            
            if (~is_waf).any():
                non_waf_ts = df.loc[~is_waf, "timestamp"].dt.tz_localize(LOCAL_TIMEZONE)
            else:
                non_waf_ts = pd.Series(dtype=f"datetime64[ns, {LOCAL_TIMEZONE}]")
            
            # Combine both series maintaining index order
            combined = pd.concat([waf_ts, non_waf_ts]).sort_index()
            df["timestamp"] = combined.reindex(df.index)
        else:
            # If they have timezone, convert to local
            df["timestamp"] = df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)

        # Dashboards tab: show KPIs for system monitoring
        if active_tab == "dashboards":
            if df.empty:
                return html.Div("No data available for dashboards yet.")

            # Ensure timestamps are sorted and used as index
            df_time = df.sort_values("timestamp").set_index("timestamp")

            # 1) Event Volume Over Time
            vol = df_time.resample("5min").size().reset_index(name="count")
            kpi1_fig = px.line(vol, x="timestamp", y="count", title="Event Volume Over Time (5 min buckets)")

            # 2) Security Alert Rate (WAF + ERROR level)
            waf_types = ["SQLi", "XSS", "CmdInjection", "PathTraversal", "NoSQLInjection", "Blocked Request", "RateLimit"]
            df_alerts = df_time[
                (df_time["event_type"].isin(waf_types)) | (df_time["level"] == "ERROR")
            ]
            if not df_alerts.empty:
                alert_rate = (
                    df_alerts
                    .groupby([pd.Grouper(freq="5min"), "event_type"])
                    .size()
                    .reset_index(name="count")
                )
                kpi2_fig = px.bar(
                    alert_rate,
                    x="timestamp",
                    y="count",
                    color="event_type",
                    title="Security Alert Rate (WAF + ERROR) per 5 min",
                )
            else:
                kpi2_fig = px.bar(title="Security Alert Rate (no alerts yet)")

            # 3) Top Talkers (sources) in recent data
            if "source" in df.columns:
                top_sources = (
                    df.groupby("source")
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                    .head(10)
                )
                kpi3_fig = px.bar(
                    top_sources,
                    x="count",
                    y="source",
                    orientation="h",
                    title="Top 10 Sources by Event Count",
                )
            else:
                kpi3_fig = px.bar(title="Top Sources (no source data)")

            # 4) WAF Block Rate vs Total Requests
            total_per_bucket = df_time.resample("5min").size().rename("total")
            blocked = df_time[df_time["event_type"] == "Blocked Request"]
            if not blocked.empty:
                blocked_per_bucket = blocked.resample("5min").size().rename("blocked")
            else:
                blocked_per_bucket = total_per_bucket.copy()
                blocked_per_bucket[:] = 0

            rate_df = (
                pd.concat([total_per_bucket, blocked_per_bucket], axis=1)
                .fillna(0)
                .reset_index()
            )
            kpi4_fig = px.line(
                rate_df,
                x="timestamp",
                y=["total", "blocked"],
                title="Total Requests vs Blocked Requests (5 min buckets)",
            )

            return html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=kpi1_fig), md=6),
                            dbc.Col(dcc.Graph(figure=kpi2_fig), md=6),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=kpi3_fig), md=6),
                            dbc.Col(dcc.Graph(figure=kpi4_fig), md=6),
                        ]
                    ),
                ]
            )

        # Rate Limited & Banned IPs tab
        if active_tab == "rate-limited":
            try:
                # Fetch both rate limited and banned IPs
                rate_resp = requests.get("http://127.0.0.1:8000/rate-limited", timeout=2)
                banned_resp = requests.get("http://127.0.0.1:8000/banned", timeout=2)
                
                rate_limited = rate_resp.json() if rate_resp.status_code == 200 else []
                banned = banned_resp.json() if banned_resp.status_code == 200 else []
                
                content = []
                
                # Banned IPs section
                if banned:
                    banned_rows = []
                    for item in banned:
                        ip = item["ip"]
                        minutes = item["remaining_seconds"] // 60
                        seconds = item["remaining_seconds"] % 60
                        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                        
                        banned_rows.append(html.Tr([
                            html.Td(ip, style={"font-weight": "bold", "color": "red"}),
                            html.Td(time_str, style={"color": "red" if item["remaining_seconds"] > 60 else "orange"}),
                            html.Td(item["unban_at"]),
                        ]))
                    
                    banned_table = dbc.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("IP Address"),
                                html.Th("Time Remaining"),
                                html.Th("Unban At"),
                            ])),
                            html.Tbody(banned_rows)
                        ],
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="mb-4"
                    )
                    
                    content.append(html.Div([
                        html.H3("Banned IPs (SOAR - 180s)", className="mb-3", style={"color": "red"}),
                        html.P(f"Total banned IPs: {len(banned)}", className="text-muted mb-3"),
                        banned_table
                    ]))
                else:
                    content.append(html.Div([
                        html.H3("Banned IPs (SOAR)", className="mb-3"),
                        html.P("No IPs are currently banned.", className="text-muted mb-4")
                    ]))
                
                # Rate Limited IPs section
                if rate_limited:
                    rate_rows = []
                    for item in rate_limited:
                        ip = item["ip"]
                        minutes = item["remaining_seconds"] // 60
                        seconds = item["remaining_seconds"] % 60
                        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                        
                        rate_rows.append(html.Tr([
                            html.Td(ip, style={"font-weight": "bold"}),
                            html.Td(item["request_count"]),
                            html.Td(time_str, style={"color": "red" if item["remaining_seconds"] > 30 else "orange"}),
                            html.Td(item["unblock_at"]),
                        ]))
                    
                    rate_table = dbc.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("IP Address"),
                                html.Th("Request Count"),
                                html.Th("Time Remaining"),
                                html.Th("Unblock At"),
                                html.Th("Action"),
                            ])),
                            html.Tbody(rate_rows)
                        ],
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True
                    )
                    
                    content.append(html.Div([
                        html.H3("Rate Limited IPs", className="mb-3"),
                        html.P(f"Total rate limited IPs: {len(rate_limited)}", className="text-muted mb-3"),
                        rate_table
                    ]))
                else:
                    content.append(html.Div([
                        html.H3("Rate Limited IPs", className="mb-3"),
                        html.P("No IPs are currently rate limited.", className="text-muted")
                    ]))
                
                return html.Div(content, className="p-4")
                
            except Exception as e:
                return html.Div(f"Error fetching IPs: {e}. Make sure WAF is running.")

        # Filter for WAF alerts if on Alerts tab
        if active_tab == "alerts":
            # WAF alerts have event_type as various attack types or "Blocked Request"
            waf_alert_types = ["SQLi", "XSS", "CmdInjection", "PathTraversal", "NoSQLInjection", "Blocked Request"]
            available_types = df["event_type"].dropna().unique().tolist()

            # Primary filter: event_type matches known WAF detections
            type_mask = df["event_type"].isin(waf_alert_types)

            # Secondary filter: message contains keywords emitted by the WAF
            messages = df["message"].fillna("")
            msg_mask = messages.str.contains("detected", case=False) | messages.str.contains("Blocked Request", case=False)

            df_alerts = df[type_mask | msg_mask].copy()
            # Ensure timestamps are preserved correctly after filtering
            if df_alerts["timestamp"].dt.tz is None:
                df_alerts["timestamp"] = df_alerts["timestamp"].dt.tz_localize(LOCAL_TIMEZONE)
            elif str(df_alerts["timestamp"].dt.tz) != LOCAL_TIMEZONE:
                df_alerts["timestamp"] = df_alerts["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
            
            if df_alerts.empty:
                return html.Div(
                    "No WAF alerts yet. Make sure WAF is running and detecting attacks."
                )

            df = df_alerts
        
        # Apply filters
        if sources:
            df = df[df["source"].isin(sources)]
        if types:
            df = df[df["event_type"].isin(types)]
        if levels:
            df = df[df["level"].isin(levels)]
        
        if df.empty:
            return html.Div("No events match the filters.")
        
        # Plot histogram of event types
        fig = px.histogram(df, x="event_type", title="Event Type Count")
        
        # Format timestamps for display (remove timezone info for cleaner display)
        df_display = df.copy()
        df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Table
        table = dbc.Table.from_dataframe(df_display[["timestamp","source","event_type","level","message"]], striped=True, bordered=True, hover=True)
        
        return html.Div([
            dcc.Graph(figure=fig),
            table
        ])
    
    except Exception as e:
        return html.Div(f"Error fetching events: {e}")



if __name__ == "__main__":
    app.run(debug=True, port=8050)

