import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
import datetime
import io

# Page Configuration
st.set_page_config(
    page_title="Retail Demand Forecasting Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection for Premium Look and Feel
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Outfit & Inter fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }

    .gradient-text {
        background: linear-gradient(135deg, #4FACFE 0%, #00F2FE 50%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #88888b;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .kpi-card {
        border: 1px solid rgba(79, 172, 254, 0.18);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.03);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        min-height: 124px;
    }

    .kpi-title {
        font-size: 0.78rem;
        color: #6b7280;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.55rem;
        color: #111827;
        font-weight: 800;
        margin-top: 0.35rem;
        overflow-wrap: anywhere;
    }

    .glow-divider {
        height: 1px;
        margin: 1.25rem 0;
        background: linear-gradient(90deg, rgba(79,172,254,0), rgba(79,172,254,.65), rgba(255,142,83,.65), rgba(255,142,83,0));
    }

    .sandbox-card {
        border: 1px solid rgba(255, 142, 83, 0.28);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(255, 142, 83, 0.04);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)
    

inject_custom_css()

# Cache Loaders for Data and Models
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/predictions.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df["Forecast_Error"] = df["Predicted_Sales"] - df["Weekly_Sales"]
    df["Absolute_Error"] = df["Forecast_Error"].abs()
    df["Absolute_Percentage_Error"] = np.where(
        df["Weekly_Sales"] != 0,
        df["Absolute_Error"] / df["Weekly_Sales"],
        np.nan
    )
    return df

@st.cache_resource
def load_model(model_name="XGBoost"):
    if model_name == "XGBoost":
        return joblib.load("models/xgboost_model.pkl")
    else:
        return joblib.load("models/random_forest_model.pkl")

# Load base data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading predictions dataset: {e}. Make sure `data/processed/predictions.csv` exists.")
    st.stop()

# Load models
xgb_model = load_model("XGBoost")
# Lazy load RF model when needed
rf_model = None

# Title Section
st.markdown('<div class="gradient-text">🛒 Retail Demand Forecasting Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-generation sales predictive modeling & macroeconomic scenario planning</div>', unsafe_allow_html=True)

# Navigation Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/dashboard.png", width=80)
st.sidebar.header("Navigation & Settings")

# Model Selection
selected_model_name = st.sidebar.selectbox("Active Forecast Model", ["XGBoost", "Random Forest"])
if selected_model_name == "XGBoost":
    active_model = xgb_model
else:
    if rf_model is None:
        with st.spinner("Loading Random Forest Model (may take a moment)..."):
            rf_model = load_model("Random Forest")
    active_model = rf_model

st.sidebar.markdown("---")
st.sidebar.info(
    "📊 **Dataset Overview**\n"
    f"- **Stores**: 45 Walmart Stores\n"
    f"- **Historical Dates**: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}\n"
    f"- **Data Points**: {len(df):,} records\n"
)

# Tabs definitions
tab_dash, tab_pred, tab_sandbox, tab_compare, tab_bulk, tab_diag, tab_powerbi = st.tabs([
    "📈 Executive Dashboard",
    "🔮 Real-Time Sales Predictor",
    "🧪 What-If Scenario Sandbox",
    "📊 Store Comparison",
    "📂 Bulk Predictions",
    "🧠 Model Diagnostics",
    "📊 Power BI Dashboard"
])

# Helpers
def format_currency(val):
    if val >= 1e9:
        return f"${val/1e9:,.2f}B"
    elif val >= 1e6:
        return f"${val/1e6:,.2f}M"
    else:
        return f"${val:,.2f}"

def render_kpi(title, value, subtitle=None):
    sub_html = f'<div style="font-size: 0.75rem; color: gray; margin-top: 4px;">{subtitle}</div>' if subtitle else ''
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """

def get_powerbi_export(data):
    export_df = data.copy()
    export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d")
    export_df["Sales_Gap"] = export_df["Predicted_Sales"] - export_df["Weekly_Sales"]
    export_df["Sales_Gap_Pct"] = np.where(
        export_df["Weekly_Sales"] != 0,
        export_df["Sales_Gap"] / export_df["Weekly_Sales"],
        np.nan
    )
    export_df["Holiday_Label"] = np.where(export_df["Holiday_Flag"] == 1, "Holiday", "Regular")
    return export_df[
        [
            "Store", "Date", "Year", "Month", "Week", "Day", "Holiday_Flag", "Holiday_Label",
            "Temperature", "Fuel_Price", "CPI", "Unemployment", "Weekly_Sales",
            "Predicted_Sales", "Sales_Gap", "Sales_Gap_Pct", "Absolute_Error",
            "Absolute_Percentage_Error"
        ]
    ]

# ==================== TAB 1: EXECUTIVE DASHBOARD ====================
with tab_dash:
    st.subheader("Retail Sales Insights & Forecast Trends")
    
    # Global KPI Calculations
    total_hist = df['Weekly_Sales'].sum()
    total_pred = df['Predicted_Sales'].sum()
    avg_weekly = df['Weekly_Sales'].mean()
    
    # Find best store
    best_store_row = df.groupby('Store')['Weekly_Sales'].mean().idxmax()
    best_store_sales = df.groupby('Store')['Weekly_Sales'].mean().max()
    
    # Display KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_kpi("Total Historical Sales", format_currency(total_hist), "Aggregate store sales"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_kpi("Total Predicted Sales", format_currency(total_pred), f"Using {selected_model_name}"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_kpi("Average Weekly Sales", format_currency(avg_weekly), "Per store week average"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_kpi(f"Top Performing Store", f"Store {best_store_row}", f"Avg: {format_currency(best_store_sales)}"), unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filter Section
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        st.markdown("### 🔍 Filters")
        selected_stores = st.multiselect("Select Stores to Compare", sorted(df['Store'].unique()), default=[1])
        
        min_date = df['Date'].min().to_pydatetime()
        max_date = df['Date'].max().to_pydatetime()
        
        date_range = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        
    with col_f2:
        # Filtered Chart Data
        filtered_df = df[
            (df['Store'].isin(selected_stores)) & 
            (df['Date'] >= date_range[0]) & 
            (df['Date'] <= date_range[1])
        ]
        
        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
        else:
            st.markdown(f"### 📈 Historical vs Predicted Weekly Sales (Stores {', '.join(map(str, selected_stores))})")
            
            # Group by Date to sum sales over selected stores
            chart_df = filtered_df.groupby('Date')[['Weekly_Sales', 'Predicted_Sales']].sum().reset_index()
            chart_df.rename(columns={'Weekly_Sales': 'Historical Sales', 'Predicted_Sales': 'Predicted Sales'}, inplace=True)
            
            # Melt for Altair format
            melted_chart_df = chart_df.melt('Date', var_name='Type', value_name='Sales')
            
            # Altair Chart
            dash_chart = alt.Chart(melted_chart_df).mark_line(strokeWidth=2).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Sales:Q', title='Weekly Sales ($)', scale=alt.Scale(zero=False)),
                color=alt.Color('Type:N', scale=alt.Scale(range=['#4FACFE', '#FF8E53']), title='Sales Stream'),
                tooltip=['Date:T', alt.Tooltip('Sales:Q', format='$,.2f'), 'Type:N']
            ).properties(
                height=350
            ).interactive()
            
            st.altair_chart(dash_chart, use_container_width=True)

# ==================== TAB 2: REAL-TIME SALES PREDICTOR ====================
with tab_pred:
    st.subheader("🔮 Demand Forecaster (Single Location)")
    
    st.markdown("Enter store details and macroeconomic indicators to generate a real-time weekly sales forecast.")
    
    col_p1, col_p2 = st.columns([1, 1])
    
    with col_p1:
        st.markdown("#### 🏪 Store Config")
        pred_store = st.selectbox("Target Store ID", sorted(df['Store'].unique()), key="pred_store_select")
        pred_date = st.date_input("Target Forecast Date", value=datetime.date(2012, 11, 2), min_value=datetime.date(2010, 1, 1), max_value=datetime.date(2015, 12, 31))
        pred_holiday = st.radio("Holiday Week?", ["No", "Yes"], index=0, horizontal=True)
        
        # Holiday Flag mapping
        holiday_flag = 1 if pred_holiday == "Yes" else 0
        
    with col_p2:
        st.markdown("#### 📊 Environmental Indicators")
        # Pre-fill averages for selected store as helper baseline
        store_base = df[df['Store'] == pred_store]
        avg_temp = float(store_base['Temperature'].mean())
        avg_fuel = float(store_base['Fuel_Price'].mean())
        avg_cpi = float(store_base['CPI'].mean())
        avg_unemp = float(store_base['Unemployment'].mean())
        
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            pred_temp = st.number_input("Temperature (°F)", value=round(avg_temp, 1))
            pred_fuel = st.number_input("Fuel Price ($/gal)", value=round(avg_fuel, 2))
        with col_sub2:
            pred_cpi = st.number_input("CPI (Consumer Price Index)", value=round(avg_cpi, 2))
            pred_unemp = st.number_input("Unemployment Rate (%)", value=round(avg_unemp, 2))
            
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    
    # Calculate Date Features
    pred_year = pred_date.year
    pred_month = pred_date.month
    pred_day = pred_date.day
    pred_week = pred_date.isocalendar()[1]
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        "Store": [pred_store],
        "Holiday_Flag": [holiday_flag],
        "Temperature": [pred_temp],
        "Fuel_Price": [pred_fuel],
        "CPI": [pred_cpi],
        "Unemployment": [pred_unemp],
        "Year": [pred_year],
        "Month": [pred_month],
        "Day": [pred_day],
        "Week": [pred_week]
    })
    
    # Predict button
    if st.button("Calculate Forecast", key="predict_single_btn"):
        with st.spinner("Running model prediction..."):
            pred_sales = active_model.predict(input_data)[0]
            
            # Compare to historical store average
            store_hist_avg = store_base['Weekly_Sales'].mean()
            diff_pct = ((pred_sales - store_hist_avg) / store_hist_avg) * 100
            
            # Visual presentation of results
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.markdown("#### 🎯 Prediction Results")
                st.metric(
                    label="Predicted Weekly Sales",
                    value=f"${pred_sales:,.2f}",
                    delta=f"{diff_pct:+.2f}% vs Store Average",
                    delta_color="normal"
                )
            with col_res2:
                st.markdown("#### Store Baseline Comparison")
                comparison_df = pd.DataFrame({
                    'Metric': ['Historical Avg', 'Current Prediction'],
                    'Sales': [store_hist_avg, pred_sales]
                })
                
                comp_chart = alt.Chart(comparison_df).mark_bar(cornerRadiusEnd=5).encode(
                    x=alt.X('Metric:N', title='', sort=None),
                    y=alt.Y('Sales:Q', title='Sales ($)'),
                    color=alt.Color('Metric:N', scale=alt.Scale(range=['#a0a0a5', '#4FACFE']), legend=None),
                    tooltip=[alt.Tooltip('Sales:Q', format='$,.2f')]
                ).properties(height=200)
                
                st.altair_chart(comp_chart, use_container_width=True)

# ==================== TAB 3: WHAT-IF SCENARIO SANDBOX ====================
with tab_sandbox:
    st.subheader("🧪 Macroeconomic Simulation Sandbox")
    st.markdown("Select an existing historical store profile as a **baseline**, then simulate economic shifts to see forecast changes.")
    
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.markdown("#### 📌 Step 1: Select Baseline Profile")
        sim_store = st.selectbox("Store Profile", sorted(df['Store'].unique()), key="sim_store_select")
        
        # Get dates for this store
        store_dates = df[df['Store'] == sim_store].sort_values('Date')
        date_options = store_dates['Date'].dt.strftime('%Y-%m-%d').tolist()
        
        sim_date_str = st.selectbox("Historical Date Profile", date_options)
        
        # Load baseline record
        baseline_record = store_dates[store_dates['Date'].dt.strftime('%Y-%m-%d') == sim_date_str].iloc[0]
        
        st.markdown("---")
        st.markdown("#### 🎚️ Step 2: Adjust Economic Controls")
        
        # Sliders for delta changes
        d_temp = st.slider("Temperature Change (°F)", min_value=-30.0, max_value=30.0, value=0.0, step=1.0)
        d_fuel = st.slider("Fuel Price Change (%)", min_value=-50, max_value=50, value=0, step=5)
        d_cpi = st.slider("CPI Adjustment (%)", min_value=-15, max_value=15, value=0, step=1)
        d_unemp = st.slider("Unemployment Shift (%)", min_value=-50, max_value=50, value=0, step=5)
        
    with col_s2:
        # Calculate simulated features
        simulated_temp = float(baseline_record['Temperature']) + d_temp
        simulated_fuel = float(baseline_record['Fuel_Price']) * (1 + d_fuel / 100.0)
        simulated_cpi = float(baseline_record['CPI']) * (1 + d_cpi / 100.0)
        simulated_unemp = float(baseline_record['Unemployment']) * (1 + d_unemp / 100.0)
        
        # Displays values side by side
        diff_table = pd.DataFrame({
            "Feature": ["Temperature (°F)", "Fuel Price ($/gal)", "CPI", "Unemployment (%)"],
            "Baseline Value": [
                f"{baseline_record['Temperature']:.2f}",
                f"${baseline_record['Fuel_Price']:.2f}",
                f"{baseline_record['CPI']:.2f}",
                f"{baseline_record['Unemployment']:.2f}%"
            ],
            "Simulated Value": [
                f"{simulated_temp:.2f}",
                f"${simulated_fuel:.2f}",
                f"{simulated_cpi:.2f}",
                f"{simulated_unemp:.2f}%"
            ]
        })
        
        st.markdown("#### 🔍 Input Comparison")
        st.dataframe(diff_table, use_container_width=True, hide_index=True)
        
        # Predict baseline and simulated
        baseline_input = pd.DataFrame([baseline_record[
            ["Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Year", "Month", "Day", "Week"]
        ]])
        
        simulated_input = pd.DataFrame({
            "Store": [int(baseline_record['Store'])],
            "Holiday_Flag": [int(baseline_record['Holiday_Flag'])],
            "Temperature": [simulated_temp],
            "Fuel_Price": [simulated_fuel],
            "CPI": [simulated_cpi],
            "Unemployment": [simulated_unemp],
            "Year": [int(baseline_record['Year'])],
            "Month": [int(baseline_record['Month'])],
            "Day": [int(baseline_record['Day'])],
            "Week": [int(baseline_record['Week'])]
        })
        
        base_pred_sales = active_model.predict(baseline_input)[0]
        sim_pred_sales = active_model.predict(simulated_input)[0]
        actual_sales = float(baseline_record['Weekly_Sales'])
        
        sim_diff_val = sim_pred_sales - base_pred_sales
        sim_diff_pct = (sim_diff_val / base_pred_sales) * 100
        
        # Display simulated metrics
        st.markdown('<div class="sandbox-card">', unsafe_allow_html=True)
        col_card1, col_card2 = st.columns(2)
        with col_card1:
            st.metric(
                label="Simulated Predicted Sales",
                value=f"${sim_pred_sales:,.2f}",
                delta=f"{sim_diff_val:+,.2f} ({sim_diff_pct:+.2f}%)",
                delta_color="normal"
            )
        with col_card2:
            st.metric(label="Actual Sales (Historical)", value=f"${actual_sales:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visual representation of impact
        sim_comp_df = pd.DataFrame({
            'Scenario': ['Actual Sales', 'Baseline Forecast', 'Simulated Scenario Forecast'],
            'Weekly Sales': [actual_sales, base_pred_sales, sim_pred_sales]
        })
        
        sim_chart = alt.Chart(sim_comp_df).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X('Scenario:N', title='', sort=None),
            y=alt.Y('Weekly Sales:Q', title='Sales ($)'),
            color=alt.Color('Scenario:N', scale=alt.Scale(range=['#a0a0a5', '#FF8E53', '#4FACFE']), legend=None),
            tooltip=[alt.Tooltip('Weekly Sales:Q', format='$,.2f')]
        ).properties(height=250)
        
        st.altair_chart(sim_chart, use_container_width=True)

# ==================== TAB 4: STORE COMPARISON ====================
with tab_compare:
    st.subheader("📊 Store Performance Comparison & Rankings")
    
    # Store sales aggregates
    store_sales = df.groupby('Store')['Weekly_Sales'].mean().reset_index()
    store_sales.rename(columns={'Weekly_Sales': 'Average Weekly Sales'}, inplace=True)
    
    # Sorting
    store_sales_sorted = store_sales.sort_values('Average Weekly Sales', ascending=False)
    
    st.markdown("#### Average Sales Leaderboard (All 45 Stores)")
    
    leader_chart = alt.Chart(store_sales_sorted).mark_bar(color='#4FACFE', cornerRadiusEnd=3).encode(
        x=alt.X('Store:O', title='Store ID', sort='-y'),
        y=alt.Y('Average Weekly Sales:Q', title='Average Weekly Sales ($)'),
        tooltip=['Store:O', alt.Tooltip('Average Weekly Sales:Q', format='$,.2f')]
    ).properties(height=280)
    
    st.altair_chart(leader_chart, use_container_width=True)
    
    # Detail columns
    st.markdown("---")
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("##### 🏆 Top 5 Performing Stores")
        st.dataframe(
            store_sales_sorted.head(5).style.format({'Average Weekly Sales': '${:,.2f}'}),
            use_container_width=True,
            hide_index=True
        )
    with col_c2:
        st.markdown("##### 📉 Bottom 5 Performing Stores")
        st.dataframe(
            store_sales_sorted.tail(5).style.format({'Average Weekly Sales': '${:,.2f}'}),
            use_container_width=True,
            hide_index=True
        )

# ==================== TAB 5: BULK FORECASTS ====================
with tab_bulk:
    st.subheader("📂 Batch Demand Forecasting")
    st.markdown("Process multi-store forecasts by uploading a CSV file. The file must match the template schema.")
    
    # Template structure
    template_df = pd.DataFrame({
        "Store": [1, 2],
        "Date": ["2012-11-02", "2012-11-02"],
        "Holiday_Flag": [0, 1],
        "Temperature": [55.3, 62.1],
        "Fuel_Price": [3.58, 3.71],
        "CPI": [223.4, 222.9],
        "Unemployment": [6.57, 6.18]
    })
    
    # Download helper
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    
    st.download_button(
        label="Download CSV Template 📥",
        data=csv_str,
        file_name="forecast_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Forecast Features CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            st.success("CSV file successfully uploaded!")
            
            # Validation
            required_cols = ["Store", "Date", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
            missing_cols = [c for c in required_cols if c not in bulk_df.columns]
            
            if missing_cols:
                st.error(f"Missing columns in uploaded CSV: {missing_cols}. Please use the template format.")
            else:
                # Preprocess dates
                bulk_df['Parsed_Date'] = pd.to_datetime(bulk_df['Date'])
                bulk_df['Year'] = bulk_df['Parsed_Date'].dt.year
                bulk_df['Month'] = bulk_df['Parsed_Date'].dt.month
                bulk_df['Day'] = bulk_df['Parsed_Date'].dt.day
                bulk_df['Week'] = bulk_df['Parsed_Date'].dt.isocalendar().week.astype(int)
                
                # Make prediction input
                pred_features = [
                    "Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Year", "Month", "Day", "Week"
                ]
                
                # Make predictions
                with st.spinner("Generating batch predictions..."):
                    predictions = active_model.predict(bulk_df[pred_features])
                    bulk_df["Predicted_Sales"] = predictions
                    
                    # Drop temporary columns for output presentation
                    output_df = bulk_df.drop(columns=['Parsed_Date', 'Year', 'Month', 'Day', 'Week'])
                    
                    st.markdown("#### 🎯 Prediction Results Preview")
                    st.dataframe(output_df.head(20).style.format({'Predicted_Sales': '${:,.2f}'}), use_container_width=True)
                    
                    # Download predicted CSV
                    out_buffer = io.StringIO()
                    output_df.to_csv(out_buffer, index=False)
                    out_str = out_buffer.getvalue()
                    
                    st.download_button(
                        label="Download Predicted Sales CSV 🚀",
                        data=out_str,
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

# ==================== TAB 6: MODEL DIAGNOSTICS ====================
with tab_diag:
    st.subheader("🧠 Machine Learning Model Diagnostic Metrics")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("#### Performance Metrics")
        
        # Static table of metrics from training script results
        metrics_df = pd.DataFrame({
            "Model Selection": ["XGBoost", "Random Forest"],
            "R² Score (Variance Explained)": [0.9779, 0.9657],
            "Mean Absolute Error (MAE)": ["$49,358.11", "$53,528.90"],
            "Model Size on Disk": ["1.3 MB", "93.8 MB"]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown(
            "💡 **Insights**:\n"
            "- **XGBoost** provides higher forecasting accuracy (R² = 97.79%) while being substantially faster and taking up 98% less memory space compared to Random Forest.\n"
            "- Holiday weeks dramatically increase demand. Adjusting the What-If holiday flag validates this behavior."
        )
        
    with col_d2:
        st.markdown("#### Feature Importances")
        # Load and draw feature importances
        features_list = [
            "Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Year", "Month", "Day", "Week"
        ]
        
        if hasattr(active_model, 'feature_importances_'):
            importances = active_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                "Feature": features_list,
                "Importance": importances
            }).sort_values('Importance', ascending=False)
            
            imp_chart = alt.Chart(feat_imp_df).mark_bar(color='#FF8E53', cornerRadiusEnd=3).encode(
                x=alt.X('Importance:Q', title='Relative Importance Score'),
                y=alt.Y('Feature:N', title='Feature Name', sort='-x'),
                tooltip=['Feature', alt.Tooltip('Importance:Q', format='.4f')]
            ).properties(height=280)
            
            st.altair_chart(imp_chart, use_container_width=True)
        else:
            st.warning("Feature importance is not available for this model type.")

# ==================== TAB 7: POWER BI DASHBOARD ====================
with tab_powerbi:
    st.subheader("📊 Power BI Analysis Pack")
    st.markdown("Use the prepared dataset below to build a Power BI dashboard with sales performance, forecast accuracy, and store-level drilldowns.")

    powerbi_df = get_powerbi_export(df)

    pbi_col1, pbi_col2, pbi_col3 = st.columns(3)
    with pbi_col1:
        st.metric("Power BI Rows", f"{len(powerbi_df):,}")
    with pbi_col2:
        st.metric("Average Forecast Error", f"${df['Absolute_Error'].mean():,.2f}")
    with pbi_col3:
        st.metric("Mean APE", f"{df['Absolute_Percentage_Error'].mean() * 100:.2f}%")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    pbi_filter_col, pbi_chart_col = st.columns([1, 2])
    with pbi_filter_col:
        st.markdown("#### Export Filters")
        pbi_stores = st.multiselect(
            "Stores",
            sorted(powerbi_df["Store"].unique()),
            default=sorted(powerbi_df["Store"].unique())[:5],
            key="powerbi_store_filter"
        )
        pbi_years = st.multiselect(
            "Years",
            sorted(powerbi_df["Year"].unique()),
            default=sorted(powerbi_df["Year"].unique()),
            key="powerbi_year_filter"
        )

    filtered_powerbi_df = powerbi_df[
        (powerbi_df["Store"].isin(pbi_stores)) &
        (powerbi_df["Year"].isin(pbi_years))
    ]

    with pbi_chart_col:
        st.markdown("#### Store Forecast Accuracy Preview")
        if filtered_powerbi_df.empty:
            st.warning("Select at least one store and year to preview the Power BI export.")
        else:
            accuracy_preview = (
                filtered_powerbi_df
                .groupby("Store", as_index=False)
                .agg(
                    Actual_Sales=("Weekly_Sales", "sum"),
                    Predicted_Sales=("Predicted_Sales", "sum"),
                    Mean_Absolute_Error=("Absolute_Error", "mean")
                )
            )
            accuracy_chart = alt.Chart(accuracy_preview).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("Store:O", title="Store"),
                y=alt.Y("Mean_Absolute_Error:Q", title="Mean Absolute Error ($)"),
                color=alt.Color("Store:N", legend=None),
                tooltip=[
                    "Store:O",
                    alt.Tooltip("Actual_Sales:Q", format="$,.2f"),
                    alt.Tooltip("Predicted_Sales:Q", format="$,.2f"),
                    alt.Tooltip("Mean_Absolute_Error:Q", format="$,.2f")
                ]
            ).properties(height=300)
            st.altair_chart(accuracy_chart, use_container_width=True)

    st.markdown("#### Recommended Power BI Pages")
    st.dataframe(
        pd.DataFrame({
            "Page": ["Executive Overview", "Store Drilldown", "Forecast Accuracy", "Economic Drivers"],
            "Visuals": [
                "KPI cards, sales trend line, top and bottom stores",
                "Store slicer, actual vs predicted trend, weekly table",
                "MAE by store, error percentage by month, outlier weeks",
                "Fuel price, CPI, unemployment, temperature vs weekly sales"
            ],
            "Primary Slicers": [
                "Date, store, holiday label",
                "Store, date",
                "Store, year, holiday label",
                "Store, year, month"
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    csv_bytes = powerbi_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Power BI Dataset",
        data=csv_bytes,
        file_name="retail_demand_powerbi_dataset.csv",
        mime="text/csv"
    )

    st.markdown("#### DAX Measures")
    st.code(
        """Total Actual Sales = SUM(retail_demand_powerbi_dataset[Weekly_Sales])
Total Predicted Sales = SUM(retail_demand_powerbi_dataset[Predicted_Sales])
Forecast Gap = [Total Predicted Sales] - [Total Actual Sales]
Forecast Gap % = DIVIDE([Forecast Gap], [Total Actual Sales])
Mean Absolute Error = AVERAGE(retail_demand_powerbi_dataset[Absolute_Error])
Mean Absolute Percentage Error = AVERAGE(retail_demand_powerbi_dataset[Absolute_Percentage_Error])
Holiday Sales = CALCULATE([Total Actual Sales], retail_demand_powerbi_dataset[Holiday_Label] = "Holiday")""",
        language="DAX"
    )
