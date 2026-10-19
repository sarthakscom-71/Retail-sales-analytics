import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="Retail Sales Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales Analytics & Demand Forecasting")
st.caption("Sales intelligence and 30-day demand prediction")

URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00352/Online%20Retail.xlsx"
)

@st.cache_data
def load_data():
    return pd.read_excel(URL, engine="openpyxl")

uploaded = st.sidebar.file_uploader(
    "Upload retail Excel or CSV",
    type=["xlsx", "csv"]
)

try:
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    else:
        df = load_data()

    required = ["InvoiceNo", "StockCode", "Description",
                "Quantity", "InvoiceDate", "UnitPrice"]

    if not all(c in df.columns for c in required):
        st.error("Dataset columns do not match the expected retail format.")
        st.stop()

    df = df.dropna(subset=["InvoiceDate", "Description"]).copy()
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
    df = df[
        (~df["InvoiceNo"].astype(str).str.startswith("C")) &
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.normalize()
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

except Exception as e:
    st.error(f"Unable to load data: {e}")
    st.stop()

st.sidebar.header("Dashboard Filters")

countries = sorted(df["Country"].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Country", countries, default=countries[:5]
)

products = sorted(df["Description"].dropna().unique())
selected_products = st.sidebar.multiselect(
    "Products", products, default=[]
)

start = st.sidebar.date_input(
    "Start date", value=df["Date"].min().date()
)
end = st.sidebar.date_input(
    "End date", value=df["Date"].max().date()
)

if start > end:
    st.error("Start date must be before end date.")
    st.stop()

filtered = df[
    df["Date"].between(pd.Timestamp(start), pd.Timestamp(end)) &
    df["Country"].isin(selected_countries)
].copy()

if selected_products:
    filtered = filtered[
        filtered["Description"].isin(selected_products)
    ]

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

revenue = filtered["Revenue"].sum()
units = filtered["Quantity"].sum()
invoices = filtered["InvoiceNo"].nunique()
product_count = filtered["StockCode"].nunique()

a, b, c, d = st.columns(4)
a.metric("Total Revenue", f"£{revenue:,.2f}")
b.metric("Units Sold", f"{units:,.0f}")
c.metric("Invoices", f"{invoices:,}")
d.metric("Unique Products", f"{product_count:,}")

left, right = st.columns(2)

with left:
    monthly = filtered.groupby("Month", as_index=False)["Revenue"].sum()
    st.plotly_chart(
        px.line(monthly, x="Month", y="Revenue",
                markers=True, title="Monthly Revenue"),
        use_container_width=True
    )

with right:
    top = (
        filtered.groupby("Description", as_index=False)["Revenue"]
        .sum().nlargest(10, "Revenue")
    )
    st.plotly_chart(
        px.bar(top, x="Revenue", y="Description",
               orientation="h", title="Top 10 Products"),
        use_container_width=True
    )

country_sales = (
    filtered.groupby("Country", as_index=False)["Revenue"]
    .sum().nlargest(10, "Revenue")
)
st.plotly_chart(
    px.bar(country_sales, x="Country", y="Revenue",
           title="Revenue by Country"),
    use_container_width=True
)

st.header("🔮 30-Day Demand Forecast")

daily = (
    filtered.groupby("Date")["Quantity"]
    .sum().sort_index()
    .asfreq("D", fill_value=0)
)

if len(daily) < 60:
    st.warning("Select at least 60 days of data for forecasting.")
else:
    try:
        model = ExponentialSmoothing(
            daily.astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated"
        ).fit(optimized=True)

        predicted = model.forecast(30).clip(lower=0)

        forecast = pd.DataFrame({
            "Date": predicted.index,
            "Forecast Units": predicted.values
        })

        st.metric(
            "Expected demand over 30 days",
            f"{forecast['Forecast Units'].sum():,.0f} units"
        )

        fig = px.line(
            daily.tail(90).rename("Actual Units").reset_index(),
            x="Date", y="Actual Units",
            title="Historical Demand"
        )

        fig.add_scatter(
            x=forecast["Date"],
            y=forecast["Forecast Units"],
            mode="lines+markers",
            name="30-Day Forecast"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(forecast, use_container_width=True)

        st.download_button(
            "Download 30-Day Forecast",
            forecast.to_csv(index=False),
            "retail_forecast.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Forecast could not be generated: {e}")

st.subheader("Retail Transaction Data")
st.dataframe(filtered.head(500), use_container_width=True)

st.download_button(
    "Download Filtered Sales Data",
    filtered.to_csv(index=False),
    "retail_sales_report.csv",
    "text/csv"
)
