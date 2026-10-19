
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="General Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 General-Purpose Data Analysis Dashboard")
st.write(
    "Upload a CSV or Excel file to explore, clean, "
    "visualize, and analyze your data."
)

uploaded = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded is None:
    st.info("Upload a CSV or Excel file using the sidebar to begin.")
    st.markdown("""
    ### Features
    - Automatic dataset summary
    - Missing-value and duplicate detection
    - Descriptive statistics
    - Interactive charts
    - Download cleaned data
    """)
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        excel_file = pd.ExcelFile(uploaded)
        sheet = st.sidebar.selectbox(
            "Choose Excel sheet",
            excel_file.sheet_names
        )
        df = pd.read_excel(excel_file, sheet_name=sheet)

except Exception as e:
    st.error(f"Could not read this file: {e}")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("This file contains no usable data.")
    st.stop()

df.columns = df.columns.astype(str)

# Optionally remove duplicates and rows with missing values
st.sidebar.header("Data cleaning")
remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows", value=False
)
remove_missing = st.sidebar.checkbox(
    "Remove rows with missing values", value=False
)

data = df.copy()

if remove_duplicates:
    data = data.drop_duplicates()

if remove_missing:
    data = data.dropna()

# Dashboard metrics
st.header("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{data.shape[0]:,}")
c2.metric("Columns", f"{data.shape[1]:,}")
c3.metric("Missing Cells", f"{data.isna().sum().sum():,}")
c4.metric("Duplicate Rows", f"{data.duplicated().sum():,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Preview",
    "Data Quality",
    "Statistics",
    "Visualizations"
])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.head(100), use_container_width=True)

    st.subheader("Column Information")
    info = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Missing Values": data.isna().sum().values,
        "Unique Values": data.nunique().values
    })
    st.dataframe(info, use_container_width=True)

    st.download_button(
        "Download Current Dataset as CSV",
        data.to_csv(index=False).encode("utf-8"),
        "analyzed_dataset.csv",
        "text/csv"
    )

with tab2:
    st.subheader("Missing Values by Column")
    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("No missing values found.")
    else:
        missing_df = missing.rename("Missing Count").reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        st.dataframe(missing_df, use_container_width=True)

        st.plotly_chart(
            px.bar(
                missing_df,
                x="Column",
                y="Missing Count",
                title="Missing Values by Column"
            ),
            use_container_width=True
        )

    st.write("Duplicate rows:", int(data.duplicated().sum()))

with tab3:
    st.subheader("Descriptive Statistics")
    st.dataframe(
        data.describe(include="all").T,
        use_container_width=True
    )

    numeric = data.select_dtypes(include="number")
    if len(numeric.columns) >= 2:
        st.subheader("Correlation Matrix")
        corr = numeric.corr()
        st.plotly_chart(
            px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Numeric Column Correlations"
            ),
            use_container_width=True
        )

with tab4:
    st.subheader("Create Your Own Chart")

    columns = data.columns.tolist()
    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    chart_type = st.selectbox(
        "Choose chart type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"]
    )

    x_col = st.selectbox("Choose X-axis column", columns)

    if chart_type == "Histogram":
        if pd.api.types.is_numeric_dtype(data[x_col]):
            fig = px.histogram(
                data, x=x_col, title=f"Distribution of {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Choose a numeric column for a histogram.")

    elif chart_type == "Scatter Plot":
        possible_y = [
            c for c in numeric_columns
            if c != x_col
        ]
        if (
            x_col in numeric_columns
            and possible_y
        ):
            y_col = st.selectbox("Choose Y-axis column", possible_y)
            st.plotly_chart(
                px.scatter(
                    data, x=x_col, y=y_col,
                    title=f"{y_col} vs {x_col}"
                ),
                use_container_width=True
            )
        else:
            st.warning(
                "Scatter plots require numeric X and Y columns. "
                "Choose a numeric X-axis column."
            )

    else:
        possible_y = [
            c for c in numeric_columns
            if c != x_col
        ]

        if possible_y:
            y_col = st.selectbox("Choose Y-axis column", possible_y)

            if chart_type == "Bar Chart":
                fig = px.bar(
                    data, x=x_col, y=y_col,
                    title=f"{y_col} by {x_col}"
                )
            else:
                fig = px.line(
                    data, x=x_col, y=y_col,
                    title=f"{y_col} over {x_col}"
                )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "This chart needs a numeric Y-axis column. "
                "Choose a different X-axis column."
            )

st.caption(
    "General-purpose analysis tool | "
    "Results depend on the uploaded dataset."
)

