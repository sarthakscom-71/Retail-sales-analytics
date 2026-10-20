import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(
    page_title="DataNova | Data Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- UI theme ----------
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">DN</div>
      <div><div class="brand-name">DataNova</div><div class="brand-sub">INTELLIGENCE STUDIO</div></div>
    </div>
    """, unsafe_allow_html=True)

theme = st.sidebar.selectbox("Appearance", ["Light", "Dark"], index=0)
if theme == "Dark":
    bg, panel, text, muted, border = "#0d1424", "#151f33", "#edf3ff", "#a5b4cc", "#2b3a53"
    plot_template = "plotly_dark"
else:
    bg, panel, text, muted, border = "#f4f7fb", "#ffffff", "#17243a", "#65748b", "#e2e8f0"
    plot_template = "plotly_white"

st.markdown(f"""
<style>
:root {{ color-scheme: {'dark' if theme == 'Dark' else 'light'}; }}
.stApp {{ background: {bg}; color: {text}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }}
.brand {{ display:flex; align-items:center; gap:11px; padding: 6px 0 22px 0; }}
.brand-icon {{ background:linear-gradient(135deg,#3b82f6,#06b6d4); color:white; font-weight:800; border-radius:12px; width:43px; height:43px; display:flex; align-items:center; justify-content:center; letter-spacing:-1px; }}
.brand-name {{ font-size:20px; font-weight:800; color:{text}; line-height:1.1; }}
.brand-sub {{ font-size:9px; color:{muted}; letter-spacing:1.6px; margin-top:4px; }}
.hero {{ background:linear-gradient(115deg,#172554 0%,#1d4ed8 60%,#0891b2 100%); padding:28px 30px; border-radius:22px; color:#fff; margin-bottom:20px; box-shadow:0 12px 30px rgba(29,78,216,.18); }}
.hero-kicker {{ text-transform:uppercase; letter-spacing:2px; font-size:11px; color:#bfdbfe; font-weight:700; }}
.hero h1 {{ color:white !important; font-size:32px !important; line-height:1.2; margin:8px 0 8px 0; }}
.hero p {{ color:#dbeafe; margin:0; font-size:15px; }}
.hero-pill {{ display:inline-block; border:1px solid rgba(255,255,255,.35); border-radius:999px; padding:6px 11px; margin-top:17px; font-size:11px; color:#eff6ff; }}
.section-head {{ font-size:20px; font-weight:750; margin: 8px 0 4px 0; color:{text}; }}
.section-sub {{ color:{muted}; font-size:13px; margin-bottom:15px; }}
[data-testid="stMetric"] {{ background:{panel}; padding:18px 18px 14px 18px; border:1px solid {border}; border-radius:16px; box-shadow:0 4px 14px rgba(15,23,42,.035); }}
[data-testid="stMetricLabel"] {{ color:{muted}; font-size:12px; }}
[data-testid="stMetricValue"] {{ color:{text}; font-weight:750; }}
[data-testid="stTabs"] button {{ border-radius:10px 10px 0 0; font-weight:650; }}
div[data-testid="stDataFrame"] {{ border:1px solid {border}; border-radius:12px; overflow:hidden; }}
div.stButton > button, div.stDownloadButton > button {{ border-radius:10px; font-weight:650; min-height:42px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:16px; }}
div[data-testid="stSidebarContent"] {{ background:{panel}; }}
.small-note {{ color:{muted}; font-size:12px; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Your workspace for smarter decisions</div>
  <h1>Data intelligence, made interactive.</h1>
  <p>Explore datasets, find data quality issues, build visual insights, and test predictive models — all in one place.</p>
  <div class="hero-pill">✦ ANALYTICS &nbsp; · &nbsp; VISUALIZATION &nbsp; · &nbsp; MACHINE LEARNING</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Your workspace")
    uploaded = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"], help="Supported formats: CSV and Excel")
    st.markdown("---")
    st.markdown("### Data preparation")
    dedupe = st.checkbox("Remove duplicate rows", value=False)
    missing_mode = st.selectbox("Missing-value strategy", [
        "Keep unchanged",
        "Remove rows with missing values",
        "Fill numeric median and text mode"
    ])
    st.markdown("---")
    st.markdown('<div class="small-note">DATANOVA · COLLEGE PROJECT EDITION</div>', unsafe_allow_html=True)

if not uploaded:
    st.markdown('<div class="section-head">Welcome to your analytics workspace</div><div class="section-sub">Upload a dataset to activate your tools. You can use student marks, retail sales, or another tabular CSV/Excel file.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        ("01", "Explore your data", "Preview records, inspect data types, and review useful summary statistics."),
        ("02", "Find data issues", "Check missing values, duplicate records, and potential numeric outliers."),
        ("03", "Build insights", "Create interactive charts, compare correlations, and train prediction models.")
    ]
    for col, (n, title, desc) in zip([c1,c2,c3], cards):
        with col:
            with st.container(border=True):
                st.markdown(f"### <span style='color:#3b82f6'>{n}</span> · {title}", unsafe_allow_html=True)
                st.write(desc)
    st.info("Use the upload control in the sidebar to get started.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        xls = pd.ExcelFile(uploaded)
        sheet = st.sidebar.selectbox("Choose worksheet", xls.sheet_names)
        raw = pd.read_excel(xls, sheet_name=sheet)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

if raw.empty:
    st.warning("This file contains no data rows.")
    st.stop()

df = raw.copy()
if dedupe:
    df = df.drop_duplicates()
if missing_mode == "Remove rows with missing values":
    df = df.dropna()
elif missing_mode == "Fill numeric median and text mode":
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            modes = df[col].mode(dropna=True)
            if not modes.empty:
                df[col] = df[col].fillna(modes.iloc[0])

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
missing_total = int(df.isna().sum().sum())
duplicate_total = int(df.duplicated().sum())

top_left, top_right = st.columns([3,1])
with top_left:
    st.markdown(f'<div class="section-head">Workspace overview</div><div class="section-sub">Currently analyzing <b>{uploaded.name}</b></div>', unsafe_allow_html=True)
with top_right:
    st.markdown(f'<div style="text-align:right;padding-top:12px;color:{muted};font-size:12px">● Dataset loaded &nbsp; <b>{len(df):,}</b> rows</div>', unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Data Explorer", "Data Quality", "Visual Analytics", "Machine Learning", "Ask Your Data", "Reports & Export"])

with tabs[0]:
    st.markdown('<div class="section-head">At a glance</div><div class="section-sub">A quick snapshot of the dataset after applying your selected preparation options.</div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric("Total records", f"{len(df):,}", help="Number of rows currently in the dataset")
    b.metric("Data fields", f"{len(df.columns):,}", help="Number of columns")
    c.metric("Missing cells", f"{missing_total:,}", help="Cells that still have missing values")
    d.metric("Duplicate rows", f"{duplicate_total:,}", help="Duplicate rows remaining")
    left,right = st.columns([1.35,1])
    with left:
        st.markdown("#### Dataset preview")
        st.dataframe(df.head(12), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Data type distribution")
        type_counts = df.dtypes.astype(str).value_counts().rename_axis("Data type").reset_index(name="Columns")
        fig = px.pie(type_counts, names="Data type", values="Columns", hole=.62)
        fig.update_layout(template=plot_template, margin=dict(t=20,b=20,l=5,r=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    if numeric_cols:
        st.markdown("#### Numeric summary")
        st.dataframe(df[numeric_cols].describe().T.round(3), use_container_width=True)

with tabs[1]:
    st.markdown('<div class="section-head">Data Explorer</div><div class="section-sub">Search fields and inspect records in your dataset.</div>', unsafe_allow_html=True)
    search = st.text_input("Search for a column", placeholder="Start typing a column name...")
    choices = [c for c in df.columns if search.lower() in str(c).lower()]
    selected = st.multiselect("Columns to display", choices, default=choices)
    if selected:
        st.dataframe(df[selected], use_container_width=True, hide_index=True)
    profile = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(float(df[c].isna().mean()*100),2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    st.markdown("#### Column profile")
    st.dataframe(profile, use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown('<div class="section-head">Data Quality</div><div class="section-sub">Understand completeness and identify records that may need attention.</div>', unsafe_allow_html=True)
    l,r = st.columns(2)
    with l:
        st.markdown("#### Missing values")
        miss = df.isna().sum().rename_axis("Column").reset_index(name="Missing values")
        miss = miss[miss["Missing values"] > 0].sort_values("Missing values", ascending=False)
        if miss.empty:
            st.success("No missing values found in the current dataset.")
        else:
            fig = px.bar(miss, x="Column", y="Missing values")
            fig.update_layout(template=plot_template)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss, use_container_width=True, hide_index=True)
    with r:
        st.markdown("#### Duplicate records")
        st.metric("Duplicate rows remaining", f"{duplicate_total:,}")
        st.markdown("#### Potential numeric outliers (IQR)")
        out = []
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) >= 4:
                q1, q3 = s.quantile(.25), s.quantile(.75)
                iqr = q3 - q1
                count = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
                out.append({"Column": col, "Potential outliers": count})
        if out:
            st.dataframe(pd.DataFrame(out).sort_values("Potential outliers", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No suitable numeric columns for outlier scanning.")

with tabs[3]:
    st.markdown('<div class="section-head">Visual Analytics</div><div class="section-sub">Create charts and explore relationships interactively.</div>', unsafe_allow_html=True)
    chart_type = st.selectbox("Visualization", ["Histogram", "Bar chart", "Line chart", "Scatter plot", "Box plot"])
    x_col = st.selectbox("X-axis / category", df.columns.tolist())
    y_candidates = [c for c in numeric_cols if c != x_col]
    y_col = st.selectbox("Numeric Y-axis (if needed)", y_candidates) if y_candidates else None
    color_choices = ["None"] + [c for c in df.columns if c != x_col]
    color_choice = st.selectbox("Color grouping (optional)", color_choices)
    color_col = None if color_choice == "None" else color_choice
    try:
        fig = None
        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
        elif chart_type == "Bar chart":
            if y_col:
                agg = st.selectbox("Aggregation", ["Sum", "Mean", "Count"])
                grouped = df.groupby(x_col, dropna=False)[y_col].agg({"Sum":"sum","Mean":"mean","Count":"count"}[agg]).reset_index(name="Value")
                fig = px.bar(grouped, x=x_col, y="Value", title=f"{agg} of {y_col} by {x_col}")
            else:
                counts = df[x_col].astype(str).value_counts().head(30).rename_axis(x_col).reset_index(name="Count")
                fig = px.bar(counts, x=x_col, y="Count", title=f"Count by {x_col}")
        elif chart_type == "Line chart":
            if y_col: fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
            else: st.warning("A numeric Y-axis column is required for a line chart.")
        elif chart_type == "Scatter plot":
            if y_col and pd.api.types.is_numeric_dtype(df[x_col]):
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
            else: st.warning("Choose numeric columns for both axes to create a scatter plot.")
        else:
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title="Box plot")
        if fig is not None:
            fig.update_layout(template=plot_template, margin=dict(t=55,b=20,l=15,r=15))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to create this chart: {e}")
    if len(numeric_cols) >= 2:
        st.markdown("#### Correlation heatmap")
        fig = px.imshow(df[numeric_cols].corr(numeric_only=True), text_auto=True, aspect="auto", title="Relationships between numeric fields")
        fig.update_layout(template=plot_template)
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.markdown('<div class="section-head">Machine Learning Lab</div><div class="section-sub">Train a baseline model and evaluate how well it predicts a selected target.</div>', unsafe_allow_html=True)
    if len(df) < 10 or len(df.columns) < 2:
        st.warning("At least 10 rows and 2 columns are needed.")
    else:
        target = st.selectbox("Target column", df.columns.tolist(), key="ml_target")
        task = st.radio("Prediction task", ["Auto-detect", "Classification", "Regression"], horizontal=True)
        features = st.multiselect("Input features", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target][:min(8,len(df.columns)-1)])
        test_pct = st.slider("Test set size (%)", 10, 40, 20, 5)
        model_name = st.selectbox("Model", ["Random Forest", "Decision Tree", "Linear / Logistic Regression"])
        if st.button("Train and evaluate model", type="primary"):
            if not features:
                st.error("Select at least one input feature.")
            else:
                work = df[features+[target]].dropna(subset=[target]).copy()
                X, y = work[features], work[target]
                actual_task = ("Classification" if (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 15) else "Regression") if task == "Auto-detect" else task
                if len(work) < 10:
                    st.error("Not enough usable rows after removing missing target values.")
                elif actual_task == "Classification" and y.nunique() < 2:
                    st.error("Classification requires at least two target classes.")
                else:
                    num_features = X.select_dtypes(include=np.number).columns.tolist()
                    cat_features = [c for c in features if c not in num_features]
                    transformers = []
                    if num_features:
                        transformers.append(("numeric", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num_features))
                    if cat_features:
                        transformers.append(("categorical", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat_features))
                    preprocessor = ColumnTransformer(transformers=transformers)
                    if actual_task == "Classification":
                        models = {
                            "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42),
                            "Decision Tree": DecisionTreeClassifier(random_state=42),
                            "Linear / Logistic Regression": LogisticRegression(max_iter=1000)
                        }
                        stratify = y if y.value_counts().min() >= 2 and y.nunique() < len(y)*.5 else None
                    else:
                        models = {
                            "Random Forest": RandomForestRegressor(n_estimators=120, random_state=42),
                            "Decision Tree": DecisionTreeRegressor(random_state=42),
                            "Linear / Logistic Regression": LinearRegression()
                        }
                        stratify = None
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_pct/100,random_state=42,stratify=stratify)
                        pipeline = Pipeline([("preprocess",preprocessor),("model",models[model_name])])
                        pipeline.fit(X_train,y_train)
                        pred = pipeline.predict(X_test)
                        st.markdown("#### Model performance")
                        if actual_task == "Classification":
                            a,b = st.columns(2)
                            a.metric("Accuracy", f"{accuracy_score(y_test,pred):.3f}")
                            b.metric("Weighted F1 score", f"{f1_score(y_test,pred,average='weighted',zero_division=0):.3f}")
                        else:
                            a,b,c = st.columns(3)
                            a.metric("MAE", f"{mean_absolute_error(y_test,pred):.3f}")
                            b.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test,pred)):.3f}")
                            c.metric("R² score", f"{r2_score(y_test,pred):.3f}")
                        predictions = pd.DataFrame({"Actual":y_test.reset_index(drop=True),"Predicted":pd.Series(pred).reset_index(drop=True)})
                        st.dataframe(predictions.head(100),use_container_width=True)
                        st.download_button("Download predictions CSV",predictions.to_csv(index=False).encode("utf-8"),"predictions.csv","text/csv")
                    except Exception as e:
                        st.error(f"Model training failed: {e}")
    st.caption("Model scores depend on the dataset, feature choices, and target definition.")

with tabs[5]:
    st.markdown('<div class="section-head">Ask Your Data</div><div class="section-sub">Ask common questions about your current dataset. This built-in helper uses rules and statistics, not a generative AI service.</div>', unsafe_allow_html=True)
    question = st.text_input("Your question", placeholder="e.g. Which column has the most missing values?")
    if st.button("Find answer"):
        q = question.lower().strip()
        if not q:
            st.warning("Type a question first.")
        elif "missing" in q or "null" in q:
            m = df.isna().sum().sort_values(ascending=False)
            st.write(f"Total missing cells: **{int(m.sum()):,}**. Highest: **{m.index[0]}** ({int(m.iloc[0])}).")
            if m.sum(): st.dataframe(m[m>0].rename("Missing values").reset_index(),use_container_width=True)
        elif "duplicate" in q:
            st.write(f"The dataset has **{duplicate_total:,} duplicate rows**.")
        elif "row" in q or "record" in q:
            st.write(f"The dataset has **{len(df):,} rows** and **{len(df.columns):,} columns**.")
        elif "column" in q:
            st.write("Columns: " + ", ".join(map(str,df.columns.tolist())))
        elif "average" in q or "mean" in q:
            if numeric_cols: st.dataframe(df[numeric_cols].mean().rename("Average").reset_index(),use_container_width=True)
            else: st.info("No numeric columns are available.")
        elif "correlation" in q and len(numeric_cols)>=2:
            corr = df[numeric_cols].corr(numeric_only=True)
            pairs = corr.where(np.triu(np.ones(corr.shape),1).astype(bool)).stack()
            if len(pairs):
                best = pairs.abs().idxmax()
                st.write(f"Strongest numeric correlation: **{best[0]}** and **{best[1]}** (r = {pairs.loc[best]:.3f}).")
            else: st.info("No correlation pairs found.")
        elif any(k in q for k in ["highest","maximum","lowest","minimum"]) and numeric_cols:
            col = st.selectbox("Numeric column",numeric_cols)
            value = df[col].max() if any(k in q for k in ["highest","maximum"]) else df[col].min()
            st.write(f"Value: **{value}** in **{col}**.")
        else:
            st.info("Try: row count, column names, missing values, duplicates, average, correlation, highest, or lowest.")

with tabs[6]:
    st.markdown('<div class="section-head">Reports & Export</div><div class="section-sub">Download the prepared dataset or create a compact summary report.</div>', unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### Cleaned dataset")
        st.download_button("Download CSV",df.to_csv(index=False).encode("utf-8"),"cleaned_dataset.csv","text/csv",use_container_width=True)
    with col2:
        st.markdown("#### Excel workbook")
        excel = BytesIO()
        with pd.ExcelWriter(excel,engine="openpyxl") as writer:
            df.to_excel(writer,sheet_name="Cleaned Data",index=False)
            profile = pd.DataFrame({"Column":df.columns.astype(str),"Type":[str(df[c].dtype) for c in df.columns],"Missing":[int(df[c].isna().sum()) for c in df.columns],"Unique":[int(df[c].nunique()) for c in df.columns]})
            profile.to_excel(writer,sheet_name="Column Profile",index=False)
        st.download_button("Download Excel report",excel.getvalue(),"data_analysis_report.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    st.markdown("---")
    if st.button("Prepare PDF summary",type="primary"):
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer,pagesize=A4)
            styles = getSampleStyleSheet()
            story = [
                Paragraph("DataNova — Dataset Summary",styles["Title"]),Spacer(1,12),
                Paragraph(f"File: {uploaded.name}",styles["Normal"]),
                Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}",styles["Normal"]),
                Paragraph(f"Rows: {len(df):,} | Columns: {len(df.columns):,}",styles["Normal"]),
                Paragraph(f"Missing cells: {missing_total:,} | Duplicate rows: {duplicate_total:,}",styles["Normal"]),
                Spacer(1,12),Paragraph("Column profile (first 30 columns)",styles["Heading2"])
            ]
            data = [["Column","Type","Missing","Unique"]]
            for c in df.columns[:30]:
                data.append([str(c)[:35],str(df[c].dtype),str(int(df[c].isna().sum())),str(int(df[c].nunique()))])
            table = Table(data,repeatRows=1,colWidths=[220,90,65,65])
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1d4ed8")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8),("PADDING",(0,0),(-1,-1),5)]))
            story.append(table)
            doc.build(story)
            st.download_button("Download PDF report",buffer.getvalue(),"datanova_summary.pdf","application/pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

st.markdown(f'<div style="text-align:center;color:{muted};font-size:11px;padding:26px 0 4px">DATANOVA DATA INTELLIGENCE STUDIO · BUILT FOR LEARNING & ANALYSIS</div>', unsafe_allow_html=True)

