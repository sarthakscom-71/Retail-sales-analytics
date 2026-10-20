
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

st.set_page_config(page_title="Data Intelligence Platform", page_icon="📊", layout="wide")
st.markdown("""<style>
.main{background:#f7f9fc}.hero{padding:22px;border-radius:16px;background:linear-gradient(120deg,#173b70,#2878b5);color:white}
.hero h1{color:white}.hero p{color:#e8f2ff}
[data-testid="stMetric"]{background:white;padding:14px;border:1px solid #e8edf5;border-radius:12px}
</style><div class="hero"><h1>📊 Data Intelligence Platform</h1><p>Explore data · Improve quality · Discover insights · Build predictive models</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Workspace")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
    st.divider()
    dedupe = st.checkbox("Remove duplicate rows")
    missing_mode = st.selectbox("Missing values", ["Keep unchanged","Remove rows with missing values","Fill numeric median and text mode"])

if not uploaded:
    st.info("Upload a CSV or Excel file from the sidebar to begin.")
    st.markdown("### Available modules")
    st.write("Dashboard • Data explorer • Data quality • Interactive charts • Machine learning • Dataset Q&A • CSV/Excel/PDF reports")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        xls = pd.ExcelFile(uploaded)
        sheet = st.sidebar.selectbox("Excel worksheet", xls.sheet_names)
        raw = pd.read_excel(xls, sheet_name=sheet)
except Exception as e:
    st.error(f"Unable to read file: {e}")
    st.stop()

df = raw.copy()
if dedupe: df = df.drop_duplicates()
if missing_mode == "Remove rows with missing values": df = df.dropna()
elif missing_mode.startswith("Fill"):
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().any(): df[c] = df[c].fillna(df[c].median())
        else:
            m = df[c].mode(dropna=True)
            if len(m): df[c] = df[c].fillna(m.iloc[0])

num = df.select_dtypes(include=np.number).columns.tolist()
cat = [c for c in df.columns if c not in num]
missing = int(df.isna().sum().sum())
dupes = int(df.duplicated().sum())
st.caption(f"Dataset: **{uploaded.name}** · {len(df):,} rows · {len(df.columns):,} columns")
tabs = st.tabs(["🏠 Overview","🔎 Data Explorer","🧹 Data Quality","📊 Visual Analytics","🤖 Machine Learning","💬 Ask Your Data","📤 Reports & Export"])

with tabs[0]:
    st.subheader("Dataset overview")
    a,b,c,d=st.columns(4)
    a.metric("Rows",f"{len(df):,}"); b.metric("Columns",f"{len(df.columns):,}")
    c.metric("Missing cells",f"{missing:,}"); d.metric("Duplicate rows",f"{dupes:,}")
    left,right=st.columns([1.4,1])
    with left:
        st.markdown("#### Data preview"); st.dataframe(df.head(20),use_container_width=True,hide_index=True)
    with right:
        types=df.dtypes.astype(str).value_counts().rename_axis("Data type").reset_index(name="Columns")
        st.plotly_chart(px.pie(types,names="Data type",values="Columns",hole=.45,title="Column types"),use_container_width=True)
    if num: st.dataframe(df[num].describe().T,use_container_width=True)

with tabs[1]:
    st.subheader("Data explorer")
    search=st.text_input("Search columns")
    choices=[c for c in df.columns if search.lower() in str(c).lower()]
    selected=st.multiselect("Columns to display",choices,default=choices)
    if selected: st.dataframe(df[selected],use_container_width=True,hide_index=True)
    profile=pd.DataFrame({"Column":df.columns.astype(str),"Type":[str(df[c].dtype) for c in df.columns],"Missing":[int(df[c].isna().sum()) for c in df.columns],"Missing %":[round(df[c].isna().mean()*100,2) for c in df.columns],"Unique":[int(df[c].nunique()) for c in df.columns]})
    st.markdown("#### Column profile"); st.dataframe(profile,use_container_width=True,hide_index=True)

with tabs[2]:
    st.subheader("Data quality")
    l,r=st.columns(2)
    with l:
        miss=df.isna().sum().sort_values(ascending=False); miss=miss[miss>0]
        if len(miss): st.plotly_chart(px.bar(miss.reset_index(),x="index",y="count" if "count" in miss.reset_index().columns else miss.name,title="Missing values by column"),use_container_width=True)
        else: st.success("No missing values found.")
        st.write(f"Duplicate rows: **{dupes}**")
    with r:
        st.markdown("#### Potential numeric outliers (IQR)")
        out=[]
        for c in num:
            s=df[c].dropna()
            if len(s)>=4:
                q1,q3=s.quantile(.25),s.quantile(.75); i=q3-q1
                out.append({"Column":c,"Potential outliers":int(((s<q1-1.5*i)|(s>q3+1.5*i)).sum())})
        if out: st.dataframe(pd.DataFrame(out),use_container_width=True,hide_index=True)
        else: st.info("No suitable numeric columns.")

with tabs[3]:
    st.subheader("Interactive visual analytics")
    chart=st.selectbox("Chart type",["Histogram","Bar chart","Line chart","Scatter plot","Box plot"])
    x=st.selectbox("X / category",df.columns.tolist())
    ys=[c for c in num if c!=x]
    y=st.selectbox("Numeric Y (if required)",ys) if ys else None
    color_choice=st.selectbox("Color grouping",["None"]+list(df.columns))
    color=None if color_choice=="None" else color_choice
    try:
        if chart=="Histogram": fig=px.histogram(df,x=x,color=color,title=f"Distribution of {x}")
        elif chart=="Bar chart":
            if y:
                agg=st.selectbox("Aggregation",["Sum","Mean","Count"])
                grouped=df.groupby(x,dropna=False)[y].agg({"Sum":"sum","Mean":"mean","Count":"count"}[agg]).reset_index(name="Value")
                fig=px.bar(grouped,x=x,y="Value",title=f"{agg} of {y} by {x}")
            else:
                counts=df[x].astype(str).value_counts().head(30).rename_axis(x).reset_index(name="Count")
                fig=px.bar(counts,x=x,y="Count",title=f"Count by {x}")
        elif chart=="Line chart":
            if not y: st.warning("A numeric Y column is needed."); fig=None
            else: fig=px.line(df,x=x,y=y,color=color,title=f"{y} by {x}")
        elif chart=="Scatter plot":
            if not y or not pd.api.types.is_numeric_dtype(df[x]): st.warning("Choose numeric X and Y columns."); fig=None
            else: fig=px.scatter(df,x=x,y=y,color=color,title=f"{y} vs {x}")
        else: fig=px.box(df,x=x,y=y,color=color,title="Box plot")
        if fig is not None: fig.update_layout(template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
    except Exception as e: st.error(f"Chart error: {e}")
    if len(num)>=2: st.plotly_chart(px.imshow(df[num].corr(numeric_only=True),text_auto=True,aspect="auto",title="Correlation heatmap"),use_container_width=True)

with tabs[4]:
    st.subheader("Machine learning lab")
    st.caption("Choose a target column, select features, and train a model. Use numeric targets for regression and categories for classification.")
    if len(df)<10 or len(df.columns)<2: st.warning("At least 10 rows and 2 columns are required.")
    else:
        target=st.selectbox("Target to predict",df.columns,key="target")
        task=st.radio("Task",["Auto-detect","Classification","Regression"],horizontal=True)
        features=st.multiselect("Input features",[c for c in df.columns if c!=target],default=[c for c in df.columns if c!=target][:min(8,len(df.columns)-1)])
        test_pct=st.slider("Test set (%)",10,40,20,5)
        model_name=st.selectbox("Model",["Random Forest","Decision Tree","Logistic Regression / Linear Regression"])
        if st.button("Train and evaluate",type="primary"):
            if not features: st.error("Select at least one feature.")
            else:
                md=df[features+[target]].dropna(subset=[target]).copy(); X=md[features]; yy=md[target]
                actual=("Classification" if (not pd.api.types.is_numeric_dtype(yy) or yy.nunique()<=15) else "Regression") if task=="Auto-detect" else task
                if len(md)<10: st.error("Not enough rows with a target value.")
                elif actual=="Classification" and yy.nunique()<2: st.error("Classification needs at least two target classes.")
                else:
                    ncols=X.select_dtypes(include=np.number).columns.tolist(); ccols=[c for c in features if c not in ncols]; trans=[]
                    if ncols: trans.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),ncols))
                    if ccols: trans.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),ccols))
                    prep=ColumnTransformer(trans)
                    if actual=="Classification":
                        models={"Random Forest":RandomForestClassifier(n_estimators=120,random_state=42),"Decision Tree":DecisionTreeClassifier(random_state=42),"Logistic Regression / Linear Regression":LogisticRegression(max_iter=1000)}
                        strat=yy if yy.value_counts().min()>=2 and yy.nunique()<len(yy)*.5 else None
                    else:
                        models={"Random Forest":RandomForestRegressor(n_estimators=120,random_state=42),"Decision Tree":DecisionTreeRegressor(random_state=42),"Logistic Regression / Linear Regression":LinearRegression()}
                        strat=None
                    try:
                        Xtr,Xte,ytr,yte=train_test_split(X,yy,test_size=test_pct/100,random_state=42,stratify=strat)
                        pipe=Pipeline([("prep",prep),("model",models[model_name])]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
                        st.markdown("#### Performance")
                        if actual=="Classification":
                            a,b=st.columns(2); a.metric("Accuracy",f"{accuracy_score(yte,pred):.3f}"); b.metric("Weighted F1",f"{f1_score(yte,pred,average='weighted',zero_division=0):.3f}")
                        else:
                            a,b,c=st.columns(3); a.metric("MAE",f"{mean_absolute_error(yte,pred):.3f}"); b.metric("RMSE",f"{np.sqrt(mean_squared_error(yte,pred)):.3f}"); c.metric("R²",f"{r2_score(yte,pred):.3f}")
                        results=pd.DataFrame({"Actual":yte.reset_index(drop=True),"Predicted":pd.Series(pred).reset_index(drop=True)})
                        st.dataframe(results.head(100),use_container_width=True)
                        st.download_button("Download predictions CSV",results.to_csv(index=False).encode(), "predictions.csv","text/csv")
                    except Exception as e: st.error(f"Training failed: {e}")
    st.caption("Performance depends on data quality and target selection; it is not a guarantee of future outcomes.")

with tabs[5]:
    st.subheader("Ask your data")
    st.write("Rule-based Q&A for common questions. Open-ended AI answers require connecting an AI API.")
    question=st.text_input("Ask a question",placeholder="Which column has the most missing values?")
    if st.button("Get answer"):
        q=question.lower()
        if not q: st.warning("Enter a question.")
        elif "missing" in q or "null" in q:
            m=df.isna().sum().sort_values(ascending=False)
            st.write(f"Total missing cells: **{int(m.sum())}**. Highest: **{m.index[0]}** ({int(m.iloc[0])}).")
            st.dataframe(m[m>0].rename("Missing values").reset_index(),use_container_width=True)
        elif "duplicate" in q: st.write(f"Duplicate rows: **{dupes}**")
        elif "row" in q or "record" in q: st.write(f"Dataset has **{len(df)} rows** and **{len(df.columns)} columns**.")
        elif "column" in q: st.write(", ".join(map(str,df.columns)))
        elif "average" in q or "mean" in q:
            if num: st.dataframe(df[num].mean().rename("Average").reset_index(),use_container_width=True)
            else: st.info("No numeric columns.")
        elif "correlation" in q and len(num)>=2:
            cor=df[num].corr(numeric_only=True); pairs=cor.where(np.triu(np.ones(cor.shape),1).astype(bool)).stack()
            if len(pairs): st.write(f"Strongest pair: **{pairs.abs().idxmax()[0]}** and **{pairs.abs().idxmax()[1]}** (r={pairs.loc[pairs.abs().idxmax()]:.3f}).")
            else: st.info("No pairs available.")
        elif ("highest" in q or "maximum" in q or "lowest" in q or "minimum" in q) and num:
            col=st.selectbox("Choose numeric column",num)
            st.write(f"{'Maximum' if ('highest' in q or 'maximum' in q) else 'Minimum'} for {col}: {df[col].max() if ('highest' in q or 'maximum' in q) else df[col].min()}")
        else: st.info("Try asking about rows, columns, missing values, duplicates, averages, correlation, maximum, or minimum.")

with tabs[6]:
    st.subheader("Reports and export")
    st.download_button("⬇️ Download cleaned CSV",df.to_csv(index=False).encode("utf-8"),"cleaned_dataset.csv","text/csv")
    excel=BytesIO()
    with pd.ExcelWriter(excel,engine="openpyxl") as writer:
        df.to_excel(writer,sheet_name="Cleaned Data",index=False)
        pd.DataFrame({"Column":df.columns.astype(str),"Type":[str(df[c].dtype) for c in df.columns],"Missing":[int(df[c].isna().sum()) for c in df.columns],"Unique":[int(df[c].nunique()) for c in df.columns]}).to_excel(writer,sheet_name="Column Profile",index=False)
    st.download_button("⬇️ Download Excel report",excel.getvalue(),"data_analysis_report.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if st.button("Generate PDF report",type="primary"):
        try:
            buf=BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4); styles=getSampleStyleSheet()
            story=[Paragraph("Data Intelligence Platform — Dataset Report",styles["Title"]),Spacer(1,12),
                   Paragraph(f"File: {uploaded.name}",styles["Normal"]),Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}",styles["Normal"]),
                   Paragraph(f"Rows: {len(df)} | Columns: {len(df.columns)} | Missing cells: {missing} | Duplicate rows: {dupes}",styles["Normal"]),Spacer(1,12),
                   Paragraph("Column profile (first 30 columns)",styles["Heading2"])]
            data=[["Column","Type","Missing","Unique"]]
            for c in df.columns[:30]: data.append([str(c)[:35],str(df[c].dtype),str(df[c].isna().sum()),str(df[c].nunique())])
            table=Table(data,repeatRows=1,colWidths=[220,90,65,65])
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#173b70")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8),("PADDING",(0,0),(-1,-1),5)]))
            story.append(table); doc.build(story); buf.seek(0)
            st.download_button("⬇️ Download PDF report",buf.getvalue(),"data_analysis_report.pdf","application/pdf")
        except Exception as e: st.error(f"PDF report failed: {e}")

st.sidebar.divider()
st.sidebar.caption("College major project · Data is processed in the current app session.")
