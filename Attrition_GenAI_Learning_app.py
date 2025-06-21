import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lifelines import CoxPHFitter
from sklearn.preprocessing import LabelEncoder, StandardScaler
import google.generativeai as genai
import base64

# ==== STREAMLIT CONFIG & STYLE ====
st.set_page_config(page_title="AttriSense HR: Employee Attrition Risk & Management Portal", layout="wide")

st.markdown("""
<style>
body, .main, .stApp { padding-top: 0 !important; margin-top: 0 !important; background: #0e0f17 !important;}
section.main > div:first-child { padding-top: 0 !important; margin-top: 0 !important;}
header, #MainMenu, footer, [data-testid="stStatusWidget"] { display: none !important;}
.stDeployButton, .stDeployButton__content, [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important;}
html, body, .stApp, [data-testid="stAppViewContainer"] {margin-top: 0 !important; padding-top: 0 !important; background: #0e0f17 !important; color: #fff !important;}
body {margin:0 !important;}
.banner {background:linear-gradient(90deg,#3a267b,#512e89);padding:2rem 2rem 1.2rem 2rem;margin:0 0 1rem 0;border-radius:0 0 12px 12px;}
.banner h1{margin:0;font-size:2.6rem;font-weight:900;}
.banner p{margin:.5rem 0 0;font-size:1.15rem;font-weight:600;color:#2ee0e0;display:flex;align-items:center;gap:0.5rem;}
.section-header{font-size:1.3rem;font-weight:700;margin-top:.5rem;margin-bottom:.2rem;display:flex;align-items:center;gap:8px;}
.stTabs [data-baseweb="tab-list"]{justify-content:center!important;}
.stTabs [data-baseweb="tab"]{color:#ddd!important;font-size:1rem;}
.stTabs [aria-selected="true"]{background:#221e3b!important;color:#fff!important;}
.kpi-row{display:flex;gap:12px;margin-bottom:1rem;}
.metric{flex:1;background:#181824;padding:1rem;border-radius:8px;text-align:center;}
.metric .val{font-size:1.8rem;font-weight:700;color:#b794f4;}
.metric .lab{font-size:1rem;color:#aaa;}
.stSelectbox,.stSlider,.stTextInput,.stButton>button {background:#1e1932!important;color:#fff!important;border:1px solid #2ee0e0!important;}
.stSelectbox label,.stSlider label,.stTextInput label {color:#fff!important;}
.stDataFrame table{background:#181824;color:#fff;}
.stDataFrame th,.stDataFrame td{padding:.3rem .6rem;font-size:.9rem;}
.stDataFrame th{font-weight:700;}
.stDataFrame tr:nth-child(even) {background: #232540;}
.ai-answer-box {background:#181824;border-left:6px solid #FFD700;border-radius:8px;padding:1.2rem 1rem 1rem 1.2rem;margin:1rem 0;color:white;}
.ai-answer-box strong {color:#FFD700;}
.stSpinner > div{color:#2ee0e0!important;}
.icon {font-size:1.5rem;vertical-align:middle;display:inline-block;margin-right:.5rem;}
.descbox {background:#102634;border-left:6px solid #2ee0e0;padding:.8rem 1rem .8rem 1.5rem;margin-bottom:1rem;border-radius:8px;display:flex;align-items:center;gap:12px;font-size:1.05rem;color:#b7edfa;}
#plan_md {width:100vw !important; max-width:90vw;}
[data-baseweb="radio"] label {color: #fff !important;}
</style>
""", unsafe_allow_html=True)

# ==== DATA GENERATION ====
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 400
    dept_roles = {
        "Tech":      ["Data Scientist", "Software Engineer", "QA Engineer", "DevOps Engineer"],
        "Business":  ["Business Analyst", "Product Manager", "Strategy Lead"],
        "Finance":   ["Accountant", "Financial Analyst", "Controller"],
        "Marketing": ["Digital Marketer", "PR Specialist", "Content Lead"],
        "HR":        ["HR Manager", "Recruiter"],
        "Operations":["Operations Manager", "Logistics Lead"],
        "Legal":     ["Legal Counsel", "Compliance Officer"]
    }
    departments, jobroles = [], []
    for _ in range(n):
        dept = np.random.choice(list(dept_roles.keys()), p=[.21,.19,.17,.13,.10,.11,.09])
        role = np.random.choice(dept_roles[dept])
        departments.append(dept)
        jobroles.append(role)
    df = pd.DataFrame({
        "EmployeeNumber": np.arange(1, n+1),
        "Department": departments,
        "JobRole": jobroles,
        "Gender": np.random.choice(["Male","Female"], n, p=[.54,.46]),
        "Age": np.random.normal(35, 8, n).round().clip(22, 60),
        "MonthlyIncome": np.random.gamma(2.5, 29000, n).clip(35000, 255000).round(-2),
        "YearsAtCompany": np.random.geometric(0.16, n).clip(1, 19),
        "OverTime": np.random.choice(["Yes","No"], n, p=[.29,.71]),
        "JobSatisfaction": np.random.choice([1,2,3,4], n, p=[.14,.25,.38,.23]),
        "EnvironmentSatisfaction": np.random.choice([1,2,3,4], n, p=[.13,.22,.37,.28]),
        "WorkLifeBalance": np.random.choice([1,2,3,4], n, p=[.14,.36,.36,.14]),
        "NumCompaniesWorked": np.random.poisson(2.6, n).clip(0, 10),
        "TrainingTimesLastYear": np.random.poisson(2.2, n).clip(0, 7),
        "CoursesCompleted": np.random.randint(0, 6, n),
    })
    base_prob = (
        0.13
        + 0.10 * (df["JobSatisfaction"]<=2)
        + 0.09 * (df["OverTime"]=="Yes")
        + 0.09 * (df["EnvironmentSatisfaction"]<=2)
        + 0.08 * (df["YearsAtCompany"]<2)
        + 0.05 * (df["NumCompaniesWorked"]>=5)
        + 0.03 * (df["WorkLifeBalance"]<=2)
        - 0.09 * (df["YearsAtCompany"]>8)
        - 0.07 * (df["JobSatisfaction"]==4)
    )
    df["Attrition"] = np.random.binomial(1, base_prob.clip(0.06,0.37))
    return df

df = load_data()

# ==== PREPROCESSING ====
cat_cols = ["Department","JobRole","Gender","OverTime"]
num_cols = ["Age","MonthlyIncome","YearsAtCompany","NumCompaniesWorked","TrainingTimesLastYear","CoursesCompleted"]
ordinal_cols = ["JobSatisfaction","EnvironmentSatisfaction","WorkLifeBalance"]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

df_enc = pd.get_dummies(df_scaled, columns=cat_cols, drop_first=True)
features = [c for c in df_enc if c not in ["EmployeeNumber","Attrition"]]
dummy_template = pd.get_dummies(df[cat_cols], drop_first=True)
all_dummy_cols = list(dummy_template.columns)

# ==== MODEL TRAINING ====
@st.cache_resource
def train_xgb(X, y):
    model = XGBClassifier(n_estimators=80, random_state=2, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    return model

xgb_model = train_xgb(df_enc[features], df_enc.Attrition)
df["RiskScore"] = xgb_model.predict_proba(df_enc[features])[:,1]

# CoxPH model (not cached, for speed and no hash error)
role_encoder = LabelEncoder().fit(df.JobRole)
cox_df = df.copy()
cox_df["JobRoleCode"] = role_encoder.transform(cox_df.JobRole)
cox_df["GenderCode"] = LabelEncoder().fit_transform(cox_df.Gender)
cph = CoxPHFitter().fit(
    cox_df[["YearsAtCompany","MonthlyIncome","Age","JobRoleCode","GenderCode","Attrition"]],
    "YearsAtCompany","Attrition"
)

# ==== GLOBAL FEATURE IMPORTANCE ====
def get_top_features(xgb_model, features, topn=7):
    importances = xgb_model.feature_importances_
    idxs = np.argsort(importances)[::-1][:topn]
    return [(features[i], round(float(importances[i]),3)) for i in idxs]

global_feature_importance = get_top_features(xgb_model, features, 7)

def get_cohort_features(df_sub, topn=5):
    X = pd.get_dummies(df_sub[cat_cols], drop_first=True)
    for col in all_dummy_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[all_dummy_cols]
    for col in num_cols + ordinal_cols:
        X[col] = df_sub[col].values
    y = df_sub["Attrition"]
    if len(df_sub) < 12: return []
    model = XGBClassifier(n_estimators=30, random_state=12, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    importances = model.feature_importances_
    idxs = np.argsort(importances)[::-1][:topn]
    return [(X.columns[i], round(float(importances[i]),3)) for i in idxs]

# ==== GEMINI SETUP ====
genai.configure(api_key=os.getenv("GEMINI_API_KEY",""))

def ask_dashboard_ai(q, df=df, features=features, global_feats=global_feature_importance):
    lower_q = q.lower()
    all_depts = list(df.Department.unique())
    all_roles = list(df.JobRole.unique())
    found_depts = [d for d in all_depts if d.lower() in lower_q]
    found_roles = [r for r in all_roles if r.lower() in lower_q]
    compare_items = found_depts + found_roles

    response = ""
    if "global" in lower_q or "overall" in lower_q or ("top" in lower_q and "driver" in lower_q):
        response += "<b>🌐 Global Attrition Drivers:</b><br><ul style='margin-bottom:0'>"
        for f, imp in global_feats:
            response += f"<li><b>{f}</b> <span style='color:#FFD700'>{imp*100:.1f}%</span></li>"
        response += "</ul>"
        topf = global_feats[0][0]
        response += f"\n<div style='margin-top:.7em'>Most important driver: <b style='color:#FFD700'>{topf}</b>.</div>"
    elif compare_items:
        for item in compare_items:
            if item in all_depts:
                subset = df[df.Department==item]
                at_rate = subset.Attrition.mean()*100
                features_cohort = get_cohort_features(subset, 5)
                response += f"<b style='color:#FFD700'>{item}</b> Attrition Rate: <b style='color:#FFD700'>{at_rate:.1f}%</b><br>Top Drivers: "
                if features_cohort:
                    response += ", ".join([f"{n} (<b>{v*100:.1f}%</b>)" for n,v in features_cohort]) + "<br>"
                else:
                    response += "_Not enough data_<br>"
            elif item in all_roles:
                subset = df[df.JobRole==item]
                at_rate = subset.Attrition.mean()*100
                features_cohort = get_cohort_features(subset, 5)
                response += f"<b style='color:#FFD700'>{item}</b> Attrition Rate: <b style='color:#FFD700'>{at_rate:.1f}%</b><br>Top Drivers: "
                if features_cohort:
                    response += ", ".join([f"{n} (<b>{v*100:.1f}%</b>)" for n,v in features_cohort]) + "<br>"
                else:
                    response += "_Not enough data_<br>"
        if len(compare_items) == 2:
            v1 = df[df.Department==compare_items[0]].Attrition.mean()*100 if compare_items[0] in all_depts else \
                 df[df.JobRole==compare_items[0]].Attrition.mean()*100
            v2 = df[df.Department==compare_items[1]].Attrition.mean()*100 if compare_items[1] in all_depts else \
                 df[df.JobRole==compare_items[1]].Attrition.mean()*100
            response += f"<b>Comparison:</b> <b style='color:#FFD700'>{compare_items[0]}</b> is {'higher' if v1>v2 else 'lower'} (<b>{v1:.1f}%</b>) than <b style='color:#FFD700'>{compare_items[1]}</b> (<b>{v2:.1f}%</b>)."
    else:
        gfi = ", ".join([f"{n} (<b>{v*100:.1f}%</b>)" for n,v in global_feats])
        context = (
            f"Global attrition drivers (by model importance): {gfi}\n"
            f"User's question: {q}\n"
            "Answer as a data-driven HR analytics expert, using these drivers, and cohort-level attrition rates if groups are mentioned."
        )
        ans = genai.GenerativeModel("gemini-2.0-flash").generate_content(context)
        response = ans.text
    return response

def ask_employee_ai(profile, question):
    prompt = f"""You are an HR insights assistant.
Employee data: {profile}
User question: {question}
Answer in clear, structured manner. Always use feature values from profile if relevant. Highlight numbers in <b style='color:#FFD700'>color</b> and always summarize key insights at end."""
    resp = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return resp.text

def gen_plan(profile, aspiration, gaps, weeks, refine=""):
    prompt = (
        f"Act as an L&D expert career coach.\n"
        f"Given this EMPLOYEE PROFILE: {profile}\n"
        f"Aspiration: {aspiration}\n"
        f"Skill Gaps: {gaps}\n"
        f"Timeline: {weeks} weeks\n"
        f"Refine: {refine}\n\n"
        f"**1. Search and recommend the best, most relevant LinkedIn, Udemy, and Coursera courses for these skill gaps, for this employee's job and aspiration. Include course links.**\n"
        f"**2. Generate a week-by-week learning plan, specifying which course/module to take each week, with links.**\n"
        f"**3. For each course, specify what outcome/skills will be gained and why it's recommended for this role and career stage.**\n"
        f"**4. Output plan using bullet points or tables, with links in Markdown format, and summarize expected impact at the end.**\n"
        f"**Do NOT leave the left of the screen empty; plan content should be left aligned and fully utilize available width.**\n"
    )
    resp = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return resp.text

# ==== UI: HEADER ====
def get_image_base64(img_path):
    with open(img_path, "rb") as img_file:
        img_bytes = img_file.read()
        encoded = base64.b64encode(img_bytes).decode("utf-8")
    return encoded

icon_path = "hr_soln.png"
icon_b64 = get_image_base64(icon_path)

banner_html = f"""
<div style='
    position: relative;
    background: linear-gradient(90deg,#3a267b,#512e89);
    padding: 2rem 2rem 1.2rem 2rem;
    margin: 0 0 1rem 0;
    border-radius: 0 0 18px 18px;
    min-height: 110px;
    display: flex;
    align-items: center;
'>
    <div style="flex: 1;">
        <h1 style='margin:0;font-size:2.0rem;font-weight:900;color:#fff;'>AttriSense HR: Employee Attrition Risk & Management Portal</h1>
        <p style='margin:.5rem 0 0;font-size:1.15rem;font-weight:600;color:#2ee0e0;display:flex;align-items:center;gap:0.5rem;'>
            <span style="font-size:1.5rem;vertical-align:middle;display:inline-block;margin-right:.5rem;">🤖</span>
            AI-driven risk analytics & career growth.
        </p>
    </div>
    <img src="data:image/png;base64,{icon_b64}" style="
        height:76px; 
        width:76px; 
        object-fit:contain; 
        border-radius:18px; 
        margin-left:2rem;
        background:rgba(0,0,0,0.11); 
        box-shadow:0 2px 16px #2225;
        display:block;
    " alt="Brand Icon" />
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📊Dashboard & KPIs", "🧑‍💼Employee Finder", "🚀Upskilling Assistant"
])

# ==== TAB 1: DASHBOARD & KPIs ====
with tab1:
    st.markdown(
        '<div class="descbox"><span class="icon">📊</span>'
        '<b>Dashboard:</b> Explore attrition, risk, and top drivers by slicing and get instant insights from Dashboard AI.'
        '</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        d = st.selectbox("Department", ["All"] + sorted(df.Department.unique()), key="dash_dept")
    with col2:
        r = st.selectbox("Job Role", ["All"] + sorted(df.JobRole.unique()), key="dash_role")
    with col3:
        g = st.selectbox("Gender", ["All"] + sorted(df.Gender.unique()), key="dash_gender")

    filt = df.copy()
    if d!="All": filt = filt[filt.Department==d]
    if r!="All": filt = filt[filt.JobRole==r]
    if g!="All": filt = filt[filt.Gender==g]

    kpis = [
        (f"{filt.Attrition.mean()*100:.1f}%","Attrition Rate"),
        ((filt.RiskScore>.5).sum(),"High-Risk Emp."),
        (f"{filt[filt.Attrition==1].YearsAtCompany.mean():.1f} yrs","Avg Years→Leave"),
        (filt.groupby("Department").RiskScore.mean().idxmax() if not filt.empty else "-","Top Risk Dept."),
        (len(filt),"Employees")
    ]
    st.markdown(
        '<div class="kpi-row">'+
        ''.join(f'<div class="metric"><div class="val">{v}</div><div class="lab">{l}</div></div>' for v,l in kpis)+
        '</div>', unsafe_allow_html=True
    )

    dept_tbl = filt.groupby("Department").agg(Emps=("EmployeeNumber","count"), Leavers=("Attrition","sum"))
    dept_tbl["Attr%"] = (dept_tbl.Leavers/dept_tbl.Emps).round(3)

    r1, r2 = st.columns(2)
    with r1:
        fig,ax = plt.subplots(figsize=(4,3))
        grp = filt.groupby("Department").Attrition.mean()*100
        ax.bar(grp.index, grp.values, color="#6f5cff")
        ax.set_facecolor("#181824"); ax.figure.patch.set_facecolor("#181824")
        ax.set_title("Attrition % by Department",fontsize=11,color="white")
        ax.tick_params(axis='x', colors="white", labelsize=8, rotation=45)
        ax.tick_params(axis='y', colors="white", labelsize=8)
        ax.set_xlabel("Department", fontsize=8, color="white")
        ax.set_ylabel("Attrition %", fontsize=8, color="white")
        for s in ax.spines.values(): s.set_color("white")
        for xi,yi in zip(grp.index,grp.values):
            ax.text(xi,yi*1.01,f"{yi:.1f}%",ha="center",color="white",fontsize=7,rotation=0)
        st.pyplot(fig)
    with r2:
        jobrole_counts = filt['JobRole'].value_counts().nlargest(10).index
        grp = filt[filt['JobRole'].isin(jobrole_counts)].groupby("JobRole").Attrition.mean()*100
        fig,ax = plt.subplots(figsize=(4,3))
        ax.bar(grp.index, grp.values, color="#6f5cff")
        ax.set_facecolor("#181824"); fig.patch.set_facecolor("#181824")
        ax.set_title("Attrition % by Top 10 Job Roles",fontsize=11,color="white")
        ax.tick_params(axis='x', colors="white", labelsize=8, rotation=45)
        ax.tick_params(axis='y', colors="white", labelsize=8)
        ax.set_xlabel("Job Role", fontsize=8, color="white")
        ax.set_ylabel("Attrition %", fontsize=8, color="white")
        for s in ax.spines.values(): s.set_color("white")
        for xi,yi in zip(grp.index,grp.values):
            ax.text(xi,yi*1.01,f"{yi:.1f}%",ha="center",color="white",fontsize=7,rotation=0)
        st.pyplot(fig)

    r3, r4 = st.columns(2)
    with r3:
        grp = filt.groupby("Department").CoursesCompleted.mean()
        fig,ax = plt.subplots(figsize=(4,3))
        ax.bar(grp.index,grp.values,color="#FFD700")
        ax.set_facecolor("#181824"); fig.patch.set_facecolor("#181824")
        ax.set_title("Avg Courses Completed",fontsize=11,color="white")
        ax.tick_params(axis='x', colors="white", labelsize=8, rotation=45)
        ax.tick_params(axis='y', colors="white", labelsize=8)
        ax.set_xlabel("Department", fontsize=8, color="white")
        ax.set_ylabel("Avg Courses", fontsize=8, color="white")
        for s in ax.spines.values(): s.set_color("white")
        for xi,yi in zip(grp.index,grp.values):
            ax.text(xi,yi*1.01,f"{yi:.1f}",ha="center",color="white",fontsize=7,rotation=0)
        st.pyplot(fig)
    with r4:
        grp = filt.Gender.value_counts()
        fig,ax = plt.subplots(figsize=(4,3))
        ax.bar(grp.index,grp.values,color="#FFD700")
        ax.set_facecolor("#181824"); fig.patch.set_facecolor("#181824")
        ax.set_title("Head-count by Gender",fontsize=11,color="white")
        ax.tick_params(axis='x', colors="white", labelsize=8, rotation=45)
        ax.tick_params(axis='y', colors="white", labelsize=8)
        ax.set_xlabel("Gender", fontsize=8, color="white")
        ax.set_ylabel("Count", fontsize=8, color="white")
        for s in ax.spines.values(): s.set_color("white")
        for xi,yi in zip(grp.index,grp.values):
            ax.text(xi,yi*1.01,f"{yi}",ha="center",color="white",fontsize=7,rotation=0)
        st.pyplot(fig)

    st.markdown('<span class="section-header">🏅 Top-10 Feature Importances</span>', unsafe_allow_html=True)
    import matplotlib.colors as mcolors
    from collections import defaultdict
    
    def get_parent_feature(col):
        if '_' in col:
            return col.split('_')[0]
        else:
            return col
    
    def compute_aggregated_importances(model, features):
        importances = model.feature_importances_
        agg = defaultdict(float)
        for f, imp in zip(features, importances):
            parent = get_parent_feature(f)
            agg[parent] += imp
        return sorted(agg.items(), key=lambda x: -x[1])
    
    agg_imp = compute_aggregated_importances(xgb_model, features)
    agg_imp_top = agg_imp[:10]
    
    labels = [f for f, v in agg_imp_top][::-1]
    values = [v*100 for f, v in agg_imp_top][::-1]
    
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    colors = [cmap(norm(v)) for v in values]
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", ha="left", fontsize=6, color="white", fontweight="bold")
    ax.set_facecolor("#181824")
    fig.patch.set_facecolor("#181824")
    ax.set_xlabel("Importance %", color="white", fontsize=9)
    ax.tick_params(axis='y', colors="white", labelsize=9)
    ax.tick_params(axis='x', colors="white", labelsize=9)
    for s in ax.spines.values(): s.set_color("white")
    plt.tight_layout()
    st.pyplot(fig)
   
    tbl = dept_tbl[["Emps","Leavers","Attr%"]].rename(columns={"Attr%":"Attrition%"})
    tbl["Retention%"] = 1 - tbl["Attrition%"]
    st.dataframe(tbl.style.format({"Attrition%":"{:.1%}","Retention%":"{:.1%}"}),height=220)

    topd = tbl["Attrition%"].idxmax()
    lowd = tbl["Attrition%"].idxmin()
    hrisk= filt.groupby("Department").RiskScore.mean().idxmax() if not filt.empty else "-"
    avg_ttl = round(float(filt[filt.Attrition==1].YearsAtCompany.mean()),1)
    st.markdown(
        f'<div class="descbox"><span class="icon">🔎</span><b>Key Insights:</b>'
        f'<ul style="margin-bottom:0"><li>Highest attrition: <b>{topd}</b></li><li>Lowest attrition: <b>{lowd}</b></li>'
        f'<li>Highest avg risk: <b>{hrisk}</b></li><li>Filtered attrition rate: <b>{filt.Attrition.mean()*100:.1f}%</b></li>'
        f'<li>Avg years-to-leave: <b>{avg_ttl} yrs</b></li></ul></div>',
        unsafe_allow_html=True)

    st.markdown('<span class="section-header">🤖 Dashboard AI Insight</span>', unsafe_allow_html=True)
    q = st.text_input("Ask Dashboard AI…", key="dash_ai_q")
    if st.button("Ask AI", key="dash_ai_btn"):
        with st.spinner("🤖 AI is analyzing dashboard metrics and drivers..."):
            ans = ask_dashboard_ai(q)
        st.markdown(f"<div class='ai-answer-box'>{ans}</div>", unsafe_allow_html=True)

# ==== TAB 2: EMPLOYEE FINDER ====
with tab2:
    st.markdown(
        '<div class="descbox"><span class="icon">🧑‍💼</span>'
        '<b>Employee Finder:</b> View, sort, and query individual employee risk, tenure and AI-powered insights.'
        '</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        d2 = st.selectbox("Department", ["All"]+sorted(df.Department.unique()), key="emp_dept")
    with c2:
        r2 = st.selectbox("Job Role", ["All"]+sorted(df.JobRole.unique()), key="emp_role")
    with c3:
        g2 = st.selectbox("Gender", ["All"]+sorted(df.Gender.unique()), key="emp_gender")
    st.markdown(
        "<div style='color:white;margin-bottom:.2rem;font-size:1rem'><b>Sort employees by:</b></div>",
        unsafe_allow_html=True
    )
    sort_mode = st.radio("",["Risk %","ID"],horizontal=True,key="emp_sort")
    sel = df.copy()
    if d2!="All": sel = sel[sel.Department==d2]
    if r2!="All": sel = sel[sel.JobRole==r2]
    if g2!="All": sel = sel[sel.Gender==g2]
    sel = sel.sort_values("RiskScore",ascending=(sort_mode=="ID"))
    emp_id = st.selectbox("Employee #", sel.EmployeeNumber, key="emp_id")
    row    = sel[sel.EmployeeNumber==emp_id].iloc[0]
    st.markdown(f"<b>Risk Score:</b> <span style='color:#FFD700'>{row.RiskScore*100:.1f}%</span>",unsafe_allow_html=True)
    enc = {
        "YearsAtCompany": row.YearsAtCompany,
        "MonthlyIncome":  row.MonthlyIncome,
        "Age":            row.Age,
        "JobRoleCode":    role_encoder.transform([row.JobRole])[0],
        "GenderCode":     0 if row.Gender=="Male" else 1
    }
    surv = cph.predict_survival_function(pd.DataFrame([enc]), [3,5,10,15])
    probs = (1-surv.values.flatten())*100
    st.markdown(f"<b>Stay ≤3y:</b> <span style='color:#2ee0e0'>{probs[0]:.1f}%</span> | "
                f"<b>≤5y:</b> <span style='color:#2ee0e0'>{probs[1]:.1f}%</span> | "
                f"<b>≤10y:</b> <span style='color:#2ee0e0'>{probs[2]:.1f}%</span> | "
                f"<b>≤15y:</b> <span style='color:#2ee0e0'>{probs[3]:.1f}%</span>",unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6,3))
    full = cph.predict_survival_function(pd.DataFrame([enc]), np.arange(0,20))
    ax.step(full.index, full.values.flatten(), where="post", color="#FFD700", linewidth=2)
    ax.set_facecolor("#181824"); fig.patch.set_facecolor("#181824")
    ax.set_title("Survival Curve", color="white", fontsize=11); ax.tick_params(colors="white")
    ax.set_xlabel("Years", fontsize=8, color="white")
    ax.set_ylabel("Stay Probability", fontsize=8, color="white")
    ax.tick_params(axis='x',labelsize=8,rotation=45)
    ax.tick_params(axis='y',labelsize=8)
    for s in ax.spines.values(): s.set_color("white")
    st.pyplot(fig)
    st.markdown('<span class="section-header">🤖 Employee AI Insight</span>', unsafe_allow_html=True)
    eq = st.text_input("Ask AI about this employee…", key="emp_ai_q")
    if st.button("Ask Employee AI", key="emp_ai_btn"):
        with st.spinner("🤖 AI is finding details about this employee..."):
            profile = row.to_dict()
            profile["RiskScore"] = round(float(row.RiskScore), 3)
            ans2 = ask_employee_ai(profile, eq)
        st.markdown(f"<div class='ai-answer-box'>{ans2}</div>", unsafe_allow_html=True)

# ==== TAB 3: UPSKILLING ASSISTANT ====
with tab3:
    st.markdown(
        '<div class="descbox"><span class="icon">🎯</span>'
        '<b>Upskilling Assistant:</b> Select employees by risk or ID and generate a fully personalized learning plan using AI and curated web courses.'
        '</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        d3 = st.selectbox("Department", ["All"]+sorted(df.Department.unique()), key="up_dept")
    with c2:
        r3 = st.selectbox("Job Role", ["All"]+sorted(df.JobRole.unique()), key="up_role")
    with c3:
        g3 = st.selectbox("Gender", ["All"]+sorted(df.Gender.unique()), key="up_gender")
    st.markdown(
        "<div style='color:white;margin-bottom:.2rem;font-size:1rem'><b>Sort employees by:</b></div>",
        unsafe_allow_html=True
    )
    sort_mode2 = st.radio("",["Risk %","ID"],horizontal=True,key="up_sort")
    sub = df.copy()
    if d3!="All": sub = sub[sub.Department==d3]
    if r3!="All": sub = sub[sub.JobRole==r3]
    if g3!="All": sub = sub[sub.Gender==g3]
    sub = sub.sort_values("RiskScore",ascending=(sort_mode2=="ID"))
    emp2 = st.selectbox("Employee #", sub.EmployeeNumber, key="up_emp")
    rec  = sub[sub.EmployeeNumber==emp2].iloc[0]
    L, R = st.columns([1,2])
    with L:
        st.markdown(f"""<div style="background:#291f46;padding:1rem;border-radius:8px;">
<b>ID</b>: {rec.EmployeeNumber}<br>
<b>Dept</b>: {rec.Department}<br>
<b>Role</b>: {rec.JobRole}<br>
<b>Age</b>: {rec.Age}<br>
<b>Income</b>: ₹{rec.MonthlyIncome:,}<br>
<b>Years</b>: {rec.YearsAtCompany} yrs<br>
<b>OverTime</b>: {rec.OverTime}<br>
<b>CoursesDone</b>: {rec.CoursesCompleted}/6<br>
<b>Risk</b>: {rec.RiskScore*100:.1f}%<br>
</div>""", unsafe_allow_html=True)
    with R:
        asp    = st.text_input("Career Aspiration","Advance to Manager", key="up_asp")
        gaps   = st.multiselect("Skill Gaps",["Sales","Negotiation","Leadership","Analytics","Empathy","Harassment","Digital Marketing","Excel","Project Management"], key="up_gaps")
        weeks  = st.slider("Timeline (weeks)",4,24,12, key="up_weeks")
        refine = st.text_input("Refine Plan (optional)", key="up_ref")
        if st.button("Generate Plan", key="up_btn"):
            with st.spinner("🎓 AI is generating your learning plan..."):
                plan_md = gen_plan(rec.to_dict(), asp, gaps, weeks, refine)
            st.markdown("<span class='section-header'>🎓 Learning Plan</span>",unsafe_allow_html=True)
            st.markdown(f"<div id='plan_md' class='ai-answer-box'>{plan_md}</div>",unsafe_allow_html=True)

st.caption("Built with Streamlit • XGBoost • Lifelines • Gemini AI • Dark Theme")
