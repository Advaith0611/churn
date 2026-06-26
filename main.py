import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence · TelCo",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0F1117; color: #E8EAF0; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1A1F2E 0%, #141824 100%);
        border: 1px solid #2A3045;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .app-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #7B8299;
        margin: 4px 0 0 0;
        font-size: 0.92rem;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 16px; margin-bottom: 24px; }
    .metric-card {
        flex: 1;
        background: #1A1F2E;
        border: 1px solid #2A3045;
        border-radius: 10px;
        padding: 20px 24px;
    }
    .metric-card .label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #7B8299;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #FFFFFF;
        line-height: 1;
    }
    .metric-card .value.red   { color: #FF5C5C; }
    .metric-card .value.amber { color: #FFB547; }
    .metric-card .value.green { color: #4ADE80; }
    .metric-card .sub {
        font-size: 0.78rem;
        color: #7B8299;
        margin-top: 6px;
    }

    /* Risk badge */
    .risk-high   { background:#FF5C5C22; color:#FF5C5C; border:1px solid #FF5C5C55;
                   padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .risk-medium { background:#FFB54722; color:#FFB547; border:1px solid #FFB54755;
                   padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .risk-low    { background:#4ADE8022; color:#4ADE80; border:1px solid #4ADE8055;
                   padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1A1F2E;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #2A3045;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        color: #7B8299;
        font-weight: 500;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background: #2A3045 !important;
        color: #FFFFFF !important;
    }

    /* Upload zone */
    .uploadedFile { background: #1A1F2E; border: 1px dashed #2A3045; border-radius: 10px; }

    /* Section label */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7B8299;
        margin-bottom: 12px;
        margin-top: 8px;
    }

    /* Info callout */
    .callout {
        background: #1A2535;
        border-left: 3px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #93B4D4;
        margin-bottom: 20px;
    }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stButton button {
        background: #3B82F6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
    }
    .stButton button:hover { background: #2563EB; }

    /* Hide default Streamlit header */
    #MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING  (unchanged from original — only touched frontend below)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_model():
    df = pd.read_csv("churn.csv")
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    le = LabelEncoder()
    obj_cols = df.select_dtypes(include='object').columns
    label_maps = {}
    for col in obj_cols:
        df[col] = le.fit_transform(df[col])
        label_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    X = df.drop(columns=['Churn'])
    Y = df['Churn']

    total_score = total_loss = count = 0
    total_TP = total_TN = total_FP = total_FN = 0
    best_model = None

    for i in range(1, 10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        Best_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5,
                       'min_samples_leaf': 2, 'max_features': 'sqrt'}
        BEST_THRESHOLD = 0.3
        model = RandomForestClassifier(**Best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, Y_train)
        score = model.score(X_test, Y_test) * 100
        Y_prob = model.predict_proba(X_test)[:, 1]
        Y_pred = (Y_prob > BEST_THRESHOLD).astype(int)
        loss = root_mean_squared_error(Y_test, Y_pred)
        matrix = confusion_matrix(Y_test, Y_pred)
        TN, FP, FN, TP = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        total_TP += TP; total_TN += TN; total_FP += FP; total_FN += FN
        total_score += score; total_loss += loss; count += 1
        best_model = model

    return (best_model, BEST_THRESHOLD, X.columns.tolist(), label_maps,
            total_score/count, total_loss/count,
            total_TP/count, total_TN/count, total_FP/count, total_FN/count,
            X_test, Y_test)

with st.spinner("Loading model…"):
    (model, BEST_THRESHOLD, feature_cols, label_maps,
     avg_score, avg_loss, avg_TP, avg_TN, avg_FP, avg_FN,
     X_test, Y_test) = train_model()

feature_importance = pd.Series(
    model.feature_importances_, index=feature_cols
).sort_values(ascending=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div>
        <h1>📡 Churn Intelligence</h1>
        <p>Random Forest · Threshold 0.30 · Built for retention teams</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL METRICS BAR
# ═══════════════════════════════════════════════════════════════════════════════
precision = avg_TP / (avg_TP + avg_FP) if (avg_TP + avg_FP) > 0 else 0
recall    = avg_TP / (avg_TP + avg_FN) if (avg_TP + avg_FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="label">Model Accuracy</div>
        <div class="value">{avg_score:.1f}<span style="font-size:1rem;color:#7B8299">%</span></div>
        <div class="sub">Avg over 9 runs</div>
    </div>
    <div class="metric-card">
        <div class="label">F1 Score</div>
        <div class="value">{f1:.3f}</div>
        <div class="sub">Harmonic mean of precision & recall</div>
    </div>
    <div class="metric-card">
        <div class="label">Precision</div>
        <div class="value">{precision:.3f}</div>
        <div class="sub">Of flagged customers, % actually churn</div>
    </div>
    <div class="metric-card">
        <div class="label">Recall</div>
        <div class="value amber">{recall:.3f}</div>
        <div class="sub">% of churners the model catches</div>
    </div>
    <div class="metric-card">
        <div class="label">RMSE Loss</div>
        <div class="value">{avg_loss:.3f}</div>
        <div class="sub">Lower is better</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_bulk, tab_single, tab_model = st.tabs([
    "  📂  Bulk Scoring  ",
    "  🔍  Single Customer  ",
    "  📊  Model Report  "
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1: BULK CSV UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
with tab_bulk:
    st.markdown('<div class="callout">Upload a CSV exported from your CRM — no Churn column needed. '
                'The model will score every customer and return a prioritised retention list.</div>',
                unsafe_allow_html=True)

    col_up, col_info = st.columns([2, 1])

    with col_info:
        st.markdown('<div class="section-label">Expected columns</div>', unsafe_allow_html=True)
        st.code("\n".join(feature_cols), language="text")

    with col_up:
        uploaded = st.file_uploader(
            "Drop your customer CSV here",
            type=["csv"],
            label_visibility="collapsed"
        )

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)

            # Drop customerID if present
            cid_col = None
            if 'customerID' in raw.columns:
                cid_col = raw['customerID'].copy()
                raw = raw.drop(columns=['customerID'])
            if 'Churn' in raw.columns:
                raw = raw.drop(columns=['Churn'])

            # Encode categoricals using same approach as training
            df_score = raw.copy()
            df_score['TotalCharges'] = pd.to_numeric(df_score['TotalCharges'], errors='coerce')
            df_score['TotalCharges'].fillna(df_score['TotalCharges'].median(), inplace=True)
            le_score = LabelEncoder()
            for col in df_score.select_dtypes(include='object').columns:
                df_score[col] = le_score.fit_transform(df_score[col].astype(str))

            # Score
            probs = model.predict_proba(df_score[feature_cols])[:, 1]
            preds = (probs >= BEST_THRESHOLD).astype(int)

            results = raw.copy()
            if cid_col is not None:
                results.insert(0, 'customerID', cid_col.values)
            results['Churn Probability'] = (probs * 100).round(1)
            results['Prediction']        = np.where(preds == 1, 'Will Churn', 'Will Stay')
            results['Risk Tier'] = pd.cut(
                probs,
                bins=[0, 0.4, 0.65, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            results = results.sort_values('Churn Probability', ascending=False).reset_index(drop=True)

            # Summary metrics
            n_high   = (results['Risk Tier'] == 'High').sum()
            n_medium = (results['Risk Tier'] == 'Medium').sum()
            n_low    = (results['Risk Tier'] == 'Low').sum()
            n_total  = len(results)

            st.markdown(f"""
            <div class="metric-row" style="margin-top:20px">
                <div class="metric-card">
                    <div class="label">Total Customers</div>
                    <div class="value">{n_total}</div>
                </div>
                <div class="metric-card">
                    <div class="label">High Risk</div>
                    <div class="value red">{n_high}</div>
                    <div class="sub">{n_high/n_total*100:.1f}% of upload</div>
                </div>
                <div class="metric-card">
                    <div class="label">Medium Risk</div>
                    <div class="value amber">{n_medium}</div>
                    <div class="sub">{n_medium/n_total*100:.1f}% of upload</div>
                </div>
                <div class="metric-card">
                    <div class="label">Low Risk</div>
                    <div class="value green">{n_low}</div>
                    <div class="sub">{n_low/n_total*100:.1f}% of upload</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Filter controls
            st.markdown('<div class="section-label">Filter results</div>', unsafe_allow_html=True)
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                tier_filter = st.multiselect(
                    "Risk tier",
                    options=['High', 'Medium', 'Low'],
                    default=['High', 'Medium'],
                    label_visibility="collapsed"
                )
            with f_col2:
                prob_min = st.slider("Min churn probability %", 0, 100, 30, label_visibility="collapsed")

            filtered = results[
                results['Risk Tier'].isin(tier_filter) &
                (results['Churn Probability'] >= prob_min)
            ]

            st.markdown(f'<div class="section-label">Showing {len(filtered)} customers</div>',
                        unsafe_allow_html=True)

            # Display table — only key columns up front, rest collapsible
            display_cols = (
                (['customerID'] if cid_col is not None else []) +
                ['Churn Probability', 'Risk Tier', 'Prediction', 'tenure',
                 'MonthlyCharges', 'TotalCharges', 'Contract']
            )
            display_cols = [c for c in display_cols if c in filtered.columns]

            st.dataframe(
                filtered[display_cols].rename(columns={
                    'Churn Probability': 'Churn Prob %',
                    'tenure': 'Tenure (mo)',
                    'MonthlyCharges': 'Monthly $',
                    'TotalCharges': 'Total $',
                }),
                use_container_width=True,
                height=400,
                column_config={
                    "Churn Prob %": st.column_config.ProgressColumn(
                        "Churn Prob %", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "Risk Tier": st.column_config.TextColumn("Risk Tier"),
                }
            )

            # Download
            csv_out = filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇ Download scored list ({len(filtered)} customers)",
                data=csv_out,
                file_name="churn_scores.csv",
                mime="text/csv"
            )

            # Top drivers chart for the high-risk cohort
            if n_high > 0:
                st.markdown('<div class="section-label" style="margin-top:24px">Top churn drivers — high risk cohort</div>',
                            unsafe_allow_html=True)
                top_features = feature_importance.head(8)
                fig, ax = plt.subplots(figsize=(8, 3.5))
                fig.patch.set_facecolor('#1A1F2E')
                ax.set_facecolor('#1A1F2E')
                bars = ax.barh(top_features.index[::-1], top_features.values[::-1],
                               color='#3B82F6', alpha=0.85, height=0.6)
                ax.set_xlabel('Feature Importance', color='#7B8299', fontsize=9)
                ax.tick_params(colors='#7B8299', labelsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2A3045')
                ax.xaxis.label.set_color('#7B8299')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        except Exception as e:
            st.error(f"Could not score this file: {e}")

    else:
        st.markdown("""
        <div style="background:#1A1F2E;border:1px dashed #2A3045;border-radius:12px;
                    padding:48px;text-align:center;color:#7B8299;margin-top:8px">
            <div style="font-size:2.5rem;margin-bottom:12px">📂</div>
            <div style="font-weight:600;color:#A0A8BF;margin-bottom:6px">No file uploaded yet</div>
            <div style="font-size:0.85rem">Upload a CSV from your CRM — customerID column optional, Churn column not needed</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2: SINGLE CUSTOMER
# ─────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown('<div class="callout">Use this when you\'re on a call with a customer and need '
                'a quick churn risk score before deciding whether to offer a retention deal.</div>',
                unsafe_allow_html=True)

    with st.form(key="single_customer"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
            gender         = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner        = st.selectbox("Partner", ["No", "Yes"])
            dependents     = st.selectbox("Dependents", ["No", "Yes"])
            tenure         = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown('<div class="section-label">Services</div>', unsafe_allow_html=True)
            phone_service     = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines    = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security   = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup     = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support      = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv      = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with c3:
            st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)
            contract          = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method    = st.selectbox("Payment Method",
                                ["Electronic check", "Mailed check",
                                 "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges   = st.slider("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total_charges     = st.slider("Total Charges ($)", 0.0, 10000.0, 2000.0)

        submitted = st.form_submit_button("Score this customer", use_container_width=True)

    if submitted:
        user_input = (
            1 if gender == "Male" else 0,
            senior_citizen,
            1 if partner == "Yes" else 0,
            1 if dependents == "Yes" else 0,
            tenure,
            1 if phone_service == "Yes" else 0,
            {"No": 0, "No phone service": 1, "Yes": 2}[multiple_lines],
            {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service],
            {"No": 0, "No internet service": 1, "Yes": 2}[online_security],
            {"No": 0, "No internet service": 1, "Yes": 2}[online_backup],
            {"No": 0, "No internet service": 1, "Yes": 2}[device_protection],
            {"No": 0, "No internet service": 1, "Yes": 2}[tech_support],
            {"No": 0, "No internet service": 1, "Yes": 2}[streaming_tv],
            {"No": 0, "No internet service": 1, "Yes": 2}[streaming_movies],
            {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
            1 if paperless_billing == "Yes" else 0,
            {"Electronic check": 0, "Mailed check": 1,
             "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[payment_method],
            monthly_charges,
            total_charges,
        )

        proba   = model.predict_proba([user_input])[0]
        prob_yes = proba[1]
        risk = "High" if prob_yes >= 0.65 else "Medium" if prob_yes >= 0.4 else "Low"
        risk_color = "#FF5C5C" if risk == "High" else "#FFB547" if risk == "Medium" else "#4ADE80"

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:32px">
                <div class="label">Churn Probability</div>
                <div class="value" style="font-size:3rem;color:{risk_color}">{prob_yes*100:.1f}<span style="font-size:1.2rem">%</span></div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:32px">
                <div class="label">Risk Tier</div>
                <div class="value" style="font-size:2rem;color:{risk_color}">{risk}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            action = ("Prioritise for retention call" if risk == "High"
                      else "Monitor — consider proactive outreach" if risk == "Medium"
                      else "Low priority — standard engagement")
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:32px">
                <div class="label">Recommended Action</div>
                <div style="font-size:0.95rem;color:#E8EAF0;margin-top:8px;font-weight:500">{action}</div>
            </div>""", unsafe_allow_html=True)

        # Top 5 factors for this customer
        importances = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False).head(5)

        st.markdown('<div class="section-label" style="margin-top:24px">Top 5 factors influencing this score</div>',
                    unsafe_allow_html=True)
        for feat, imp in importances.items():
            bar_pct = int(imp / importances.max() * 100)
            st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:0.85rem;color:#A0A8BF">{feat}</span>
                    <span style="font-size:0.85rem;color:#7B8299;font-family:monospace">{imp:.3f}</span>
                </div>
                <div style="background:#2A3045;border-radius:4px;height:6px">
                    <div style="background:#3B82F6;width:{bar_pct}%;height:6px;border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3: MODEL REPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown('<div class="section-label">Confusion matrix (normalised)</div>', unsafe_allow_html=True)
        vals = [
            [avg_TN / (avg_TN + avg_FP), avg_FP / (avg_TN + avg_FP)],
            [avg_FN / (avg_FN + avg_TP), avg_TP / (avg_FN + avg_TP)]
        ]
        colors_cm = [["#1B4332", "#7F1D1D"], ["#7C3800", "#1E3A5F"]]
        accent_cm = [["#4ADE80", "#FF5C5C"], ["#FFB547", "#3B82F6"]]
        labels_cm = [["True Negative\nCorrectly kept", "False Positive\nWrong alarm"],
                     ["False Negative\nMissed churner", "True Positive\nCaught churner"]]

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#1A1F2E')
        ax2.set_facecolor('#1A1F2E')
        ax2.set_xlim(0, 2); ax2.set_ylim(0, 2)
        ax2.set_aspect('equal'); ax2.axis('off')

        import matplotlib.patches as patches
        for i in range(2):
            for j in range(2):
                rect = patches.FancyBboxPatch(
                    (j+0.05, 1-i+0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=colors_cm[i][j], alpha=0.9,
                    edgecolor=accent_cm[i][j], linewidth=2
                )
                ax2.add_patch(rect)
                ax2.text(j+0.5, 1-i+0.62, f"{vals[i][j]:.1%}",
                         ha='center', va='center', fontsize=18,
                         fontweight='bold', color=accent_cm[i][j])
                ax2.text(j+0.5, 1-i+0.28, labels_cm[i][j],
                         ha='center', va='center', fontsize=8,
                         color='#A0A8BF', linespacing=1.4)

        ax2.text(0.5, 2.08, "No Churn", ha='center', fontsize=11, fontweight='bold', color='#4ADE80')
        ax2.text(1.5, 2.08, "Churn",    ha='center', fontsize=11, fontweight='bold', color='#FF5C5C')
        ax2.text(-0.2, 1.5, "No\nChurn", ha='center', va='center', fontsize=10, fontweight='bold', color='#4ADE80', rotation=90)
        ax2.text(-0.2, 0.5, "Churn",     ha='center', va='center', fontsize=10, fontweight='bold', color='#FF5C5C', rotation=90)
        ax2.text(1.0, 2.22, "Predicted", ha='center', fontsize=11, fontweight='bold', color='#7B8299')
        ax2.text(-0.48, 1.0, "Actual",   ha='center', fontsize=11, fontweight='bold', color='#7B8299', rotation=90)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_fi:
        st.markdown('<div class="section-label">Feature importance (top 10)</div>', unsafe_allow_html=True)
        top10 = feature_importance.head(10)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor('#1A1F2E')
        ax3.set_facecolor('#1A1F2E')
        colors_fi = ['#3B82F6' if i < 3 else '#2A3A55' for i in range(len(top10))]
        ax3.barh(top10.index[::-1], top10.values[::-1], color=colors_fi[::-1], height=0.6)
        ax3.set_xlabel('Importance', color='#7B8299', fontsize=9)
        ax3.tick_params(colors='#7B8299', labelsize=9)
        for spine in ax3.spines.values():
            spine.set_edgecolor('#2A3045')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.markdown('<div class="section-label" style="margin-top:8px">Model configuration</div>', unsafe_allow_html=True)
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.json({
            "n_estimators": 200, "max_depth": 10,
            "min_samples_split": 5, "min_samples_leaf": 2,
            "max_features": "sqrt", "churn_threshold": BEST_THRESHOLD,
            "train_runs": 9, "test_size": "20%"
        })
    with cfg_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Why threshold 0.30?</div>
            <div style="font-size:0.85rem;color:#A0A8BF;margin-top:8px;line-height:1.6">
                A lower threshold means the model flags more customers as at-risk.
                For a TelCo retention team, a missed churner (false negative) is typically
                more costly than a wrong alarm (false positive) — so we bias toward recall.
                At 0.30, the model catches <strong style="color:#4ADE80">{recall*100:.0f}%</strong>
                of actual churners.
            </div>
        </div>
        """, unsafe_allow_html=True)