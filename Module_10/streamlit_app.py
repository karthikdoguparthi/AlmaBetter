import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="CSAT Predictor AI", layout="wide", initial_sidebar_state="collapsed")

# Injecting Custom Premium CSS
st.markdown("""
<style>
    /* Premium Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #818cf8, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
        text-align: center;
    }
    .sub-header {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Hide the top padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Custom Tabs Styling to look like a Top Nav */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2);
        border-bottom: 2px solid #818cf8;
    }

    /* Style the form directly instead of using a dummy div */
    div[data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    /* Tightening the form layout */
    .stSelectbox, .stNumberInput {
        margin-bottom: -5px;
    }
    
    /* Result styling */
    .result-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        text-align: center;
        margin-top: 0px;
        animation: fadeIn 0.5s ease-in-out;
    }
    .result-score {
        font-size: 3rem;
        font-weight: 900;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    .score-out-of-5 {
        font-size: 2rem;
        font-weight: bold;
        color: #fcd34d;
        text-shadow: 0 0 10px rgba(252, 211, 77, 0.4);
    }
    .satisfied { 
        color: #10b981; 
        text-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    .neutral { 
        color: #f59e0b; 
        text-shadow: 0 0 15px rgba(245, 158, 11, 0.4);
    }
    .dissatisfied { 
        color: #ef4444; 
        text-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }
    .conf-text {
        font-size: 1.2rem;
        color: #94a3b8;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Center the submit button */
    div[data-testid="stFormSubmitButton"] > button, .big-btn {
        width: 100%;
        height: 60px;
        font-size: 1.2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border: none;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
        color: white;
    }
    div[data-testid="stFormSubmitButton"] > button:hover, .big-btn:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        transform: translateY(-2px);
    }
    
    /* Explicitly make the form submit button full width */
    div[data-testid="stFormSubmitButton"] {
        width: 100%;
    }
    div[data-testid="stFormSubmitButton"] button {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>✨ CSAT Predictor AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Real-time customer satisfaction scoring</p>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = joblib.load('csat_prediction_model.pkl')
    scaler = joblib.load('csat_scaler.pkl')
    mean_mappings = joblib.load('csat_mean_mappings.pkl')
    label_encoders = joblib.load('csat_label_encoders.pkl')
    return model, scaler, mean_mappings, label_encoders

model, scaler, mean_mappings, label_encoders = load_models()

# Extract dropdown options
channel_opts = list(label_encoders['channel_name'].classes_)
category_opts = list(label_encoders['category'].classes_)
sub_category_opts = list(label_encoders['Sub-category'].classes_)
tenure_bucket_opts = list(label_encoders['Tenure Bucket'].classes_)
agent_shift_opts = list(label_encoders['Agent Shift'].classes_)

agent_name_opts = sorted(list(mean_mappings['Agent_name'].keys()))
supervisor_opts = sorted(list(mean_mappings['Supervisor'].keys()))
manager_opts = sorted(list(mean_mappings['Manager'].keys()))

day_map = {
    "Sunday": 6, "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5
}
day_opts = list(day_map.keys())

def get_prediction(channel_name, category, sub_category, tenure_bucket, agent_shift,
                   agent_name, supervisor, manager, response_time_mins, issue_hour,
                   issue_dayofweek, has_remark, remark_word_count):
    def safe_encode(le, val):
        if val in le.classes_:
            return le.transform([val])[0]
        return 0
        
    channel_name_enc = safe_encode(label_encoders['channel_name'], channel_name)
    category_enc = safe_encode(label_encoders['category'], category)
    sub_category_enc = safe_encode(label_encoders['Sub-category'], sub_category)
    tenure_bucket_enc = safe_encode(label_encoders['Tenure Bucket'], tenure_bucket)
    agent_shift_enc = safe_encode(label_encoders['Agent Shift'], agent_shift)
    
    agent_name_mean = mean_mappings['Agent_name'].get(agent_name, 0.5)
    supervisor_mean = mean_mappings['Supervisor'].get(supervisor, 0.5)
    manager_mean = mean_mappings['Manager'].get(manager, 0.5)
    
    rt_capped = min(response_time_mins, mean_mappings.get('response_time_mins_p99', 9999))
    rt_transformed = np.log1p(rt_capped)
    remark_transformed = np.log1p(remark_word_count)
    
    feature_cols = [
        'channel_name_enc', 'category_enc', 'Sub-category_enc',
        'Tenure Bucket_enc', 'Agent Shift_enc',
        'Agent_name_csat_mean', 'Supervisor_csat_mean', 'Manager_csat_mean',
        'response_time_mins', 'issue_hour', 'issue_dayofweek',
        'has_remark', 'remark_word_count'
    ]
    
    data = [[
        channel_name_enc, category_enc, sub_category_enc,
        tenure_bucket_enc, agent_shift_enc,
        agent_name_mean, supervisor_mean, manager_mean,
        rt_transformed, issue_hour, issue_dayofweek,
        has_remark, remark_transformed
    ]]
    
    df = pd.DataFrame(data, columns=feature_cols)
    X_scaled = scaler.transform(df)
    
    prob = model.predict_proba(X_scaled)[0][1]
    return prob

def render_result(prob):
    score_out_of_5 = 1.0 + (prob * 4.0)
    
    if score_out_of_5 < 3.0:
        status_class = "dissatisfied"
        status_label = "Dissatisfied"
        emoji = "😞"
    elif score_out_of_5 > 4.0:
        status_class = "satisfied"
        status_label = "Satisfied"
        emoji = "⭐"
    else:
        status_class = "neutral"
        status_label = "Neutral"
        emoji = "😐"
    
    st.markdown(f"""
    <div class="result-card">
        <h3 style="color: #cbd5e1; margin-bottom: 0;">Prediction Score</h3>
        <div class="result-score {status_class}">{emoji}<br>{status_label}</div>
        <div class="score-out-of-5">{score_out_of_5:.1f} / 5.0</div>
        <div class="conf-text" style="margin-top: 10px;">AI Confidence: <b>{prob*100:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)


# Top Navigation using Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Predictor Model", "🏆 Optimal Strategy", "⚡ Instant Optimal Score"])

with tab1:
    main_col, result_col = st.columns([2.5, 1], gap="large")

    with main_col:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                channel_name = st.selectbox("Channel", channel_opts)
                category = st.selectbox("Category", category_opts)
                sub_category = st.selectbox("Sub-category", sub_category_opts)
                tenure_bucket = st.selectbox("Tenure Bucket", tenure_bucket_opts)
                agent_shift = st.selectbox("Agent Shift", agent_shift_opts)
                
            with c2:
                agent_name = st.selectbox("Agent Name", agent_name_opts)
                supervisor = st.selectbox("Supervisor", supervisor_opts)
                manager = st.selectbox("Manager", manager_opts)
                has_remark_str = st.selectbox("Has Remark", ["Yes", "No"])
                
            with c3:
                response_time_mins = st.number_input("Response Time (mins)", min_value=0, value=15)
                issue_hour = st.selectbox("Issue Hour (0-23)", list(range(24)))
                issue_day_str = st.selectbox("Issue Day of Week", day_opts)
                remark_word_count = st.number_input("Remark Word Count", min_value=0, value=20)
                
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Predict CSAT", use_container_width=True)

    with result_col:
        if submitted:
            issue_dayofweek = day_map[issue_day_str]
            has_remark = 1 if has_remark_str == "Yes" else 0
            
            prob = get_prediction(channel_name, category, sub_category, tenure_bucket, agent_shift,
                                  agent_name, supervisor, manager, response_time_mins, issue_hour,
                                  issue_dayofweek, has_remark, remark_word_count)
            render_result(prob)
        else:
            st.markdown("""
            <div class="result-card" style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
                <div style="color: #64748b; font-size: 1.1rem;">
                    👈 Fill the form and click <br><br><b>Predict CSAT</b><br><br>to see the magic happen!
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
<div class="result-card" style="max-width: 800px; margin: 0 auto; text-align: left; padding: 40px;">
<h2 style="color: #818cf8;">🏆 Optimal Strategy for Max CSAT</h2>
<p style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 30px;">
Based on our Machine Learning model's feature importance and probability mapping, these are the mathematically optimal parameters to achieve a near-perfect CSAT score (5.0).
</p>
<ul style="font-size: 1.2rem; line-height: 2; list-style-type: none; padding-left: 0;">
<li style="margin-bottom: 10px;"><b style="color: white; background: #334155; padding: 5px 10px; border-radius: 5px; margin-right: 10px;">Channel:</b> Email</li>
<li style="margin-bottom: 10px;"><b style="color: white; background: #334155; padding: 5px 10px; border-radius: 5px; margin-right: 10px;">Category:</b> Others</li>
<li style="margin-bottom: 10px;"><b style="color: white; background: #334155; padding: 5px 10px; border-radius: 5px; margin-right: 10px;">Agent Shift:</b> Night</li>
<li style="margin-bottom: 10px;"><b style="color: white; background: #334155; padding: 5px 10px; border-radius: 5px; margin-right: 10px;">Tenure Bucket:</b> &gt;90</li>
<li style="margin-bottom: 10px;"><b style="color: white; background: #334155; padding: 5px 10px; border-radius: 5px; margin-right: 10px;">Agent Name:</b> Brian Williams (Top Performer)</li>
<li style="margin-bottom: 10px;"><b style="color: white; background: #334155; padding: 5px 10px; border-radius: 5px; margin-right: 10px;">Response Time:</b> Minimize as much as possible (0 mins)</li>
<li style="margin-bottom: 10px;"><b style="color: white; background: #334155; padding: 5px 10px; border-radius: 5px; margin-right: 10px;">Has Remark:</b> Yes</li>
</ul>
<p style="margin-top: 30px; font-style: italic; color: #94a3b8;">
*Head over to the Instant Optimal Score tab to see this in action!
</p>
</div>
""", unsafe_allow_html=True)

with tab3:
    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        st.markdown("""
<div class="result-card" style="text-align: left; padding: 30px;">
<h3 style="color: #f472b6; margin-bottom: 20px;">⚡ Optimal Parameters</h3>
<p style="color: #cbd5e1; font-size: 1.1rem; line-height: 1.8;">
<b style="color: white;">Channel:</b> Email<br>
<b style="color: white;">Category:</b> Others<br>
<b style="color: white;">Sub-category:</b> Account updation<br>
<b style="color: white;">Tenure Bucket:</b> &gt;90<br>
<b style="color: white;">Agent Shift:</b> Night<br>
<b style="color: white;">Agent Name:</b> Brian Williams<br>
<b style="color: white;">Supervisor:</b> Abigail Suzuki<br>
<b style="color: white;">Manager:</b> Emily Chen<br>
<b style="color: white;">Response Time:</b> 0 mins<br>
<b style="color: white;">Issue Hour:</b> 10<br>
<b style="color: white;">Issue Day:</b> Monday<br>
<b style="color: white;">Has Remark:</b> Yes<br>
<b style="color: white;">Remark Words:</b> 50<br>
</p>
</div>
""", unsafe_allow_html=True)
        
    with col_b:
        # Moved the button to the top of the right column
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        opt_submit = st.button("🚀 Calculate Optimal Score", use_container_width=True)
        
        if opt_submit:
            prob = get_prediction(
                "Email", "Others", sub_category_opts[0], ">90", "Night",
                "Brian Williams", supervisor_opts[0], manager_opts[0],
                0, 10, 1, 1, 50
            )
            render_result(prob)
        else:
            st.markdown("""
<div class="result-card" style="display: flex; flex-direction: column; justify-content: center; min-height: 200px;">
<div style="color: #64748b; font-size: 1.1rem;">
Click the button above to instantly evaluate the optimal combination!
</div>
</div>
""", unsafe_allow_html=True)
