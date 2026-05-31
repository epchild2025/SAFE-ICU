import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import joblib
import shap
import matplotlib.pyplot as plt

######################## 1. 基础配置 ########################
# 加载训练好的最佳模型
model = joblib.load('xgb_model.pkl')

# 【关键修复】: 动态获取模型训练时的确切特征名称，避免手动硬编码导致的大小写或拼写不匹配
try:
    # 针对 scikit-learn API (XGBClassifier 等)
    expected_feature_names = model.feature_names_in_
except AttributeError:
    # 针对原生 XGBoost API
    expected_feature_names = model.get_booster().feature_names

######################## 2. Streamlit 页面配置 ########################
st.set_page_config(page_title="SAFE-ICU Predictor", layout="wide") 
st.title("SAFE-ICU Predictor")
st.subheader("Survival Assessment for Elderly Atrial Fibrillation Patients in the Intensive Care Unit")
st.markdown("Please input the required clinical parameters to generate a **SAFE-ICU** 1-year mortality risk assessment.")

######################## 3. 特征输入组件 ########################
# 1. age (连续变量：年龄)
age = st.number_input(
    "What is the patient's age?",
    min_value=65,     
    max_value=90,   
    value=70,        
    step=1
)

# 2. bmi (连续变量：bmi)
bmi = st.number_input(
    "What is the patient's bmi?",
    min_value=0.0,     
    max_value=100.0,   
    value=20.0,        
    step=0.1        
)

# 3. sbp(连续变量：收缩压)
sbp = st.number_input(
    "What is the patient's systolic blood presssure (mmHg)?",
    min_value=20,     
    max_value=300,   
    value=120,        
    step=1
)

# 4. dbp(连续变量：舒张压)
dbp = st.number_input(
    "What is the patient's diastolic blood presssure (mmHg)?",
    min_value=20,     
    max_value=200,   
    value=70,        
    step=1
)

# 5. hf(0：没有合并心衰，1：合并心衰) 
hf = st.selectbox(
    "Heart Failure (HF) Status", 
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 6. stroke_tia (0：没有合并脑卒中，1：合并脑卒中)
stroke_tia = st.selectbox(
    "Stroke or TIA Status", 
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 7. med_status
MED_OPTIONS = {
    0: "None",
    1: "Antiplatelet only",
    2: "Anticogulation"
}
med_status = st.selectbox(
    "Anticoagulation or Antiplatelet Therapy",
    options=list(MED_OPTIONS.keys()),
    format_func=lambda x: MED_OPTIONS.get(x, "Unknown")
)

# 8. metoprolol(0：未使用美托洛尔，1：使用过美托洛尔)
metoprolol = st.selectbox(
    "Metoprolol Use",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 9. ucr (连续变量：尿素氮与肌酐比值)
ucr = st.number_input(
    "What is the patient's ucr?",
    min_value=0.0,     
    max_value=200.0,   
    value=20.0,        
    step=0.1          
)

# 10. anion_gap (连续变量：阴离子间隙)
anion_gap = st.number_input(
    "What is the patient's anion gap (mmol/L)?",
    min_value=0,     
    max_value=30,   
    value=15,        
    step=1                     
)

# 11. platelets(连续变量：血小板)
platelets = st.number_input(
    "What is the patient's platelet count (10⁹/L)?",
    min_value=0.0,     
    max_value=500.0,
    value=150.0,
    step=1.0        
)

# 12. mech_vent(0：未进行过机械通气，1：进行过机械通气)
mech_vent = st.selectbox(
    "Machine ventilation Use",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 13. apsiii(连续变量：apsiii评分)
apsiii = st.number_input(
    "What is the patient's apsiii score?",
    min_value=0,     
    max_value=110,   
    value=45,        
    step=1
)

# 14. sofa_score
sofa_score = st.number_input(
    "What is the patient's sofa score?",
    min_value=0,     
    max_value=12,   
    value=2,        
    step=1
)

######################## 4. 数据处理与预测 ########################
# 点击预测按钮后执行逻辑
if st.button("Predict"):
    
    # 确保此处的变量顺序与模型训练时的列顺序绝对一致！
    features_list = [age, bmi, sbp, dbp, hf, stroke_tia, med_status, metoprolol, ucr, anion_gap, platelets, mech_vent, apsiii, sofa_score]

    # 【关键修复】: 使用直接从模型中提取的确切列名
    features_df = pd.DataFrame([features_list], columns=expected_feature_names)
    
    # 提取模型预测结果
    predicted_class = model.predict(features_df)[0] 
    predicted_proba = model.predict_proba(features_df)[0]

    # 显示风险类别
    st.subheader("Assessment Results")
    risk_label = "High Risk" if predicted_class == 1 else "Low Risk" 
    st.metric(label="Predicted Risk Category", value=risk_label)
    
    # 显示概率分布
    st.markdown(f"""
    **Probability Distribution:**
    * Low Risk: {predicted_proba[0]:.2%}
    * High Risk: {predicted_proba[1]:.2%}
    """)
    
    # 临床建议解读
    st.subheader("Clinical Interpretation")
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1: 
        advice = (
            f"The model indicates a **high-risk** prognosis for 1-year mortality (probability: {probability:.1f}%). "
            "It is recommended to implement intensive monitoring and prioritize clinical management strategies "
            "as per institutional guidelines for high-risk elderly AFib ICU patients."
        )
        st.warning(advice)
    else:
        advice = (
            f"The model indicates a **low-risk** prognosis for 1-year mortality (probability: {probability:.1f}%). "
            "Continue standard care and ongoing clinical observation according to the patient's routine status."
        )
        st.success(advice)

    ######################## 5. SHAP 解释图 ########################
    st.subheader("SHAP Interpretation") 

    # 初始化 Explainer
    explainer = shap.TreeExplainer(model) 
    shap_values = explainer(features_df)

    # 绘制瀑布图
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 对于 XGBoost，大部分情况下返回的 shap_values[0] 就是对应正类的二维数据切片
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)

    # 在 Streamlit 中渲染图像
    st.pyplot(fig)

    # 清理画布内存
    plt.clf()
