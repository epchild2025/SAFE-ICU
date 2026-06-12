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
model = joblib.load('lgb_model.pkl')

# 定义特征名称(与模型训练时的列名严格对应)
feature_names = [
    'gender', 'hf', 'prior_bleeding', 'stroke_tia', 'med_status', 'amiodarone', 
    'metoprolol', 'mech_vent', 'rrt', 'age', 'bmi', 'anion_gap', 'ucr', 
    'platelets', 'pt', 'tbil', 'wbc', 'dbp', 'hr', 'sbp', 'spo2', 'apsiii', 
    'sofa_score', 'copd', 'htn', 'hemoglobin'
]

######################## 2. Streamlit 页面配置 ########################
st.set_page_config(page_title="SAFE-ICU Predictor", layout="wide") 
st.title("SAFE-ICU Predictor")
st.subheader("Survival Assessment for Elderly Atrial Fibrillation Patients in the Intensive Care Unit")
st.markdown("Please input the required clinical parameters to generate a **SAFE-ICU** 1-year mortality risk assessment.")

######################## 3. 特征输入组件 ########################

# 1. gender (0: Female, 1: Male)
gender = st.selectbox(
    "What is the patient's gender?", 
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

# 2. hf (0: No, 1: Yes) 
hf = st.selectbox(
    "Heart Failure (HF) Status", 
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 3. prior_bleeding (0: No, 1: Yes) 
prior_bleeding = st.selectbox(
    "Prior Bleeding Status", 
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 4. stroke_tia (0: No, 1: Yes)
stroke_tia = st.selectbox(
    "Stroke or TIA Status", 
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 5. med_status
MED_OPTIONS = {
    0: "None",
    1: "Antiplatelet only",
    2: "Anticoagulant"
}
med_status = st.selectbox(
    "Anticoagulation or Antiplatelet Therapy",
    options=list(MED_OPTIONS.keys()),
    format_func=lambda x: MED_OPTIONS.get(x, "Unknown")
)

# 6. amiodarone (0: No, 1: Yes) 
amiodarone = st.selectbox(
    "Amiodarone Use",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 7. metoprolol (0: No, 1: Yes)
metoprolol = st.selectbox(
    "Metoprolol Use",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 8. mech_vent (0: No, 1: Yes)
mech_vent = st.selectbox(
    "Machine ventilation Use",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 9. rrt (0: No, 1: Yes)
rrt = st.selectbox(
    "Renal Replacement Therapy",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 10. age
age = st.number_input(
    "What is the patient's age?",
    min_value=65,     
    max_value=90,   
    value=70,        
    step=1
)

# 11. bmi
bmi = st.number_input(
    "What is the patient's BMI?",
    min_value=0.0,     
    max_value=100.0,   
    value=20.0,        
    step=0.1
)

# 12. anion_gap
anion_gap = st.number_input(
    "What is the patient's anion gap (mmol/L)?",
    min_value=0,     
    max_value=30,   
    value=15,        
    step=1            
)

# 13. ucr
ucr = st.number_input(
    "What is the patient's BUN/Cr ratio (UCR)?",
    min_value=0.0,     
    max_value=200.0,   
    value=20.0,        
    step=0.1          
)

# 14. platelets
platelets = st.number_input(
    "What is the patient's platelet count (10⁹/L)?",
    min_value=0.0,     
    max_value=500.0,   
    value=200.0,        
    step=1.0        
)

# 15. pt
pt = st.number_input(
    "What is the patient's prothrombin time (s)?",
    min_value=0.0,     
    max_value=100.0,   
    value=13.0,        
    step=0.1 
)

# 16. tbil
tbil = st.number_input(
    "What is the patient's total bilirubin (μmol/L)?",
    min_value=0.0,     
    max_value=10.0,   
    value=3.0,        
    step=0.1 
)

# 17. wbc
wbc = st.number_input(
    "What is the patient's blood white cell count (10⁹/L)?",
    min_value=0.0,     
    max_value=100.0,   
    value=8.0,        
    step=0.1 
)

# 18. dbp
dbp = st.number_input(
    "What is the patient's diastolic blood pressure (mmHg)?",
    min_value=20,     
    max_value=200,   
    value=70,        
    step=1
)

# 19. hr
hr = st.number_input(
    "What is the patient's heart rate (bpm)?",
    min_value=0,     
    max_value=200,   
    value=70,        
    step=1
)

# 20. sbp
sbp = st.number_input(
    "What is the patient's systolic blood pressure (mmHg)?",
    min_value=20,     
    max_value=300,   
    value=120,        
    step=1
)

# 21. spo2
spo2 = st.number_input(
    "What is the patient's SpO2 (%)?",
    min_value=0,     
    max_value=100,   
    value=98,        
    step=1
)

# 22. apsiii
apsiii = st.number_input(
    "What is the patient's APS III score?",
    min_value=0,     
    max_value=110,   
    value=45,        
    step=1
)

# 23. sofa_score
sofa_score = st.number_input(
    "What is the patient's SOFA score?",
    min_value=0,     
    max_value=12,   
    value=2,        
    step=1
)

# 24. copd (0: No, 1: Yes)
copd = st.selectbox(
    "COPD Status",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 25. htn (0: No, 1: Yes)
htn = st.selectbox(
    "Hypertension Status",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# 26. hemoglobin
hemoglobin = st.number_input(
    "What is the patient's hemoglobin (g/L)?",
    min_value=0.0,     
    max_value=20.0,   
    value=10.0,        
    step=0.1
)

######################## 4. 数据处理与预测 ########################

# 确保这里的变量顺序与 feature_names 严格一致
features_list = [
    gender, hf, prior_bleeding, stroke_tia, med_status, amiodarone, 
    metoprolol, mech_vent, rrt, age, bmi, anion_gap, ucr, 
    platelets, pt, tbil, wbc, dbp, hr, sbp, spo2, apsiii, 
    sofa_score, copd, htn, hemoglobin
]

# 转换为 DataFrame 格式，保留特征列名
features_df = pd.DataFrame([features_list], columns=feature_names)

# 点击预测按钮后执行逻辑
if st.button("Predict"):
    
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
