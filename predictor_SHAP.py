# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:26:01 2026

@author: LENOVO
"""

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

# 定义特征名称(替换为业务相关列名，与编码规则对应) 
feature_names = ['gender', 'hf', 'prior_bleeding', 'stroke_tia', 'med_status', 'amiodarone', 'metoprolol', 'mech_vent', 'rrt', 'age', 'bmi', 'anion_gap', 'ucr', 'platelets', 'pt', 'tbil', 'wbc', 'dbp', 'hr', 'sbp', 'spo2', 'apsiii', 'sofa_score', 'copd', 'htn', 'hemoglobin']

######################## 2. Streamlit页面配置 ########################
st.set_page_config(page_title="SAFE-ICU Predictor", layout="wide") 
st.title("SAFE-ICU Predictor")
st.subheader("Survival Assessment for Elderly Atrial Fibrillation Patients in the Intensive Care Unit")
st.markdown("Please input the required clinical parameters to generate a **SAFE-ICU** 1-year mortality risk assessment.")

######################## 3. 特征输入组件 ########################
#1.gender(0：女性，1：男性) 
male = st.selectbox(
    "What is the patient's gender?", 
    options = [0,1],
    format_func=lambda x:"Female" if x == 0 else"Male")

# 2. age (连续变量：年龄)
age = st.number_input(
    "What is the patient's age?",
    min_value=65,     
    max_value=90,   
    value=70,        
    step=1
)

# 3. bmi (连续变量：bmi)
age = st.number_input(
    "What is the patient's bmi?",
    min_value=0.0,     
    max_value=100.0,   
    value=20.0,        
    step=0.1        # 统一为浮点数
)

# 4. sbp(连续变量：收缩压)
sbp = st.number_input(
    "What is the patient's systolic blood presssure(mmHg)?",
    min_value=20,     
    max_value=300,   
    value=120,        
    step=1
)

# 5.dbp(连续变量：舒张压)
dbp = st.number_input(
    "What is the patient's diastolic blood presssure(mmHg)?",
    min_value=20,     
    max_value=200,   
    value=70,        
    step=1
)

# 6. hr(连续变量：心率)
hr = st.number_input(
    "What is the patient's heart rate(bpm)?",
    min_value=0,     
    max_value=200,   
    value=70,        
    step=1
)

# 7. spo2(连续变量：氧饱和度)
spo2 = st.number_input(
    "What is the patient's spo2(%)?",
    min_value=0,     
    max_value=100,   
    value=98,        
    step=1
)

# 8.copd(分类变量，copd：0：不合并copd，1：合并copd)
copd = st.selectbox(
    "COPD Status",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")

# 9. htn(分类变量：高血压：0：不合并高血压，1：合并高血压)
htn = st.selectbox(
    "Hypertension Status",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")

# 10. hf(0：没有合并心衰，1：合并心衰) 
hf = st.selectbox(
    "Heart Failure (HF) Status", 
    options = [0,1],
    format_func=lambda x:"No" if x == 0 else"Yes")

# 11. prior_bleeding(0：不合并出血，1：合并出血) 
prevalentStroke = st.selectbox(
    "Prior Bleeding Status", 
    options = [0,1],
    format_func=lambda x:"No" if x == 0 else"Yes")

# 12. stroke_tia (0：没有合并脑卒中，1：合并脑卒中)
stroke_tia = st.selectbox(
    "Stroke or TIA Status", 
    options = [0,1],
    format_func=lambda x:"No" if x == 0 else"Yes")

# 13. med_status(0：没有使用任何抗凝或抗板药物，1：仅使用了阿司匹林或氯吡格雷，2：使用了抗凝药物（华法林/利伐沙班/达比加群/艾多沙班/阿哌沙班) 
MED_OPTIONS = {
    0: "None",
    1: "Antiplatelet only",
    2: "Anticoagulant"
}
med_status = st.selectbox(
    "Anticoagulation or Antiplatelet Therapy",
    options=list(MED_OPTIONS.keys()),
    format_func=lambda x: MED_OPTIONS.get(x, "Unknown") # 使用 .get() 更安全
)

# 14. amiodarone(0：未使用胺碘酮，1：使用过胺碘酮） 
amiodarone = st.selectbox(
    "Amiodarone Use",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")

# 15. metoprolol(0：未使用美托洛尔，1：使用过美托洛尔)
metoprolol = st.selectbox(
    "Metoprolol Use",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")

# 16. mech_vent(0：未进行过机械通气，1：进行过机械通气)
mech_vent = st.selectbox(
    "Machine ventilation Use",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")

# 17. rrt(0：未进行过肾脏替代治疗，1：进行过肾脏替代治疗)
rrt = st.selectbox(
    "Renal Replacement Therapy",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")

# 18. anion_gap (连续变量：阴离子间隙)
anion_gap = st.number_input(
    "What is the patient's anion gap(mmol/L)?",
    min_value=0,     
    max_value=30,   
    value=15,        
    step=1           
)

# 19. ucr (连续变量：尿素氮与肌酐比值)
ucr = st.number_input(
    "What is the patient's ucr?",
    min_value=0.0,     
    max_value=200.0,   
    value=20.0,        
    step=0.1          
)

# 20. platelets(连续变量：血小板)
platelets = st.number_input(
    "What is the patient's platelet count (10⁹/L)?",
    min_value=0.0,     
    max_value=500.0,   
    value=200.0,        
    step=1.0        
)

# 21. pt(连续变量：凝血酶原时间)
platelets = st.number_input(
    "What is the patient's prothrombin time (s)?",
    min_value=0.0,     
    max_value=100.0,   
    value=13.0,        
    step=0.1 
)

# 22. tbil(连续变量：总胆红素)
tbil = st.number_input(
    "What is the patient's total bilirubin (μmol/L)?",
    min_value=0.0,     
    max_value=10.0,   
    value=3.0,        
    step=0.1 
)

# 23. wbc(连续变量：白细胞)
wbc = st.number_input(
    "What is the patient's blood white cell(10⁹/L)?",
    min_value=0.0,     
    max_value=100.0,   
    value=8.0,        
    step=0.1 
)

# 24. hemoglobin
hemoglobin = st.number_input(
    "What is the patient's hemoglobin(g/L)?",
    min_value=0.0,     
    max_value=20.0,   
    value=10.0,        
    step=0.1
)

# 25. apsiii(连续变量：apsiii评分)
apsiii = st.number_input(
    "What is the patient's apsiii score?",
    min_value=0,     
    max_value=110,   
    value=45,        
    step=1
)

# 26. sofa_score
sofa_score = st.number_input(
    "What is the patient's sofa score?",
    min_value=0,     
    max_value=12,   
    value=2,        
    step=1
)
######################## 4. 数据处理与预测 ########################
feature_values = ['gender', 'hf', 'prior_bleeding', 'stroke_tia', 'med_status', 'amiodarone', 'metoprolol', 'mech_vent', 'rrt', 'age', 'bmi', 'anion_gap', 'ucr', 'platelets', 'pt', 'tbil', 'wbc', 'dbp', 'hr', 'sbp', 'spo2', 'apsiii', 'sofa_score', 'copd', 'htn', 'hemoglobin']    
features = np.array([feature_values])

# 点击预测按钮后，执行下方所有缩进的代码
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0] 
    predicted_proba = model.predict_proba(features)[0] 

    # 显示预测结果
    st.subheader("Assessment Results")
    risk_label = "High Risk" if predicted_class == 1 else "Low Risk" 
    st.metric(label="Predicted Risk Category", value=risk_label)
    
    st.markdown(f"""
    **Probability Distribution:**
    * Low Risk: {predicted_proba[0]:.2%}
    * High Risk: {predicted_proba[1]:.2%}
    """)
    
    # Clinical Interpretation
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
    ######################## 5. SHAP (单样本特征贡献解释) ########################
    # 建议放在 if st.button("预测"): 里面
    st.subheader("SHAP explain") 

    # 1. 初始化 Explainer
    # 注意：如果是随机森林、XGBoost、LightGBM 等树模型，直接用 TreeExplainer。
    # 这样不需要传入 training_data，直接传模型即可，
    explainer = shap.TreeExplainer(model) 

    # 如果是逻辑回归等线性模型，请改用： 
    # explainer = shap.LinearExplainer(model, x_test_filtered.values)

    # 2. 计算当前患者的 SHAP 值
    # 确保 features 是二维数组，例如 shape 为 (1, 12)
    shap_values = explainer(features.reshape(1, -1))

    # 3. 绘制并展示瀑布图
    fig, ax = plt.subplots(figsize=(8, 5))

    # 二分类模型通常会输出两个类别的 SHAP 值。
    # 假设你想解释的是“高心脏病风险” (通常对应索引 1)
    # 如果是 XGBoost，shap_values 可能直接是一维的；如果是 RandomForest，可能需要取 shap_values[0, :, 1]
    # 这里以最常见的标准输出为例 (取第0个样本)：
    shap.plots.waterfall(shap_values[0], max_display=12, show=False) # max_display=特征变量数

    # 将 matplotlib 图像传递给 Streamlit 显示
    st.pyplot(fig)

    # 清理内存，防止连续点击预测导致图表重叠或内存泄漏
    plt.clf()
