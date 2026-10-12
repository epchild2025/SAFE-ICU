# SAFE-ICU：随机森林网页版本

本版本使用重新分析后的固定随机森林模型，预测老年房颤ICU患者的院内死亡。预测时点为ICU入住后24小时，适用于在该时点仍符合研究入选条件的患者。

## 上传GitHub

将以下文件放在GitHub仓库同一个目录，建议仓库根目录：

- `app.py`：网页主体。
- `predictor_SHAP.py`：兼容原有Streamlit入口，转到新版网页。
- `rf_predictor.py`：模型读取、输入检查、预测和SHAP处理。
- `rf_model.pkl`：已核对的最终随机森林，约5 MB。
- `requirements.txt`：部署依赖版本。
- `README.md`：本说明。
- `.gitignore`：排除本地测试文件。
