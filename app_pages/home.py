import pandas as pd
import streamlit as st

from config import CLASS_IDS, CLASS_NAMES_CN


st.markdown(
    "本系统使用 PlantVillage 公开数据训练番茄叶片十分类模型，"
    "用于展示从数据准备、迁移学习到可解释预测的完整课程实验流程。"
)

with st.container(horizontal=True):
    st.metric("识别类别", "10 类", border=True)
    st.metric("主模型", "ResNet18", border=True)
    st.metric("基线模型", "SimpleCNN", border=True)

with st.container(border=True):
    st.subheader("使用步骤")
    st.markdown(
        "1. 在“叶片识别”页面上传一张清晰的番茄叶片图片。\n"
        "2. 查看 Top-3 类别、置信度和 Grad-CAM 关注区域。\n"
        "3. 在“模型分析”页面查看实际评估指标与错误样本。"
    )

classes = pd.DataFrame(
    {
        "中文类别": [CLASS_NAMES_CN[class_id] for class_id in CLASS_IDS],
        "数据标签": list(CLASS_IDS),
    }
)
st.subheader("支持的类别")
st.dataframe(classes, hide_index=True, alt="系统支持的十个番茄叶片类别")

st.info(
    "本系统是课程实验演示，不替代农业专家的现场诊断，也不提供农药、剂量或治疗处方。",
    icon=":material/info:",
)
