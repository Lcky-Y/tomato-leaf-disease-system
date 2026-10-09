from pathlib import Path

import streamlit as st

from app import load_analysis_artifacts


with st.sidebar:
    output_text = st.text_input(
        "评估结果目录",
        value="outputs/evaluation/resnet18-main",
        help="目录内包含 metrics.json、混淆矩阵和 errors.csv。",
        key="analysis_output_dir",
    )
    training_text = st.text_input(
        "训练结果目录",
        value="outputs/resnet18-main",
        help="目录内可包含 training_curves.png。",
        key="analysis_training_dir",
    )
    comparison_text = st.text_input(
        "模型对比表",
        value="outputs/evaluation/comparison.csv",
        help="由 compare_runs 生成的 comparison.csv。",
        key="analysis_comparison_path",
    )

artifacts = load_analysis_artifacts(
    Path(output_text),
    training_dir=Path(training_text),
    comparison_path=Path(comparison_text),
)

if artifacts.metrics is not None:
    metrics = artifacts.metrics
    with st.container(horizontal=True):
        st.metric("Accuracy", f"{float(metrics.get('accuracy', 0)):.2%}", border=True)
        st.metric("Macro-F1", f"{float(metrics.get('macro_f1', 0)):.2%}", border=True)
        st.metric(
            "CPU 单图延迟",
            f"{float(metrics.get('cpu_latency_ms_per_image', metrics.get('latency_ms_per_image', 0))):.1f} ms",
            border=True,
        )
        st.metric("运行设备", str(metrics.get("device", "未知")), border=True)
else:
    st.warning("未找到 metrics.json；请先运行独立评估。", icon=":material/warning:")

if artifacts.comparison is not None:
    st.subheader("模型对比")
    st.dataframe(
        artifacts.comparison,
        hide_index=True,
        alt="SimpleCNN 与 ResNet18 的测试指标和效率对比",
    )
else:
    st.info("未找到 comparison.csv，暂不显示模型对比表。")

left, right = st.columns(2)
with left:
    st.subheader("训练曲线")
    if artifacts.training_curve is not None:
        st.image(
            artifacts.training_curve,
            alt="训练集与验证集的损失、准确率和 Macro-F1 曲线",
            width="stretch",
        )
    else:
        st.info("未找到 training_curves.png。")
with right:
    st.subheader("归一化混淆矩阵")
    if artifacts.confusion_matrix is not None:
        st.image(
            artifacts.confusion_matrix,
            alt="十个番茄叶片类别的归一化混淆矩阵",
            width="stretch",
        )
    else:
        st.info("未找到 confusion_matrix_normalized.png。")

st.subheader("高置信度错误样本")
if artifacts.errors is not None:
    st.dataframe(
        artifacts.errors,
        hide_index=True,
        alt="按置信度从高到低排列的错误分类样本",
    )
else:
    st.info("未找到 errors.csv。")
