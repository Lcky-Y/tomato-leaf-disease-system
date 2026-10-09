from pathlib import Path

import pandas as pd
import streamlit as st

from app import cached_inference_bundle, low_confidence_message, validate_upload
from disease_info import DISEASE_INFO
from explain import UnsupportedExplanationError, generate_gradcam, overlay_heatmap
from predict import predict_image
from utils.checkpoints import CheckpointCompatibilityError


with st.sidebar:
    checkpoint_text = st.text_input(
        "模型检查点",
        value="outputs/resnet18-main/best_model.pth",
        help="填写训练生成的 best_model.pth 路径。",
        key="recognition_checkpoint",
    )

with st.form("recognition_form", border=True):
    uploaded = st.file_uploader(
        "上传番茄叶片图片",
        type=["jpg", "jpeg", "png"],
        help="支持 JPG、JPEG、PNG，文件最大 10 MB。",
    )
    submitted = st.form_submit_button(
        "开始识别", type="primary", icon=":material/search:"
    )

st.caption("推理会使用检查点中保存的类别顺序、输入尺寸和归一化参数。")

if submitted:
    if uploaded is None:
        st.warning("请先上传一张叶片图片。", icon=":material/warning:")
    else:
        try:
            image = validate_upload(uploaded.name, uploaded.getvalue())
            checkpoint = Path(checkpoint_text)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"未找到模型权重：{checkpoint}")
            bundle = cached_inference_bundle(str(checkpoint.resolve()))
            result = predict_image(image, bundle)

            st.image(
                image,
                caption="上传的叶片图片",
                alt="待识别的番茄叶片",
                width="stretch",
            )
            primary = result.items[0]
            with st.container(horizontal=True):
                st.metric("预测类别", primary.display_name, border=True)
                st.metric("最高置信度", f"{primary.probability:.2%}", border=True)
                st.metric("推理耗时", f"{result.elapsed_ms:.1f} ms", border=True)

            warning = low_confidence_message(result.low_confidence)
            if warning:
                st.warning(warning, icon=":material/warning:")

            probabilities = pd.DataFrame(
                {
                    "类别": [item.display_name for item in result.items],
                    "概率": [item.probability for item in result.items],
                }
            )
            st.subheader("Top-3 预测")
            st.bar_chart(
                probabilities,
                x="类别",
                y="概率",
                horizontal=True,
                alt="模型置信度最高的三个番茄叶片类别",
            )

            info = DISEASE_INFO.get(primary.class_id)
            if info is not None:
                with st.container(border=True):
                    st.subheader("类别说明")
                    st.markdown(info.description)

            try:
                heatmap = generate_gradcam(image, bundle)
                overlay = overlay_heatmap(image, heatmap)
                st.subheader("Grad-CAM 关注区域")
                st.image(
                    overlay,
                    caption="暖色区域代表模型更关注的位置",
                    alt="模型在番茄叶片上的 Grad-CAM 关注区域",
                    width="stretch",
                )
            except UnsupportedExplanationError as exc:
                st.info(str(exc), icon=":material/info:")
        except (FileNotFoundError, CheckpointCompatibilityError, ValueError, OSError) as exc:
            st.error(str(exc), icon=":material/error:")

st.info(
    "识别结果仅用于课程实验观察，不构成农业专业诊断。",
    icon=":material/info:",
)
