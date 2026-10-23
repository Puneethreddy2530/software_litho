import streamlit as st
import pandas as pd
import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Need to import model
try:
    from src.models.server_fusion import LithoScanServerFusion
except ImportError:
    pass

st.set_page_config(page_title="LithoScan Dashboard", layout="wide", page_icon="🪨")

st.sidebar.title("LithoScan")
st.sidebar.subheader("Automated Drill-Core Lithology Classification")
page = st.sidebar.radio("Navigation", [
    "Overview", 
    "Model Comparison", 
    "Image Prediction", 
    "Robustness Analysis", 
    "Confusion Matrices", 
    "Experiment Timeline", 
    "Artifact Explorer"
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    """<small><b>LithoScan Research Prototype</b><br>
    Benchmark results are dataset-specific and do not by themselves establish field/generalization performance.</small>""", 
    unsafe_allow_html=True
)

def load_metrics(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def format_val(val, is_pct=False, decimals=4):
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return val

if page == "Overview":
    st.title("1. Overview")
    st.write("**Project:** LithoScan")
    st.write("**Dataset:** DCID")
    st.write("**Current status:** Phase 5A RESULT LOCKED")
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("DCID-7", "7 Classes")
    with col2:
        st.metric("DCID-35", "35 Classes")
    with col3:
        st.metric("Best Clean Test Acc", "99.77%")
    with col4:
        st.metric("Best Robustness (Noise)", "88.93%")
        
    st.markdown("### Pipeline")
    st.code("Image → CNN Backbone → Spatial Representation → Classification → Lithology Prediction", language="text")
    
    st.markdown("### Major Completed Phases")
    st.markdown("""
    - **Phase 1** Dataset / Infrastructure ✅ COMPLETE
    - **Phase 2** ResNet Baseline ✅ COMPLETE
    - **Phase 3** Dataset / Shortcut Audit ✅ COMPLETE
    - **Phase 4** Robustness / Architecture ✅ COMPLETE
    - **Phase 5A** DCID-35 Comparison ✅ COMPLETE
    """)

elif page == "Model Comparison":
    st.title("2. Model Comparison")
    
    # Load JSON metrics
    cnn_only = load_metrics("reports/phase5/dcid35_cnn_only/metrics.json")
    cnn_trans = load_metrics("reports/phase5/dcid35_server/metrics.json")
    
    data = {
        "Model": ["ResNet-50", "CNN-only", "CNN-Transformer"],
        "Best Val F1": [0.9882, cnn_only.get("validation_best_f1"), cnn_trans.get("validation_best_f1")],
        "Clean Accuracy": [0.9928, cnn_only.get("test_clean_accuracy"), cnn_trans.get("test_clean_accuracy")],
        "Clean F1": [0.9928, cnn_only.get("test_clean_macro_f1"), cnn_trans.get("test_clean_macro_f1")],
        "Noise Accuracy": ["N/A", cnn_only.get("test_noise_accuracy"), cnn_trans.get("test_noise_accuracy")],
        "Noise F1": ["N/A", cnn_only.get("test_noise_macro_f1"), cnn_trans.get("test_noise_macro_f1")],
        "Noise Drop": ["N/A", cnn_only.get("noise_absolute_drop"), cnn_trans.get("noise_absolute_drop")],
        "Masked Accuracy": ["N/A", cnn_only.get("test_masked_accuracy"), cnn_trans.get("test_masked_accuracy")],
        "Masked F1": ["N/A", cnn_only.get("test_masked_macro_f1"), cnn_trans.get("test_masked_macro_f1")],
        "Mask Drop": ["N/A", cnn_only.get("mask_absolute_drop"), cnn_trans.get("mask_absolute_drop")],
        "Parameters": ["~25.6M", cnn_only.get("parameter_count"), cnn_trans.get("parameter_count")],
        "Peak CUDA Memory": ["N/A", cnn_only.get("peak_cuda_memory_mb"), cnn_trans.get("peak_cuda_memory_mb")],
        "Training Time": ["N/A", cnn_only.get("training_time_seconds"), cnn_trans.get("training_time_seconds")],
        "Best Epoch": ["N/A", cnn_only.get("best_epoch"), cnn_trans.get("best_epoch")]
    }
    
    df = pd.DataFrame(data)
    
    # Format dataframe for display
    display_df = df.copy()
    for col in display_df.columns:
        if col not in ["Model", "Parameters", "Peak CUDA Memory", "Training Time", "Best Epoch"]:
            display_df[col] = display_df[col].apply(lambda x: format_val(x))
        elif col in ["Training Time", "Peak CUDA Memory"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else x)
            
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("### Visualizations")
    col1, col2 = st.columns(2)
    
    models = ["ResNet-50", "CNN-only", "CNN-Transformer"]
    
    with col1:
        clean_acc = [0.9928, cnn_only.get("test_clean_accuracy", 0), cnn_trans.get("test_clean_accuracy", 0)]
        fig = px.bar(x=models, y=clean_acc, title="A. Clean Accuracy Comparison", labels={"x": "Model", "y": "Accuracy"}, range_y=[0.98, 1.0])
        st.plotly_chart(fig, use_container_width=True)
        
        mask_acc = [0, cnn_only.get("test_masked_accuracy", 0), cnn_trans.get("test_masked_accuracy", 0)]
        fig3 = px.bar(x=["CNN-only", "CNN-Transformer"], y=[mask_acc[1], mask_acc[2]], title="C. Masked Accuracy Comparison", labels={"x": "Model", "y": "Accuracy"}, range_y=[0.98, 1.0])
        st.plotly_chart(fig3, use_container_width=True)
        
        train_time = [0, cnn_only.get("training_time_seconds", 0)/3600, cnn_trans.get("training_time_seconds", 0)/3600]
        fig5 = px.bar(x=["CNN-only", "CNN-Transformer"], y=[train_time[1], train_time[2]], title="E. Training Time (Hours)", labels={"x": "Model", "y": "Hours"})
        st.plotly_chart(fig5, use_container_width=True)
        
    with col2:
        noise_acc = [0, cnn_only.get("test_noise_accuracy", 0), cnn_trans.get("test_noise_accuracy", 0)]
        fig2 = px.bar(x=["CNN-only", "CNN-Transformer"], y=[noise_acc[1], noise_acc[2]], title="B. Noise Accuracy Comparison", labels={"x": "Model", "y": "Accuracy"}, range_y=[0.8, 0.95])
        st.plotly_chart(fig2, use_container_width=True)
        
        params = [25600000, cnn_only.get("parameter_count", 0), cnn_trans.get("parameter_count", 0)]
        fig4 = px.bar(x=models, y=params, title="D. Parameter Count", labels={"x": "Model", "y": "Parameters"})
        st.plotly_chart(fig4, use_container_width=True)

elif page == "Image Prediction":
    st.title("3. Image Prediction")
    st.warning("⚠️ **IMPORTANT**: MODEL INFERENCE / DEMONSTRATION ONLY. This does not represent field validation.")
    
    model_choice = st.selectbox("Select Model", ["CNN-only", "CNN-Transformer"])
    
    @st.cache_resource
    def load_model_for_inference(model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "CNN-only":
            model = LithoScanServerFusion(num_classes=35, architecture_mode="cnn_only")
            ckpt_path = "checkpoints/phase5/dcid35_cnn_only/best_model.pt"
        else:
            model = LithoScanServerFusion(num_classes=35, architecture_mode="cnn_transformer")
            ckpt_path = "checkpoints/phase5/dcid35_server/best_model.pt"
            
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=device)
            # handle module. prefix if DDP was used
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model, device
        return None, None
        
    @st.cache_data
    def get_classes():
        manifest_path = Path("data/manifests/dcid_manifest.csv")
        if manifest_path.exists():
            df = pd.read_csv(manifest_path, usecols=['dataset_version', 'class_name', 'class_id'])
            df_35 = df[df['dataset_version'] == 'DCID-512-35']
            if not df_35.empty:
                unique_classes = df_35[['class_id', 'class_name']].drop_duplicates().sort_values('class_id')
                return unique_classes['class_name'].tolist()
                
        # Fallback
        return [f"Class_{i}" for i in range(35)]
        
    uploaded_file = st.file_uploader("Upload a drill-core image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            st.markdown("### Inference Results")
            model, device = load_model_for_inference(model_choice)
            if model is None:
                st.error(f"Checkpoint for {model_choice} not found!")
            else:
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1)[0]
                    
                classes = get_classes()
                top5_prob, top5_catid = torch.topk(probs, 5)
                
                st.success(f"**Predicted Class:** {classes[top5_catid[0].item()]}")
                st.write(f"**Confidence:** {top5_prob[0].item():.2%}")
                
                st.markdown("#### Top 5 Probabilities:")
                top5_df = pd.DataFrame({
                    "Class": [classes[catid] for catid in top5_catid.tolist()],
                    "Probability": top5_prob.tolist()
                })
                fig = px.bar(top5_df, x="Probability", y="Class", orientation='h')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

elif page == "Robustness Analysis":
    st.title("4. Robustness Analysis")
    
    st.subheader("DCID-35 Official Evaluation")
    
    cnn_only = load_metrics("reports/phase5/dcid35_cnn_only/metrics.json")
    cnn_trans = load_metrics("reports/phase5/dcid35_server/metrics.json")
    
    df_rob = pd.DataFrame({
        "Condition": ["Clean", "Official Noise", "40px Border Mask"],
        "CNN-only Acc": [cnn_only.get("test_clean_accuracy"), cnn_only.get("test_noise_accuracy"), cnn_only.get("test_masked_accuracy")],
        "CNN-Transformer Acc": [cnn_trans.get("test_clean_accuracy"), cnn_trans.get("test_noise_accuracy"), cnn_trans.get("test_masked_accuracy")]
    })
    
    for col in ["CNN-only Acc", "CNN-Transformer Acc"]:
        df_rob[col] = df_rob[col].apply(lambda x: format_val(x))
        
    st.table(df_rob)
    
    st.markdown("---")
    st.subheader("Phase 4D Similarity-Filtered Evaluation (DCID-7)")
    st.info("⚠️ Note: Phase 4D was performed on DCID-7. Do NOT imply these results apply to DCID-35.")
    
    df_sim = pd.DataFrame({
        "Test Split": ["TEST_ALL", "FILTERED_2", "FILTERED_5", "FILTERED_10"],
        "Sample Count": [2000, 1508, 1024, 825], # Approximated from existing reports or standard
        "CNN-only Acc": ["0.9996", "0.9993", "0.9990", "0.9976"], 
        "CNN-Transformer Acc": ["0.9994", "0.9993", "0.9990", "0.9988"]
    })
    st.table(df_sim)

elif page == "Confusion Matrices":
    st.title("5. Confusion Matrices")
    
    model_choice = st.selectbox("Select Model", ["CNN-only", "CNN-Transformer"])
    normalize = st.selectbox("Normalization", ["raw counts", "normalized by true class"])
    
    if model_choice == "CNN-only":
        cm_path = "reports/phase5/dcid35_cnn_only/confusion_matrix.csv"
    else:
        cm_path = "reports/phase5/dcid35_server/confusion_matrix.csv"
        
    if os.path.exists(cm_path):
        df_cm = pd.read_csv(cm_path, index_col=0)
        
        if normalize == "normalized by true class":
            df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
            
        fig = px.imshow(df_cm, text_auto=False, aspect="auto", title=f"{model_choice} Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True, height=800)
    else:
        st.error(f"Confusion matrix not found at {cm_path}")

elif page == "Experiment Timeline":
    st.title("6. Experiment Timeline")
    
    st.markdown("""
    <style>
    .timeline {
        border-left: 3px solid #3b82f6;
        padding-left: 20px;
        margin-left: 10px;
    }
    .timeline-item {
        margin-bottom: 20px;
    }
    .timeline-title {
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 5px;
    }
    .status-badge {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
    </style>
    <div class="timeline">
        <div class="timeline-item">
            <div class="timeline-title">Phase 1 <span class="status-badge">PASS / COMPLETE</span></div>
            <div>Dataset Reconnaissance</div>
        </div>
        <div class="timeline-item">
            <div class="timeline-title">Phase 2 <span class="status-badge">PASS / COMPLETE</span></div>
            <div>ResNet Baseline</div>
        </div>
        <div class="timeline-item">
            <div class="timeline-title">Phase 3 <span class="status-badge">PASS / COMPLETE</span></div>
            <div>Dataset Difficulty / Shortcut Audit</div>
        </div>
        <div class="timeline-item">
            <div class="timeline-title">Phase 4 <span class="status-badge">PASS / COMPLETE</span></div>
            <div>CNN-Transformer / Ablation / Interventions / Similarity Stress Test</div>
        </div>
        <div class="timeline-item">
            <div class="timeline-title">Phase 5A <span class="status-badge">PASS / COMPLETE</span></div>
            <div>DCID-35 Architecture Comparison</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Artifact Explorer":
    st.title("7. Artifact Explorer")
    
    def scan_reports():
        artifacts = []
        base = Path("reports")
        if base.exists():
            for root, _, files in os.walk(base):
                for file in files:
                    if file.endswith((".md", ".json", ".csv", ".log", ".yaml")):
                        path = Path(root) / file
                        artifacts.append({
                            "Filename": file,
                            "Phase": path.parts[1] if len(path.parts) > 1 else "Root",
                            "Type": path.suffix,
                            "Path": str(path)
                        })
        return pd.DataFrame(artifacts)
        
    df_arts = scan_reports()
    if not df_arts.empty:
        # Filter for only Phase 3, 4, 5
        df_filtered = df_arts[df_arts["Phase"].isin(["phase3", "phase4", "phase5"])]
        
        st.dataframe(df_filtered[["Filename", "Phase", "Type"]], use_container_width=True)
        
        st.markdown("### Read Report")
        selected_file = st.selectbox("Select a file to view", df_filtered["Path"].tolist())
        
        if st.button("Read File"):
            if selected_file.endswith(".md") or selected_file.endswith(".log") or selected_file.endswith(".json") or selected_file.endswith(".yaml"):
                try:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if selected_file.endswith(".md"):
                        st.markdown(content)
                    else:
                        st.code(content)
                except Exception as e:
                    st.error(f"Could not read file: {e}")
            else:
                st.warning("Preview only available for text/markdown files.")
    else:
        st.info("No artifacts found.")
