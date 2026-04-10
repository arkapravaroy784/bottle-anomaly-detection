# Bottle Anomaly Detection (MVTec)

Unsupervised basic anomaly detection system using
deep feature extraction and reconstruction-based techniques, deployed as an
interactive **Streamlit web application**.

## Methods
- YOLOv8-based object localization (optional preprocessing)
- ResNet18 (ImageNet pretrained) for deep feature extraction
- Feature-level anomaly scoring using distance-based metrics
- Patch-wise feature aggregation for fine grained anomaly localization
- ROC-AUC evaluation for quantitative performance analysis

## Deployment
- Streamlit-based web application
- Upload and analyze bottle images in real time
- Visual anomaly heatmaps and confidence scores
- Optimized inference for CPU and GPU environments

## Key Insights
- Smooth and structured objects like bottles are well suited for feature based
  anomaly detection
- Pretrained CNN features capture subtle surface defects effectively
- Simpler models with strong representations outperform deeper but unfocused architectures

## Dataset
- **MVTec Anomaly Detection Dataset**
- Category used: `Bottle`
- Training data contains **only normal samples**

## Demo
🔗 **Live Demo:** https://basicbottle-anomaly-detectionand-deployment-hvnnjozy4ltweupntu.streamlit.app
<img width="757" height="961" alt="569821745-e9d41b30-e92c-4d6b-84e6-6107b154e682" src="https://github.com/user-attachments/assets/4c3c9b71-814b-445c-a9d8-41be9997ebe2" />

## Author
Arkaprava Roy
