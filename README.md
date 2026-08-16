# Two-Phase Weakly Supervised Framework for Anomaly Detection and Fine-Grained Crime Classification in Surveillance Videos

> **Automated Video Data Processing, Workflow Automation, and Comparative Analysis**

An end-to-end **Python-based video data processing and analysis workflow** for transforming large-scale surveillance video datasets into structured frames, extracted features, standardized clips, model outputs, and analytical results.

This project was developed as a Graduation Thesis at the **University of Information Technology – Vietnam National University, Ho Chi Minh City (UIT-VNUHCM)**.

The research proposes a two-phase coarse-to-fine framework for detecting, temporally localizing, and classifying abnormal events in surveillance videos. From a data and automation perspective, this repository highlights the workflow used to process large unstructured video datasets, automate repetitive preparation steps, compare multiple model configurations, and expose structured analytical outputs.

---

## Project Highlights

* Built a **Python-based video processing workflow** for UCF-Crime and XD-Violence, representing more than **345 hours of video data**.
* Automated preprocessing steps including **motion-based frame selection, feature extraction, temporal clip generation, and input standardization**.
* Transformed unstructured surveillance videos into structured intermediate representations including frames, feature matrices, anomaly scores, temporal clips, and predictions.
* Designed a two-phase workflow separating **Temporal Anomaly Localization** from **Fine-Grained Crime Classification**.
* Compared multiple model configurations under consistent preprocessing and evaluation settings.
* Summarized experimental results through **tables, visualizations, cross-validation results, and performance metrics**.
* Built a web-based prototype supporting **video upload, processing monitoring, analytical result tracking, dashboard visualization, alerts, feedback, and CSV export**.

---

# End-to-End Data Workflow

The proposed framework processes long surveillance videos through two connected phases.

Phase 1 reduces redundant video data, extracts structured features, and identifies suspicious temporal locations. Phase 2 converts these locations into standardized short clips for fine-grained classification.

![End-to-End Video Data Processing Pipeline](assets/images/pipeline_architecture.png)

### Workflow Overview

```text
Raw Surveillance Video
        ↓
Motion-Based Frame Selection & Feature Extraction
        ↓
Temporal Anomaly Localization
        ↓
Localization-Guided Clip Generation
        ↓
Fine-Grained Crime Classification
        ↓
Predicted Category + Confidence
```

---

# Datasets

The framework was evaluated on two widely used Video Anomaly Detection datasets.

| Dataset         | Scale                              | Classes               |
| --------------- | ---------------------------------- | --------------------- |
| **UCF-Crime**   | ~128 hours of surveillance footage | 13 anomaly categories |
| **XD-Violence** | 4,754 videos, >217 hours           | 6 anomaly categories  |

UCF-Crime contains real-world surveillance scenarios and primarily provides video-level labels for training.

XD-Violence contains more diverse video sources, viewpoints, camera movements, lighting conditions, and image quality.

Together, the two datasets provide more than **345 hours of unstructured video data** for processing and evaluation.

---

# Automated Data Processing

The preprocessing workflow transforms long raw videos into standardized representations suitable for downstream analysis.

## Processing Steps

1. **Motion-based frame selection**
   GMM-based background subtraction is used to estimate foreground activity and rank frames by motion intensity. The **Top-30 frames** are retained.

2. **Frame standardization**
   Selected frames are resized to **224 × 224 pixels**.

3. **Feature extraction**
   NASNetMobile converts each selected frame into a **1056-dimensional feature vector**.

4. **Temporal localization**
   The resulting **30 × 1056 feature matrix** is passed to temporal models to generate anomaly scores.

5. **Automated clip generation**
   The **Top-8 anomaly peaks** are mapped back to the original video. A 64-frame window is extracted around each candidate and uniformly sampled into a **16-frame clip**.

6. **Downstream classification**
   Standardized clips are classified into fine-grained abnormal-event categories.

---

## Intermediate Data Representations

| Processing Stage      | Output                 |
| --------------------- | ---------------------- |
| Motion analysis       | Motion scores          |
| Frame selection       | Top-30 frames          |
| Standardization       | 224 × 224 images       |
| Feature extraction    | 1056-D feature vectors |
| Temporal localization | Anomaly scores         |
| Candidate selection   | Temporal anchors       |
| Temporal cropping     | 64-frame windows       |
| Temporal sampling     | 16-frame clips         |
| Classification        | Category + confidence  |

The workflow separates **data preparation** from **model-specific analysis**, enabling multiple model configurations to use consistent prepared inputs.

---

# Phase 1 — Temporal Anomaly Localization

Phase 1 converts structured feature matrices into anomaly scores and candidate temporal locations.

Five temporal architectures were evaluated under the same preprocessing conditions:

* LSTM
* Transformer
* TCN
* EViT
* ST-GNN

Weak supervision was handled using **Multiple Instance Learning (MIL)** with Top-5 pooling.

## UCF-Crime Results

| Model           |  Video AUC |  Frame AUC |   Frame AP | Event Recall |
| --------------- | ---------: | ---------: | ---------: | -----------: |
| LSTM            |     92.18% |     78.62% |     22.51% |        0.232 |
| **Transformer** |     90.48% | **80.29%** | **25.11%** |    **0.304** |
| TCN             |     90.43% |     73.78% |     18.32% |        0.235 |
| EViT            |     91.22% |     76.45% |     19.87% |        0.254 |
| ST-GNN          | **93.04%** |     77.17% |     19.74% |        0.301 |

![Phase 1 Frame-Level AUC Comparison on UCF-Crime](assets/images/phase1_ucf_auc_comparison.png)

### Model Selection

ST-GNN achieved the highest **Video AUC (93.04%)**, indicating stronger overall video-level discrimination.

However, Transformer achieved stronger localization-oriented metrics:

* **Frame AUC:** 80.29% vs. 77.17%
* **Frame AP:** 25.11% vs. 19.74%
* **Event Recall:** 0.304 vs. 0.301

Since downstream clip generation depends on accurate temporal localization rather than video-level classification alone, **Transformer was selected as the final Phase 1 backbone**.

---

## XD-Violence Results

| Model       |  Video AUC |  Frame AUC |   Frame AP | Event Recall |
| ----------- | ---------: | ---------: | ---------: | -----------: |
| LSTM        |     95.74% | **87.50%** | **59.28%** |        0.352 |
| Transformer |     95.30% |     86.99% |     58.18% |        0.379 |
| TCN         |     95.93% |     84.09% |     55.79% |    **0.432** |
| EViT        |     94.91% |     84.43% |     52.81% |        0.385 |
| ST-GNN      | **96.50%** |     85.73% |     56.64% |        0.382 |

Different temporal models performed best under different metrics. Transformer remained competitive across localization measures and was retained as the common Phase 1 backbone for downstream clip generation.

---

# Phase 2 — Fine-Grained Crime Classification

Phase 2 classifies the standardized clips generated from Phase 1.

Two classifiers were evaluated:

* ConvNeXt-Tiny
* Swin Transformer Tiny (Swin-T)

## UCF-Crime

| Classifier        |   Accuracy | Precision |   Recall | Macro F1 |    Inference |
| ----------------- | ---------: | --------: | -------: | -------: | -----------: |
| **ConvNeXt-Tiny** | **41.18%** |  **0.39** | **0.35** | **0.35** | **20.21 ms** |
| Swin-T            |     39.74% |      0.35 |     0.33 |     0.33 |     23.93 ms |

![Phase 2 Classification Accuracy Comparison on UCF-Crime](assets/images/phase2_ucf_accuracy_comparison.png)

ConvNeXt-Tiny achieved the strongest overall result on UCF-Crime across classification metrics and inference time.

---

## XD-Violence

| Classifier    |   Accuracy | Precision |   Recall | Macro F1 |    Inference |
| ------------- | ---------: | --------: | -------: | -------: | -----------: |
| ConvNeXt-Tiny |     72.71% |  **0.75** |     0.72 |     0.72 | **22.12 ms** |
| **Swin-T**    | **72.96%** |      0.74 | **0.73** | **0.73** |     25.97 ms |

Swin-T achieved slightly higher Accuracy, Recall, and Macro F1, while ConvNeXt-Tiny provided higher Precision and faster inference.

Considering its stronger overall performance on UCF-Crime and competitive efficiency across both datasets, **ConvNeXt-Tiny was retained as the Phase 2 classifier in the final proposed pipeline**.

---

# Final Proposed Pipeline

```text
Transformer
Temporal Anomaly Localization
        ↓
Localization-Guided Temporal Cropping
        ↓
ConvNeXt-Tiny
Fine-Grained Crime Classification
```

The final combination was selected based on the role of each component in the full workflow rather than a single evaluation metric.

* **Transformer** was selected for stronger frame-level localization.
* **ConvNeXt-Tiny** was selected for its balance between classification performance and inference efficiency.

---

# Comparative Analysis

The experiments evaluated both predictive quality and computational cost.

### Detection & Localization

* Video-level AUC
* Frame-level AUC
* Frame Average Precision
* Temporal IoU
* Event Recall
* False Alarms per Hour
* Detection Delay

### Classification

* Accuracy
* Precision
* Recall
* Macro F1-score

### Computational Performance

* Parameter count
* FLOPs
* Inference time

Results were summarized using:

* comparison tables,
* 5-fold cross-validation,
* sensitivity charts,
* confusion matrices,
* predictive metrics,
* computational metrics.

This allowed model configurations to be compared across multiple dimensions rather than accuracy alone.

---

# Processing & Monitoring System

In addition to the experimental framework, the project includes a web-based prototype for managing the video analysis workflow.

![Video Processing Progress](assets/images/processing_progress.png)

The application supports:

* video upload,
* processing queue management,
* processing progress monitoring,
* anomaly result visualization,
* predicted category and confidence display,
* anomaly alerts,
* historical results,
* user feedback,
* CSV export.

The system connects the analytical pipeline with a user-facing workflow:

```text
Upload
→ Process
→ Analyze
→ Store
→ Monitor
→ Review
→ Export
```

---

# Structured Data Storage

The application uses relational storage to separate video metadata, processing state, analytical results, and system activity.

Core entities include:

```text
USER
VIDEO
BATCHES
PROCESSING_JOBS
ANOMALY_SEGMENTS
ACTIVITY_LOG
NOTIFICATIONS
```

The design supports tracking the lifecycle of a video from ingestion through processing and result generation.

---

# Dashboard & Reporting

Processed results are exposed through a dashboard for monitoring and analysis.

![Analysis Dashboard](assets/images/dashboard_overview.png)

The dashboard supports:

* overview statistics,
* processing monitoring,
* anomaly analysis results,
* historical records,
* alerts,
* user feedback.

---

## CSV Export

Analysis results can also be exported into CSV format.

![CSV Analysis Output](assets/images/csv_export.png)

The structured output can be reused for:

* additional analysis,
* validation,
* reporting,
* spreadsheet processing,
* downstream analytical workflows.

---

# Technology Stack

## Data Processing & Automation

* Python
* OpenCV
* NumPy
* GMM-based motion filtering
* Automated frame processing
* Feature extraction
* Temporal sampling
* Automated clip generation

## Machine Learning

* PyTorch
* NASNetMobile
* Transformer
* Multiple Instance Learning
* LSTM
* TCN
* ST-GNN
* EViT
* ConvNeXt-Tiny
* Swin Transformer

## Application

* React
* TypeScript
* FastAPI
* SQLite
* REST API

## Analysis

* 5-fold Cross-Validation
* Comparative model evaluation
* Sensitivity analysis
* Confusion matrix analysis
* Performance metrics
* CSV reporting

---

# What This Project Demonstrates

From a **Data Engineering, Automation, and Analysis** perspective, the repository highlights the following transferable skills.

### Data Processing

Transforming large unstructured video datasets into standardized and reusable representations:

```text
Videos → Frames → Features → Scores → Clips → Predictions
```

### Workflow Automation

Automating repetitive preparation steps:

```text
Frame Selection
→ Feature Extraction
→ Candidate Selection
→ Clip Generation
→ Input Standardization
```

### Structured Data Transformation

Producing analysis-ready outputs including:

* standardized frames,
* numerical feature matrices,
* anomaly scores,
* fixed-length clips,
* predictions,
* CSV reports.

### Comparative Analysis

Evaluating multiple model configurations under consistent preprocessing and evaluation conditions.

### Analytical Decision-Making

Selecting models based on downstream requirements and trade-offs across multiple metrics rather than using a single performance value.

### Monitoring & Reporting

Exposing processing outputs through:

* job monitoring,
* dashboards,
* structured database records,
* CSV export,
* alert history.

---

# Key Results

| Stage                 | Selected Model    | Key Result                              |
| --------------------- | ----------------- | --------------------------------------- |
| Phase 1 — UCF-Crime   | **Transformer**   | Frame AUC **80.29%**                    |
| Phase 2 — UCF-Crime   | **ConvNeXt-Tiny** | Accuracy **41.18%**                     |
| Phase 2 — XD-Violence | **ConvNeXt-Tiny** | Accuracy **72.71%**, Precision **0.75** |

---

# Challenges & Limitations

* Long video files require substantial storage and preprocessing resources.
* Abnormal events occupy only a small portion of long surveillance videos.
* Several anomaly categories contain visually similar patterns.
* The current framework primarily relies on visual information.
* Experiments were conducted mainly on benchmark datasets rather than live multi-camera streams.

---

# Future Improvements

Potential improvements include:

* scalable batch processing for larger video collections,
* stronger processing-job orchestration,
* improved metadata and experiment tracking,
* automated data-quality validation,
* cloud-based storage for raw and processed data,
* distributed preprocessing,
* model and inference optimization,
* richer analytical dashboards and monitoring.

---

# Summary

This project combines automated video data processing, temporal anomaly localization, fine-grained classification, and analytical reporting into an end-to-end workflow.

```text
Raw Video Data
      ↓
Automated Processing
      ↓
Structured Frames & Features
      ↓
Temporal Localization
      ↓
Automated Clip Generation
      ↓
Classification
      ↓
Structured Results
      ↓
Dashboard / CSV / Monitoring
```

From a data and automation perspective, the repository highlights three main areas:

**Data Processing → Workflow Automation → Comparative Analysis**
