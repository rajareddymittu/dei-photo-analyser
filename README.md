# Wardrobe & Brand Intelligence with Persistent Face Recognition

A multi-modal computer vision dashboard and analytics platform built with Streamlit, PyTorch, OpenCLIP, and AuraFace. It automatically localizes people in images, zero-shot classifies clothing styles, colors, and sportswear brands, and maps recurring individuals across photos into a persistent face identity library.

---

## 1. System Architecture

![System Architecture](assets/system_architecture.svg)

<details>
<summary><b>Click to expand Mermaid source text</b></summary>

```mermaid
graph TB
    subgraph Ingestion["1. Ingestion & Hashing"]
        Upload["Photo Uploads (JPG/PNG/WEBP)"]
        Samples["Dataset Gallery (train/)"]
        Prep["prepare_source() (SHA-256 Hashing)"]
    end

    subgraph Vision["2. Dual-Vision AI Engine"]
        FRCNN["Faster R-CNN (ResNet-50-FPN)"]
        CLIP["OpenCLIP (ViT-B-32 @ LAION-2B)"]
        SCRFD["SCRFD 10G Facial Landmark Detector"]
        AuraFace["AuraFace-v1 glintr100 Recognizer"]
    end

    subgraph Storage["3. Persistent Storage"]
        CSV[("classification_results.csv")]
        Registry[("face_registry.json")]
        Occurrences[("face_occurrences.csv")]
        Templates[("face_embeddings.npz")]
    end

    subgraph UI["4. Streamlit Dashboard"]
        Tab1["Upload & Classify"]
        Tab2["Analysis Analytics"]
        Tab3["Classification Records"]
        Tab4["Face Library & Dialog"]
        Tab5["AI Assistant Chat"]
    end

    Upload --> Prep
    Samples --> Prep
    Prep --> FRCNN
    FRCNN --> CLIP
    Prep --> SCRFD
    SCRFD --> AuraFace
    CLIP --> CSV
    AuraFace --> Registry
    AuraFace --> Occurrences
    AuraFace --> Templates

    CSV --> Tab2
    CSV --> Tab3
    CSV --> Tab5
    Registry --> Tab4
    Occurrences --> Tab4
```

</details>

---

## 2. End-to-End Processing Flowchart

![Processing Pipeline](assets/process_flowchart.svg)

<details>
<summary><b>Click to expand Mermaid source text</b></summary>

```mermaid
flowchart TD
    Start(["Click 'Classify Wardrobe & Detect Faces'"]) --> Step1["Step 1: Ingestion & Deduplication Check"]
    Step1 --> Step2["Step 2: Person Localization (Faster R-CNN)"]
    Step2 --> Step3["Step 3: Wardrobe & Brand Classification (OpenCLIP)"]
    Step3 --> Step4["Step 4: Face Detection & Recognition (AuraFace)"]
    Step4 --> Done(["Hot-Reload Streamlit Views"])
```

</details>

---

## 3. Key AI Models & Specifications

| Pipeline Stage | Model Architecture | Pretrained Source / Backbone | Key Technical Parameters |
| :--- | :--- | :--- | :--- |
| **Person Detection** | **Faster R-CNN** | `ResNet-50-FPN` (TorchVision) | Default sensitivity `0.50`, 12% width / 6% height margin crops |
| **Wardrobe Classification** | **OpenCLIP** | `ViT-B-32` (`laion2b_s34b_b79k`) | Zero-shot cosine similarity, temperature scaling, top-3 attributes |
| **Brand Recognition** | **OpenCLIP** | `ViT-B-32` (`laion2b_s34b_b79k`) | Tuned brand threshold `0.55` (Nike, Adidas, Puma, UA, New Balance) |
| **Face Detection** | **SCRFD 10G** | InsightFace ONNX (`scrfd_10g_bnkps.onnx`) | Detection size `(1024, 1024)`, 5-point facial landmark regression |
| **Face Recognition** | **AuraFace-v1** | `glintr100.onnx` (`fal/AuraFace-v1`) | 512-D normalized embeddings, `82.0%` match threshold, `4.0%` margin |
| **AI Assistant** | **Llama 3.3 70B** | Groq Cloud API | Natural language querying over structured dataset context |

---

## 4. Quick Start & Running Locally

```bash
# 1. Activate python virtual environment
source .venv/bin/activate

# 2. Run the Streamlit application
streamlit run code/classifier.py
```
Open **http://localhost:8501** in your browser.
