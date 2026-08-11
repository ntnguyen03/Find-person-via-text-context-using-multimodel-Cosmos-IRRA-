### 🔎 Alpha-Gated Multimodal Person Retrieval

#### Text-Based Person Retrieval with IRRA, COSMOS and Context-Aware Fusion

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit)
![Task](https://img.shields.io/badge/Task-Text--to--Person%20Retrieval-purple)
![Status](https://img.shields.io/badge/Status-Research%20Project-orange)

</p>

---

### 📌 Overview

This project focuses on **Text-Based Person Retrieval**, where a natural-language description is used to retrieve the most relevant person images from a gallery.

For example:

> *"A man wearing a white shirt standing near a white car with green trees in the background."*

The project investigates how different types of visual information affect person retrieval, particularly:

- 👤 **Person-centric features** — clothing, appearance, accessories, etc.
- 🌳 **Background/context features** — cars, trees, roads, buildings, and surrounding environments.

Two multimodal approaches are investigated:

- **IRRA** — Implicit Relation Reasoning and Aligning
- **COSMOS** — Cross-Modality Self-Distillation

Based on the experimental results, the project proposes an **Alpha-Gated Hybrid Retrieval** mechanism that allows the user to control the balance between person-centric and context-centric information.

---

### 🎯 Research Objective

The main objective is to investigate:

> **Can visual context improve text-to-person retrieval without introducing background noise that harms person identification?**

The project therefore explores the trade-off between:

```text
Person Appearance  ←──────────────→  Visual Context
```

and proposes a controllable fusion mechanism for combining both signals.

---

### 🧠 Key Idea

A text description may contain both information about the target person and information about the surrounding environment.

```text
"A man in a white shirt standing near a white car
 with green trees in the background."

        ┌───────────────────┐
        │   Person / Subject│
        │                   │
        │   white shirt     │
        │   man             │
        └─────────┬─────────┘
                  │
                  │
        ┌─────────▼─────────┐
        │ Background /      │
        │ Context           │
        │                   │
        │ white car         │
        │ green trees       │
        └───────────────────┘
```

The proposed system separates these two signals and combines them during inference using an adjustable parameter **Alpha**.

---

### 🏗️ System Architecture

```text
                         Text Query
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Query Processing  │
                  └──────────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
          Subject / Person         Background / Context
                 │                       │
                 ▼                       ▼
           IRRA Encoder            COSMOS Encoder
                 │                       │
                 ▼                       ▼
        Subject Embedding        Context Embedding
                 │                       │
                 └───────────┬───────────┘
                             │
                             ▼
                    Alpha-Gated Fusion
                             │
                             ▼
                     Similarity Ranking
                             │
                             ▼
                        Top-K Images
```

The final similarity is calculated as:

```text
S_final = α × S_subject + (1 - α) × S_background
```

where:

| Alpha | Retrieval Behavior |
|------:|--------------------|
| `1.0` | Person-centric |
| `0.75` | Mostly person-centric |
| `0.50` | Balanced |
| `0.25` | Mostly context-centric |
| `0.0` | Context-centric |

---

### 🔬 Models

### IRRA

**IRRA (Implicit Relation Reasoning and Aligning)** is used as the primary person-centric retrieval model.

It focuses strongly on discriminative pedestrian attributes such as:

- Clothing
- Color
- Hair
- Accessories
- Appearance

IRRA serves as the main baseline for evaluating text-to-person retrieval performance.

---

### COSMOS

**COSMOS (Cross-Modality Self-Distillation)** is investigated as a multimodal model capable of capturing broader visual information.

Its representation learning includes both foreground and contextual information.

This makes COSMOS interesting for studying the role of environmental context in person retrieval.

---

### ⚠️ Background Noise Problem

The experiments revealed an important observation:

> **More visual context does not necessarily improve Person Re-ID performance.**

While contextual information can provide useful semantic cues, excessive attention to the background may introduce **background noise**.

For example, the same person may appear across different cameras:

```text
Camera A                     Camera B

   👤                           👤
  /█\                          /█\
 / █ \                        / █ \
🌳 🚗 🏠                      🛣️ 🏢 🌳
```

The person's identity remains the same while the surrounding environment changes.

Therefore, relying too heavily on contextual features can make retrieval less robust.

---

### 📊 Experimental Results

The project compares the IRRA baseline with the COSMOS-IRRA configuration.

| Model | Rank-1 | Rank-5 | Rank-10 | mAP |
|-------|-------:|-------:|--------:|----:|
| **IRRA** | **60.20** | **81.30** | **88.20** | **47.50** |
| **COSMOS-IRRA** | 55.00 | 77.05 | 84.95 | 43.60 |

Compared with the IRRA baseline, COSMOS-IRRA achieved:

- **-5.20% Rank-1**
- **-3.90% mAP**

These results motivated the development of the Alpha-Gated Hybrid Retrieval approach.

---

### 🚀 Alpha-Gated Hybrid Retrieval

Instead of forcing the system to rely entirely on either person features or context features, the proposed method allows the retrieval strategy to be controlled dynamically.

```text
                 Alpha
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
      1.0         0.5         0.0

    PERSON      BALANCED     CONTEXT
     FOCUS       FUSION       FOCUS
```

#### Person-focused retrieval

```text
Alpha = 1.0
```

The system prioritizes the appearance and identity-related features of the person.

#### Balanced retrieval

```text
Alpha = 0.5
```

Person and background information contribute equally.

#### Context-focused retrieval

```text
Alpha = 0.0
```

The retrieval is driven entirely by contextual similarity.

---

### 🖥️ Interactive Application

The proposed system is implemented as an interactive **Streamlit** application.

The interface provides:

#### 🎛️ Alpha Control

Allows users to dynamically adjust the balance between:

```text
Person Similarity  ←────────────→  Background Similarity
```

#### 🎯 Confidence Threshold

Filters retrieval results according to their similarity score.

#### 💬 Text Query

Users can enter natural-language descriptions and retrieve the most relevant pedestrian images.

---

### ⚡ Efficient Inference

To reduce inference time, image features are pre-computed for the gallery.

```text
Gallery Images
      │
      ▼
ViT-B/16 Image Encoder
      │
      ▼
Image Feature Matrix
      │
      ▼
Cached Features
```

During inference:

```text
Text Query
    │
    ▼
Text Encoder
    │
    ▼
Text Feature
    │
    ▼
Cosine Similarity
    │
    ▼
Alpha-Gated Fusion
    │
    ▼
Ranking
    │
    ▼
Top-K Results
```

This avoids repeatedly encoding the entire image gallery for every query.

---

### 📚 Dataset

The project uses **RSTPReid (Real Scenarios Text-based Person Re-identification)** for text-based person retrieval experiments.

The interactive retrieval system uses a gallery of approximately **2,000 pedestrian images**.

The dataset contains realistic variations including:

- Different camera views
- Lighting changes
- Viewpoint variations
- Clothing differences
- Background changes
- Environmental context

> Dataset files are not included in this repository because of their large size.

---

### 📂 Project Structure

```text
.
├── codeSpace/
│   └── IRRA/
│       ├── model/
│       ├── dataset/
│       ├── processor/
│       ├── utils/
│       └── ...
│
├── dataset/
│   └── RSTPReid/
│
├── paper_survey/
│
├── testfolder/
│
├── workSpace/
│
├── .gitignore
└── README.md
```

---

### ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ntnguyen03/Find-person-via-text-context-using-multimodel-Cosmos-IRRA-.git

cd Find-person-via-text-context-using-multimodel-Cosmos-IRRA-
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

#### Windows

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### ▶️ Running the Application

After configuring the dataset and model checkpoints:

```bash
streamlit run app.py
```

> The exact Streamlit entry point may vary depending on the current project structure.

---

### 🔍 Interpretability

The project also investigates model behavior through:

#### Cross-Attention Visualization

Used to analyze which regions of an image receive stronger attention.

#### t-SNE Visualization

Used to visualize the distribution of learned embeddings and identity clusters.

#### Retrieval Ranking

Used to observe how changing Alpha affects the final ranking of retrieved images.

These analyses help explain not only **which model performs better**, but also **why their retrieval behaviors differ**.

---

### 🧩 Main Contributions

#### 1. IRRA vs. COSMOS Analysis

A comparative investigation of person-centric and multimodal representation learning for text-based person retrieval.

#### 2. Background Noise Analysis

Analysis of how excessive contextual information can negatively affect Person Re-ID performance.

#### 3. Alpha-Gated Fusion

A late-fusion mechanism that dynamically combines person and background similarity.

#### 4. Human-in-the-Loop Retrieval

Users can directly control the retrieval behavior through the Alpha parameter.

#### 5. Interactive Prototype

A Streamlit-based prototype demonstrates the proposed retrieval mechanism in an interactive environment.

---

### 🛣️ Future Work

- [ ] Learnable Alpha / automatic query-aware weighting
- [ ] Improved subject/background separation
- [ ] Vietnamese language support
- [ ] More robust cross-camera evaluation
- [ ] Real-time deployment
- [ ] Edge-device optimization
- [ ] Privacy-preserving offline inference

---

### 📖 References

#### IRRA

> Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval.

#### COSMOS

> Cross-Modality Self-Distillation for Vision-Language Pre-training.

#### CLIP

> Learning Transferable Visual Models From Natural Language Supervision.

#### Vision Transformer

> An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.

---

### 👨‍💻 Author

**Nguyễn Trường Nam**

Information Technology Student  
Vietnam 🇻🇳

---

<p align="center">

⭐ If you find this project interesting, consider giving it a star.

</p>
