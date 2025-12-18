# Cost-Aware Structured Generation: High-Fidelity Synthesis via Hybrid RAG and Adaptive Conditional Compute

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18270482.svg)](https://zenodo.org/records/18270482)

**Author:** Anand Wankhade  
**Paper:** [Read the full research paper](research_publications/papers/Cost_Aware_Structured_Generation.md)

> **Abstract:** We introduce a domain-agnostic 14-stage pipeline that integrates Hybrid RAG with Reciprocal Rank Fusion, conditional compute routing based on query difficulty, and adaptive voting strategies. This architecture achieves **93.5% overall precision** with a hallucination rate of only **0.31%** (1 error in 325 human-reviewed samples) on 1,200+ complex aerospace engineering questions, reducing costs by **50.4%** compared to standard ensembles.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Innovations](#-key-innovations)
- [Dataset Details](#-dataset-details)
- [14-Stage Pipeline Architecture](#-14-stage-pipeline-architecture)
- [Installation & Setup](#-installation--setup)
- [Configuration & Domain Adaptation](#-configuration--domain-adaptation)
- [Usage](#-usage)
- [Performance & Cost Analysis](#-performance--cost-analysis)
- [Citation](#-citation)

---

## 🎯 Overview

This repository contains the production-grade implementation of the **Cost-Aware Structured Generation** paradigm. The system addresses the fundamental tension between quality and cost in generating rigorous, schema-compliant data for high-stakes domains (Aerospace Engineering, Legal, Medical, Financial).

It processes **1,200+ GATE Aerospace Engineering questions** (2007-2025) to generate comprehensive structured JSON outputs with **200+ fields** per question, ensuring deep pedagogical tagging and step-by-step reasoning.

---

## ✨ Key Innovations

### 1. Hybrid RAG Integration Framework
Systematically combines **Dense Retrieval** (BAAI/bge-m3) and **Sparse Retrieval** (BM25) using **Reciprocal Rank Fusion (RRF)**. This handles both semantic nuance and exact technical terminology (e.g., "Reynolds number", "Euler-Bernoulli").

### 2. Schema-Aware Conditional Compute
A classification-based router (Gemini 2.0 Flash) estimates query complexity on a 1-10 scale and routes to three cost-efficient tiers:
- **Tier 1 (Difficulty 1-3)**: Template + Rules (~$0.01)
- **Tier 2 (Difficulty 4-5)**: Mid-tier models (~$0.08)
- **Tier 3 (Difficulty ≥6)**: Full SOTA Ensemble + GPT-5.1 (~$0.35)

### 3. Adaptive Model Weighting
Configurable weighting maps domain priorities to model strengths (defined in `config/weights_config.yaml`):
- **DeepSeek R1**: Heavy weight for **MATH_WEIGHTED** derivation tasks.
- **Claude Sonnet 4.5**: Heavy weight for **CONCEPTUAL** and pedagogical clarity.
- **Gemini 2.5 Pro**: Heavy weight for **VISION** tasks.
- **GPT-5.1**: Utilized as a "Judge" for tie-breaking and validation.

### 4. Multi-Round Debate Orchestration
A safety valve for high-stakes accuracy. If consensus < 80%, the system triggers:
- **Round 1**: Models defend/revise answers.
- **Round 2**: GPT-5.1 acts as an impartial judge to resolve persisting disputes.

---

## 📂 Dataset Details

The release includes the full processed dataset in `data/final_consensus_responses`.

- **Total Questions**: 1,200+ (Years 2007-2025)
- **Schema**: 200+ fields across 5 hierarchical tiers
  - **Tier 0**: Metadata & Classification
  - **Tier 1**: Core Solution & Formulas
  - **Tier 2**: Pedagogical Scaffolding (Common Mistakes, Mnemonics)
  - **Tier 3**: Advanced Learning (Real-world Applications)
  - **Tier 4**: Quality Metrics & Logs
- **Modality**: Text and Image (42% of questions include diagrams)

---

## 🏗️ 14-Stage Pipeline Architecture

The implementation in `pipeline/scripts/` strictly follows the 14-stage architecture defined in the research paper.

| Phase | Stage | Script Implementation | Description |
| :--- | :--- | :--- | :--- |
| **I. Ingestion** | **S0** | `init_00_initialization.py` | Bootstraps Qdrant, Redis, and API connections. |
| | **S1** | `init_01_question_loader.py` | Loads queries, extracts metadata and attachments. |
| | **S2** | `init_02_question_classifier.py` | Classifies difficulty (1-10) and content type. |
| | **S3** | `init_03_cache_manager.py` | Checks Redis (97% similarity threshold) for hits. |
| **II. Retrieval** | **S4a** | `init_04_retrieval_dense.py` | BGE-M3 embedding retrieval (Top-10). |
| | **S4b** | `init_05_retrieval_sparse.py` | BM25 keyword retrieval (Top-10). |
| | **S4c** | `init_06_retrieval_merger.py` | Reciprocal Rank Fusion (k=60) -> Top 6 chunks. |
| | **S5** | `init_07_image_consensus.py` | *Multimodal Consensus* for diagram analysis. |
| **III. Generation** | **S6** | `init_08_model_orchestrator.py` | Parallel execution of routed model tier. |
| | **S7** | `init_09_voting_engine.py` | Field-level weighted voting (80% threshold). |
| | **S8** | `init_10_debate_orchestrator.py` | Multi-round adversarial debate resolution. |
| | **S9** | `init_11_synthesis_engine.py` | Deterministic merge of best-of-breed fields. |
| **IV. Output** | **S10** | `init_12_output_manager.py` | *Validation*: Schema & content quality checks. |
| | **S11** | `init_12_output_manager.py` | *Export*: JSON persistence and S3 upload. |
| | **Ops** | `init_13`, `init_14` | Checkpointing and Health Monitoring. |

---

## 🚀 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Anand0008/cost-aware-structured-generation.git
    cd cost-aware-structured-generation
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**
    Copy `.env.example` to `.env` and configure your API keys:
    ```bash
    GOOGLE_API_KEY=...       # For Gemini 2.5 Pro / 2.0 Flash
    AWS_ACCESS_KEY_ID=...    # For Claude Sonnet 4.5 (Bedrock)
    DEEPSEEK_API_KEY=...     # For DeepSeek R1
    OPENAI_API_KEY=...       # For GPT-5.1
    QDRANT_URL=...           # Vector DB Endpoint
    REDIS_URL=...            # Cache Endpoint
    ```

4.  **Initialize System**
    ```bash
    python pipeline/scripts/init_00_setup_production.py
    ```

---

## ⚙️ Configuration & Domain Adaptation

The architecture is **domain-agnostic**. You can adapt it to Legal, Medical, or Financial domains by modifying the configuration files in `config/`:

-   **`config/weights_config.yaml`**: The heart of the **Adaptive Weighting Framework**. Define strategies that map domain priorities to model strengths.
    ```yaml
    # Example: Aerospace Math Strategy
    MATH_WEIGHTED:
      deepseek_r1: 0.40       # Strength: Derivations
      claude_sonnet_4.5: 0.20 # Strength: Pedagogy
      gemini_2.5_pro: 0.20
      gpt_5_1: 0.20
    ```
-   **`config/models_config.yaml`**: Define the specific model identifiers and API parameters.
-   **`config/schema_definition.json`**: (Located in `pipeline/schema`) Define the Tier 0-4 structure for your target domain output.

---

## 💻 Usage

**Run the Full Pipeline (All Questions)**
```bash
python pipeline/scripts/pipeline_runner.py --all
```

**Process a Specific Year**
```bash
python pipeline/scripts/pipeline_runner.py --year 2024
```

**Process a Single Question (Debug Mode)**
```bash
python pipeline/scripts/pipeline_runner.py --question-id GATE_AE_2024_Q15
```

---

## 💰 Performance & Cost Analysis

Validated on the Aerospace Engineering domain (N=1,200), the system yields:

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Precision** | **93.5%** | Human-verified (N=325, 2021-2025) |
| **Hallucination Rate** | **0.31%** | 1 error in 325 samples (vs 2-5% industry std) |
| **Content Richness** | **+84%** | Improvement in mnemonics vs single-model baseline |
| **Avg Cost** | **~$0.22** | Per Question (50.4% savings vs Self-Consistency) |
| **Debate Rate** | **17.0%** | Only hard queries trigger debate (Round 1: 11.9%, R2: 5.1%) |

**Cost Breakdown:**
- **Model Invocations**: 87.7%
- **Debate**: 3.5%
- **Classification**: 0.5%
- **Other**: 8.3%

---

## 📜 Citation

If you use this codebase or dataset in your research, please cite:

**Dataset (Zenodo):**
```bibtex
@dataset{wankhade_2026_18270482,
  author       = {Wankhade, Anand},
  title        = {Cost-Aware Structured Generation: High-Fidelity Synthesis via Hybrid RAG and Adaptive Conditional Compute},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.0},
  url          = {https://zenodo.org/records/18270482}
}
```

**Research Paper:**
```bibtex
@misc{wankhade2026costaware,
  title={Cost-Aware Structured Generation: High-Fidelity Synthesis via Hybrid RAG and Adaptive Conditional Compute},
  author={Wankhade, Anand},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Anand0008/cost-aware-structured-generation}}
}
```
