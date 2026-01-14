# Cost-Aware Structured Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Fidelity Synthesis via Hybrid RAG and Adaptive Conditional Compute**

A 14-stage pipeline for generating high-quality structured outputs from LLMs with 93.5% precision, 0.31% hallucination rate, and 50% cost reduction vs. baseline ensembles.

---

## 🎯 Key Features

- **Hybrid RAG** with Reciprocal Rank Fusion (dense + sparse retrieval)
- **Conditional Compute Routing** based on query difficulty (3 tiers)
- **Adaptive Model Weighting** per domain-specific priorities
- **Multi-Round Debate** for high-stakes conflict resolution
- **Schema-Enforced Outputs** with 200+ field validation

## 📊 Results

| Metric | Value |
|--------|-------|
| **Precision** | 93.5% |
| **Hallucination Rate** | 0.31% (1 in 325) |
| **Content Richness** | +84% vs. single model |
| **Cost Reduction** | 50.4% vs. Self-Consistency |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/anandwankhade/cost-aware-structured-generation.git
cd cost-aware-structured-generation
pip install -e .
```

### Basic Usage

```python
from pipeline import PipelineRunner

# Initialize with your domain corpus
runner = PipelineRunner(config_path="config/pipeline_config.yaml")

# Process a query
result = runner.process_question(
    question="Explain the relationship between Mach number and compressibility effects.",
    year=2024
)

print(result.json())
```

### Configuration

Edit `config/weights_config.yaml` to customize model weighting:

```yaml
weight_strategies:
  MATH_WEIGHTED:
    deepseek_r1: 0.40      # Boost math specialist
    claude_sonnet_4.5: 0.30
    gemini_2.5_pro: 0.30
    
  CONCEPTUAL:
    claude_sonnet_4.5: 0.40  # Boost pedagogical clarity
    deepseek_r1: 0.30
    gemini_2.5_pro: 0.30
```

---

## 📁 Project Structure

```
cost-aware-structured-generation/
├── pipeline/
│   ├── scripts/
│   │   ├── init_00_setup_production.py      # Stage 0: Initialization
│   │   ├── init_02_question_classifier.py   # Stage 2: Classification
│   │   ├── init_04_rag_retrieval.py         # Stage 4: Hybrid RAG
│   │   ├── init_06_model_orchestrator.py    # Stage 6: Model Invocation
│   │   ├── init_07_voting_engine.py         # Stage 7: Consensus Voting
│   │   ├── init_08_debate_orchestrator.py   # Stage 8: Debate Resolution
│   │   └── init_09_synthesis_engine.py      # Stage 9: Output Synthesis
│   └── utils/
├── config/
│   ├── models_config.yaml       # Model API configurations
│   ├── weights_config.yaml      # Adaptive weighting strategies
│   ├── thresholds_config.yaml   # Consensus thresholds
│   └── prompts_config.yaml      # Prompt templates
├── data/
│   └── corpus/                  # Domain corpus (chunked)
├── examples/
│   └── demo_pipeline.py         # Quick start demo
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔧 Domain Adaptation

The pipeline is **domain-agnostic**. To adapt to a new domain:

### 1. Replace Corpus
```bash
# Add your documents to data/corpus/
python scripts/build_index.py --input data/corpus/ --output data/index/
```

### 2. Configure Weights
Edit `config/weights_config.yaml`:
```yaml
# Example: Legal Contract Analysis
weight_strategies:
  REGULATORY_PRECISION:
    gpt_5_1: 0.40        # Citation accuracy
    claude_sonnet_4.5: 0.35
    gemini_2.5_pro: 0.25
```

### 3. Define Schema
Create your output schema in `config/schema_definition.json`:
```json
{
  "contract_parties": {"type": "array"},
  "obligations": {"type": "array"},
  "termination_conditions": {"type": "object"}
}
```

---

## 📈 Evaluation

### Human Evaluation (N=325)
```bash
python evaluation/run_human_eval.py --samples data/eval/human_eval_325.json
```

### Ablation Study (N=200)
```bash
python evaluation/run_ablation.py --config config/ablation_config.yaml
```

### LLM-as-Judge Pairwise (N=1,270)
```bash
python evaluation/run_llm_judge.py --comparisons 1270
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{wankhade2025costaware,
  title={Cost-Aware Structured Generation: High-Fidelity Synthesis via Hybrid RAG and Adaptive Conditional Compute},
  author={Wankhade, Anand},
  year={2025},
  url={https://github.com/anandwankhade/cost-aware-structured-generation}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

---

## 📧 Contact

**Anand Wankhade**  
📧 anandwankhade0008@gmail.com

---

## Acknowledgments

- Hybrid RAG methodology adapted from RetHyb-RRF [Mala et al., 2025]
- RRF fusion based on [Cormack et al., 2009]
- Multi-agent debate inspired by [Du et al., 2023]
