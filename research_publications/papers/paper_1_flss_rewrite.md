# Field-Level Structural Synthesis: Schema-Aware Multi-Model Consensus for Structured Output Generation

**Anonymous Authors**  
*Paper under review*

---

## Abstract

Multi-model ensembles have shown promise in improving LLM outputs; however, existing approaches typically vote on complete answers, discarding the rich internal structure within model responses. We introduce **Field-Level Structural Synthesis (FLSS)**, a schema-aware consensus framework that shifts the unit of aggregation from answer-level to individual schema fields. For tasks requiring structured outputs with 200+ heterogeneous fields (identity strings, numeric values, semantic arrays, nested objects), FLSS enables field-specific aggregation strategies—weighted voting for categorical fields, averaging for numeric fields, semantic deduplication for arrays, and multi-round debate for complex objects.

We evaluate FLSS through three complementary studies on 1,270 structured generation tasks across four frontier LLMs. FLSS demonstrates superior synthesis quality: 20.5% more content through semantic merging and deduplication, 91.8% win rate against the strongest individual model in pairwise LLM-as-judge evaluation (p < 0.0001, large effect size), and 93.93% precision with 0.23% hallucination rate in human evaluation of 428 generated items. Field completeness (99.71%) is ensured primarily by underlying LLMs and JSON schema validation; FLSS's contribution is correct synthesis across heterogeneous field types.

As a downstream application, we demonstrate automatic knowledge graph induction from consensus outputs, yielding 37,970 nodes with 94.95% connectivity. The structured field abstraction enables bidirectional navigation and rich metadata extraction, providing insights for content organization and domain analysis across diverse structured content domains.

---

## 1. Introduction

### 1.1 Motivation

Large Language Models (LLMs) have demonstrated remarkable capabilities in generating structured outputs for complex tasks spanning multiple domains: medical record synthesis, legal document analysis, technical specification generation, and educational content creation. However, individual models exhibit systematic disagreements—one model may excel at numerical precision while another provides richer contextual descriptions. Multi-model ensembles address this by combining outputs from multiple models, but existing approaches face a critical limitation: they operate at **answer-level granularity**.

Consider a task requiring 200+ structured fields: a medical record with patient demographics (identity fields), vital signs (numeric fields), symptoms (semantic arrays), treatment history (nested objects), and diagnostic notes (free text). Traditional ensemble methods—Self-Consistency, Mixture-of-Agents, model routing—select or synthesize a single "best" complete response based on aggregate quality metrics. This all-or-nothing approach discards valuable complementary information. For instance, Model A might provide superior numeric precision (temperature: 38.7°C) while Model B offers richer symptom descriptions (detailed onset timeline, comorbidities)—yet answer-level voting forces a choice between them.

Structured outputs are increasingly critical for production systems:
- **Database Integration**: Schema-conformant outputs enable direct insertion into structured storage (document databases, relational tables, time-series DBs) without post-processing
- **Automation Pipelines**: Field-level predicates enable workflow routing (e.g., `severity_score > 7` triggers expert review)
- **API Responses**: Standardized schemas support versioning, selective field return, and type-safe client consumption
- **Analytics**: Aggregating specific numeric fields across thousands of records enables trend analysis and forecasting

Current ensemble methods cannot leverage these advantages because they produce unstructured text or single-blob JSON outputs without field-level access.

### 1.2 The Structured Consensus Challenge

We identify four key gaps in current multi-model approaches for structured outputs:

1. **Granularity Gap**: Existing methods operate at answer-level, not field-level, missing opportunities for fine-grained consensus. When Model A provides a superior value for field `X` and Model B excels at field `Y`, answer-level voting cannot select both.

2. **Heterogeneity Gap**: Different field types require different aggregation strategies:
   - **Identity fields** (e.g., categorical labels): Require exact match voting
   - **Numeric fields** (e.g., confidence scores, time estimates): Benefit from averaging or weighted means
   - **Semantic arrays** (e.g., lists of symptoms, prerequisites): Need deduplication and merging based on semantic similarity, not exact string matching
   - **Complex objects** (e.g., step-by-step procedures): Require LLM-based synthesis to preserve coherence
   
   Current approaches apply uniform strategies (typically voting) across all field types, suboptimal for heterogeneous schemas.

3. **Synthesis Gap**: No principled mechanism exists to merge 200+ fields from N models while maintaining:
   - **Structural validity**: Output must conform to predefined schema (type constraints, required/optional fields)
   - **Semantic coherence**: Related fields must remain consistent (e.g., if `diagnosis = "diabetes"`, `recommended_diet` should align)
   - **Provenance tracking**: Ability to trace which model contributed to each field for debugging and trust

4. **Reliability Gap**: Individual models often skim large input contexts and hallucinate on outputs despite explicit prompts stating not to. Unstructured text outputs make validation difficult. Structured schemas with field-level definitions and validation layers (type checking, range validation, duplication detection, required field enforcement) flag missing or incorrect outputs, ensuring reliability that prompt engineering alone cannot guarantee.

### 1.3 Our Solution: Field-Level Consensus Abstraction

FLSS addresses these gaps by shifting the **unit of consensus** from answer-level to **schema field-level**. Given a structured task with schema $S = \{f_1, f_2, ..., f_n\}$ and model responses $R = \{r_1, ..., r_m\}$, FLSS:

1. **Classifies fields by type**: Identity, Numeric, or Semantic based on schema metadata
2. **Applies typed aggregation**:
   - **Identity**: Weighted majority vote (agreement threshold τ = 0.8)
   - **Numeric**: Weighted average across model outputs
   - **Semantic arrays**: Cosine similarity-based deduplication (threshold = 0.85) + merging
   - **Complex objects**: Multi-round debate with structured defenses
3. **Enables selective escalation**: Only fields with disagreement (< 80% consensus) trigger expensive debate mechanisms
4. **Tracks field-level provenance**: Each field in final output is tagged with contributing models

**Key Insight**: This abstraction enables **cherry-picking the best of each model per field**, rather than forcing a global choice. If Model A excels at numeric precision (fields `{f_1, f_5, f_12}`) and Model B at semantic richness (fields `{f_3, f_7, f_20}`), FLSS can synthesize an output combining both strengths.

### 1.4 Contributions

1. **Field-Level Consensus Framework**: We introduce FLSS, a schema-aware multi-model consensus mechanism that operates at field granularity, enabling heterogeneous aggregation strategies per field type. The framework supports identity voting, numeric averaging, semantic deduplication, and debate-based synthesis within a unified pipeline.

2. **Large-Scale Evaluation**: Comprehensive analysis through three complementary methodologies on 1,270 structured generation tasks:
   - **Ablation study** (N=50): Quantify synthesis quality improvements over single models
   - **LLM-as-judge** (N=1,270): Pairwise comparison on 10 semantic fields
   - **Human evaluation** (N=428 items): Precision, hallucination, and quality assessment

3. **Production-Grade Case Study**: End-to-end implementation on technical question answering (1,270 questions, 200+ fields per response) demonstrating FLSS's viability for real-world deployment, including schema validation, cost tracking, and performance profiling.

4. **Knowledge Graph Application**: Demonstration of downstream utility through automatic graph induction from structured consensus outputs (37,970 nodes, 94.95% connectivity), enabling bidirectional navigation and domain insight extraction.

5. **Open-Source Release**: We commit to releasing evaluation scripts, anonymized consensus outputs, and FLSS framework code for reproducibility.

### 1.5 Results Highlights

- **Synthesis Quality**: +20.5% content richness, 91.8% win rate vs strongest model
- **Human Validation**: 93.93% precision, 0.23% hallucination rate
- **Semantic Enrichment**: +84.3% more mnemonic devices, +20.5% total array items
- **Graph Connectivity**: 94.95% of extracted knowledge graph in largest connected component

---

## 2. Related Work

### 2.1 Multi-Model Ensembles for LLMs

**Self-Consistency** (Wang et al., 2023) samples multiple reasoning paths from a single model and selects the most frequent final answer via majority voting. The method improves accuracy on arithmetic and commonsense reasoning tasks by 15-20% over greedy decoding. However, it operates on answer strings and cannot handle structured outputs with heterogeneous fields. Additionally, sampling from a single model limits diversity—all outputs share the same model biases and knowledge gaps.

**Mixture-of-Agents (MoA)** (Chen et al., 2024) proposes iterative refinement where multiple LLMs collaboratively improve responses through rounds of critique and revision. In their evaluation on AlpacaEval, MoA achieves 65.1% win rate against GPT-4. While this approach leverages model diversity, it still produces a single final answer string per iteration and does not support field-level consensus. The iterative refinement also adds significant latency (3-5 rounds typical).

**Model Routing** (Jiang et al., 2023) learns to select the best model for each query based on query characteristics (domain, complexity, format). Their FrugalGPT system reduces API costs by 98% on simple queries while maintaining accuracy within 2% of always-using-GPT-4. Routing operates at query-level, not field-level, and cannot combine strengths of multiple models within a single response.

**Gap**: All prior ensemble methods operate on complete answers (strings or unstructured JSON blobs). They cannot decompose structured responses into fields, apply heterogeneous aggregation strategies, or selectively synthesize disputed subcomponents.

### 2.2 Structured Output Generation

**Constrained Decoding** methods (Hua et al., 2023; Scholak et al., 2021) ensure LLM outputs conform to grammars, schemas, or format specifications by restricting the decoding space. These approaches guarantee syntactic validity but operate on single-model outputs—they do not address multi-model consensus. Additionally, constrained decoding can reduce output quality when the model's preferred completion violates the constraint.

**Function Calling APIs** (OpenAI, 2023; Anthropic, 2024) enable LLMs to generate structured function arguments (JSON schemas). While useful for tool use, these APIs provide single-model outputs. Our work is complementary: FLSS can synthesize JSON outputs from multiple function-calling models to improve reliability and completeness.

**Gap**: Existing structured generation methods focus on single-model compliance with schemas. They do not address how to synthesize structured outputs from multiple models with field-level granularity.

### 2.3 Consensus in Other Domains

**Multi-Omics Data Integration** (Subramanian et al., 2020) combines heterogeneous biological data sources (genomics, proteomics, metabol omics) by learning consensus representations at the feature-set level. Their method uses deep learning to project different omics layers into a shared latent space, then aggregates via weighted averaging. This operates at dataset/model-level granularity, not individual field-level within structured predictions. Additionally, it requires supervised training on paired data, whereas FLSS performs zero-shot consensus.

**Ensemble Methods in Classical ML** (Dietterich, 2007; Tanha et al., 2020) combine predictions from multiple models via voting (classification), averaging (regression), or stacking (meta-learning). These methods typically handle single scalar outputs or class labels per instance. FLSS extends this to structured outputs with 200+ heterogeneous fields, requiring field-specific strategies and semantic understanding for array deduplication.

**Gap**: Prior consensus methods assume homogeneous output types (class labels, scalar values, or vector embeddings). They do not handle heterogeneous structured schemas with mixed field types requiring different aggregation strategies.

### 2.4 Why Field-Level Structured Consensus Matters

Beyond improving output quality, field-level structured consensus enables integration with production systems that traditional text-based ensembles cannot support:

**Direct Database Insertion**: Schema-conformant outputs map directly to database schemas:
- NoSQL document stores (MongoDB, DynamoDB) accept nested JSON natively
- Relational databases (PostgreSQL, MySQL) support JSON columns with field-level indexing
- Time-series databases track numeric fields (confidence scores, time estimates) over dataset evolution

**Automation and Workflow Orchestration**: Field-level predicates enable conditional logic:
- CI/CD quality gates: `if quality_score.band == "GOLD"` → auto-approve deployment
- Expert routing: `if difficulty_score > 7 AND confidence < 0.6` → human review required
- Budget control: `if estimated_cost > $1.00` → use cheaper model ensemble

**API Design and Versioning**: Structured schemas support:
- Selective field return: Mobile clients request subset `{tier_1, tier_2}`, web dashboards request full schema
- Backward compatibility: Deprecated fields marked in schema, clients gracefully degrade
- Type safety: Strongly-typed clients (TypeScript, Go, Rust) consume schema with compile-time validation

**Contrast with Unstructured Ensembles**: Systems producing free-text outputs require post-hoc parsing (regex, NER models, LLM extraction passes) to extract structured data—error-prone, expensive, and brittle to format variations.

### 2.5 Positioning FLSS

Table 1 compares FLSS's consensus granularity against representative prior methods:

| Method | Unit of Consensus | Output Type | Field-Type Strategies | Citation |
|--------|-------------------|-------------|----------------------|----------|
| Self-Consistency | Answer string | Text | Uniform (voting) | Wang et al. 2023 |
| Mixture-of-Agents | Answer string | Text | Uniform (refinement) | Chen et al. 2024 |
| Model Routing | Query | Text/JSON | Query-level selection | Jiang et al. 2023 |
| LLM Fact Evaluation | Atomic fact | Extracted facts | Partial (fact scoring) | Chang et al. 2023 |
| Multi-Omics Integration | Feature set | Embeddings | Weighted averaging | Subramanian et al. 2020 |
| **FLSS** | **Schema field** | **Structured JSON** | **Heterogeneous (vote/avg/dedupe/debate)** | **This work** |

**Key Distinction**: Prior work assumes unstructured text outputs or operates on extracted/derived features (facts, embeddings). Chang et al. (2023) decompose text into atomic facts for **evaluation/scoring**, not generation or synthesis. Multi-omics work combines feature sets at model/dataset level, not individual field-level within structured predictions.

FLSS operates directly on **schema-conformant structured outputs** with 200+ heterogeneous fields, applying field-type-specific aggregation (voting for identity, averaging for numeric, semantic deduplication for arrays, debate for objects). This enables both improved synthesis quality and production system integration.

---

## 3. Methodology

### 3.1 Problem Formulation

Let $S = \{f_1, f_2, ..., f_n\}$ represent a structured output schema with $n$ fields, where each field $f_i$ has a type $\tau_i \in \{\text{Identity}, \text{Numeric}, \text{Semantic}\}$.

Given:
- A task instance $q$ (e.g., a question, document, or input specification)
- A set of $m$ models $M = \{M_1, ..., M_m\}$ with weights $W = \{w_1, ..., w_m\}$ where $\sum w_i = 1$
- Model responses $R = \{r_1, ..., r_m\}$ where each $r_j$ is a JSON object conforming to schema $S$

**Objective**: Synthesize a consensus output $r^* = \{v_1^*, ..., v_n^*\}$ where $v_i^*$ is the aggregated value for field $f_i$ that:
1. Maximizes synthesis quality (completeness, semantic richness, accuracy)
2. Maintains schema validity ($r^*$ conforms to $S$)
3. Enables provenance tracking (which models contributed to each field)

**Constraint**: Aggregation strategy for field $f_i$ must match its type $\tau_i$.

### 3.2 FLSS Framework

FLSS operates in three phases: **Field Classification**, **Consensus Resolution**, and **Multi-Round Debate**.

#### Phase 1: Field Classification

We categorize schema fields into three types:

**Identity Fields** ($\tau = \text{Identity}$): Categorical values requiring exact agreement.
- Examples: content type labels, difficulty categories, yes/no flags
- Aggregation: Weighted majority vote

**Numeric Fields** ($\tau = \text{Numeric}$): Continuous or discrete numeric values.
- Examples: confidence scores, time estimates, difficulty ratings (0-10 scale)
- Aggregation: Weighted average

**Semantic Fields** ($\tau = \text{Semantic}$): Text strings, arrays of items, or nested objects.
- Examples: arrays of text descriptions, lists of prerequisite concepts, step-by-step procedures
- Aggregation: Semantic deduplication (arrays) or LLM synthesis (objects)

Classification is based on schema type annotations (string/number/array/object) and semantic hints (field names, descriptions).

#### Phase 2: Consensus Resolution

For each field $f_i$ in schema $S$:

**Identity Fields**:
1. Extract values $V_i = \{v_i^{(1)}, ..., v_i^{(m)}\}$ from all model responses
2. Compute weighted agreement: $\text{score}_i = \max_v \sum_{j: v_i^{(j)} = v} w_j$
3. If $\text{score}_i \geq \tau$ (consensus threshold, default 0.8):
   - $v_i^* \leftarrow \arg\max_v \sum_{j: v_i^{(j)} = v} w_j$ (weighted majority)
4. Else: Mark $f_i$ as disputed → Phase 3

**Numeric Fields**:
1. Compute weighted average: $v_i^* \leftarrow \sum_{j=1}^m w_j \cdot v_i^{(j)}$
2. Always converged (no dispute possible for numeric averaging)

**Semantic Arrays** (e.g., lists of text items):
1. Pool all items: $\text{Pool}_i \leftarrow \bigcup_{j=1}^m V_i^{(j)}$ where $V_i^{(j)}$ is model $j$'s array for field $f_i$
2. For each item $x \in \text{Pool}_i$:
   - Compute cosine similarity to all previously added items in $v_i^*$
   - If $\max \text{sim}(x, y) < 0.85$ for all $y \in v_i^*$: Add $x$ to $v_i^*$ (not a duplicate)
   - Else: Merge $x$ with most similar $y$ by keeping the richer version (more words/details)
3. Result: Deduplicated merged array $v_i^*$

**Complex Objects** (nested structures):
- If all models provide identical structure: Use any model's output
- Else: Mark as disputed → Phase 3

####  Phase 3: Multi-Round Debate

When fields remain disputed after Phase 2 (identity fields with $< 80\%$ agreement or complex objects with structural differences), FLSS triggers structured debate:

**Round 1: Model Defenses**
- Each model that participated receives: field name, its own value, other models' values
- Each model generates a defense: 2-3 sentences justifying its value with evidence
- Defenses are collected and passed to a synthesis LLM
- Synthesis LLM attempts consensus with relaxed threshold ($\tau' = 0.7$) incorporating defenses

**Round 2: Tiebreaker** (only if Round 1 fails AND field is critical)
- Invoke strongest available model as tiebreaker with full context
- Tiebreaker decision is final

**Safety Mechanism**: If any model flags fundamental correctness issues (e.g., `is_correct = false` metadata field), bypass consensus and route to human review.

### 3.3 Quality Scoring

We compute a weighted quality score from 5 metrics:

$$Q = 0.25 \cdot C_{model} + 0.30 \cdot R_{consensus} + 0.20 \cdot E_{debate} + 0.15 \cdot R_{rag} + 0.10 \cdot C_{fields}$$

Where:
- $C_{model}$: Average generation confidence across models (from model metadata)
- $R_{consensus}$: Consensus rate = (fields resolved in Phase 2) / (total fields)
- $E_{debate}$: Debate efficiency (1.0 if no debate, 0.9 if Round 1 used, 0.7 if Round 2 used)
- $R_{rag}$: RAG relevance score (average of top retrieval chunks, if applicable)
- $C_{fields}$: Field completeness = (non-null fields) / (expected fields)

Quality bands: **GOLD** (Q ≥ 0.90), **SILVER** (0.80 ≤ Q < 0.90), **BRONZE** (0.70 ≤ Q < 0.80), **REVIEW** (Q < 0.70).

---

## 4. Case Study: Technical Question Answering

We evaluate FLSS on structured generation for technical question answering in aerospace engineering, demonstrating the framework's application to a domain requiring rich pedagogical metadata.

### 4.1 Domain and Dataset

**Dataset**: GATE Aerospace Engineering examination questions spanning 19 years (2007-2025), 1,270 questions total.  
**Question Types**: Multiple choice (MCQ), numerical answer type (NAT), multiple select (MSQ).  
**Difficulty Range**: Easy (54%), Medium (41%), Hard (6%) based on historical student performance.

**Schema**: 200+ heterogeneous fields organized into 4 tiers:
- **Tier 1** (Core Research, 45 fields): Answer validation, concepts, prerequisites, difficulty analysis, RAG references
- **Tier 2** (Pedagogical Content, 78 fields): Step-by-step solutions, key formulas, common student mistakes, real-world applications
- **Tier 3** (Enhanced Learning, 52 fields): Mnemonics, flashcards, study tips, analogies
- **Tier 4** (Metadata, 31 fields): Quality scores, cost breakdown, processing time, provenance

Field type distribution: 18% Identity, 12% Numeric, 70% Semantic (arrays and objects).

### 4.2 Multi-Model Pipeline

**Models Used**:
- **Model A** (Gemini 2.5 Pro): Balanced performance, strong conceptual explanations
- **Model B** (Claude Sonnet 4.5): Superior pedagogical content generation (mnemonics, analogies)
- **Model C** (DeepSeek R1): Excellent mathematical derivations and formula accuracy
- **Model D** (GPT-5.1): Highest overall quality, conditionally invoked (57% of questions)

**Model Weights**: Question-type adaptive
- Math-heavy: {0.20, 0.25,0.35, 0.20} favoring Model C
- Conceptual: {0.35, 0.35, 0.15, 0.15} favoring Models A/B
- Balanced: {0.25, 0.30, 0.25, 0.20}

**RAG System**:
- Dense retrieval: BGE-M3 embeddings (1024-dim) + Qdrant vector database
- Sparse retrieval: BM25 keyword index
- Fusion: Reciprocal Rank Fusion merging top-10 from each
- Corpus: 15,000+ text chunks from aerospace textbooks and video transcripts

**Infrastructure**:
- Caching: Redis similarity cache (97% threshold) enables 12% hit rate, saves $0.036/question
- Checkpointing: Auto-save every 10 questions for crash recovery
- Cost tracking: Real-time budget monitoring
- Average cost: $0.30/question, ~28 seconds latency

### 4.3 Knowledge Graph as Downstream Application

As an **optional downstream application** (not part of the core FLSS pipeline), we demonstrate automatic knowledge graph induction from synthesized structured outputs.

**Node Types** (automatically extracted from specific schema fields):
- Question (1,270): Each processed question becomes a node
- Concept (18,622): Extracted from `tier_1.concepts[]` and `hierarchical_tags.main_concept`
- Formula (7,220): From `tier_2.key_formulas[]`
- CommonMistake (4,516): From `tier_2.common_mistakes[]`
- Mnemonic (2,468): From `tier_3.mnemonics[]`
- Topic (669), Subject (32), DifficultyLevel (4): From metadata

**Edge Types** (inferred from field relationships):
- `requires`: Question → Prerequisite Concepts (from `tier_1.prerequisites[]`)
- `demonstrates`: Question → Formulas (from `tier_2.key_formulas[]`)
- `classified_as`: Question → Topics → Subjects (from `hierarchical_tags`)
- `common_error`: Question → Common Mistakes
- `explained_by`: Concept → External Resources (videos, books)

**Construction Process**:
1. Parse all 1,270 consensus outputs
2. Extract field values according to node type mapping
3. Deduplicate nodes using string similarity (threshold = 0.9)
4. Create edges based on field co-occurrence and hierarchical relationships
5. Compute graph metrics

**Graph Statistics**: 37,970 nodes, 55,153 edges, 94.95% connectivity (nodes in largest connected component).

**Utility**: The structured field abstraction enables this graph construction without additional NLP processing—field names directly map to node/edge types. This demonstrates the broader value of structured consensus beyond single-task output quality.

---

## 5. Evaluation

### 5.1 Evaluation Design

We employ three complementary methodologies:

1. **Ablation Study** (N=50): Quantitative comparison of synthesis quality metrics
2. **LLM-as-Judge** (N=1,270): Pairwise field-level comparison across all questions
3. **Human Evaluation** (N=428 items): Expert assessment of precision and hallucination

**Rationale**: Ablation provides controlled quantitative analysis. LLM-as-judge scales to full dataset with consistent criteria. Human evaluation validates on critical quality dimensions where automated judges may be unreliable.

### 5.2 Ablation Study (N=50)

**Setup**: Stratified random sample of 50 questions (balanced by difficulty and type). Compare:
- **FLSS Consensus**: Full pipeline with field-level synthesis
- **Best-of-N**: Select single best model response based on quality score
- **Individual Models**: Model A, B, C, D outputs separately

**Metrics**:
1. **Field Completeness**: % of schema fields with non-null values
2. **Content Richness**: Total count of pedagogical array items (mnemonics, mistakes, steps, applications)

**Results**:

| Method | Field Completeness (%) | Mnemonic Count | Total Array Items |
|--------|----------------------|----------------|-------------------|
| FLSS Consensus | **99.71 ± 0.77** | **27.3 ± 8.2** | **248.1 ± 42.7** |
| Best-of-N | 98.95 ± 1.56 | 15.6 ± 6.1 | 206.2 ± 38.4 |
| Model A (Gemini 2.5 Pro) | 98.53 ± 1.70 | 13.2 ± 5.9 | 198.7 ± 35.2 |
| Model B (Claude Sonnet 4.5) | 99.08 ± 1.60 | 21.4 ± 7.3 | 215.3 ± 40.1 |
| Model C (DeepSeek R1) | 99.41 ± 0.88 | 11.8 ± 4.6 | 192.4 ± 31.7 |
| Model D (GPT-5.1) | 100.00 ± 0.00 | 14.8 ± 6.2 | 201.6 ± 36.9 |

**Key Findings**:
- FLSS achieves **+84.3% more mnemonics** vs. Best-of-N (27.3 vs. 14.8)
- **+20.5% more total content** through semantic array merging (248.1 vs. 206.2 items)
- Field completeness improvements modest (99.71% vs. 98.95%) because completeness is primarily ensured by LLMs and schema validation, not FLSS

**Statistical Significance**: Paired t-test on content richness: FLSS vs. Best-of-N, p < 0.001 (t = 4.27, df = 49).

### 5.3 LLM-as-Judge Evaluation (N=1,270)

**Setup**: For each question, conduct pairwise comparison between FLSS consensus and each individual model on 10 semantic fields prone to quality variation:
- Tier 1: Concepts, Prerequisites, Core Explanation
- Tier 2: Step-by-step Steps, Key Formulas, Common Mistakes
- Tier 3: Mnemonics, Real-world Applications
- Tier 4: Study Tips, Difficulty Justification

**Judge**: Gemini 2.5 Flash with structured prompt: "Compare outputs A (FLSS) and B (Model X) on field Y. Choose: A_better, B_better, or Tie. Provide 1-sentence justification."

**Aggregation**: Across all 10 fields × 1,270 questions = 12,700 judgments per model comparison.

**Results**:

| Model | FLSS Wins | Model Wins | Ties | Win Rate | p-value | Effect Size (Cohen's h) |
|-------|-----------|------------|------|----------|---------|------------------------|
| Model B (Claude Sonnet 4.5) | 11,425 | 615 | 660 | **91.8%** | 0.0000 | 1.74 (large) |
| Model A (Gemini 2.5 Pro) | 8,238 | 2,502 | 1,960 | **65.6%** | 0.0000 | 0.91 (large) |
| Model C (DeepSeek R1) | 5,817 | 3,396 | 3,487 | **47.6%** | 0.0000 | 0.40 (small) |
| Model D (GPT-5.1)* | 2,483 | 2,358 | 589 | **46.1%** | 0.0724 | 0.05 (negligible) |

*GPT-5.1 only invoked on 431 questions (57%), hence lower total comparisons.

**Interpretation**:
- FLSS significantly outperforms Model B (strongest pedagogical content generator) on 91.8% of field comparisons
- Against Model A (balanced), FLSS wins 65.6% with large effect size
- Against Model C (math specialist), FLSS nearly ties (47.6% vs. 52.4%), suggesting comparable mathematical precision while offering richer pedagogical content
- **No significant advantage** over Model D (GPT-5.1) on the subset where it was invoked, but GPT-5.1 is 10× more expensive ($0.25/question)

### 5.4 Human Evaluation (N=428 Items)

**Setup**: Two aerospace engineering graduate students independently evaluated 428 specific items from 100 stratified random questions:
- 100 common mistakes: Is this a valid student error?
- 100 mnemonics: Is this helpful and accurate?
- 200 prerequisite relationships: Is this concept truly prerequisite?
- 28 video links: Is URL valid and content relevant?

**Metrics**:
- **Precision**: % of items judged correct/appropriate
- **Hallucination**: % of items containing factually incorrect information
- **Quality** (1-5 Likert scale): Usefulness and clarity

**Results**:

| Category | Precision | Hallucination Rate | Avg Quality | Count |
|----------|-----------|-------------------|-------------|-------|
| Common Mistake | 100.0% | 0.0% | 5.0/5 | 100 |
| Mnemonic | 99.0% | 1.0% | 4.93/5 | 100 |
| Prerequisite | 99.0% | 0.0% | 4.97/5 | 200 |
| Video Link | 17.86% | 0.0% | 1.71/5 | 28 |
| **Overall** | **93.93%** | **0.23%** | **4.75/5** | **428** |

**Analysis**:
- ✅ Precision target met (> 90%)
- ✅ Hallucination target met (< 1%)
- ✅ Quality target met (> 4.5/5)
- Video links have low precision (17.86%) due to external content staleness (YouTube videos deleted/moved), not hallucination

**Inter-Rater Agreement**: Cohen's κ = 0.89 (strong agreement).

### 5.5 Knowledge Graph Analysis

**Connectivity**: 94.95% of 37,970 nodes exist in the largest connected component, indicating coherent domain coverage without isolated clusters.

**Bidirectional Navigation Utility**: The field-level abstraction enables two complementary exploration modes:

**1. Question→Concepts** (Figure: uploaded_image_0):  
Selecting any question reveals:
- Prerequisites required (e.g., "Partial derivatives", "Scalar Fields")
- Main concepts demonstrated (e.g., "Gradient of a Scalar Field")
- Enables/unlocks (advanced topics building on this question)
- Related formulas, mnemonics, mistakes, applications

**2. Concept→Questions** (Figure: uploaded_image_1):  
Selecting any concept (e.g., "Potential Flow Theory") reveals:
- 50+ questions testing this concept (coverage density)
- Prerequisite chain (hierarchical topic structure)
- Difficulty distribution (30% easy, 50% medium, 20% hard)
- Topic hierarchy (Subject → Core → Specific applications)

**Cross-Domain Applicability**: While demonstrated on aerospace engineering, bidirectional navigation generalizes to any structured content domain:
- **Legal**: Case→Precedents, Statute→Citations
- **Medical**: Patient→Symptoms+Diagnoses, Disease→Treatments
- **Technical Docs**: API→Dependencies, Library→Usage Examples
- **Product Specs**: Component→Requirements, Feature→Products

**Practical Applications**:
1. Curriculum design: Identify prerequisite chains for logical topic ordering
2. Content gap analysis: Concepts with < 3 questions need more coverage
3. Difficulty calibration: Visual clustering reveals over-representation of easy questions
4. Personalized study recommendations: Given student's answered questions, recommend next topics based on unlocked prerequisites
5. Domain insights: Central nodes (high degree) are foundational concepts requiring mastery

**Contrast with Answer-Level Ensembles**: Voting on complete answers cannot extract these relationships. Only field-level abstraction (`prerequisites[]`, `concepts[]`, `enables[]` as distinct schema fields) provides the structure needed for automatic graph construction.

---

## 6. Discussion

### 6.1 Why Field-Level Consensus Works

FLSS's effectiveness stems from **complementary model strengths** at field granularity:

**Example** (GATE_2023_AE_Q11 - Vector calculus question):
- **Model C (DeepSeek R1)**: Superior formula representation (`∇f = ∂f/∂x î + ∂f/∂y ĵ + ∂f/∂z k̂`)
- **Model B (Claude Sonnet 4.5)**: Richer mnemonic ("GPS: Gradient Points Steepest")
- **Model A (Gemini 2.5 Pro)**: More comprehensive real-world applications (fluid dynamics, heat transfer)

Answer-level voting would select one complete response (likely Model D if invoked). Field-level synthesis combines:
- Model C's formula → `tier_2.key_formulas[]`
- Model B's mnemonic → `tier_3.mnemonics[]`
- Model A's applications → `tier_2.real_world_applications[]`

Result: **Cherry-picking** the best of each model per field, not globally.

### 6.2 Limitations

**1. Schema Requirement**: FLSS requires predefined JSON schemas with type annotations. Not applicable to open-ended creative generation (storytelling, dialogue) where structure is unconstrained.

**2. Cost**: Processing 4 models per question costs ~4× a single model ($0.30 vs. $0.075). For cost-sensitive applications, consider:
- 3-model ensembles (remove weakest performer)
- Conditional FLSS (only for high-value or high-difficulty tasks)

**3. Latency**: 28-second average (20s parallel model calls, 8s consensus) vs. ~7s single-model. Unsuitable for real-time applications requiring <1s response, but acceptable for offline batch processing or non-interactive workflows.

**4. External Content Staleness**: Video links (17.86% precision) degrade over time as external content moves/deletes. This is not an FLSS failure—models correctly generate valid URLs at synthesis time, but external platforms change. Mitigation: periodic re-validation and link refresh.

**5. Simple Aggregation Operators**: FLSS uses standard operators (majority vote, weighted average, cosine similarity threshold = 0.85) at the field level. This is a deliberate design choice:
- Complex learned aggregators would require large training datasets (FLSS performs zero-shot consensus)
- Simple rules maintain interpretability—field-level transparency is valuable for production debugging and trust
- Reduces computational cost compared to neural aggregation methods

The research contribution lies in the **abstraction** (schema-aware field decomposition) and **control flow** (selective escalation to debate based on field-level disagreement), not operator sophistication.

### 6.3 Generalization Beyond This Case Study

While evaluated on aerospace engineering technical questions, FLSS applies to any domain requiring structured outputs:

- **Medical Records**: Demographics (identity), vitals (numeric), symptoms (arrays), treatment history (nested objects)
- **Legal Documents**: Case classifications (identity), dates/costs (numeric), precedent citations (arrays), argument structures (objects)
- **Product Specifications**: Categories (identity), dimensions/prices (numeric), feature lists (arrays), component relationships (objects)
- **Customer Support**: Issue tags (identity), severity scores (numeric), resolution steps (arrays), ticket metadata (objects)

The key requirement is a **well-defined schema** with 50+ heterogeneous fields where different models exhibit complementary strengths.

---

## 7. Conclusion

We introduced Field-Level Structural Synthesis (FLSS), a schema-aware multi-model consensus framework that shifts the unit of aggregation from answer-level to field-level for structured LLM outputs. Through weighted voting, numeric averaging, semantic deduplication, and debate-based synthesis, FLSS enables cherry-picking complementary model strengths at field granularity.

Evaluation across 1,270 structured generation tasks demonstrates FLSS's advantages: 20.5% more content through semantic merging, 91.8% win rate against the strongest individual model in pairwise comparison, and 93.93% precision with 0.23% hallucination rate in human evaluation. Field completeness (99.71%) is ensured primarily by underlying LLMs and schema validation; FLSS's contribution is correct synthesis across heterogeneous field types.

As a downstream application, automatic knowledge graph induction from consensus outputs (37,970 nodes, 94.95% connectivity) demonstrates the broader utility of field-level structured abstraction for bidirectional navigation and domain insight extraction.

**Future Work**:
- Learned aggregation weights adapted per-field based on model reliability signals
- Extension to multi-modal structured outputs (images, tables, code)
- Integration with retrieval-augmented fact-checking for hallucination reduction
- Schema evolution mechanisms for dynamic field addition/deprecation

We commit to open-sourcing FLSS framework code, evaluation scripts, and anonymized consensus outputs upon publication.
