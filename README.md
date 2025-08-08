# Brain-Inspired Cognitive Architecture Analysis of Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper: IEEE Big Data 2025](https://img.shields.io/badge/Paper-IEEE%20Big%20Data%202025-green.svg)](https://doi.org/your-paper-doi)

## 📖 Overview

This repository contains the implementation and analysis framework for **"Cognitive Architecture Analysis of Large Language Models: A Brain-Inspired Computational Framework for Understanding Neural Network Behavior"**, presented at IEEE International Conference on Big Data 2025.

We introduce a novel framework that maps Large Language Model (LLM) computations to brain-inspired cognitive architectures, evaluating 21 state-of-the-art language models across 21 cognitive tasks and 10 distinct brain regions.

### 🎯 Key Findings

- **Top Performer**: BLOOM-560M with 1558.42 total activity score
- **Central Hub**: Prefrontal cortex analog (hub score: 0.859)
- **Most Challenging Task**: Sentiment analysis (difficulty score: 7.49)
- **Total Data Points**: 8,820 model-task-region measurements

## 🚀 Features

- **Brain Region Mapping**: Maps LLM activations to 10 cognitive brain regions
- **Comprehensive Evaluation**: 21 models × 21 tasks × 10 regions analysis
- **Cognitive Metrics**: Novel metrics including cognitive balance, hub scores, and task difficulty
- **Visualization Tools**: Brain network connectivity maps and performance visualizations
- **Architectural Insights**: Identifies optimal model families for specific cognitive tasks

## 📊 Extended Analysis Features

### Model Families Tested (50+ Models)

- **BERT Family**: BERT, DistilBERT, RoBERTa, ALBERT, DeBERTa
- **GPT Family**: GPT-2 (all sizes), DialoGPT, GPT-Neo (125M to 2.7B)
- **BLOOM Family**: BLOOM-560M, 1.1B, 1.7B, 3B
- **OPT Family**: OPT-125M, 350M, 1.3B, 2.7B
- **Specialized Models**: CodeBERT, SciBERT, BioBERT, FinBERT, LegalBERT
- **Multilingual Models**: XLM-RoBERTa, mDeBERTa, MuRIL
- **Sentence Transformers**: all-MiniLM, all-mpnet

### Cognitive Tasks (20+ Datasets)

| Category | Tasks | Datasets |
|----------|-------|----------|
| Language Understanding | Sentiment, Classification, NLI | GLUE, AG News |
| Reasoning & Logic | Commonsense, Causal, Textual Entailment | WinoGrande, COPA, RTE |
| Memory & Context | Long Context, Episodic Memory | NarrativeQA, SQuAD |
| Mathematical | Arithmetic, Word Problems | MathQA, DROP |
| Creative | Story Completion, Dialogue | WritingPrompts, DailyDialog |
| Specialized | Emotion, Irony, Fact-Checking | Emotion, TweetEval, FEVER |

### Brain Region Mapping

The framework maps model activations to 10 brain regions:
- **Visual Cortex**: Early pattern recognition layers
- **Temporal Lobe**: Language and semantic processing
- **Parietal Lobe**: Attention and numerical processing
- **Prefrontal Cortex**: Executive function and reasoning
- **Hippocampus**: Memory and sequence learning
- **Broca's Area**: Language production
- **Wernicke's Area**: Language comprehension
- **Anterior Cingulate**: Error detection
- **Cerebellum**: Prediction and coordination
- **Amygdala**: Emotional processing

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or Apple Silicon Mac (MPS support)
- At least 16GB RAM (32GB recommended for larger models)
- 50GB+ free disk space for model downloads

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/BrightMindAI/brain-llm-analysis.git
cd neural-cognitive-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
# CUDA 11.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB+ |
| GPU VRAM | 8GB | 16GB+ |
| Storage | 50GB | 100GB+ |
| CPU | 4 cores | 8+ cores |

## 📁 Project Structure

```
neural-cognitive-analysis/
├── brain.py                                    # Main analysis script
├── comprehensive_brain_analysis_results.xlsx   # Original analysis data
├── comprehensive_brain_analysis_tables.xlsx    # Processed results tables
├── README.md                                   # Project documentation
├── requirements.txt                            # Python dependencies
├── extended_llm_brain_analysis/               # Output directory
│   ├── comprehensive_results.json             # Complete analysis results
│   ├── brain_specialization.html             # Interactive brain specialization chart
│   ├── cognitive_radar.html                  # Cognitive profile radar charts
│   ├── mega_brain_heatmap.html              # Comprehensive activity heatmap
│   ├── model_similarity.html                 # Model similarity visualization
│   └── [task]_brain_activity.json           # Individual task results
│       ├── sentiment_analysis_brain_activity.json
│       ├── causal_reasoning_brain_activity.json
│       ├── linguistic_acceptability_brain_activity.json
│       ├── emotion_recognition_brain_activity.json
│       ├── mathematical_reasoning_brain_activity.json
│       └── ... (20+ cognitive task results)
├── brain_analysis_summary.txt                # Analysis summary report
└── paper_analysis_BERT/                      # Additional analysis outputs
```

## 🔧 Usage

### Quick Start

```bash
# Clone the repository
git clone https://github.com/BrightMindAI/brain-llm-analysis.git
cd neural-cognitive-analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python brain.py
```

### Running the Extended Analysis

The `brain.py` script contains the `ExtendedLLMBrainAnalyzer` class that:
- Tests 50+ language models across multiple families (BERT, GPT, BLOOM, OPT, etc.)
- Evaluates models on 20+ cognitive tasks
- Maps activations to 10 distinct brain regions
- Generates interactive visualizations

```python
from brain import ExtendedLLMBrainAnalyzer

# Initialize analyzer
analyzer = ExtendedLLMBrainAnalyzer(device='auto', max_models_parallel=2)

# Run comprehensive analysis
results = analyzer.run_comprehensive_analysis(output_dir='extended_llm_brain_analysis')
```

### Analyzing Specific Tasks

```python
# Test a specific cognitive task
task_results = analyzer.test_cognitive_task(
    task_name='sentiment_analysis',
    dataset_name='glue/sst2',
    num_samples=30
)

# Extract brain activations for a specific model
activations = analyzer.extract_brain_activations(
    model_name='bert-base-uncased',
    text='Example text for analysis'
)
```

### Visualizing Results

After running the analysis, open the generated HTML files:
- `mega_brain_heatmap.html` - Overall brain activity patterns
- `cognitive_radar.html` - Task-specific cognitive profiles
- `brain_specialization.html` - Brain region specializations
- `model_similarity.html` - Model similarity based on brain patterns

## 📈 Results

### Model Performance

| Model | Family | Total Activity | Cognitive Balance |
|-------|--------|----------------|-------------------|
| bloom-560m | BLOOM | 1558.42 | 0.598 |
| gpt-neo-125M | GPT | 963.79 | 0.583 |
| DialoGPT-small | GPT | 686.11 | 0.542 |

### Task Difficulty Rankings

| Task | Difficulty Score | Complexity | Primary Region |
|------|------------------|------------|----------------|
| Sentiment Analysis | 7.49 | 4 | Amygdala |
| Causal Reasoning | 6.74 | 4 | Prefrontal |
| Linguistic Acceptability | 6.48 | 4 | Broca's |

## 🧠 Methodology

### Brain Region Mapping Function

```
φ: M × T → R^10
```

Where:
- M: Model space
- T: Task space  
- R: 10-dimensional brain region representation

### Cognitive Metrics

1. **Cognitive Balance Score (CBS)**:
   ```
   CBS(m) = 1 - Σ|p_r - 1/10|/2
   ```

2. **Hub Score**:
   ```
   Hub(r) = Σ corr(a_r, a_r')/9
   ```

3. **Task Difficulty**:
   ```
   Difficulty(t) = α·avg(a_t) + β·var(a_t) + γ·complexity(t)
   ```

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{moghadasi2025cognitive,
  title={Cognitive Architecture Analysis of Large Language Models: A Brain-Inspired Computational Framework for Understanding Neural Network Behavior},
  author={Moghadasi, Mahdi Naser},
  booktitle={2025 IEEE International Conference on Big Data (Big Data)},
  pages={xxx--xxx},
  year={2025},
  organization={IEEE}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Mahdi Naser Moghadasi** - *Principal Investigator* - [BrightMind AI](https://brightmindai.com)

## 🙏 Acknowledgments

- BrightMind AI research team for computational resources and insights
- Open-source community for model access
- IEEE Big Data 2025 reviewers for valuable feedback

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```python
   # Reduce batch size or number of parallel models
   analyzer = ExtendedLLMBrainAnalyzer(max_models_parallel=1)
   ```

2. **Model Loading Failures**
   - Some models require `trust_remote_code=True`
   - Encoder-decoder models (T5, BART) are skipped in current version
   - Check internet connection for model downloads

3. **MPS Backend Issues (Apple Silicon)**
   ```python
   # Force CPU if MPS causes issues
   analyzer = ExtendedLLMBrainAnalyzer(device='cpu')
   ```

4. **Missing Dependencies**
   ```bash
   # Install additional tokenizers
   pip install sentencepiece protobuf
   ```

### Performance Optimization

- Use GPU/MPS for faster processing
- Reduce `num_samples` for quicker testing
- Process models in batches to manage memory
- Use `comprehensive_brain_analysis_results.xlsx` for pre-computed results

## 🔬 Research Applications

This framework can be used for:
- Understanding LLM internal representations
- Comparing cognitive architectures across model families
- Identifying optimal models for specific cognitive tasks
- Developing brain-inspired AI architectures
- Studying emergence of cognitive specialization in LLMs
