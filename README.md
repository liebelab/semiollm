# 🧠 SemioLLM: Evaluating Large Language Models for Diagnostic Reasoning from Unstructured Clinical Narratives in Epilepsy

## Overview

SemioLLM is a comprehensive evaluation framework for assessing Large Language Models (LLMs) in medical diagnostic reasoning tasks. This project specifically focuses on diagnostic reasoning from unstructured clinical narratives in epilepsy, while also demonstrating domain transfer capabilities to other medical domains such as dermatology.

## 📁 Project Structure

```
semiollm/
├── README.md                     # This file
├── LICENSE                       # License information
├── data_files/                   # Required data files for main figure visualizations
│   ├── f1_dict_nature_bootstrap.json
│   ├── mean_conf_score_bootstrap.json
│   └── mean_dict_nature_bootstrap.json
├── visualization/                # Code to reproduce figures from the main paper
│   ├── fig_2_a.ipynb            # Figure 2a Seizure Onset Zone (SOZ) localization performance
│   ├── fig_2_b.ipynb            # Figure 2b SOZ Confidence 
│   ├── fig_2_c.ipynb            # Figure 2c SOZ Calibration - Brier Score Loss
│   ├── fig_3_b.ipynb            # Figure 3b Reasoning Correctness/Completeness performance
│   ├── fig_3_c.ipynb            # Figure 3c Reasoning : comprehension, knowledge recall and reasoning step correct/incorrect assessment
│   ├── fig_3_d.ipynb            # Figure 3d source citation verification result
│   ├── fig_4_a.ipynb            # Figure 4a Impact of symptom length
│   ├── fig_4_b.ipynb            # Figure 4b impact of persona adaptation - both SOZ localization and confidence.
│   └── fig_4_c.ipynb            # Figure 4c impact of language (English, French, Spanish and Chinese)
├── domain_transfer/              # Domain transfer experiments
│   ├── dermatology_zeroshot.ipynb    # Zero-shot dermatology experiments
│   ├── derma_CoT.ipynb              # Chain-of-thought dermatology experiments
│   └── resources/                   # Symptom2Disease dataset file used 
└── src/                          # Source code modules
    ├── models/                   # LLM model implementations
    └── helper/                   # Utility functions and parsers
```

## 🎯 Key Features

### Main Research Focus
- **Epilepsy Diagnostic Reasoning**: Comprehensive evaluation of LLMs for epilepsy diagnosis from clinical narratives
- **Multi-Model Comparison**: Comparative analysis across different LLM architectures
- **Reasoning Analysis**: In-depth interpretability into models performance via reasoning and source citation analysis

### Domain Transfer Capabilities
- **Cross-Domain Generalization**: Demonstration of model performance across different medical specialties
- **Dermatology Case Study**: Proof-of-concept implementation for skin condition diagnosis. We evaluate LLM performance on mapping skin anomaly symptoms to likelihood of acne, psoriasis, fungal infection, impetigo, chicken pox, or others. We find that CoT prompting showed a 15% improvement over zero-shot for domain experts and, different AI personas affect diagnostic performance.


## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab
- Required packages (install via pip):
  ```bash
  pip install pandas numpy matplotlib seaborn openai
  ```

### Configuration
- Set up your OpenAI API key for LLM experiments


*This repository contains the implementation and experimental code for evaluating LLMs in medical diagnostic reasoning, with a focus on epilepsy and domain transfer capabilities.*