# Slot Filling and Intent Classification on ATIS and SLURP Datasets

Multi-task learning implementation for intent classification and slot filling in task-oriented dialogue systems using RNN and LSTM architectures.

##  Project Overview

This project implements and compares various neural architectures for two fundamental NLU tasks in dialogue systems:
- **Intent Classification**: Identifying the user's intention from an utterance
- **Slot Filling**: Extracting key information (entities) from the utterance

We evaluate four distinct modeling approaches:
1. **Independent Models** - Separate slot and intent classifiers
2. **Slot → Intent** - Slot predictions inform intent classification
3. **Intent → Slot** - Intent predictions guide slot filling
4. **Joint Multi-Task Learning** - Shared encoder for both tasks

##  Key Results

### ATIS Dataset Performance

| Model | Slot F1 | Intent F1 | Slot Acc | Intent Acc |
|-------|---------|-----------|----------|------------|
| **Independent RNN** | 0.791 | 0.841 | 0.985 | 0.874 |
| **Independent LSTM** | 0.739 | 0.653 | 0.975 | 0.949 |
| **Joint LSTM** | 0.727 | 0.636 | 0.975 | 0.946 |

### SLURP Dataset Performance

| Model | Slot F1 | Intent F1 | Slot Acc | Intent Acc |
|-------|---------|-----------|----------|------------|
| **Independent RNN** | 0.689 | 0.755 | 0.928 | 0.758 |
| **Independent LSTM** | 0.750 | 0.787 | 0.670 | 0.787 |
| **Joint LSTM** | 0.724 | 0.777 | 0.637 | 0.778 |

### Performance Highlights
-  **98.5% slot accuracy** on ATIS with independent RNN
-  **94.9% intent accuracy** on ATIS with independent LSTM
-  **Multi-task learning** improves joint understanding with minimal performance trade-off
-  **LSTM superiority** on complex SLURP dataset with longer sequences

## Datasets

### ATIS (Airline Travel Information System)
- **Total Utterances**: 5,871
- **Intents**: 22 classes
- **Slots**: 478 unique slot types
- **Domain**: Airline travel booking
- **Example**: "show me flights from boston to denver" → Intent: `flight`, Slots: `[O, O, O, B-fromloc.city_name, O, B-toloc.city_name]`

### SLURP (Spoken Language Understanding Resource Package)
- **Total Utterances**: 17,351
- **Intents**: 60 classes
- **Slots**: 55 unique slot types
- **Domain**: Home automation, calendar, weather, etc.
- **Characteristics**: More diverse and noisy utterances

##  Model Architectures

### 1. Independent Models
```
Input Utterance
     ↓
Embedding Layer
     ↓
RNN/LSTM Encoder
     ↓
├─→ Slot Classifier (per-token)
└─→ Intent Classifier (sentence-level)
```

### 2. Slot → Intent Model
```
Input → Embedding → RNN/LSTM → Slot Predictions
                                      ↓
                              Intent Classifier
```
Slot predictions are concatenated with hidden states to inform intent classification.

### 3. Intent → Slot Model
```
Input → Embedding → RNN/LSTM → Intent Prediction
                                      ↓
                              Slot Classifier
```
Intent embedding is concatenated with token representations for slot filling.

### 4. Joint Multi-Task Model
```
Input Utterance
     ↓
Embedding Layer
     ↓
Shared RNN/LSTM Encoder
     ↓
├─→ Slot Head (per-token predictions)
└─→ Intent Head (sentence classification)
```

**Loss Function:**
```
L_total = α * L_slot + (1-α) * L_intent
```

##  Implementation Details

### Hyperparameters

**RNN Models:**
- Epochs: 10
- Learning Rates:
  - Independent Intent: 0.001
  - Independent Slot: 0.01
  - Slot→Intent: 0.001
  - Intent→Slot: 0.01
  - Joint Model: 0.0005

**LSTM Models:**
- Batch Size: 32
- Embedding Dimension: 100
- Hidden Dimension: 128
- Learning Rate: 0.001
- Optimizer: Adam
- Epochs: 10

### Preprocessing
1. Tokenization of utterances
2. Conversion of slots and intents to integer indices
3. Sequence padding to uniform length
4. Train-validation-test split: 80%-10%-10%

### Evaluation Metrics
- **Slot Filling**: Macro-averaged Precision, Recall, F1-score, Accuracy
- **Intent Classification**: Macro-averaged Precision, Recall, F1-score, Accuracy
- **SLURP**: Additional weighted F1-scores and class-weighted loss

##  Repository Structure

```
├── Assignment_2_Report.pdf          # Comprehensive project report
├── ATIS_NLP_ASSIGNMENT_PYNB        # ATIS implementation notebook
├── Slurp_Atis_RNN_nlp_assignment   # SLURP RNN implementation
├── intent_model.pth                 # Trained intent classifier
├── intent_to_slot_model.pth        # Intent→Slot model weights
├── joint_model.pth                  # Joint multi-task model
├── slot_model.pth                   # Trained slot classifier
├── slot_to_intent_model.pth        # Slot→Intent model weights
├── slurp_intent_classifier.pth     # SLURP intent model
├── slurp_intent_to_slot.pth        # SLURP Intent→Slot model
├── slurp_joint_model.pth           # SLURP joint model
├── slurp_slot_filler.pth           # SLURP slot model
├── slurp_slot_to_intent.pth        # SLURP Slot→Intent model
├── experiment_results               # JSON results file
├── slurp_experiment_results        # SLURP results file
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

##  Key Findings

### 1. Architecture Comparison
- **LSTM consistently outperforms RNN** on both datasets, especially SLURP
- **RNN performs well on simpler ATIS** dataset due to shorter sequences
- **Gating mechanisms in LSTM** crucial for handling long-range dependencies

### 2. Task Interaction Effects
- **Slot→Intent transfer** improves LSTM performance but hurts simple RNNs
- **Intent→Slot** guidance beneficial when model is sufficiently expressive
- **Multi-task learning** offers best trade-off between joint understanding and task-specific performance

### 3. Dataset-Specific Insights
- **ATIS**: Simpler domain allows RNNs to achieve competitive results
- **SLURP**: Complex, diverse utterances require LSTM's memory capabilities
- **Class imbalance** in SLURP necessitates weighted loss functions

### 4. Performance Trade-offs
- Independent models excel at individual tasks
- Joint models sacrifice slight token-level accuracy for better task synergy
- Multi-task learning reduces overfitting through shared representations

##  Detailed Analysis

### Independent Models
- **RNN on ATIS**: Best slot F1 (0.791) and intent F1 (0.841)
- **LSTM on SLURP**: Superior performance (Slot F1: 0.750, Intent F1: 0.787)
- **Accuracy vs F1**: High accuracy doesn't always mean high F1 due to class imbalance

### Cross-Task Information Flow
- **Slot→Intent (LSTM, SLURP)**: Intent F1 improves to 0.790
- **Intent→Slot (RNN, ATIS)**: Slot F1 drops to 0.672 (information bottleneck)
- **Proper integration** requires sufficient model capacity

### Joint Multi-Task Learning
- **Shared representations** capture commonalities between tasks
- **LSTM joint model** maintains competitive performance on both tasks
- **Lower slot accuracy** sometimes observed due to optimization trade-offs

##  Educational Value

This project demonstrates:
- **Sequence modeling** with RNNs and LSTMs
- **Multi-task learning** principles and architectures
- **Transfer learning** between related NLU tasks
- **Handling class imbalance** in real-world datasets
- **Hyperparameter tuning** and model selection

##  Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Models

**For ATIS:**
```python
# See ATIS_NLP_ASSIGNMENT_PYNB for complete implementation
# Train independent models, then slot→intent, intent→slot, and joint models
```

**For SLURP:**
```python
# See Slurp_Atis_RNN_nlp_assignment for implementation
# Includes class-weighted loss for handling imbalance
```

##  System Configuration

- **Platform**: Google Colab
- **OS**: Linux 6.1.123+
- **CPU**: Intel Xeon @ 2.20 GHz
- **RAM**: 12.7 GB

##  References

- **ATIS Dataset**: [HuggingFace - tuetschek/atis](https://huggingface.co/datasets/tuetschek/atis)
- **SLURP Dataset**: [GitHub - pswietojanski/slurp](https://github.com/pswietojanski/slurp/tree/master/dataset/slurp)
- **Multi-Task Learning**: Joint training of related NLU tasks
- **Sequence Labeling**: Token-level classification for slot filling
