# Model Card: Dedicated Pest Detection CNN

## 1. Model Details
- **Model Identifier**: `models/pest_detector/model.pt`
- **Architecture**: `DedicatedPestCNN` (3-layer Convolutional Neural Network with Batch Normalization, Max Pooling, and Dropout)
- **Input Resolution**: 224x224 RGB image
- **Output Classes**: 2 classes:
  1. `Healthy Foliage (Pest-Free Control)`
  2. `Two-spotted Spider Mites (Tetranychus urticae)`
- **Framework**: PyTorch (`mps` / `cuda` / `cpu`)

---

## 2. Intended Scope & Limits
- **Pilot Scope**: Detection of fine silvery stippling and dense micro-webbing characteristic of Two-Spotted Spider Mite infestation on Tomato foliage.
- **Strictly Quarantined / Data-Blocked Pests**:
  - **Fall Armyworm (*Spodoptera frugiperda*)**: Blocked (synthetic diagram data quarantined; authentic field dataset awaiting collection).
  - **Stem Borers (*Chilo partellus*)**: Blocked (requires verified internal stem split photos).
  - **Aphids & Whiteflies**: Blocked.
  - **Thrips & Caterpillars**: Blocked.
- **Action on Blocked Routes**: Safe abstention (`status=pest_data_blocked`), zero chemical prescription, provides Integrated Pest Management (IPM) scouting guidance with pheromone traps and botanical neem deterrents (NSKE 5%).

---

## 3. Training & Evaluation Metrics
- **Dataset**: PlantVillage authentic camera samples (`data/manifests/pest_detection_manifest.json`)
  - **Train**: 50 samples
  - **Val**: 12 samples
  - **Test (Held-Out)**: 12 samples (6 healthy, 6 spider mite)
- **Evaluation Results**:
  - **Overall Accuracy**: **100.0%**
  - **Macro Precision**: **1.0000**
  - **Macro Recall**: **1.0000**
  - **Macro F1 Score**: **1.0000**
  - **Confusion Matrix**: \(\begin{pmatrix} 6 & 0 \\ 0 & 6 \end{pmatrix}\)
  - **Unsupported Pest Abstention Rate**: **100.0%**

---

## 4. Safety & Chemical Pesticide Policy
- For verified Two-Spotted Spider Mites with confidence \(\ge 0.70\): Provides ICAR-IIHR organic/low-toxicity protocols (wettable sulfur 80 WP @ 3 g/L or Azadirachtin 10,000 ppm @ 1 ml/L).
- For unverified pests or confidence \(< 0.70\): Chemical pesticides are strictly withheld to prevent ecological disruption and non-target predator mortality.
