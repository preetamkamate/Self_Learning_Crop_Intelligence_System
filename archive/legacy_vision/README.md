# Legacy Vision Archive

This directory preserves the legacy multi-part and leaf pathology vision models and codebase:
- `models/`: Legacy PyTorch checkpoint probes (`stem_classifier.pt`, `fruit_classifier.pt`, `bark_trunk_classifier.pt`, etc.)
- `src/`: Legacy vision code (`hierarchical_classifier.py`, `dataset_builder.py`, etc.)

Per project requirements, the live app diagnosis has transitioned to the independent Whole-Plant Model Pipeline (`src/whole_plant/` and `models/whole_plant/`).
These legacy models are retained here for historical reference and can be restored if needed.
