# Deprecation Notice: Synthetic Whole-Plant Diagrams

**Status**: DEPRECATED & EXCLUDED FROM TRAINING  
**Date of Deprecation**: 2026-11-25  
**Audit Finding**: Synthetic / Procedural Diagrammatic Origin  
**Resolution**: Excluded from active model training, validation, and held-out evaluation splits.

---

## 1. Summary of Deprecated Assets

The following 9 classes and their 225 associated image files were moved from `data/raw/whole_plant/` to this deprecation archive:

1. `Corn_Fall_Armyworm_Damage`
2. `Corn_Nitrogen_Deficiency_Chlorosis`
3. `Corn_Stunt_Disease`
4. `Healthy_Corn_Whole_Plant`
5. `Healthy_Tomato_Whole_Plant`
6. `Non-Plant_Background_Object`
7. `Tomato_Fusarium_Wilt`
8. `Tomato_Heat_and_Drought_Stress`
9. `Unknown_Plant_Foliage`

---

## 2. Technical Evidence of Synthetic Origin

Forensic image analysis verified that these images were procedurally generated diagrams rather than camera-captured agricultural photographs:

1. **Color Histogram Diversity**:
   - Synthetic diagrams: Between **1,279 and 8,677 unique RGB colors** per 256×256 image.
   - Real photographic images: **25,000 to 55,000+ unique RGB colors** per image due to continuous natural lighting, sub-pixel camera sensor Bayer filter interpolation, and atmospheric scattering.
2. **Laplacian Edge Variance (Texture & Sensor Noise)**:
   - Synthetic diagrams: Laplacian variance \(\sigma^2_{\text{Lap}} \in [22.3, 87.5]\), showing flat color fills, sharp geometric polygon margins, and absence of ISO sensor noise.
   - Real photographic images: Laplacian variance \(\sigma^2_{\text{Lap}} \in [217.0, 3195.0]\), demonstrating organic cellular leaf venation, trichomes, fungal spore pustules, and natural depth-of-field defocus blur.
3. **Generator Provenance**:
   - Traceable to procedural Python scripts utilizing PIL `ImageDraw.Draw()` (`_generate_synthetic_whole_plant_image`, `draw.rectangle`, `draw.polygon`, `draw.ellipse`).

---

## 3. Policy & Enforcement

- **Training Exclusion**: These images MUST NOT be loaded into `WholePlantDataset`, `train.py`, or `evaluate.py`.
- **Active Dataset**: Replaced by authentic camera photographs from verified open-access agricultural datasets (`PlantVillage`, `PlantDoc`, `Plant Pathology 2021 FGVC8`, and `Cassava Leaf Disease Classification`).
- **Audit Gate**: Any image exhibiting fewer than 10,000 unique colors or flat polygon fill signatures is automatically flagged by `scripts/reproducible_audit.py` and barred from training.
