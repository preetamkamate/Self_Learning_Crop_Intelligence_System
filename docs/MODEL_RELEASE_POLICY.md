# Strict Model-Release Policy & Validation Gates

## Executive Summary
This document establishes the mandatory operational safety envelope for the local farmer advisory system. Because the active foliar pathology model (`models/whole_plant/model.pt`) achieves a held-out test accuracy of **58.33%**, it is designated as an **Experimental Research Prototype**. Under no circumstances is it presented as a dependable, certified diagnostic authority.

---

## 1. Classification of Routes: Supported vs. Data-Blocked

| Category | Viewpoint / Organ | Crop Species | Operational Status | Release Gate & Behavior |
| :--- | :--- | :--- | :---: | :--- |
| **Foliar Pathology** | `leaf` | Apple, Cassava, Corn, Tomato | 🧪 **Experimental** | Active inference; high confidence (\(\ge 0.70\)) and margin (\(\ge 0.15\)) required for chemical guidance. |
| **Foliar Pathology** | `leaf` | Grape, Potato, Citrus, Banana, etc. | 🛑 **Data-Blocked** | Safe abstention; returns `disease=None`, `supported=False`; photo guidance for unsupported crops. |
| **Stem / Branch Canker** | `stem_branch` | All crops | 🛑 **Data-Blocked** | Safe abstention; advises internal vascular browning inspection & SAU diagnostic clinic referral. |
| **Fruit Spot / Borer** | `fruit` | All crops | 🛑 **Data-Blocked** | Safe abstention; calyx close-up and borehole inspection guidance. |
| **Canopy Field View** | `canopy` | All crops | 🛑 **Data-Blocked** | Safe abstention; requests step-in foliar close-up photo. |
| **Whole Plant Overview**| `whole_plant` | All crops | 🛑 **Data-Blocked** | Safe abstention; full plant architecture guidance; requests single affected leaf photo. |
| **Root / Tuber Rots** | `root` | All crops | 🛑 **Data-Blocked** | Safe abstention; root gall wash guidance & certified extension laboratory recommendation. |
| **Entomology (Mites)** | `leaf` (underside) | Tomato | 🧪 **Pilot** | Dedicated binary CNN (`models/pest_detector/model.pt`); ICAR-IIHR IPM protocol. |
| **Entomology (Insects)**| `leaf`, `stem`, `whorl`| All crops (Armyworm, Borers, Aphids)| 🛑 **Data-Blocked** | Safe abstention; zero chemical pesticide prescription; IPM pheromone trap & neem extract scouting. |

---

## 2. Chemical Treatment Withholding Rules
1. **Rule 1 (Data-Blocked Routes)**: Any query involving a data-blocked organ, unverified crop, or unverified pest **MUST NEVER** output synthetic chemical fungicides, bactericides, or insecticides.
2. **Rule 2 (Low-Confidence Foliar Routes)**: For supported foliar crops, if top-1 confidence is below **0.70** or decision margin (difference between top-1 and top-2) is below **0.15**, chemical treatments are **strictly withheld**. The system presents the result as a "Tentative Observation" and supplies only cultural/sanitation methods.
3. **Rule 3 (Mandatory PPE & KVK Referral)**: Whenever chemical recommendations are displayed (only at high confidence and margin), they must carry mandatory Pre-Harvest Interval (PHI) warnings, Personal Protective Equipment (PPE) requirements, and explicit instructions to verify symptoms with the nearest Krishi Vigyan Kendra (KVK).

---

## 3. Privacy & Location Architecture
1. **Zero Raw GPS Persistence**: Latitude and longitude captured via browser geolocation are held in-memory solely long enough to calculate Haversine distance to the nearest Karnataka district and KVK center.
2. **Immediate Purge**: Coordinates are cleared from URL parameters, session objects, and local logs immediately after resolution.
3. **Offline Climatology**: Weather intelligence operates entirely on local historical monthly climate tables with zero external API calls or secret keys.
