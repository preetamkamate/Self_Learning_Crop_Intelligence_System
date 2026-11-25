"""
Honest Photo Guidance Engine for Crop Pathology and Field Diagnostics.
Provides specific, actionable, expert photographic instructions for farmers
whenever a diagnosis cannot be made with verified real labeled data.
Never guesses diseases, pests, or chemical treatments.
"""
from typing import Any, Dict, List, Optional


PHOTO_GUIDANCE_CATALOG: Dict[str, Dict[str, Any]] = {
    "leaf": {
        "title_en": "Photo Guidance for Leaf / Foliar Inspection",
        "title_kn": "ಎಲೆಯ ಫೋಟೋ ತೆಗೆಯುವ ನಿಖರ ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Distance & Focus**: Hold your camera 15-25 cm from the leaf and tap the screen to focus on the border between healthy and affected tissue.\n"
            "2. **Lighting**: Take the photo under bright, indirect daylight. Avoid strong camera flash which washes out lesion halos and sporulation.\n"
            "3. **Leaf Underside**: Many fungal mildews, rust pustules, and spider mites live on the underside of the leaf. If symptoms are faint on top, turn the leaf over and take a second photo.\n"
            "4. **Framing**: Place a single leaf flat against a neutral background (such as your hand or a plain notebook) to help the AI isolate the foliar margins."
        ),
        "instructions_kn": (
            "೧. **ಅಂತರ ಮತ್ತು ಫೋಕಸ್**: ಕ್ಯಾಮೆರಾವನ್ನು ಎಲೆಯಿಂದ ೧೫-೨೫ ಸೆಂ.ಮೀ ದೂರದಲ್ಲಿಟ್ಟು, ರೋಗದ ಗಡಿಭಾಗದ ಮೇಲೆ ಬೆರಳು ತಟ್ಟಿ ಫೋಕಸ್ ಮಾಡಿ.\n"
            "೨. **ಬೆಳಕು**: ಸಹಜ ಹಗಲು ಬೆಳಕಿನಲ್ಲಿ ಫೋಟೋ ತೆಗೆಯಿರಿ. ಬಲವಾದ ಫ್ಲಾಶ್ ಬೆಳಕು ಕಲೆಯ ಸೂಕ್ಷ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಮರೆಮಾಡುತ್ತದೆ.\n"
            "೩. **ಎಲೆಯ ಹಿಂಭಾಗ**: ತುಕ್ಕು ರೋಗ, ಬೂದಿ ರೋಗ ಮತ್ತು ನುಸಿ ಕೀಟಗಳು ಎಲೆಯ ಕೆಳಭಾಗದಲ್ಲಿರುತ್ತವೆ. ಎಲೆಯನ್ನು ತಿರುಗಿಸಿ ಕೆಳಭಾಗದ ಫೋಟೋ ಕೂಡ ತೆಗೆಯಿರಿ.\n"
            "೪. **ಹಿನ್ನೆಲೆ**: ಎಲೆಯನ್ನು ಕೈಯಲ್ಲಾಗಲಿ ಅಥವಾ ಬಿಳಿ ಹಾಳೆಯ ಮೇಲಾಗಲಿ ಇಟ್ಟು ಸ್ಪಷ್ಟವಾಗಿ ಫೋಟೋ ತೆಗೆಯಿರಿ."
        ),
        "checklist": [
            "In-focus lesion border",
            "Natural daylight (no harsh flash)",
            "Leaf upper and lower surfaces inspected",
            "Single leaf centered",
        ],
        "lab_test_recommended": False,
    },
    "stem_branch": {
        "title_en": "Photo Guidance for Stem & Branch Inspection",
        "title_kn": "ಕಾಂಡ ಮತ್ತು ಕೊಂಬೆಗಳ ಫೋಟೋ ತೆಗೆಯುವ ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Canker & Lesion Focus**: Focus closely (15-30 cm) on the transition zone where healthy bark meets discolored, sunken, or cracked stem tissue.\n"
            "2. **Base of Plant (Collar Region)**: If the whole plant is wilting, inspect the stem base at soil level (collar rot / damping off). Clear away dry debris before taking the photo.\n"
            "3. **Internal Stem Test (Field Tip)**: For suspected bacterial or fusarium wilt, split an affected lower stem lengthwise with a clean knife. Look for dark brown vascular streaking inside.\n"
            "4. **Lab Confirmation**: Stem cankers and vascular wilts often require laboratory culture or Krishi Vigyan Kendra (KVK) testing for definitive diagnosis."
        ),
        "instructions_kn": (
            "೧. **ಕಾಂಡದ ಹುಣ್ಣು / ಕಲೆ**: ಕಾಂಡದ ಆರೋಗ್ಯಕರ ಭಾಗ ಮತ್ತು ಕೊಳೆತ/ಕಪ್ಪಾದ ಭಾಗದ ಸಂಧಿಸ್ಥಾನದ ಮೇಲೆ ೧೫-೩೦ ಸೆಂ.ಮೀ ಅಂತರದಲ್ಲಿ ಫೋಕಸ್ ಮಾಡಿ.\n"
            "೨. **ಬುಡದ ಭಾಗ (ಕಾಲರ್ ಭಾಗ)**: ಗಿಡ ಬಾಡುತ್ತಿದ್ದರೆ, ಮಣ್ಣಿನ ಮಟ್ಟದಲ್ಲಿರುವ ಕಾಂಡದ ಬುಡವನ್ನು ಪರೀಕ್ಷಿಸಿ. ಕಸಕಡ್ಡಿ ಸರಿಸಿ ಫೋಟೋ ತೆಗೆಯಿರಿ.\n"
            "೩. **ಒಳಭಾಗದ ಪರೀಕ್ಷೆ**: ಸೊರಗು ರೋಗದ ಅನುಮಾನವಿದ್ದರೆ, ಕಾಂಡವನ್ನು ಉದ್ದಕ್ಕೆ ಸೀಳಿ ಒಳಭಾಗ ಕಂದುಬಣ್ಣಕ್ಕೆ ತಿರುಗಿದೆಯೇ ಎಂದು ಪರೀಕ್ಷಿಸಿ.\n"
            "೪. **ಪ್ರಯೋಗಾಲಯ ದೃಢೀಕರಣ**: ಕಾಂಡದ ಆಳವಾದ ರೋಗಗಳಿಗೆ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (KVK) ಅಥವಾ ಲ್ಯಾಬ್ ಪರೀಕ್ಷೆ ಅಗತ್ಯವಿರುತ್ತದೆ."
        ),
        "checklist": [
            "Focus on stem collar/canker margins",
            "Clear away soil debris around base",
            "Check for internal vascular discoloration",
            "Note any sap oozing or boreholes",
        ],
        "lab_test_recommended": True,
    },
    "fruit": {
        "title_en": "Photo Guidance for Fruit & Pod Inspection",
        "title_kn": "ಹಣ್ಣು ಮತ್ತು ಕಾಯಿಗಳ ಫೋಟೋ ತೆಗೆಯುವ ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Lesion Texture & Margins**: Capture a clear close-up (10-20 cm) showing the texture of the spot (sunken, water-soaked, target-board concentric rings, or fungal velvet).\n"
            "2. **Blossom End vs. Calyx**: Take photos of both the blossom end (bottom) and the stem end (calyx). Blossom-end rot is physiological (calcium deficiency), whereas anthracnose occurs across fruit walls.\n"
            "3. **Boreholes & Frass**: Inspect for tiny entry pinholes, gummy exudates, or insect droppings (frass) indicating fruit borer pests.\n"
            "4. **Lighting**: Avoid bright sun reflection on glossy fruit skins (e.g., tomatoes or apples) by angling slightly away from direct sun."
        ),
        "instructions_kn": (
            "೧. **ಕಲೆಯ ವಿನ್ಯಾಸ**: ಹಣ್ಣಿನ ಮೇಲಿನ ಕಲೆಯು ತಗ್ಗು ಬಿದ್ದಿದೆಯೇ, ಉಂಗುರಾಕಾರವಾಗಿದೆಯೇ ಅಥವಾ ಶಿಲೀಂಧ್ರ ಬೆಳೆದಿದೆಯೇ ಎಂದು ಹತ್ತಿರದಿಂದ (೧೦-೨೦ ಸೆಂ.ಮೀ) ಫೋಟೋ ತೆಗೆಯಿರಿ.\n"
            "೨. **ಹಣ್ಣಿನ ತುದಿ ಮತ್ತು ತೊಟ್ಟು**: ಹಣ್ಣಿನ ಕೆಳತುದಿ (ಬ್ಲಾಸಮ್ ಎಂಡ್) ಮತ್ತು ತೊಟ್ಟಿನ ಭಾಗ ಎರಡನ್ನೂ ತೋರಿಸಿ. ಕೆಳತುದಿ ಕೊಳೆತವು ಸುಣ್ಣದ ಕೊರತೆಯಿಂದಾಗಬಹುದು.\n"
            "೩. **ಕೀಟದ ರಂಧ್ರಗಳು**: ಹಣ್ಣು ಕೊರೆಯುವ ಹುಳುಗಳ ಲಕ್ಷಣಗಳಾದ ಸಣ್ಣ ರಂಧ್ರಗಳು ಅಥವಾ ಮಲದ ಕಣಗಳು ಇವೆಯೇ ಎಂದು ಪರೀಕ್ಷಿಸಿ.\n"
            "೪. **ಬೆಳಕಿನ ಪ್ರತಿಫಲನ ತಪ್ಪಿಸಿ**: ಹಣ್ಣಿನ ಹೊಳಪಿನ ಮೇಲೆ ನೇರ ಸೂರ್ಯನ ಬೆಳಕು ಬೀಳದಂತೆ ಸ್ವಲ್ಪ ಓರೆಯಾಗಿ ಫೋಟೋ ತೆಗೆಯಿರಿ."
        ),
        "checklist": [
            "Close-up of lesion texture and margin",
            "Show both blossom-end and stem-end",
            "Check for insect entry holes or gummy ooze",
            "Avoid shiny specular glare",
        ],
        "lab_test_recommended": False,
    },
    "canopy": {
        "title_en": "Photo Guidance for Field Canopy & Plot Inspection",
        "title_kn": "ಬೆಳೆ ಮೇಲ್ಛಾವಣಿ (Canopy) ಮತ್ತು ಕ್ಷೇತ್ರದ ಫೋಟೋ ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Broad Framing**: Step back 2-3 meters and take an angled (45-degree) photo capturing 4-6 adjacent plants across rows.\n"
            "2. **Distribution Pattern**: Note whether symptoms appear in circular patches (soilborne root rot/nematodes), scattered randomly (insect vector virus), or uniformly across the whole field (nutrient deficiency/drought).\n"
            "3. **Lighting**: Capture under uniform morning or late afternoon daylight. Harsh mid-day sun casts deep shadows through the canopy that obscure foliar symptoms.\n"
            "4. **Supplement with Close-up**: A canopy photo shows symptom distribution, but AI disease diagnosis requires an accompanying close-up of an affected leaf or stem."
        ),
        "instructions_kn": (
            "೧. **ವ್ಯಾಪಕ ಚೌಕಟ್ಟು**: ಗಿಡಗಳಿಂದ ೨-೩ ಮೀಟರ್ ಹಿಂದೆ ನಿಂತು, ೪೫ ಡಿಗ್ರಿ ಕೋನದಲ್ಲಿ ೪-೬ ಗಿಡಗಳು ಕಾಣುವಂತೆ ಫೋಟೋ ತೆಗೆಯಿರಿ.\n"
            "೨. **ರೋಗ ಹರಡುವಿಕೆಯ ಮಾದರಿ**: ರೋಗವು ಗುಂಪು-ಗುಂಪಾಗಿ ಹರಡಿದೆಯೇ (ಮಣ್ಣಿನ ಶಿಲೀಂಧ್ರ), ಅಲ್ಲಲ್ಲಿ ಚದುರಿದೆಯೇ (ಕೀಟ ರೋಗವಾಹಕ) ಅಥವಾ ಇಡೀ ಹೊಲಕ್ಕೆ ಸಮಾನವಾಗಿದೆಯೇ (ಪೋಷಕಾಂಶ ಕೊರತೆ) ಎಂಬುದನ್ನು ಗಮನಿಸಿ.\n"
            "೩. **ಬೆಳಕು**: ಮುಂಜಾನೆ ಅಥವಾ ಸಂಜೆಯ ಸಮಾನ ಬೆಳಕಿನಲ್ಲಿ ಫೋಟೋ ತೆಗೆಯಿರಿ. ಮಧ್ಯಾಹ್ನದ ನೆರಳು ತಪ್ಪಿಸಿ.\n"
            "೪. **ಸೂಕ್ಷ್ಮ ಫೋಟೋ ಸೇರಿಸಿ**: ಕ್ಯಾನೋಪಿ ಫೋಟೋದೊಂದಿಗೆ ಬಾಧಿತ ಎಲೆಯ ಹತ್ತಿರದ ಫೋಟೋವನ್ನು ಸಹ ನೀಡುವುದು ನಿಖರ ರೋಗನಿರ್ಣಯಕ್ಕೆ ಅತ್ಯಗತ್ಯ."
        ),
        "checklist": [
            "2-3 meter distance capturing row context",
            "45-degree angle showing upper canopy",
            "Uniform daytime illumination (no heavy shadows)",
            "Supplement with leaf close-up for pathology",
        ],
        "lab_test_recommended": False,
    },
    "whole_plant": {
        "title_en": "Photo Guidance for Whole-Plant Inspection",
        "title_kn": "ಸಂಪೂರ್ಣ ಗಿಡದ ಫೋಟೋ ತೆಗೆಯುವ ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Full Stature Framing**: Position yourself so the entire plant from root collar/soil line to the growing apical tip is in frame.\n"
            "2. **Background Simplicity**: If possible, place a plain board, cardboard, or clean cloth behind the plant so background soil and neighboring weeds do not distract the model.\n"
            "3. **Symptom Localization**: Look at where symptoms start: lower/older leaves (nitrogen/potassium deficiency, soil splash blights) vs. top/young leaves (calcium/iron deficiency, virus stunting).\n"
            "4. **Complementary Leaf Shot**: For automated diagnosis, please provide both a whole-plant frame (for stature/wilting) and a clear single-leaf close-up (for lesion pathology)."
        ),
        "instructions_kn": (
            "೧. **ಸಂಪೂರ್ಣ ಗಿಡದ ಫ್ರೇಮ್**: ಮಣ್ಣಿನ ಬುಡದಿಂದ ತುದಿಯ ಚಿಗುರಿನವರೆಗೆ ಇಡೀ ಗಿಡವು ಒಂದೇ ಫ್ರೇಮ್‌ನಲ್ಲಿ ಬರುವಂತೆ ನಿಲ್ಲಿ.\n"
            "೨. **ಸರಳ ಹಿನ್ನೆಲೆ**: ಸಾಧ್ಯವಾದರೆ ಗಿಡದ ಹಿಂದೆ ಬಿಳಿ ಅಥವಾ ಸರಳ ಬಣ್ಣದ ಹಲಗೆ/ಬಟ್ಟೆ ಇಡಿ, ಇದರಿಂದ ಕಳೆಗಳು ಮತ್ತು ಮಣ್ಣಿನ ಗೊಂದಲ ಉಂಟಾಗುವುದಿಲ್ಲ.\n"
            "೩. **ರೋಗಲಕ್ಷಣಗಳ ಹರಡುವಿಕೆ**: ರೋಗವು ಕೆಳಗಿನ ಹಳೆಯ ಎಲೆಗಳಲ್ಲಿ ಪ್ರಾರಂಭವಾಗಿದೆಯೇ (ಸಾರಜನಕ/ಪೊಟ್ಯಾಶ್ ಕೊರತೆ) ಅಥವಾ ಮೇಲಿನ ಚಿಗುರು ಎಲೆಗಳಲ್ಲಿ ಪ್ರಾರಂಭವಾಗಿದೆಯೇ (ವೈರಸ್/ಕಬ್ಬಿಣದ ಕೊರತೆ) ಗಮನಿಸಿ.\n"
            "೪. **ಎಲೆಯ ಪೂರಕ ಫೋಟೋ**: ಸಂಪೂರ್ಣ ಗಿಡದ ಫೋಟೋದೊಂದಿಗೆ ಒಂದು ಬಾಧಿತ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ಒದಗಿಸಿ."
        ),
        "checklist": [
            "Plant framed from soil line to apex",
            "Clear background separation",
            "Observe symptom gradient (lower vs upper leaves)",
            "Attach leaf close-up for AI classification",
        ],
        "lab_test_recommended": False,
    },
    "root": {
        "title_en": "Photo Guidance for Root & Tuber Inspection",
        "title_kn": "ಬೇರು ಮತ್ತು ಗೆಡ್ಡೆಗಳ ಫೋಟೋ ತೆಗೆಯುವ ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Careful Excavation**: Gently dig around the plant base (15-20 cm radius) without slicing the main taproot or lateral root system.\n"
            "2. **Gentle Water Rinse**: Rinse clinging soil particles away with clean water so root surfaces, lesions, and nodules are visible.\n"
            "3. **Examine Root Architecture**: Inspect for dark brown/black water-soaked rot, galls/knots (root-knot nematodes), stubby stunted feeder roots, or healthy white root tips.\n"
            "4. **Laboratory Testing Notice**: Subterranean root pathogens (Phytophthora, Pythium, Rhizoctonia, Fusarium, and plant-parasitic nematodes) cannot be diagnosed with certainty by visual photos alone. A soil/root sample should be sent to an accredited ICAR/SAU plant health clinic."
        ),
        "instructions_kn": (
            "೧. **ಎಚ್ಚರಿಕೆಯ ಅಗೆಯುವಿಕೆ**: ಗಿಡದ ಸುತ್ತ ೧೫-೨೦ ಸೆಂ.ಮೀ ಅಂತರದಲ್ಲಿ ಬೇರಿಗೆ ಪೆಟ್ಟಾಗದಂತೆ ನಿಧಾನವಾಗಿ ಮಣ್ಣನ್ನು ಅಗೆಯಿರಿ.\n"
            "೨. **ನೀರಿನಿಂದ ತೊಳೆಯಿರಿ**: ಬೇರಿಗೆ ಅಂಟಿದ ಮಣ್ಣನ್ನು ಸ್ವಚ್ಛ ನೀರಿನಿಂದ ನಿಧಾನವಾಗಿ ತೊಳೆದು ಬೇರಿನ ಮೇಲ್ಮೈ ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣುವಂತೆ ಮಾಡಿ.\n"
            "೩. **ಬೇರಿನ ರಚನೆ ಪರೀಕ್ಷೆ**: ಬೇರು ಕಪ್ಪಾಗಿ ಕೊಳೆತಿದೆಯೇ, ಗಂಟುಗಳಿವೆಯೇ (ಜಂತುಹುಳು/ನೆಮಟೋಡ್) ಅಥವಾ ಆರೋಗ್ಯಕರ ಬಿಳಿ ಬೇರುಗಳಿವೆಯೇ ಗಮನಿಸಿ.\n"
            "೪. **ಪ್ರಯೋಗಾಲಯ ಪರೀಕ್ಷೆ ಕಡ್ಡಾಯ**: ಬೇರಿನ ಆಂತರಿಕ ಶಿಲೀಂಧ್ರ ಮತ್ತು ಜಂತುಹುಳು ರೋಗಗಳನ್ನು ಕೇವಲ ಕಣ್ಣಿನ ಫೋಟೋದಿಂದ ದೃಢಪಡಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ. ಮಣ್ಣು ಮತ್ತು ಬೇರಿನ ಮಾದರಿಯನ್ನು ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರಕ್ಕೆ (KVK) ಕಳುಹಿಸಿ."
        ),
        "checklist": [
            "Root excavated without cutting taproot",
            "Soil rinsed with water",
            "Galls, rot, and root tips visible",
            "Certified soil/root testing recommended",
        ],
        "lab_test_recommended": True,
    },
    "pest": {
        "title_en": "Photo Guidance for Insect Pest & Damage Detection",
        "title_kn": "ಕೀಟ ಮತ್ತು ಕೀಟಹಾನಿಯ ಫೋಟೋ ತೆಗೆಯುವ ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Macro Close-Up of Pest**: Get as close as possible (5-15 cm) using macro mode. Ensure the insect body, legs, wings, or caterpillar head capsule are sharply in focus.\n"
            "2. **Inspect Leaf Underside & Growing Tips**: Aphids, whiteflies, thrips, and mites congregate on young tender shoots and the underside of leaves.\n"
            "3. **Photograph Damage Symptoms**: If the pest has flown away, capture the distinct feeding symptom: windowpane scraping, circular leaf notches, boreholes with sawdust-like frass, or silvery stippling.\n"
            "4. **No Premature Chemical Sprays**: Never apply broad-spectrum chemical insecticides without positive pest identification. Collect a physical specimen in a transparent vial if unsure."
        ),
        "instructions_kn": (
            "೧. **ಕೀಟದ ಮ್ಯಾಕ್ರೋ ಫೋಟೋ**: ಕ್ಯಾಮೆರಾವನ್ನು ೫-೧೫ ಸೆಂ.ಮೀ ಹತ್ತಿರವಿಟ್ಟು ಕೀಟದ ದೇಹ, ಕಾಲುಗಳು ಅಥವಾ ಮರಿಹುಳು ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣುವಂತೆ ಫೋಕಸ್ ಮಾಡಿ.\n"
            "೨. **ಎಲೆಯ ಹಿಂಭಾಗ ಮತ್ತು ಚಿಗುರು**: ಜೇಡರ ನುಸಿ, ಬಿಳಿ ನೊಣ, ಹೇನು ಮತ್ತು ಥ್ರಿಪ್ಸ್ ಕೀಟಗಳು ಚಿಗುರು ಮತ್ತು ಎಲೆಯ ಕೆಳಭಾಗದಲ್ಲಿರುತ್ತವೆ.\n"
            "೩. **ಕೀಟ ಹಾನಿಯ ಲಕ್ಷಣಗಳು**: ಕೀಟ ಕಾಣದಿದ್ದರೆ, ಎಲೆ ತಿಂದ ಗುರುತುಗಳು, ರಂಧ್ರಗಳು, ಎಲೆಯ ಮೇಲಿನ ಬೆಳ್ಳನೆಯ ಗೆರೆಗಳು ಅಥವಾ ಮಲದ ಕಣಗಳ ಫೋಟೋ ತೆಗೆಯಿರಿ.\n"
            "೪. **ಅನಗತ್ಯ ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಬೇಡಿ**: ಕೀಟ ನಿಖರವಾಗಿ ದೃಢಪಡುವ ಮುನ್ನ ರಾಸಾಯನಿಕ ಸಿಂಪಡಣೆ ಮಾಡಬೇಡಿ. ಕೀಟವನ್ನು ಸಣ್ಣ ಡಬ್ಬಿಯಲ್ಲಿ ಹಿಡಿದು ತಜ್ಞರಿಗೆ ತೋರಿಸಿ."
        ),
        "checklist": [
            "Sharp macro shot of pest body/larva",
            "Leaf undersides and apical shoots checked",
            "Feeding patterns, frass, or webbing documented",
            "Specimen collection in clear container if unconfirmed",
        ],
        "lab_test_recommended": False,
    },
    "blurry": {
        "title_en": "Photo Guidance: Out-of-Focus Image",
        "title_kn": "ಫೋಟೋ ಮಸುಕಾಗಿದೆ - ಮರು-ಫೋಟೋ ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Hold Camera Steady**: Brace your elbows against your body or support your hands to prevent motion blur.\n"
            "2. **Tap to Focus**: Tap the screen directly on the disease spot or pest before clicking the shutter.\n"
            "3. **Adequate Daylight**: Low light forces a slow shutter speed causing blur. Take photos in bright daylight.\n"
            "4. **Minimum Distance**: Do not place the camera closer than 10 cm unless using dedicated macro focus mode."
        ),
        "instructions_kn": (
            "೧. **ಕ್ಯಾಮೆರಾ ಅಲುಗಾಡಿಸಬೇಡಿ**: ಕೈಗಳನ್ನು ಸ್ಥಿರವಾಗಿಟ್ಟುಕೊಂಡು ಫೋಟೋ ತೆಗೆಯಿರಿ.\n"
            "೨. **ಸ್ಕ್ರೀನ್ ಮೇಲೆ ಟ್ಯಾಪ್ ಮಾಡಿ**: ಫೋಟೋ ತೆಗೆಯುವ ಮುನ್ನ ರೋಗದ ಕಲೆಯ ಮೇಲೆ ಬೆರಳು ತಟ್ಟಿ ಫೋಕಸ್ ಸ್ಪಷ್ಟಪಡಿಸಿ.\n"
            "೩. **ಸಾಕಷ್ಟು ಹಗಲು ಬೆಳಕು**: ಕಡಿಮೆ ಬೆಳಕಿನಲ್ಲಿ ಫೋಟೋ ಮಸುಕಾಗುತ್ತದೆ. ಉತ್ತಮ ಹಗಲು ಬೆಳಕಿನಲ್ಲಿ ಫೋಟೋ ತೆಗೆಯಿರಿ.\n"
            "೪. **ಕನಿಷ್ಠ ಅಂತರ**: ಕನಿಷ್ಠ ೧೦-೧೫ ಸೆಂ.ಮೀ ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        ),
        "checklist": [
            "Sharp focus on leaf lesion or pest",
            "No motion blur or camera shake",
            "Adequate daylight",
        ],
        "lab_test_recommended": False,
    },
    "non_plant": {
        "title_en": "Photo Guidance: Non-Plant Object Detected",
        "title_kn": "ಕೃಷಿಯೇತರ ಚಿತ್ರ - ಗಿಡದ ಫೋಟೋ ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Crop Subject Only**: Ensure the photograph is centered on a living crop plant, foliar tissue, stem, fruit, or canopy.\n"
            "2. **Exclude Tools & Soil Clutter**: Do not submit photos of machinery, farm inputs, fertilizer bags, or bare field soil without crop vegetation.\n"
            "3. **Safety Protection**: Automatic disease diagnosis is safely withheld for non-crop photos to prevent incorrect chemical recommendations."
        ),
        "instructions_kn": (
            "೧. **ಕೃಷಿ ಸಸ್ಯ ಮಾತ್ರ**: ಕೃಷಿ ಬೆಳೆ, ಎಲೆ, ಕಾಂಡ ಅಥವಾ ಹಣ್ಣಿನ ಚಿತ್ರವನ್ನು ಮಾತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.\n"
            "೨. **ಇತರ ವಸ್ತುಗಳನ್ನು ಹೊರಗಿಡಿ**: ರಸಗೊಬ್ಬರದ ಚೀಲ, ಯಂತ್ರೋಪಕರಣ ಅಥವಾ ಬರಿಯ ಮಣ್ಣಿನ ಫೋಟೋ ಕಳುಹಿಸಬೇಡಿ.\n"
            "೩. **ಸುರಕ್ಷತಾ ಕ್ರಮ**: ತಪ್ಪು ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆ ತಡೆಯಲು ಕೃಷಿಯೇತರ ಚಿತ್ರಗಳಿಗೆ ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ."
        ),
        "checklist": [
            "Living agricultural crop in frame",
            "No tools, bags, or non-plant objects",
        ],
        "lab_test_recommended": False,
    },
    "unsupported_crop": {
        "title_en": "Photo Guidance: Unsupported Crop Species",
        "title_kn": "ಬೆಂಬಲವಿಲ್ಲದ ಬೆಳೆ - ಮಾರ್ಗದರ್ಶನ",
        "instructions_en": (
            "1. **Verified Real Datasets**: Currently, verified disease diagnosis models are deployed for **Apple**, **Cassava**, **Corn (Maize)**, and **Tomato**.\n"
            "2. **Zero Guessing Policy**: We strictly refuse to guess diseases or recommend fungicides for untrained crops to protect your farm from crop injury and financial loss.\n"
            "3. **Local Extension Support**: For other crops, please share your symptom photos with your local Krishi Vigyan Kendra (KVK), Raitha Samparka Kendra (RSK), or State Agricultural University extension clinic."
        ),
        "instructions_kn": (
            "೧. **ದೃಢೀಕೃತ ಬೆಳೆಗಳು**: ಪ್ರಸ್ತುತ **ಸೇಬು (Apple)**, **ಮರಗೆಣಸು (Cassava)**, **ಮೆಕ್ಕೆಜೋಳ (Corn)**, ಮತ್ತು **ಟೊಮ್ಯಾಟೊ (Tomato)** ಬೆಳೆಗಳಿಗೆ ಮಾತ್ರ ತರಬೇತಿ ನೀಡಲಾದ ಮಾದರಿಗಳಿವೆ.\n"
            "೨. **ಊಹಿಸದಿರುವ ನೀತಿ**: ತರಬೇತಿ ಇಲ್ಲದ ಬೆಳೆಗಳಿಗೆ ತಪ್ಪು ರೋಗ ಅಥವಾ ಕೀಟನಾಶಕವನ್ನು ಊಹಿಸಿ ಹೇಳುವುದಿಲ್ಲ.\n"
            "೩. **ಸ್ಥಳೀಯ ಕೃಷಿ ಕೇಂದ್ರ ಸಂಪರ್ಕಿಸಿ**: ಇತರ ಬೆಳೆಗಳಿಗಾಗಿ ನಿಮ್ಮ ಸಮೀಪದ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರ (RSK) ಅಥವಾ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರದ (KVK) ವಿಜ್ಞಾನಿಗಳನ್ನು ಸಂಪರ್ಕಿಸಿ."
        ),
        "checklist": [
            "Check supported crop list (Apple, Cassava, Corn, Tomato)",
            "Consult local RSK/KVK for unlisted crops",
        ],
        "lab_test_recommended": True,
    },
}


def get_photo_guidance(
    viewpoint: str,
    crop: Optional[str] = None,
    is_pest: bool = False,
    is_blurry: bool = False,
    is_non_plant: bool = False,
    is_unsupported_crop: bool = False,
) -> Dict[str, Any]:
    """
    Retrieve comprehensive, honest photographic guidance for the specified situation.
    """
    if is_non_plant:
        cat_key = "non_plant"
    elif is_blurry:
        cat_key = "blurry"
    elif is_pest:
        cat_key = "pest"
    elif is_unsupported_crop:
        cat_key = "unsupported_crop"
    else:
        v_clean = str(viewpoint).lower().strip().replace(" ", "_").replace("/", "_").replace("-", "_")
        if "stem" in v_clean or "branch" in v_clean or "cane" in v_clean or "twig" in v_clean or "bark" in v_clean or "trunk" in v_clean:
            cat_key = "stem_branch"
        elif "fruit" in v_clean or "pod" in v_clean or "berry" in v_clean or "grain" in v_clean:
            cat_key = "fruit"
        elif "canopy" in v_clean or "field" in v_clean or "plot" in v_clean:
            cat_key = "canopy"
        elif "whole" in v_clean or "full" in v_clean:
            cat_key = "whole_plant"
        elif "root" in v_clean or "tuber" in v_clean:
            cat_key = "root"
        elif "leaf" in v_clean or "foliar" in v_clean:
            cat_key = "leaf"
        else:
            cat_key = "leaf"

    data = PHOTO_GUIDANCE_CATALOG.get(cat_key, PHOTO_GUIDANCE_CATALOG["leaf"])
    result = {
        "viewpoint_category": cat_key,
        "crop": crop or "Crop",
        "title_en": data["title_en"],
        "title_kn": data["title_kn"],
        "instructions_en": data["instructions_en"],
        "instructions_kn": data["instructions_kn"],
        "checklist": data["checklist"],
        "lab_test_recommended": data["lab_test_recommended"],
    }
    return result
