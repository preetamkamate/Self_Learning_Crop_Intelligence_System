"""
Nutrient Deficiency & Mobility Reasoning Engine.
Provides agronomic screening hypotheses distinguishing mobile vs immobile nutrient stresses,
soil pH-induced nutrient lockout, and visual mimics of biotic diseases.

Key Agronomic Principles:
1. Mobility in Plant Vascular System:
   - Mobile Nutrients (N, P, K, Mg): Translocated to active growing points; symptoms manifest FIRST on OLDER lower leaves.
   - Immobile Nutrients (Fe, Zn, S, B, Ca, Mn, Cu): Bound in cellular structures; symptoms manifest FIRST on YOUNGEST upper leaves / growing tips.
2. Soil pH Lockout:
   - pH < 5.5 (Acidic): Induces Phosphorus (P), Calcium (Ca), Magnesium (Mg) lockout; potential Al/Mn toxicity.
   - pH 6.0 - 7.2: Optimal nutrient bioavailability for most Indian field crops.
   - pH > 7.5 (Alkaline/Calcareous): Induces Iron (Fe), Zinc (Zn), Manganese (Mn), and Phosphorus (P) lockout.
3. Differential Mimic Differentiation:
   - Distinguishes N chlorosis from waterlogging/root rot.
   - Distinguishes K marginal necrosis from salinity/drought scorch.
   - Distinguishes Fe interveinal chlorosis from viral mosaic.
   - Distinguishes Mg chlorosis from fungal leaf spot.
   - Distinguishes Ca deficiency from Blossom End Rot in Solanaceous crops.
"""
from typing import Any, Dict, List, Optional
import config


# Comprehensive agronomic profiles for key essential nutrients
NUTRIENT_DATABASE = {
    "N": {
        "name": "Nitrogen (N)",
        "name_kannada": "ಸಾರಜನಕ (Nitrogen)",
        "mobility": "mobile",
        "affected_tissue": "older_lower_leaves",
        "visual_symptoms": (
            "General, uniform pale-yellowing (chlorosis) starting at the tip and progressing along the midrib of older lower leaves. "
            "Stunted plant growth, thin spindly stems, and premature leaf drop. Leaves do not show mottled or mosaic patterns."
        ),
        "visual_symptoms_kn": (
            "ಕೆಳಭಾಗದ ಹಳೆಯ ಎಲೆಗಳು ತುದಿಯಿಂದ ಪ್ರಾರಂಭವಾಗಿ ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ. ಗಿಡದ ಬೆಳವಣಿಗೆ ಕುಂಠಿತವಾಗುತ್ತದೆ ಮತ್ತು ಕಾಂಡವು ತೆಳುವಾಗುತ್ತದೆ."
        ),
        "differential_mimics": [
            "Root rot / Waterlogging (check for waterlogged soil and root discoloration)",
            "Sulfur deficiency (S symptoms appear on younger upper leaves first, unlike N)",
            "Senescence (natural aging of the oldest solitary leaf)"
        ],
        "sau_corrective_guideline": (
            "Soil application of Urea or Ammonium Sulphate based on Soil Health Card recommendation. "
            "For immediate rescue, foliar spray of 1.0 - 2.0% Urea (10-20 g/L) or 19:19:19 soluble fertilizer @ 5 g/L during vegetative stage."
        ),
        "sau_corrective_guideline_kn": (
            "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಪ್ರಕಾರ ಯೂರಿಯಾ ಗೊಬ್ಬರ ನೀಡಿ. ತುರ್ತು ಪರಿಹಾರಕ್ಕಾಗಿ 1-2% ಯೂರಿಯಾ (10-20 ಗ್ರಾಂ/ಲೀಟರ್) ಅಥವಾ 19:19:19 ಕರಗುವ ಗೊಬ್ಬರವನ್ನು (5 ಗ್ರಾಂ/ಲೀಟರ್) ಎಲೆಗಳ ಮೇಲೆ ಸಿಂಪಡಿಸಿ."
        ),
    },
    "P": {
        "name": "Phosphorus (P)",
        "name_kannada": "ರಂಜಕ (Phosphorus)",
        "mobility": "mobile",
        "affected_tissue": "older_lower_leaves",
        "visual_symptoms": (
            "Dark green foliage turning dull bronze, purple, or reddish-violet pigmentation on leaf margins and undersides of older leaves. "
            "Poor root development, delay in flowering, and severe stunting."
        ),
        "visual_symptoms_kn": (
            "ಹಳೆಯ ಎಲೆಗಳ ಹಿಂಭಾಗ ಹಾಗೂ ಅಂಚುಗಳು ನೇರಳೆ ಅಥವಾ ಕೆಂಪು ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ. ಬೇರುಗಳ ಬೆಳವಣಿಗೆ ಕುಂಠಿತವಾಗಿ ಹೂ ಬಿಡುವುದು ವಿಳಂಬವಾಗುತ್ತದೆ."
        ),
        "differential_mimics": [
            "Cold weather stress (induces transient anthocyanin purple pigmentation in maize/tomato)",
            "Root-knot nematode injury restricting phosphorus uptake",
            "Acidic soil lockout (pH < 5.5 fixes phosphorus with iron/aluminum oxides)"
        ],
        "sau_corrective_guideline": (
            "Basal soil application of Single Super Phosphate (SSP) or DAP. "
            "For foliar correction, spray 12:61:0 (Mono Ammonium Phosphate) @ 5 g/L or 0:52:34 (MKP) @ 5 g/L."
        ),
        "sau_corrective_guideline_kn": (
            "ಡಿಎಪಿ (DAP) ಅಥವಾ ಸಿಂಗಲ್ ಸೂಪರ್ ಫಾಸ್ಫೇಟ್ (SSP) ಅನ್ನು ಬುಡಕ್ಕೆ ಹಾಕಿ. ಸಿಂಪಡಣೆಗೆ 12:61:00 ಅಥವಾ 0:52:34 ಗೊಬ್ಬರವನ್ನು 5 ಗ್ರಾಂ/ಲೀಟರ್ ನೀರಿಗೆ ಬೆರೆಸಿ ಸಿಂಪಡಿಸಿ."
        ),
    },
    "K": {
        "name": "Potassium (K)",
        "name_kannada": "ಪೊಟ್ಯಾಶ್ (Potassium)",
        "mobility": "mobile",
        "affected_tissue": "older_lower_leaves",
        "visual_symptoms": (
            "Marginal chlorosis followed by dry brown scorching/necrosis along outer margins and tips of older lower leaves ('firing' of margins). "
            "Leaves curl downward; weak stems prone to lodging; poor fruit fill and luster."
        ),
        "visual_symptoms_kn": (
            "ಕೆಳಭಾಗದ ಹಳೆಯ ಎಲೆಗಳ ಅಂಚುಗಳು ಸುಟ್ಟಂತೆ ಒಣಗುತ್ತವೆ (Marginal Scorch). ಕಾಂಡ ದುರ್ಬಲವಾಗಿ ಬಾಗಬಹುದು ಮತ್ತು ಕಾಯಿ ಕಟ್ಟುವುದು ಕಡಿಮೆಯಾಗುತ್ತದೆ."
        ),
        "differential_mimics": [
            "Salinity / Chloride injury (induces leaf margin burn)",
            "Drought stress / Desiccation (marginal leaf drying)",
            "Bacterial leaf blight (check for wavy water-soaked margins)"
        ],
        "sau_corrective_guideline": (
            "Apply Muriate of Potash (MOP) or Sulfate of Potash (SOP) to soil. "
            "For rapid canopy recovery, foliar spray Potassium Nitrate (13:0:45) @ 5-10 g/L or Potassium Sulfate (0:0:50) @ 5 g/L at fruit development stage."
        ),
        "sau_corrective_guideline_kn": (
            "ಬುಡಕ್ಕೆ ಪೊಟ್ಯಾಶ್ (MOP) ಗೊಬ್ಬರ ನೀಡಿ. ತುರ್ತಾಗಿ ಪೊಟ್ಯಾಸಿಯಮ್ ನೈಟ್ರೇಟ್ (13:0:45) 5-10 ಗ್ರಾಂ/ಲೀಟರ್ ನೀರಿಗೆ ಬೆರೆಸಿ ಎಲೆಗಳ ಮೇಲೆ ಸಿಂಪಡಿಸಿ."
        ),
    },
    "Mg": {
        "name": "Magnesium (Mg)",
        "name_kannada": "ಮೆಗ್ನೀಸಿಯಮ್ (Magnesium)",
        "mobility": "mobile",
        "affected_tissue": "older_lower_leaves",
        "visual_symptoms": (
            "Interveinal chlorosis on older leaves where main veins remain dark green while area between veins turns yellow, bronze, or orange. "
            "An inverted green 'V' pattern often remains at the leaf base. Common in tomatoes, citrus, and maize."
        ),
        "visual_symptoms_kn": (
            "ಹಳೆಯ ಎಲೆಗಳ ನಾಳಗಳ ಮಧ್ಯಭಾಗ ಹಳದಿಯಾಗಿದ್ದು, ಮುಖ್ಯ ನಾಳಗಳು ಹಸಿರಾಗಿರುತ್ತವೆ (Interveinal Chlorosis). ಎಲೆಯ ಬುಡದಲ್ಲಿ ತಲೆಕೆಳಗಾದ ಹಸಿರು 'V' ಆಕಾರ ಉಳಿಯುತ್ತದೆ."
        ),
        "differential_mimics": [
            "Spider mite stippling (inspect underside with 10x lens for webbing)",
            "Fungal leaf spot (Alternaria/Septoria produce distinct concentric dark rings, not diffuse interveinal yellowing)"
        ],
        "sau_corrective_guideline": (
            "Soil application of Agricultural Magnesium Sulfate (Epsom salt) @ 10-15 kg/acre. "
            "Foliar spray of Magnesium Sulfate (MgSO4) @ 5 g/L (0.5%) twice at 15-day intervals."
        ),
        "sau_corrective_guideline_kn": (
            "ಎಕರೆಯ ಮಣ್ಣಿಗೆ 10-15 ಕೆಜಿ ಮೆಗ್ನೀಸಿಯಮ್ ಸಲ್ಫೇಟ್ (ಎಪ್ಸಮ್ ಉಪ್ಪು) ನೀಡಿ. ಅಥವಾ ಮೆಗ್ನೀಸಿಯಮ್ ಸಲ್ಫೇಟ್ 5 ಗ್ರಾಂ/ಲೀಟರ್ ನೀರಿಗೆ ಬೆರೆಸಿ 15 ದಿನಗಳ ಅಂತರದಲ್ಲಿ ಸಿಂಪಡಿಸಿ."
        ),
    },
    "Fe": {
        "name": "Iron (Fe)",
        "name_kannada": "ಕಬ್ಬಿಣದ ಅಂಶ (Iron)",
        "mobility": "immobile",
        "affected_tissue": "young_upper_leaves",
        "visual_symptoms": (
            "Striking interveinal chlorosis on the YOUNGEST terminal leaves. Interveinal tissue turns creamy-yellow to bleach-white while even fine veins remain starkly dark green. "
            "Under severe conditions, the entire young shoot bleaches white with tip dieback. Highly prevalent in alkaline/calcareous soils (pH > 7.5)."
        ),
        "visual_symptoms_kn": (
            "ಗಿಡದ ತುದಿಯ ಅತ್ಯಂತ ಕಿರಿಯ ಎಲೆಗಳ ನಾಳಗಳು ಹಸಿರಾಗಿದ್ದು ಮಧ್ಯದ ಭಾಗ ಸಂಪೂರ್ಣ ಬಿಳಿ-ಹಳದಿಯಾಗುತ್ತದೆ. ಕ್ಷಾರೀಯ (Alkaline pH > 7.5) ಮಣ್ಣಿನಲ್ಲಿ ಇದು ಹೆಚ್ಚು."
        ),
        "differential_mimics": [
            "Viral mosaic disease (mosaic causes mottled leaf distortion, whereas iron chlorosis is uniform geometric interveinal netting)",
            "Zinc deficiency (Zn chlorosis shows broader yellow bands and smaller leaf size)"
        ],
        "sau_corrective_guideline": (
            "Foliar spray of Ferrous Sulfate (FeSO4) @ 5 g/L (0.5%) mixed with Citric Acid @ 1 g/L (or 2-3 drops lemon juice) to prevent iron precipitation. "
            "Alternatively, use Chelated Iron (Fe-EDTA 12%) @ 1.0 - 1.5 g/L."
        ),
        "sau_corrective_guideline_kn": (
            "ಕಬ್ಬಿಣದ ಸಲ್ಫೇಟ್ (FeSO4) 5 ಗ್ರಾಂ ಮತ್ತು ಸಿಟ್ರಿಕ್ ಆಮ್ಲ 1 ಗ್ರಾಂ (ಅಥವಾ ನಿಂಬೆ ರಸ) ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ ಬೆರೆಸಿ ಸಿಂಪಡಿಸಿ. ಅಥವಾ Fe-EDTA (ಚೆಲೇಟೆಡ್ ಐರನ್) 1-1.5 ಗ್ರಾಂ/ಲೀಟರ್ ಸಿಂಪಡಿಸಿ."
        ),
    },
    "Zn": {
        "name": "Zinc (Zn)",
        "name_kannada": "ಸತು (Zinc)",
        "mobility": "immobile",
        "affected_tissue": "young_upper_leaves",
        "visual_symptoms": (
            "Broad bleached or light yellow bands on both sides of the midrib in young leaves (known as 'White Bud' in maize). "
            "Shortened internodes resulting in a rosetted, bushy bunching of small distorted leaves ('Little Leaf' symptom in citrus and cotton)."
        ),
        "visual_symptoms_kn": (
            "ಮೆಕ್ಕೆಜೋಳದಲ್ಲಿ ಎಲೆಯ ನಾಳದ ಪಕ್ಕ ಬಿಳಿ ಪಟ್ಟಿಗಳು ಕಾಣುತ್ತವೆ (White Bud). ಕಿರಿಯ ಎಲೆಗಳು ಚಿಕ್ಕದಾಗಿ ಗುಂಪಾಗಿ ಬೆಳೆಯುತ್ತವೆ (Little Leaf)."
        ),
        "differential_mimics": [
            "Herbicide injury (bleaching from pigment inhibitors)",
            "Mycoplasma Little Leaf disease (transmissible via jassid vector; requires insecticidal control)"
        ],
        "sau_corrective_guideline": (
            "Soil application of Zinc Sulfate (ZnSO4 21%) @ 10 kg/acre before sowing/transplanting. "
            "Foliar spray of Zinc Sulfate @ 2.5 - 3.0 g/L neutralized with 1.25 g/L slaked lime, or Chelated Zn-EDTA (12%) @ 1.0 g/L."
        ),
        "sau_corrective_guideline_kn": (
            "ನಾಟಿಗೆ ಮುನ್ನ ಎಕರೆಗೆ 10 ಕೆಜಿ ಸತು ಸಲ್ಫೇಟ್ (ZnSO4) ಮಣ್ಣಿಗೆ ನೀಡಿ. ತುರ್ತು ಪರಿಹಾರಕ್ಕೆ ಸತು ಸಲ್ಫೇಟ್ 3 ಗ್ರಾಂ + 1.5 ಗ್ರಾಂ ಸುಣ್ಣ ಅಥವಾ Zn-EDTA 1 ಗ್ರಾಂ/ಲೀಟರ್ ನೀರಿಗೆ ಬೆರೆಸಿ ಸಿಂಪಡಿಸಿ."
        ),
    },
    "Ca": {
        "name": "Calcium (Ca)",
        "name_kannada": "ಕ್ಯಾಲ್ಸಿಯಂ (Calcium)",
        "mobility": "immobile",
        "affected_tissue": "growing_tips_and_fruits",
        "visual_symptoms": (
            "Distortion, hooking, and necrosis of young growing tips and leaf margins. "
            "Blossom End Rot in tomato, pepper, and watermelon: dark, sunken, leathery water-soaked lesions on the blossom end of the fruit."
        ),
        "visual_symptoms_kn": (
            "ಗಿಡದ ತುದಿಯ ಮೊಗ್ಗುಗಳು ಒಣಗುತ್ತವೆ. ಟೊಮೇಟೊ ಮತ್ತು ಮೆಣಸಿನ ಹಣ್ಣಿನ ಕೆಳಭಾಗದಲ್ಲಿ ಕಪ್ಪು, ಚಪ್ಪಟೆಯಾದ ಕಲೆಗಳು ಉಂಟಾಗುತ್ತವೆ (Blossom End Rot)."
        ),
        "differential_mimics": [
            "Fruit Anthracnose / Sunscald (sunscald appears on the sun-exposed upper shoulder, while Blossom End Rot is strictly on the bottom blossom scar)",
            "Irregular watering (wide swings in soil moisture inhibit calcium flow through xylem)"
        ],
        "sau_corrective_guideline": (
            "Maintain consistent soil moisture through regular drip irrigation. Avoid excessive ammonium nitrogen fertilizers. "
            "Foliar spray of Calcium Nitrate @ 4-5 g/L or Chelated Calcium (Ca-EDTA) @ 1.0 g/L directed at flowering clusters and young fruit."
        ),
        "sau_corrective_guideline_kn": (
            "ಹನಿ ನೀರಾವರಿ ಮೂಲಕ ತೇವಾಂಶವನ್ನು ಸಮನಾಗಿ ಕಾಪಾಡಿ. ಹೂವು ಮತ್ತು ಕಾಯಿ ಕಟ್ಟುವ ಹಂತದಲ್ಲಿ ಕ್ಯಾಲ್ಸಿಯಂ ನೈಟ್ರೇಟ್ (4-5 ಗ್ರಾಂ/ಲೀಟರ್) ಸಿಂಪಡಿಸಿ."
        ),
    },
    "B": {
        "name": "Boron (B)",
        "name_kannada": "ಬೋರಾನ್ (Boron)",
        "mobility": "immobile",
        "affected_tissue": "growing_tips_and_fruits",
        "visual_symptoms": (
            "Death of terminal buds causing prolific lateral branching (witch's broom). Thick, brittle, curled leaves with corky cracked veins. "
            "Fruit cracking, internal brown corkiness, and premature flower/fruit drop."
        ),
        "visual_symptoms_kn": (
            "ತುದಿಯ ಚಿಗುರು ಸಾಯುತ್ತದೆ ಮತ್ತು ಪಕ್ಕದ ರೆಂಬೆಗಳು ಅಧಿಕವಾಗಿ ಬೆಳೆಯುತ್ತವೆ. ಕಾಯಿಗಳು ಬಿರುಕು ಬಿಡುತ್ತವೆ ಮತ್ತು ಎಲೆಗಳು ಸುಲಭವಾಗಿ ಮುರಿಯುತ್ತವೆ."
        ),
        "differential_mimics": [
            "Broad mite damage (causes downward curling and brittleness)",
            "Calcium deficiency (often co-occurs in acidic sandy soils)"
        ],
        "sau_corrective_guideline": (
            "Soil application of Borax @ 4-5 kg/acre. Foliar spray of Solubor (Di-sodium octaborate tetrahydrate 20% B) @ 1.0 - 1.5 g/L prior to flowering."
        ),
        "sau_corrective_guideline_kn": (
            "ಮಣ್ಣಿಗೆ ಬೋರಾಕ್ಸ್ 4-5 ಕೆಜಿ/ಎಕರೆ ನೀಡಿ. ಹೂವಾಡುವ ಮುನ್ನ ಸಾಲ್ಯುಬೋರ್ (Solubor) 1 ಗ್ರಾಂ/ಲೀಟರ್ ನೀರಿಗೆ ಬೆರೆಸಿ ಎಲೆಗಳ ಮೇಲೆ ಸಿಂಪಡಿಸಿ."
        ),
    },
}

SCREENING_DISCLAIMER_EN = (
    "⚠️ **Visual Screening Hypothesis (Not a Confirmed Laboratory Diagnosis)**:\n"
    "Visual foliar symptoms of nutrient deficiencies frequently mimic and overlap with root rot, "
    "nematode infection, herbicide drift, waterlogging, or soil pH lockout. "
    "Before purchasing high-dose fertilizers or micronutrients, confirm through a certified "
    "Soil Health Card test and leaf tissue analysis."
)

SCREENING_DISCLAIMER_KN = (
    "⚠️ **ದೃಶ್ಯ ತಪಾಸಣಾ ಊಹೆ (ಇದು ಪ್ರಯೋಗಾಲಯ ದೃಢೀಕೃತ ವರದಿಯಲ್ಲ)**:\n"
    "ಪೋಷಕಾಂಶಗಳ ಕೊರತೆಯ ಲಕ್ಷಣಗಳು ಬೇರು ಕೊಳೆ ರೋಗ, ಜಂತುಹುಳು ಬಾಧೆ, ಕಳೆನಾಶಕದ ಹೊಡೆತ ಅಥವಾ ಮಣ್ಣಿನ ರಸಸಾರ (pH) ವ್ಯತ್ಯಾಸದ ಲಕ್ಷಣಗಳನ್ನು ಹೋಲುತ್ತವೆ. "
    "ರಾಸಾಯನಿಕ ಗೊಬ್ಬರಗಳನ್ನು ಖರೀದಿಸುವ ಮುನ್ನ ಮಣ್ಣು ಮತ್ತು ಎಲೆಯ ಲ್ಯಾಬ್ ಪರೀಕ್ಷೆ (Soil Health Card) ಮೂಲಕ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ."
)


def evaluate_soil_ph_lockout(soil_ph: Optional[float]) -> Dict[str, Any]:
    """
    Evaluate potential nutrient availability lockout based on soil pH.
    """
    if soil_ph is None:
        return {
            "status": "unknown",
            "message_en": "Soil pH not specified. Optimal range for field crops is 6.0 to 7.2.",
            "message_kn": "ಮಣ್ಣಿನ pH ತಿಳಿಸಿಲ್ಲ. ಸಾಮಾನ್ಯ ಬೆಳೆಗೆ 6.0 ರಿಂದ 7.2 ಸೂಕ್ತ.",
            "locked_out_nutrients": [],
        }

    locked_out = []
    if soil_ph < 5.5:
        locked_out = ["P", "Ca", "Mg"]
        msg_en = (
            f"Acidic soil condition (pH {soil_ph:.1f} < 5.5): Phosphorus, Calcium, and Magnesium "
            "are chemically bound and unavailable to plant roots. Potential Aluminum/Manganese toxicity. "
            "Liming (Agricultural Lime / Dolomite @ 200-400 kg/acre) is recommended before fertilizing."
        )
        msg_kn = (
            f"ಆಮ್ಲೀಯ ಮಣ್ಣು (pH {soil_ph:.1f} < 5.5): ರಂಜಕ, ಕ್ಯಾಲ್ಸಿಯಂ ಮತ್ತು ಮೆಗ್ನೀಸಿಯಮ್ ಲಭ್ಯತೆ ಕಡಿಮೆಯಾಗುತ್ತದೆ. "
            "ಕೃಷಿ ಸುಣ್ಣ ಅಥವಾ ಡಾಲೋಮೈಟ್ ಹಾಕುವುದು ಅತ್ಯಗತ್ಯ."
        )
    elif soil_ph > 7.8:
        locked_out = ["Fe", "Zn", "Mn", "P"]
        msg_en = (
            f"Alkaline/Calcareous soil condition (pH {soil_ph:.1f} > 7.8): Micronutrients (Iron, Zinc, Manganese) "
            "and Phosphorus precipitate as insoluble calcium salts. Foliar chelated sprays (Fe-EDTA, Zn-EDTA) "
            "and soil gypsum / organic compost application are strongly recommended."
        )
        msg_kn = (
            f"ಕ್ಷಾರೀಯ ಮಣ್ಣು (pH {soil_ph:.1f} > 7.8): ಕಬ್ಬಿಣ (Fe) ಮತ್ತು ಸತು (Zn) ಲಭ್ಯತೆ ಕುಂಠಿತವಾಗುತ್ತದೆ. "
            "ಚೆಲೇಟೆಡ್ (EDTA) ಗೊಬ್ಬರಗಳನ್ನು ಎಲೆಗಳ ಮೇಲೆ ಸಿಂಪಡಿಸುವುದು ಮತ್ತು ಸಾವಯವ ಗೊಬ್ಬರ ನೀಡುವುದು ಉತ್ತಮ."
        )
    else:
        msg_en = f"Optimal soil pH ({soil_ph:.1f}): Nutrients are bio-available under adequate moisture conditions."
        msg_kn = f"ಉತ್ತಮ ಮಣ್ಣಿನ pH ({soil_ph:.1f}): ಪೋಷಕಾಂಶಗಳ ಲಭ್ಯತೆಗೆ ಸೂಕ್ತವಾಗಿದೆ."

    return {
        "status": "evaluated",
        "soil_ph": soil_ph,
        "locked_out_nutrients": locked_out,
        "message_en": msg_en,
        "message_kn": msg_kn,
    }


def analyze_nutrient_symptoms(
    affected_part: str = "leaf",
    plant_region: str = "lower",  # "lower", "upper", "whole_canopy", "fruit"
    observed_signs: Optional[str] = None,
    crop_species: Optional[str] = None,
    soil_ph: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Perform agronomic screening hypothesis based on plant tissue location, mobility rules, and pH.
    """
    ph_eval = evaluate_soil_ph_lockout(soil_ph)
    hypotheses = []

    # Mobility reasoning
    is_lower = "lower" in plant_region.lower() or "old" in plant_region.lower()
    is_upper = "upper" in plant_region.lower() or "young" in plant_region.lower() or "shoot" in plant_region.lower()
    is_fruit = "fruit" in affected_part.lower()

    if is_fruit:
        hypotheses.append("Ca")
        hypotheses.append("B")
    elif is_lower:
        # Mobile elements
        hypotheses.extend(["N", "K", "Mg", "P"])
    elif is_upper:
        # Immobile elements
        hypotheses.extend(["Fe", "Zn", "B", "Ca"])
    else:
        # Generalized canopy or undetermined
        hypotheses.extend(["N", "Fe", "Zn", "K", "Mg"])

    # Rank hypotheses giving boost to pH-locked elements
    ranked = []
    for nut in hypotheses:
        rec = NUTRIENT_DATABASE.get(nut)
        if not rec:
            continue
        boost = 1.3 if nut in ph_eval["locked_out_nutrients"] else 1.0
        ranked.append({
            "code": nut,
            "name": rec["name"],
            "name_kannada": rec["name_kannada"],
            "mobility": rec["mobility"],
            "affected_tissue": rec["affected_tissue"],
            "visual_symptoms": rec["visual_symptoms"],
            "visual_symptoms_kn": rec["visual_symptoms_kn"],
            "differential_mimics": rec["differential_mimics"],
            "sau_guideline": rec["sau_corrective_guideline"],
            "sau_guideline_kn": rec["sau_corrective_guideline_kn"],
            "ph_lockout_risk": nut in ph_eval["locked_out_nutrients"],
        })

    return {
        "status": "screening_hypothesis",
        "affected_tissue_location": plant_region,
        "ph_evaluation": ph_eval,
        "disclaimer_en": SCREENING_DISCLAIMER_EN,
        "disclaimer_kn": SCREENING_DISCLAIMER_KN,
        "recommended_lab_tests": [
            "Soil Health Card (Available N, P2O5, K2O, Organic Carbon %)",
            "Soil pH and Electrical Conductivity (EC)",
            "DTPA-extractable micronutrients (Zn, Fe, Mn, Cu, Hot-water soluble B)",
            "Representative Leaf Tissue Testing (standard 3rd-4th petiole sample from active canopy)"
        ],
        "candidate_deficiencies": ranked,
    }
