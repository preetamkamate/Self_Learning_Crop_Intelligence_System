"""
ICAR and IIHR Knowledge Base Retriever for Crop Intelligence System.

Maps multimodal vision and agronomic queries across 5 authoritative datasets:
1. PlantVillage (Corn)
2. FGVC8 (Apple)
3. Cassava Leaf Disease
4. ICAR Research
5. IIHR Advisory
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

SUPPORTED_DATASETS: tuple[str, ...] = (
    "PlantVillage (Corn)",
    "FGVC8 (Apple)",
    "Cassava Leaf Disease",
    "ICAR Research",
    "IIHR Advisory",
)


@dataclass(frozen=True)
class KnowledgeEntry:
    """Structured agronomic knowledge entry grounded to one of the 5 taxonomies."""

    id: str
    dataset: str
    crop: str
    condition: str
    pathogen_or_cause: str
    symptoms: str
    chemical_solution: str
    organic_solution: str
    next_steps: str
    keywords: tuple[str, ...]
    source: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation."""
        return asdict(self)


KNOWLEDGE_BASE: tuple[KnowledgeEntry, ...] = (
    # -------------------------------------------------------------------------
    # 1. PlantVillage (Corn)
    # -------------------------------------------------------------------------
    KnowledgeEntry(
        id="pv_corn_gls",
        dataset="PlantVillage (Corn)",
        crop="Corn (Maize)",
        condition="Cercospora Leaf Spot (Gray Leaf Spot)",
        pathogen_or_cause=(
            "Fungal infection by Cercospora zeae-maydis favored by warm, humid weather, "
            "prolonged leaf wetness, and infected crop residue."
        ),
        symptoms="Rectangular tan-to-gray lesions bounded by parallel leaf veins on lower foliage.",
        chemical_solution=(
            "Apply Azoxystrobin + Propiconazole or Pyraclostrobin at early tasseling (VT/R1) "
            "as per label dosage when disease reaches ear-leaf threshold."
        ),
        organic_solution=(
            "Spray Neem oil (Azadirachtin 1500 ppm at 3–5 mL/L) or Bacillus subtilis bio-fungicide, "
            "and incorporate crop residue via deep tillage."
        ),
        next_steps=(
            "Scout upper canopy every 5–7 days, rotate away from continuous corn for 1–2 seasons, "
            "and select Gray Leaf Spot-tolerant hybrids."
        ),
        keywords=("corn", "maize", "gray leaf spot", "cercospora", "rectangular", "lesion", "plantvillage"),
        source="PlantVillage Corn Taxonomy & ICAR-IIMR Maize Pathology Guidelines",
    ),
    KnowledgeEntry(
        id="pv_corn_rust",
        dataset="PlantVillage (Corn)",
        crop="Corn (Maize)",
        condition="Common Rust",
        pathogen_or_cause=(
            "Puccinia sorghi fungus spread by windborne urediniospores under cool-to-moderate "
            "temperatures (16–23°C) and high relative humidity."
        ),
        symptoms="Cinnamon-brown, powdery pustules scattered across both upper and lower leaf surfaces.",
        chemical_solution=(
            "Apply Mancozeb 75% WP (2.5 g/L) or Tebuconazole / Azoxystrobin if pustules cover "
            "more than 10% of upper leaves before silking."
        ),
        organic_solution=(
            "Use sulfur-based wettable powder (2 g/L) or Pseudomonas fluorescens foliar spray "
            "during early pustule appearance."
        ),
        next_steps=(
            "Monitor field edges for windborne spore progression, avoid overhead evening irrigation, "
            "and record hybrid susceptibility for next planting."
        ),
        keywords=("corn", "maize", "common rust", "puccinia", "pustule", "brown", "rust", "plantvillage"),
        source="PlantVillage Corn Taxonomy & ICAR-IIMR Advisory",
    ),
    KnowledgeEntry(
        id="pv_corn_nclb",
        dataset="PlantVillage (Corn)",
        crop="Corn (Maize)",
        condition="Northern Corn Leaf Blight",
        pathogen_or_cause=(
            "Exserohilum turcicum fungal pathogen surviving in maize stubble and thriving "
            "under moderate temperatures with heavy dew or rain."
        ),
        symptoms="Long, cigar-shaped elliptical grayish-green to tan lesions (2.5–15 cm) on leaves.",
        chemical_solution=(
            "Spray Propiconazole 25% EC (1 mL/L) or Azoxystrobin + Difenoconazole at initial "
            "cigar-shaped lesion detection."
        ),
        organic_solution=(
            "Apply Trichoderma viride or Bacillus amyloliquefaciens foliar bio-formulation and "
            "remove heavily infected lower leaves."
        ),
        next_steps=(
            "Avoid late-season nitrogen excess, manage weeds, and rotate with legumes or pulses."
        ),
        keywords=("corn", "maize", "northern leaf blight", "cigar", "exserohilum", "blight", "plantvillage"),
        source="PlantVillage Corn Taxonomy & ICAR-IIMR",
    ),
    KnowledgeEntry(
        id="pv_corn_healthy",
        dataset="PlantVillage (Corn)",
        crop="Corn (Maize)",
        condition="Healthy Corn Foliage",
        pathogen_or_cause="No foliar pathogen detected; chlorophyll distribution and leaf turgor are normal.",
        symptoms="Uniform deep-green leaf blades without necrotic spots, pustules, or water-soaked lesions.",
        chemical_solution="No fungicide or pesticide spray required; avoid unnecessary chemical applications.",
        organic_solution=(
            "Maintain soil organic matter with compost or vermicompost and apply foliar sea-weed "
            "or humic supplements only if growth lags."
        ),
        next_steps=(
            "Continue balanced NPK top-dressing at knee-high and tasseling stages and scout weekly."
        ),
        keywords=("corn", "maize", "healthy", "green", "normal", "plantvillage"),
        source="PlantVillage Corn Taxonomy",
    ),
    # -------------------------------------------------------------------------
    # 2. FGVC8 (Apple)
    # -------------------------------------------------------------------------
    KnowledgeEntry(
        id="fgvc8_apple_scab",
        dataset="FGVC8 (Apple)",
        crop="Apple",
        condition="Apple Scab",
        pathogen_or_cause=(
            "Ascomycete fungus Venturia inaequalis overwintering in fallen leaf litter and releasing "
            "ascospores during cool spring rains."
        ),
        symptoms="Olive-green to dark velvety spots on leaves and scabby corky lesions on fruit.",
        chemical_solution=(
            "Apply protective Mancozeb 75% WP (2.5 g/L) or Captan 50% WP (2 g/L), rotating with "
            "Difenoconazole or Myclobutanil during primary infection windows."
        ),
        organic_solution=(
            "Spray wettable sulfur (2–3 g/L) or copper oxychloride during early bud break; "
            "apply 5% urea to fallen leaves in autumn to speed decomposition."
        ),
        next_steps=(
            "Rake and destroy fallen orchard leaves, prune canopy for airflow, and follow wet-period spray alerts."
        ),
        keywords=("apple", "scab", "venturia", "velvety", "olive", "fgvc8"),
        source="FGVC8 Plant Pathology Taxonomy & ICAR-CITH Temperate Horticulture",
    ),
    KnowledgeEntry(
        id="fgvc8_apple_black_rot",
        dataset="FGVC8 (Apple)",
        crop="Apple",
        condition="Black Rot / Frogeye Leaf Spot",
        pathogen_or_cause=(
            "Botryosphaeria obtusa fungus entering through pruning wounds, dead wood, or insect damage "
            "during warm, humid periods."
        ),
        symptoms="Purple-bordered tan circular leaf spots ('frogeye') and firm concentric black fruit rot.",
        chemical_solution=(
            "Spray Captan 50% WP (2 g/L) or Thiophanate-methyl / Pyraclostrobin + Boscalid "
            "from petal fall through cover sprays."
        ),
        organic_solution=(
            "Prune out mummified fruits and cankered twigs; apply Bordeaux mixture (1%) on dormant wood."
        ),
        next_steps=(
            "Sanitize pruning shears between trees with 70% alcohol and remove all mummified fruit from canopy."
        ),
        keywords=("apple", "black rot", "frogeye", "botryosphaeria", "canker", "fgvc8"),
        source="FGVC8 Plant Pathology Taxonomy & ICAR-CITH",
    ),
    KnowledgeEntry(
        id="fgvc8_apple_cedar_rust",
        dataset="FGVC8 (Apple)",
        crop="Apple",
        condition="Cedar Apple Rust",
        pathogen_or_cause=(
            "Heteroecious fungus Gymnosporangium juniperi-virginianae cycling between nearby "
            "juniper/cedar hosts and apple foliage."
        ),
        symptoms="Bright yellow-orange spots on upper apple leaves with raised cup-like aecia underneath.",
        chemical_solution=(
            "Apply Myclobutanil or Mancozeb at pink-bud to 2 weeks post-petal fall before spore "
            "germination peaks."
        ),
        organic_solution=(
            "Use wettable sulfur or Neem oil + Bacillus subtilis preventive sprays during spring spore release."
        ),
        next_steps=(
            "Inspect surrounding windbreaks for cedar galls and prune them before spring rain."
        ),
        keywords=("apple", "rust", "cedar apple rust", "orange", "yellow", "gymnosporangium", "fgvc8"),
        source="FGVC8 Plant Pathology Taxonomy",
    ),
    KnowledgeEntry(
        id="fgvc8_apple_powdery_mildew",
        dataset="FGVC8 (Apple)",
        crop="Apple",
        condition="Powdery Mildew (and Complex Foliar Stress)",
        pathogen_or_cause=(
            "Podosphaera leucotricha thriving in warm days and high humidity without requiring free water."
        ),
        symptoms="White powdery fungal coating on young shoots, leaf curling, and russeting on fruit.",
        chemical_solution=(
            "Apply Hexaconazole 5% EC (1 mL/L) or Trifloxystrobin + Tebuconazole at tight-cluster to petal fall."
        ),
        organic_solution=(
            "Spray potassium bicarbonate (5 g/L) or wettable sulfur (2 g/L) plus horticultural Neem oil."
        ),
        next_steps=(
            "Prune silvered terminal shoots during dormancy and avoid excessive nitrogen flush."
        ),
        keywords=("apple", "powdery mildew", "white", "complex", "multiple", "fgvc8"),
        source="FGVC8 Plant Pathology Taxonomy & ICAR-CITH",
    ),
    # -------------------------------------------------------------------------
    # 3. Cassava Leaf Disease
    # -------------------------------------------------------------------------
    KnowledgeEntry(
        id="cassava_cmd",
        dataset="Cassava Leaf Disease",
        crop="Cassava",
        condition="Cassava Mosaic Disease (CMD)",
        pathogen_or_cause=(
            "Begomoviruses (Cassava mosaic viruses) transmitted by whitefly vectors (Bemisia tabaci) "
            "and propagated via infected stem cuttings."
        ),
        symptoms="Distinctive yellow-green mosaic mottling, leaf distortion, and stunted plant growth.",
        chemical_solution=(
            "No chemical cures the virus directly; manage whitefly vectors with Imidacloprid 17.8% SL "
            "(0.3 mL/L) or Thiamethoxam 25% WG (0.25 g/L) only when vector pressure is high."
        ),
        organic_solution=(
            "Rogue out infected plants early, install yellow sticky traps (15–20/ha), and spray "
            "Neem oil (3 mL/L) or Verticillium lecanii against whiteflies."
        ),
        next_steps=(
            "Source certified virus-free stem cuttings (e.g., ICAR-CTCRI resistant lines) for next cycle."
        ),
        keywords=("cassava", "mosaic", "cmd", "whitefly", "mottling", "distortion", "virus"),
        source="Cassava Leaf Disease Taxonomy & ICAR-CTCRI Tuber Crops Advisory",
    ),
    KnowledgeEntry(
        id="cassava_cbb",
        dataset="Cassava Leaf Disease",
        crop="Cassava",
        condition="Cassava Bacterial Blight (CBB)",
        pathogen_or_cause=(
            "Bacterium Xanthomonas phaseoli pv. manihotis spread via rain splash, contaminated tools, "
            "and infected planting stakes."
        ),
        symptoms="Angular water-soaked leaf spots turning necrotic brown, wilting, and stem gum exudation.",
        chemical_solution=(
            "Apply Copper Oxychloride 50% WP (2.5 g/L) or Copper Hydroxide to slow secondary spread; "
            "dip healthy stakes in copper solution prior to planting."
        ),
        organic_solution=(
            "Remove and burn blighted leaves/stems, treat cuttings with Pseudomonas fluorescens talc "
            "formulation (10 g/L), and improve field drainage."
        ),
        next_steps=(
            "Disinfect machetes/knives between plants, rotate with non-host crops, and plant at onset of rains."
        ),
        keywords=("cassava", "bacterial blight", "cbb", "angular", "water-soaked", "wilt"),
        source="Cassava Leaf Disease Taxonomy & ICAR-CTCRI",
    ),
    KnowledgeEntry(
        id="cassava_cbsd",
        dataset="Cassava Leaf Disease",
        crop="Cassava",
        condition="Cassava Brown Streak Disease (CBSD)",
        pathogen_or_cause=(
            "Ipomoviruses transmitted by Bemisia tabaci whiteflies and infected stem cuttings."
        ),
        symptoms="Feathery yellow chlorosis along secondary/tertiary leaf veins and dry brown root necrosis.",
        chemical_solution=(
            "No curative viricide exists; control whitefly vectors with Acetamiprid or Imidacloprid "
            "during early establishment if thresholds are exceeded."
        ),
        organic_solution=(
            "Uproot and destroy symptomatic plants immediately; use Neem-based anti-feedants and "
            "certified clean stakes."
        ),
        next_steps=(
            "Inspect roots at harvest for dry brown corky lesions and isolate clean seed-nursery blocks."
        ),
        keywords=("cassava", "brown streak", "cbsd", "vein", "chlorosis", "necrosis"),
        source="Cassava Leaf Disease Taxonomy",
    ),
    KnowledgeEntry(
        id="cassava_cgm",
        dataset="Cassava Leaf Disease",
        crop="Cassava",
        condition="Cassava Green Mottle / Green Mite Damage (CGM)",
        pathogen_or_cause=(
            "Cassava green mite (Mononychellus tanajoa) feeding or viral mottling aggravated by "
            "prolonged dry spells."
        ),
        symptoms="Fine chlorotic pinprick speckling, puckered apical leaves, and shoot-tip stunting.",
        chemical_solution=(
            "Apply Abamectin 1.9% EC (0.5 mL/L) or Spiromesifen / Wettable Sulfur when mite counts "
            "exceed economic threshold on young shoots."
        ),
        organic_solution=(
            "Spray Neem Seed Kernel Extract (NSKE 5%) or release predatory phytoseiid mites; "
            "maintain adequate soil moisture."
        ),
        next_steps=(
            "Monitor apical shoots weekly during dry weather and avoid dusty road-edge stress."
        ),
        keywords=("cassava", "green mottle", "cgm", "mite", "speckling", "puckered"),
        source="Cassava Leaf Disease Taxonomy",
    ),
    # -------------------------------------------------------------------------
    # 4. ICAR Research
    # -------------------------------------------------------------------------
    KnowledgeEntry(
        id="icar_rice_blast",
        dataset="ICAR Research",
        crop="Rice (Paddy)",
        condition="Rice Blast & Bacterial Leaf Blight",
        pathogen_or_cause=(
            "Magnaporthe oryzae (blast fungus) or Xanthomonas oryzae (BLB), triggered by excess "
            "nitrogen, high humidity, and cloudy weather."
        ),
        symptoms="Spindle/diamond-shaped lesions with gray centers (blast) or wavy yellow leaf margins (BLB).",
        chemical_solution=(
            "For blast: Tricyclazole 75% WP (0.6 g/L) or Isoprothiolane 40% EC (1.5 mL/L). "
            "For BLB: Copper Oxychloride (2.5 g/L) + Streptocycline (as per ICAR recommendation)."
        ),
        organic_solution=(
            "Spray Pseudomonas fluorescens (10 g/L) or Neemazal, drain excess standing water for 2–3 days, "
            "and split nitrogen applications."
        ),
        next_steps=(
            "Avoid excessive urea top-dressing, maintain 2–3 cm shallow water, and adopt ICAR resistant varieties."
        ),
        keywords=("rice", "paddy", "blast", "bacterial leaf blight", "spindle", "icar", "nitrogen"),
        source="ICAR-NRRI / ICAR-IIRR Package of Practices",
    ),
    KnowledgeEntry(
        id="icar_maize_faw",
        dataset="ICAR Research",
        crop="Maize (Corn)",
        condition="Fall Armyworm (Spodoptera frugiperda)",
        pathogen_or_cause=(
            "Invasive lepidopteran larvae feeding inside maize whorls and damaging developing cobs."
        ),
        symptoms="Window-pane leaf scraping, ragged whorl feeding, and abundant wet sawdust-like frass.",
        chemical_solution=(
            "Apply Emamectin Benzoate 5% SG (0.4 g/L) or Chlorantraniliprole 18.5% SC (0.4 mL/L) "
            "directed directly into the central leaf whorl."
        ),
        organic_solution=(
            "Install pheromone traps (10–15/ha), apply Metarhizium anisopliae / Bacillus thuringiensis "
            "(2 g/L), or place sand + lime (9:1) in whorls at early instar stage."
        ),
        next_steps=(
            "Scout 20 plants per plot twice weekly during vegetative V2–V8 stages and conserve natural parasitoids."
        ),
        keywords=("maize", "corn", "fall armyworm", "worm", "whorl", "frass", "icar", "pest"),
        source="ICAR-NBAIR & ICAR-IIMR Fall Armyworm IPM Advisory",
    ),
    KnowledgeEntry(
        id="icar_soil_npk",
        dataset="ICAR Research",
        crop="General Field Crops",
        condition="Soil Nutrient & NPK / pH Fertility Management",
        pathogen_or_cause=(
            "Suboptimal Nitrogen, Phosphorus, Potassium balance or soil pH outside 6.0–7.5 "
            "restricting root nutrient uptake and increasing susceptibility to foliar stress."
        ),
        symptoms="Uniform pale chlorosis (N deficiency), purplish lower leaves (P), or scorched margins (K).",
        chemical_solution=(
            "Follow ICAR Soil Health Card dose: split Urea (N) into 2–3 splits, apply DAP/SSP (P) "
            "and MOP (K) basally; amend pH < 5.5 with agricultural lime or pH > 8.2 with gypsum."
        ),
        organic_solution=(
            "Incorporate 5–10 t/ha well-rotted FYM/compost, Azotobacter/Rhizobium biofertilizers, "
            "Phosphate Solubilizing Bacteria (PSB), and green manuring (Dhaincha/Sunhemp)."
        ),
        next_steps=(
            "Conduct soil testing before every Kharif/Rabi season and adjust top-dressing via leaf color chart."
        ),
        keywords=("soil", "npk", "nitrogen", "phosphorus", "potassium", "ph", "deficiency", "fertilizer", "icar"),
        source="ICAR Soil Health Card & Nutrient Management Framework",
    ),
    # -------------------------------------------------------------------------
    # 5. IIHR Advisory
    # -------------------------------------------------------------------------
    KnowledgeEntry(
        id="iihr_tomato_blight_curl",
        dataset="IIHR Advisory",
        crop="Tomato",
        condition="Tomato Early/Late Blight & Leaf Curl Management",
        pathogen_or_cause=(
            "Alternaria solani / Phytophthora infestans (blights) or Tomato Leaf Curl Virus (ToLCV) "
            "transmitted by whiteflies."
        ),
        symptoms="Concentric target-ring brown spots on lower leaves (early blight) or upward leaf curling.",
        chemical_solution=(
            "For blight: Mancozeb (2.5 g/L) or Azoxystrobin (1 mL/L). For whitefly/ToLCV vector: "
            "Imidacloprid 17.8% SL (0.3 mL/L) or Spiromesifen."
        ),
        organic_solution=(
            "Apply IIHR Arka Microbial Consortium (soil drench), Trichoderma harzianum, and weekly "
            "Neem soap / Pongamia soap (5 g/L) sprays."
        ),
        next_steps=(
            "Stake plants for aeration, remove infected basal leaves, and use IIHR resistant hybrids (e.g., Arka Rakshak)."
        ),
        keywords=("tomato", "blight", "early blight", "late blight", "leaf curl", "iihr", "arka"),
        source="ICAR-IIHR Bengaluru Horticultural Advisory",
    ),
    KnowledgeEntry(
        id="iihr_chilli_thrips_anthracnose",
        dataset="IIHR Advisory",
        crop="Chilli",
        condition="Chilli Thrips, Mites & Fruit Anthracnose",
        pathogen_or_cause=(
            "Scirtothrips dorsalis / Polyphagotarsonemus latus causing upward/downward curl, "
            "and Colletotrichum capsici causing sunken fruit lesions."
        ),
        symptoms="Boat-shaped upward leaf curling (thrips), downward curl (mites), or sunken black fruit spots.",
        chemical_solution=(
            "Rotate Spinosad 45% SC (0.3 mL/L) or Fipronil for thrips, Spiromesifen for mites, and "
            "Azoxystrobin / Carbendazim + Mancozeb for anthracnose."
        ),
        organic_solution=(
            "Spray IIHR Neem Soap (10 g/L) or Pongamia Soap, erect blue/yellow sticky traps, and "
            "drench soil with Arka Microbial Consortium."
        ),
        next_steps=(
            "Avoid waterlogging, harvest ripe infected pods separately, and maintain border trap crops like maize."
        ),
        keywords=("chilli", "pepper", "thrips", "mite", "anthracnose", "curl", "iihr"),
        source="ICAR-IIHR Bengaluru Chilli IPM Package",
    ),
    KnowledgeEntry(
        id="iihr_mango_anthracnose",
        dataset="IIHR Advisory",
        crop="Mango & Banana",
        condition="Mango Anthracnose / Hopper & Banana Sigatoka",
        pathogen_or_cause=(
            "Colletotrichum gloeosporioides in mango under humid flowering/fruiting, and "
            "Mycosphaerella musicola causing yellow-brown streaks in banana."
        ),
        symptoms="Dark sunken spots on mango panicles/fruits or linear yellow-to-necrotic streaks on banana leaves.",
        chemical_solution=(
            "Spray Carbendazim + Mancozeb (2 g/L) or Propiconazole (1 mL/L) at panicle emergence "
            "or early banana streak stage; add Imidacloprid if mango hoppers are active."
        ),
        organic_solution=(
            "Apply IIHR Arka Banana Special foliar nutrition + Pseudomonas fluorescens, prune dense "
            "mango canopy for sunlight penetration, and spray Neem oil."
        ),
        next_steps=(
            "Maintain orchard sanitation by collecting fallen twigs/leaves and schedule preventive pre-monsoon sprays."
        ),
        keywords=("mango", "banana", "anthracnose", "sigatoka", "hopper", "fruit", "iihr"),
        source="ICAR-IIHR Bengaluru Fruit Crop Advisory",
    ),
)


class KnowledgeBaseRetriever:
    """Retrieves grounded ICAR, IIHR, and benchmark dataset advisories."""

    def __init__(self, entries: Optional[tuple[KnowledgeEntry, ...]] = None):
        self.entries: tuple[KnowledgeEntry, ...] = entries or KNOWLEDGE_BASE

    @staticmethod
    def list_datasets() -> List[str]:
        """Return the 5 supported datasets/taxonomies."""
        return list(SUPPORTED_DATASETS)

    def get_entries_by_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Return all entries belonging to a specific dataset taxonomy."""
        target = dataset_name.strip().lower()
        return [
            entry.to_dict()
            for entry in self.entries
            if entry.dataset.lower() == target
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Export the full knowledge base as a pandas DataFrame."""
        return pd.DataFrame([entry.to_dict() for entry in self.entries])

    def retrieve(
        self,
        query: str,
        crop_hint: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Rank and retrieve the top-k relevant entries across the 5 datasets."""
        combined_query = f"{query or ''} {crop_hint or ''}".strip().lower()
        tokens = set(re.findall(r"[a-z0-9]+", combined_query))

        scored: List[tuple[float, KnowledgeEntry]] = []
        for entry in self.entries:
            score = 0.0
            searchable = (
                f"{entry.dataset} {entry.crop} {entry.condition} "
                f"{entry.symptoms} {entry.pathogen_or_cause}"
            ).lower()

            for kw in entry.keywords:
                if kw in combined_query:
                    score += 3.0 if " " in kw else 2.0

            for tok in tokens:
                if len(tok) >= 3 and tok in searchable:
                    score += 1.0

            if crop_hint and crop_hint.strip().lower() in entry.crop.lower():
                score += 4.0

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)

        if not scored:
            # Always return balanced ICAR + IIHR general grounding when query is generic
            defaults = [
                entry.to_dict()
                for entry in self.entries
                if entry.id in {"icar_soil_npk", "iihr_tomato_blight_curl", "pv_corn_gls"}
            ]
            return defaults[:top_k]

        return [entry.to_dict() for _, entry in scored[:top_k]]

    @staticmethod
    def get_soil_advisory(
        n: float = 90.0,
        p: float = 42.0,
        k: float = 43.0,
        ph: float = 6.5,
    ) -> Dict[str, Any]:
        """Classify soil NPK and pH parameters using ICAR Soil Health Card benchmarks."""
        n_status = "Low" if n < 50 else ("Optimal" if n <= 140 else "High")
        p_status = "Low" if p < 25 else ("Optimal" if p <= 60 else "High")
        k_status = "Low" if k < 30 else ("Optimal" if k <= 80 else "High")

        if ph < 5.5:
            ph_status = "Acidic"
            ph_note = "Apply agricultural lime to raise pH toward 6.0–7.0."
        elif ph <= 7.5:
            ph_status = "Neutral / Optimal"
            ph_note = "Soil pH is ideal for nutrient availability."
        else:
            ph_status = "Alkaline"
            ph_note = "Incorporate gypsum and organic compost to buffer alkalinity."

        notes: List[str] = [ph_note]
        if n_status == "Low":
            notes.append("Nitrogen is low: prioritize split urea or Rhizobium/Azotobacter biofertilizer.")
        elif n_status == "High":
            notes.append("Nitrogen is high: avoid excess urea to prevent lush, disease-prone foliage.")
        if p_status == "Low":
            notes.append("Phosphorus is low: apply basal DAP/SSP with Phosphate Solubilizing Bacteria (PSB).")
        if k_status == "Low":
            notes.append("Potassium is low: supplement MOP or potash-rich organic ash to strengthen disease resistance.")

        return {
            "n": round(float(n), 1),
            "p": round(float(p), 1),
            "k": round(float(k), 1),
            "ph": round(float(ph), 2),
            "n_status": n_status,
            "p_status": p_status,
            "k_status": k_status,
            "ph_status": ph_status,
            "recommendations": notes,
        }

    @staticmethod
    def format_context(entries: List[Dict[str, Any]]) -> str:
        """Format retrieved knowledge entries into concise grounded prompt context."""
        if not entries:
            return "No specific KB entries matched."
        blocks: List[str] = []
        for idx, item in enumerate(entries, start=1):
            blocks.append(
                f"[{idx}] Dataset: {item['dataset']} | Crop: {item['crop']} | "
                f"Condition: {item['condition']}\n"
                f"    Cause: {item['pathogen_or_cause']}\n"
                f"    Chemical: {item['chemical_solution']}\n"
                f"    Organic: {item['organic_solution']}\n"
                f"    Next Steps: {item['next_steps']}\n"
                f"    Source: {item['source']}"
            )
        return "\n".join(blocks)
