"""RAG engine — FAISS vector store + farming knowledge base."""
from __future__ import annotations

from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Knowledge base: crops, diseases, fertilizers, irrigation, market context
# ---------------------------------------------------------------------------
FARMING_KNOWLEDGE: List[dict] = [
    # ── Crops ───────────────────────────────────────────────────────────────
    {
        "id": "crop_wheat",
        "category": "crop_recommendation",
        "content": (
            "Wheat (Triticum aestivum) grows best in loamy soil with pH 6.0–7.5. "
            "It is a Rabi crop sown in October–December and harvested in March–April. "
            "Suitable regions: Punjab, Haryana, UP, MP, Rajasthan. "
            "Water requirement: 450–650 mm. Yield: 4–6 tonnes/hectare. "
            "Key nutrients: NPK 120:60:40 kg/ha. Major diseases: rust, smut, blight."
        ),
    },
    {
        "id": "crop_rice",
        "category": "crop_recommendation",
        "content": (
            "Rice (Oryza sativa) thrives in clayey or loamy soil with pH 5.5–7.0. "
            "It is a Kharif crop transplanted in June–July and harvested in October–November. "
            "Suitable regions: West Bengal, Andhra Pradesh, Punjab, Odisha, Tamil Nadu. "
            "Water requirement: 1200–1400 mm. Yield: 3.5–5 tonnes/hectare. "
            "Key nutrients: NPK 100:50:50 kg/ha. Major diseases: blast, bacterial blight, sheath blight."
        ),
    },
    {
        "id": "crop_sugarcane",
        "category": "crop_recommendation",
        "content": (
            "Sugarcane (Saccharum officinarum) requires deep, well-drained loamy soil with pH 6.0–8.0. "
            "Planting: February–March (spring) or October–November (autumn). Harvest: 10–12 months. "
            "Suitable regions: UP, Maharashtra, Karnataka, Tamil Nadu, Gujarat. "
            "Water requirement: 1500–2500 mm. Yield: 70–100 tonnes/hectare. "
            "Key nutrients: NPK 250:115:115 kg/ha. Major pests: borer, whitefly, mealybug."
        ),
    },
    {
        "id": "crop_cotton",
        "category": "crop_recommendation",
        "content": (
            "Cotton (Gossypium hirsutum) grows best in black/regur soil with pH 5.8–8.0. "
            "Kharif crop sown in May–June and picked in November–January. "
            "Suitable regions: Gujarat, Maharashtra, Telangana, Andhra Pradesh, Punjab. "
            "Water requirement: 700–1200 mm. Yield: 1.5–2.5 tonnes/hectare lint. "
            "Key nutrients: NPK 150:60:60 kg/ha. Major pests: bollworm, aphid, whitefly."
        ),
    },
    {
        "id": "crop_maize",
        "category": "crop_recommendation",
        "content": (
            "Maize (Zea mays) grows in well-drained sandy loam to loam soil with pH 5.8–7.0. "
            "Can be grown Kharif (June–July), Rabi (November), and Zaid (February–March). "
            "Suitable regions: Karnataka, AP, Rajasthan, MP, Bihar, Himachal Pradesh. "
            "Water requirement: 500–800 mm. Yield: 5–8 tonnes/hectare. "
            "Key nutrients: NPK 120:60:40 kg/ha. Major diseases: downy mildew, stem rot."
        ),
    },
    {
        "id": "crop_tomato",
        "category": "crop_recommendation",
        "content": (
            "Tomato (Solanum lycopersicum) requires well-drained sandy loam soil with pH 6.0–7.0. "
            "Can be grown year-round; main season June–September. "
            "Suitable across India. Water requirement: 400–600 mm. Yield: 25–40 tonnes/hectare. "
            "Key nutrients: NPK 150:100:100 kg/ha + calcium. "
            "Major diseases: early blight, late blight, wilt. Major pests: fruit borer, whitefly."
        ),
    },
    {
        "id": "crop_soybean",
        "category": "crop_recommendation",
        "content": (
            "Soybean (Glycine max) grows in well-drained clay loam to loam soil with pH 6.0–6.8. "
            "Kharif crop sown in June–July and harvested in September–October. "
            "Suitable regions: MP, Maharashtra, Rajasthan, Karnataka. "
            "Water requirement: 450–700 mm. Yield: 2–3 tonnes/hectare. "
            "Key nutrients: NPK 30:80:40 kg/ha (nitrogen-fixing legume). "
            "Major diseases: yellow mosaic virus, charcoal rot."
        ),
    },
    {
        "id": "crop_mustard",
        "category": "crop_recommendation",
        "content": (
            "Mustard (Brassica juncea) grows in well-drained sandy loam soil with pH 6.0–7.5. "
            "Rabi crop sown in October–November and harvested in February–March. "
            "Suitable regions: Rajasthan, UP, Haryana, MP, Bihar. "
            "Water requirement: 250–400 mm. Yield: 1.5–2.5 tonnes/hectare. "
            "Key nutrients: NPK 80:40:40 kg/ha + sulphur 30 kg/ha."
        ),
    },
    # ── Soils ────────────────────────────────────────────────────────────────
    {
        "id": "soil_alluvial",
        "category": "soil_type",
        "content": (
            "Alluvial soil is the most fertile soil found in the Indo-Gangetic plains (UP, Punjab, Bihar, Assam). "
            "It is rich in potash but poor in phosphorus. pH: 6.5–8.0. "
            "Best crops: wheat, rice, sugarcane, maize, cotton, jute, pulses, oilseeds. "
            "Management: regular organic matter addition, balanced NPK."
        ),
    },
    {
        "id": "soil_black",
        "category": "soil_type",
        "content": (
            "Black soil (Regur) is found in Maharashtra, Gujarat, MP, Telangana, AP. "
            "It has high clay content, good moisture retention, rich in calcium and magnesium, "
            "but deficient in nitrogen and phosphorus. pH: 7.0–8.5. "
            "Best crops: cotton, sorghum, wheat, groundnut, pulses. "
            "Management: gypsum for structure, phosphorus supplementation."
        ),
    },
    {
        "id": "soil_red",
        "category": "soil_type",
        "content": (
            "Red soil is common in Odisha, Karnataka, AP, Tamil Nadu, and Jharkhand. "
            "It is porous, friable, low in lime and nitrogen. pH: 5.5–7.5. "
            "Best crops: rice, wheat, millets, tobacco, groundnut, potato. "
            "Management: heavy manuring, liming if pH < 6.0."
        ),
    },
    {
        "id": "soil_laterite",
        "category": "soil_type",
        "content": (
            "Laterite soil found in Kerala, Goa, Western Ghats, Chhattisgarh. "
            "Acidic pH 4.5–6.0, low in nitrogen, calcium, magnesium. "
            "Best crops: cashew, tea, coffee, rubber, coconut, rice. "
            "Management: heavy liming, organic manure, mulching."
        ),
    },
    # ── Diseases & Pests ────────────────────────────────────────────────────
    {
        "id": "disease_wheat_rust",
        "category": "disease_pest",
        "content": (
            "Wheat Rust (Puccinia species) — yellow, brown, or black pustules on leaves and stems. "
            "Cause: fungal infection favoured by cool humid weather 15–25°C. "
            "Prevention: use rust-resistant varieties (HD 3086, PBW 343), seed treatment with Carboxin. "
            "Control: spray Propiconazole 25% EC @ 0.1% or Mancozeb 75% WP @ 0.2% at first symptom."
        ),
    },
    {
        "id": "disease_rice_blast",
        "category": "disease_pest",
        "content": (
            "Rice Blast (Magnaporthe oryzae) — diamond-shaped lesions on leaves, neck rot at panicle stage. "
            "Favoured by high humidity, temperature 24–28°C, excess nitrogen. "
            "Prevention: balanced fertilisation, resistant varieties (Pusa Basmati 1121, MTU 1010). "
            "Control: Tricyclazole 75% WP @ 0.06% or Isoprothiolane 40% EC @ 0.075%."
        ),
    },
    {
        "id": "pest_cotton_bollworm",
        "category": "disease_pest",
        "content": (
            "American Bollworm (Helicoverpa armigera) is the most destructive pest of cotton in India. "
            "Larvae bore into bolls, causing 30–60% yield loss. "
            "Prevention: pheromone traps @ 5/hectare, inter-cropping with coriander. "
            "Control: Spinosad 45% SC @ 0.01%, Chlorantraniliprole 18.5% SC @ 0.003%, "
            "or Bacillus thuringiensis (Bt) spray for eco-friendly management."
        ),
    },
    {
        "id": "pest_aphid",
        "category": "disease_pest",
        "content": (
            "Aphids (Aphis gossypii, Rhopalosiphum maidis) attack cotton, wheat, vegetables. "
            "Suck sap causing yellowing, stunted growth, honeydew leading to sooty mould. "
            "Prevention: reflective mulch, conserve natural enemies (ladybird beetles). "
            "Control: Dimethoate 30% EC @ 0.03%, Imidacloprid 17.8% SL @ 0.005%, "
            "or Neem oil 3000 ppm @ 2 ml/litre."
        ),
    },
    {
        "id": "disease_tomato_blight",
        "category": "disease_pest",
        "content": (
            "Tomato Early Blight (Alternaria solani) — dark brown concentric ring spots on lower leaves. "
            "Favoured by warm humid weather 24–29°C. "
            "Prevention: crop rotation, avoid overhead irrigation, remove infected debris. "
            "Control: Mancozeb 75% WP @ 0.25%, Chlorothalonil 75% WP @ 0.2% at weekly intervals."
        ),
    },
    # ── Fertilizers ─────────────────────────────────────────────────────────
    {
        "id": "fertilizer_npk",
        "category": "fertilizer",
        "content": (
            "NPK Fertilizer guidelines: Nitrogen (N) promotes leafy growth — Urea (46% N), DAP. "
            "Phosphorus (P) builds roots and flowers — DAP (46% P2O5), SSP (16% P2O5). "
            "Potassium (K) strengthens immunity and fruit quality — MOP (60% K2O). "
            "Apply N in split doses: 1/2 at basal, 1/4 at tillering/vegetative, 1/4 at grain fill. "
            "Soil test-based fertilisation is strongly recommended before each season."
        ),
    },
    {
        "id": "fertilizer_organic",
        "category": "fertilizer",
        "content": (
            "Organic Fertilizers: Farmyard Manure (FYM) @ 10–15 tonnes/hectare improves soil structure. "
            "Vermicompost @ 2–4 tonnes/hectare provides balanced macro and micro nutrients. "
            "Green manuring with Dhaincha or Sesbania fixes 60–80 kg N/ha. "
            "Bio-fertilizers: Rhizobium for legumes, Azospirillum for cereals, PSB (phosphate solubilising bacteria). "
            "Use composted city waste, press mud, or poultry manure to enrich soil organic carbon."
        ),
    },
    {
        "id": "fertilizer_micro",
        "category": "fertilizer",
        "content": (
            "Micronutrient deficiencies in Indian soils: "
            "Zinc deficiency (most common) — Khaira disease in rice, white bud in maize. "
            "Apply ZnSO4 @ 25 kg/ha basal. Iron deficiency in alkaline soils: FeSO4 @ 25 kg/ha. "
            "Boron for flowering crops: Borax @ 10 kg/ha. "
            "Foliar sprays of 0.5% ZnSO4 or 0.2% FeSO4 give quick correction."
        ),
    },
    # ── Irrigation ──────────────────────────────────────────────────────────
    {
        "id": "irrigation_drip",
        "category": "irrigation",
        "content": (
            "Drip Irrigation: 40–50% water saving over flood irrigation. Best for sugarcane, cotton, "
            "vegetables, fruit orchards. Lateral spacing: 1.2–1.5 m; emitter flow: 2–4 litres/hour. "
            "Fertigation through drip reduces fertilizer use by 25–30%. "
            "Government subsidy available under PMKSY up to 55% for small/marginal farmers. "
            "Maintain filter, flush laterals weekly to avoid clogging."
        ),
    },
    {
        "id": "irrigation_sprinkler",
        "category": "irrigation",
        "content": (
            "Sprinkler Irrigation: 25–30% water saving; suitable for groundnut, wheat, pulses, vegetables. "
            "System types: portable, semi-portable, permanent. Pressure: 2–2.5 kg/cm². "
            "Avoid sprinkler during windy conditions and midday heat. "
            "Subsidy under PMKSY up to 50% for small farmers. "
            "Ideal for undulating terrain where flood irrigation is difficult."
        ),
    },
    {
        "id": "irrigation_schedule",
        "category": "irrigation",
        "content": (
            "Critical irrigation stages: Rice — transplanting, tillering, panicle initiation, flowering. "
            "Wheat — crown root initiation (21 DAS), tillering, jointing, flowering, grain filling. "
            "Sugarcane — germination, tillering, grand growth period. "
            "Cotton — squaring, flowering, boll development. "
            "Tomato — transplanting, flowering, fruit set (avoid water stress). "
            "Always irrigate early morning or evening to minimise evaporation losses."
        ),
    },
    # ── Market & Mandi ──────────────────────────────────────────────────────
    {
        "id": "market_overview",
        "category": "market",
        "content": (
            "Mandi prices in India are regulated by state APMCs (Agricultural Produce Market Committees). "
            "e-NAM (National Agriculture Market) platform connects 1000+ mandis across India digitally. "
            "MSP (Minimum Support Price) is announced by Government of India for 23 crops. "
            "Key price portals: Agmarknet (agmarknet.gov.in), e-NAM (enam.gov.in), Kisan Suvidha app. "
            "Factors affecting prices: seasonality, rainfall, international trade, storage availability."
        ),
    },
    {
        "id": "market_msp_2024",
        "category": "market",
        "content": (
            "MSP 2024-25 Key Crops (indicative): "
            "Wheat: ₹2,275/quintal | Paddy (Common): ₹2,300/quintal | Paddy (Grade A): ₹2,320/quintal. "
            "Maize: ₹2,225/quintal | Soybean: ₹4,892/quintal | Groundnut: ₹6,783/quintal. "
            "Cotton (Medium): ₹7,121/quintal | Cotton (Long): ₹7,521/quintal. "
            "Mustard: ₹5,650/quintal | Sunflower: ₹7,280/quintal | Urad: ₹7,400/quintal."
        ),
    },
    # ── Seasons ─────────────────────────────────────────────────────────────
    {
        "id": "season_kharif",
        "category": "season",
        "content": (
            "Kharif Season (June–November): Sown at start of monsoon, harvested in autumn. "
            "Major crops: Rice, Maize, Sorghum, Bajra, Groundnut, Soybean, Cotton, Sugarcane, Turmeric. "
            "Key states: Most of India benefiting from SW monsoon. "
            "Critical activities: land preparation in May, sowing with first rain, pest management during monsoon."
        ),
    },
    {
        "id": "season_rabi",
        "category": "season",
        "content": (
            "Rabi Season (November–April): Sown in winter, harvested in spring. "
            "Major crops: Wheat, Barley, Mustard, Chickpea (Chana), Lentil (Masoor), Peas, Potato. "
            "Key states: Punjab, Haryana, UP, MP, Rajasthan, Bihar. "
            "Critical activities: sowing immediately after Kharif harvest, 4–6 irrigations needed."
        ),
    },
    {
        "id": "season_zaid",
        "category": "season",
        "content": (
            "Zaid Season (March–June): Short summer season between Rabi and Kharif. "
            "Major crops: Watermelon, Muskmelon, Cucumber, Pumpkin, Bitter Gourd, Moong (Green Gram). "
            "Requires irrigation as little to no rainfall. Short duration 60–90 days. "
            "Good opportunity for vegetable and short-duration crop production."
        ),
    },
    # ── Government Schemes ───────────────────────────────────────────────────
    {
        "id": "scheme_pmfby",
        "category": "government_scheme",
        "content": (
            "Pradhan Mantri Fasal Bima Yojana (PMFBY): Crop insurance scheme. "
            "Premium: 2% for Kharif, 1.5% for Rabi, 5% for horticulture/commercial crops. "
            "Covers loss due to non-preventable risks: natural fire, lightning, storm, hailstorm, "
            "cyclone, typhoon, flood, drought, pest/disease. "
            "Enroll via Common Service Centers, bank branches, or pmfby.gov.in before cut-off date."
        ),
    },
    {
        "id": "scheme_pmkisan",
        "category": "government_scheme",
        "content": (
            "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi): Direct income support ₹6,000/year "
            "in 3 installments of ₹2,000 to eligible farmer families. "
            "Eligibility: landholding farmer families with cultivable land. "
            "Register at pmkisan.gov.in or nearest Common Service Centre. "
            "Aadhaar-linked bank account mandatory for receiving benefits."
        ),
    },
]


class RAGEngine:
    """FAISS-based Retrieval-Augmented Generation engine."""

    def __init__(self) -> None:
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_index()

    def _build_index(self) -> None:
        texts = [doc["content"] for doc in FARMING_KNOWLEDGE]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """Return the top-k most relevant knowledge snippets."""
        q_vec = self.encoder.encode([query], show_progress_bar=False)
        q_vec = np.array(q_vec, dtype=np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if score > 0.2 and idx < len(FARMING_KNOWLEDGE):
                results.append(FARMING_KNOWLEDGE[idx]["content"])
        return results

    def build_rag_context(self, query: str, top_k: int = 4) -> str:
        """Build a formatted context string from retrieved documents."""
        snippets = self.retrieve(query, top_k)
        if not snippets:
            return ""
        return "\n\n".join(f"[Knowledge {i+1}]\n{s}" for i, s in enumerate(snippets))


# Singleton instance
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
