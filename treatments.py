"""
treatments.py
-------------
Three-tier treatment advice for each of the 21 fruit disease classes.

Keys  → raw class names (must match CLASS_NAMES in model.py exactly).
Values → dict with keys "Mild", "Moderate", "Severe" (or "Healthy" for healthy classes).

Usage in model.py:
    advice = TREATMENTS.get(class_name, {}).get(severity, "No advice available.")

To update advice: just edit the text below. No other code needs to change.
"""

TREATMENTS: dict[str, dict[str, str]] = {

    # ════════════════════════════════════════════════════════════════════
    # AVOCADO
    # ════════════════════════════════════════════════════════════════════

    "Avocado_Anthracnose": {
        "Mild": (
            "Early infection detected. Improve air circulation by light pruning "
            "around affected areas. Avoid overhead irrigation. Scout every 5–7 days "
            "and begin a copper-based fungicide spray as a precaution."
        ),
        "Moderate": (
            "Active infection. Apply copper oxychloride fungicide every 14 days. "
            "Remove all visibly infected fruit and leaves and dispose of them away "
            "from the orchard. Reduce canopy density with targeted pruning to lower "
            "humidity around remaining fruit."
        ),
        "Severe": (
            "Heavy infection — immediate action required. Remove and destroy all "
            "infected plant material. Apply systemic fungicide (carbendazim or "
            "tebuconazole) on a 10-day schedule. Post-harvest, treat remaining "
            "fruit with a hot-water dip (52°C, 5 min). Consider canopy reduction "
            "to prevent recurrence next season."
        ),
    },

    "Avocado_Healthy": {
        "Healthy": (
            "No disease detected. Maintain tree health through balanced NPK "
            "fertilisation, adequate drainage, and annual pruning. Scout regularly "
            "for early signs of anthracnose or scab, especially during wet seasons."
        ),
    },

    "Avocado_Scab": {
        "Mild": (
            "Early scab signs. Apply mancozeb or copper fungicide spray now. "
            "Prune slightly to open the canopy and reduce moisture retention. "
            "Remove and discard fallen infected leaves to reduce spore load."
        ),
        "Moderate": (
            "Moderate scab infection. Spray mancozeb or copper oxychloride every "
            "10–14 days through fruit development. Prune to improve sunlight "
            "penetration and airflow. Remove all infected fruit promptly."
        ),
        "Severe": (
            "Severe scab outbreak. Apply systemic fungicide (tebuconazole) on a "
            "10-day interval. Heavily infected fruit should be harvested early and "
            "discarded. Conduct aggressive canopy pruning after harvest and apply a "
            "dormant copper spray to reduce inoculum for the next season."
        ),
    },

    # ════════════════════════════════════════════════════════════════════
    # GUAVA
    # ════════════════════════════════════════════════════════════════════

    "Guava_Anthracnose": {
        "Mild": (
            "Early stage. Remove affected fruit and leaves. Apply copper hydroxide "
            "spray as a preventive measure. Avoid wetting foliage when irrigating."
        ),
        "Moderate": (
            "Apply carbendazim or copper hydroxide fungicide every 15 days. Remove "
            "and burn all fallen and infected fruit. Harvest at correct maturity — "
            "over-ripe guava is significantly more vulnerable."
        ),
        "Severe": (
            "Severe infection. Remove all visibly infected fruit and branches. "
            "Apply systemic fungicide (carbendazim) on a 10-day cycle. Handle fruit "
            "with care during harvest to avoid wounds. Apply post-harvest fungicide "
            "dip before storage or transport."
        ),
    },

    "Guava_Healthy": {
        "Healthy": (
            "No disease detected. Water at the base rather than overhead. Keep trees "
            "well-spaced and monitor during humid seasons for early signs of "
            "anthracnose or scab."
        ),
    },

    "Guava_Phytophthora": {
        "Mild": (
            "Early water-mould symptoms. Improve soil drainage immediately — "
            "waterlogged roots accelerate spread. Reduce irrigation frequency. "
            "Apply phosphonate fungicide (fosetyl-aluminium) as a preventive soil drench."
        ),
        "Moderate": (
            "Active Phytophthora infection. Improve drainage, avoid overwatering. "
            "Apply phosphonate-based fungicide (fosetyl-aluminium or metalaxyl) as "
            "both a soil drench and foliar spray. Remove and destroy all infected "
            "root material and mulch away from the tree base."
        ),
        "Severe": (
            "Severe Phytophthora — tree survival is at risk. Excavate and remove "
            "all visibly decayed roots. Apply metalaxyl soil drench monthly. Do not "
            "replant in the same location without full soil fumigation or solarisation. "
            "Consider removing severely affected trees to protect neighbouring plants."
        ),
    },

    "Guava_Scab": {
        "Mild": (
            "Light scab infection. Apply sulfur or copper fungicide now. Thin the "
            "canopy slightly to allow sunlight and airflow. Remove affected shoots."
        ),
        "Moderate": (
            "Apply copper or sulfur-based fungicide every 10 days during wet weather. "
            "Prune to open the canopy and reduce leaf wetness duration. Remove and "
            "dispose of infected shoots away from the orchard."
        ),
        "Severe": (
            "Heavy scab infection. Apply systemic fungicide (tebuconazole or "
            "carbendazim) on a 10-day schedule. Conduct significant canopy pruning "
            "after harvest. Clear all fallen infected material from the orchard floor."
        ),
    },

    "Guava_Styler_and_Root": {
        "Mild": (
            "Early root/styler disease. Improve drainage around the tree base. "
            "Apply carbendazim as a preventive soil drench. Avoid mechanical root "
            "damage during cultivation."
        ),
        "Moderate": (
            "Moderate infection. Apply systemic fungicide (carbendazim) as a root-zone "
            "soil drench every 3–4 weeks. Improve soil drainage. Remove and destroy "
            "all infected root material found during inspection."
        ),
        "Severe": (
            "Severe root/styler disease — tree loss is possible. Remove all infected "
            "roots and plant material. Apply soil fumigant before replanting. Do not "
            "replant guava in the same spot without full soil treatment. Destroy removed "
            "material away from the orchard."
        ),
    },

    # ════════════════════════════════════════════════════════════════════
    # MANGO
    # ════════════════════════════════════════════════════════════════════

    "Mango_Alternaria": {
        "Mild": (
            "Early Alternaria infection. Apply mancozeb spray immediately. Store "
            "harvested fruit in cool, dry conditions and avoid physical injuries "
            "to the fruit skin."
        ),
        "Moderate": (
            "Apply mancozeb or carbendazim every 15 days through fruit development. "
            "Remove and dispose of infected leaves and twigs. Ensure adequate "
            "spacing and airflow. Keep storage areas dry and well-ventilated."
        ),
        "Severe": (
            "Severe Alternaria outbreak. Apply systemic fungicide (tebuconazole) on "
            "a 10-day cycle. Remove all visibly infected plant parts. Post-harvest, "
            "treat fruit with prochloraz dip. Destroy infected material away from "
            "the orchard. Improve storage temperature management."
        ),
    },

    "Mango_Anthracnose": {
        "Mild": (
            "Early anthracnose signs. Apply copper oxychloride spray now. Prune "
            "dead wood and improve canopy airflow. Monitor every week during "
            "flowering and fruit set."
        ),
        "Moderate": (
            "Apply copper oxychloride or carbendazim every 10–15 days from panicle "
            "emergence. Remove infected panicles and mummified fruit. Prune to "
            "reduce canopy density and lower humidity."
        ),
        "Severe": (
            "Severe anthracnose — act immediately. Apply systemic fungicide "
            "(tebuconazole or propiconazole) on a 10-day schedule. Remove all "
            "infected parts. Post-harvest: hot water dip at 52°C for 5 minutes. "
            "Conduct aggressive pruning after harvest to prevent recurrence."
        ),
    },

    "Mango_Healthy": {
        "Healthy": (
            "No disease detected. Apply preventive copper sprays at flowering as a "
            "precaution during high-humidity seasons. Maintain balanced NPK "
            "fertilisation and prune annually after harvest."
        ),
    },

    "Mango_Scab": {
        "Mild": (
            "Early scab infection. Apply copper fungicide at panicle emergence. "
            "Prune lightly to open the canopy. Avoid excessive nitrogen fertilisation."
        ),
        "Moderate": (
            "Apply copper-based fungicide from panicle emergence every 14 days. "
            "Prune to improve sunlight penetration. Remove and dispose of infected "
            "leaves to reduce spore load in the orchard."
        ),
        "Severe": (
            "Heavy scab. Apply tebuconazole or carbendazim on a 10-day schedule. "
            "Conduct aggressive canopy pruning after harvest. Clear all fallen "
            "infected leaves from the orchard floor and apply dormant copper spray."
        ),
    },

    "Mango_Stem_end_Rot": {
        "Mild": (
            "Minor stem-end rot signs. Harvest fruit with care to avoid stem "
            "injuries — the fungus enters primarily through the stem scar. "
            "Ensure prompt cooling after harvest."
        ),
        "Moderate": (
            "Apply post-harvest prochloraz or thiabendazole dip immediately after "
            "harvest. Handle fruit gently to avoid stem wounds. Store in cool, "
            "well-ventilated areas. Market fruit promptly."
        ),
        "Severe": (
            "Severe post-harvest rot. Apply hot water treatment (52°C for 5 min) "
            "followed by prochloraz dip on all harvested fruit. Remove and destroy "
            "all infected fruit — do not store with healthy fruit. Disinfect all "
            "storage containers and handling equipment."
        ),
    },

    # ════════════════════════════════════════════════════════════════════
    # ORANGE
    # ════════════════════════════════════════════════════════════════════

    "Orange_Citrus_Canker": {
        "Mild": (
            "Early bacterial canker detected. Apply copper hydroxide spray "
            "immediately. Disinfect all pruning tools with 70% alcohol between "
            "cuts. Avoid working in the orchard when plants are wet."
        ),
        "Moderate": (
            "Apply copper bactericide every 3–4 weeks during the flushing season "
            "and after rain events. Remove and destroy all infected leaves, twigs, "
            "and fruit. Disinfect tools after every cut. Citrus canker is "
            "quarantine-regulated — report to local agricultural authorities."
        ),
        "Severe": (
            "Severe citrus canker — a quarantine disease. Immediately notify local "
            "agricultural authorities. Remove and destroy all infected trees or "
            "heavily infected sections. Apply copper bactericide to surrounding "
            "trees as a buffer. Disinfect all equipment. Do not move plant material "
            "off the property without official clearance."
        ),
    },

    "Orange_Citrus_Melanose": {
        "Mild": (
            "Early melanose. Remove dead wood where the Diaporthe fungus "
            "overwinters — this is the single most effective control. Apply copper "
            "fungicide within 24 hours of rain events."
        ),
        "Moderate": (
            "Apply copper fungicide sprays during young fruit development, especially "
            "within 24 hours of rain. Prune and remove all deadwood thoroughly. "
            "Keep the orchard floor clear of fallen dead twigs."
        ),
        "Severe": (
            "Heavy melanose. Conduct aggressive deadwood removal throughout the "
            "canopy. Apply copper fungicide on a 14-day schedule during fruit "
            "development. Infected fruit may still be usable — melanose affects "
            "appearance not flavour — but assess marketability carefully."
        ),
    },

    "Orange_Healthy": {
        "Healthy": (
            "No disease detected. Apply preventive copper sprays at the start of "
            "wet seasons. Prune annually to remove deadwood. Maintain balanced "
            "fertilisation and monitor for early melanose and canker signs."
        ),
    },

    # ════════════════════════════════════════════════════════════════════
    # POMEGRANATE
    # ════════════════════════════════════════════════════════════════════

    "Pomegranate_Alternaria": {
        "Mild": (
            "Early Alternaria infection. Apply mancozeb spray at fruit set. "
            "Switch to drip irrigation to keep foliage dry. "
            "Remove and destroy infected fruit and debris."
        ),
        "Moderate": (
            "Apply mancozeb or copper oxychloride every 12–15 days. Remove infected "
            "fruit promptly. Ensure adequate spacing for airflow. Keep the orchard "
            "floor free of fallen infected debris."
        ),
        "Severe": (
            "Severe Alternaria outbreak. Apply systemic fungicide (tebuconazole) "
            "every 10 days. Remove all infected fruit — do not leave mummified "
            "fruit on the tree. Destroy all removed material. Review irrigation "
            "method to eliminate overhead watering."
        ),
    },

    "Pomegranate_Anthracnose": {
        "Mild": (
            "Early anthracnose. Apply carbendazim spray now. Monitor fruit closely "
            "and harvest at the correct maturity stage — over-ripe fruit is much "
            "more susceptible."
        ),
        "Moderate": (
            "Apply carbendazim or copper fungicide every 14 days during flowering "
            "and fruit development. Remove all mummified fruit and infected twigs. "
            "Harvest promptly at maturity."
        ),
        "Severe": (
            "Heavy anthracnose infection. Apply systemic fungicide (tebuconazole) "
            "on a 10-day cycle. Remove and destroy all infected and mummified fruit. "
            "Prune infected twigs back to healthy wood. Apply post-harvest "
            "fungicide dip before storage."
        ),
    },

    "Pomegranate_Bacterial_Blight": {
        "Mild": (
            "Early bacterial blight (Xanthomonas). Apply copper bactericide now. "
            "Avoid overhead irrigation. Do not work in the orchard when plants "
            "are wet — the bacteria spread easily through water splash."
        ),
        "Moderate": (
            "Apply copper bactericide every 10 days during the rainy season. "
            "Remove and destroy all infected wood. Avoid overhead irrigation. "
            "Disinfect pruning tools between cuts with 70% alcohol."
        ),
        "Severe": (
            "Severe bacterial blight. Remove and destroy all heavily infected "
            "branches and fruit. Apply copper bactericide on a 7-day schedule. "
            "Avoid any activity in the orchard during rain. Consider removing "
            "severely affected trees to protect neighbours. Use resistant varieties "
            "for replanting."
        ),
    },

    "Pomegranate_Cercospora": {
        "Mild": (
            "Early Cercospora leaf spot. Rake and remove fallen infected leaves — "
            "they are the primary inoculum source. Apply mancozeb spray. "
            "Avoid excessive nitrogen fertilisation."
        ),
        "Moderate": (
            "Apply mancozeb or copper oxychloride every 10–14 days. Remove and "
            "destroy all fallen infected leaves. Prune to improve air circulation. "
            "Maintain balanced fertilisation — excess nitrogen promotes susceptible "
            "lush growth."
        ),
        "Severe": (
            "Heavy Cercospora infection — significant defoliation risk. Apply "
            "systemic fungicide (tebuconazole) every 10 days. Aggressively clear "
            "all fallen leaf material. Prune the canopy to reduce humidity. "
            "Reassess fertilisation programme to avoid nitrogen excess."
        ),
    },

    "Pomegranate_Healthy": {
        "Healthy": (
            "No disease detected. Use drip irrigation to keep foliage dry. "
            "Apply preventive copper sprays at the start of each wet season. "
            "Prune annually and maintain balanced fertilisation."
        ),
    },
}
