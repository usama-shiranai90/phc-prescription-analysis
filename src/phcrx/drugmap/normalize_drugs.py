"""Brand -> molecule -> therapeutic class normalisation.

Why this exists. The model predicts 719 commercial brand names, and that label
space is the study's biggest structural weakness:

  * brand-level micro-F1 is 0.208; the same model at class level scores 0.348
  * brand-level performance collapses forward in time (-66%)
  * measured cross-era stability, train <=2015 vs test >=2017:

        level           Jaccard   unseen-order rate   top-10 shared
        brand (719)      0.462          12.2%             2/10
        category (89)    0.545           3.0%             7/10

    Collapsing to class cuts structurally unpredictable orders four-fold. The
    prescribing *shape* is stable across eras even though the brands churn.

The existing `rx_category` map covers only 86.8% of orders and has three
defects this module fixes:

  1. **Case/whitespace duplicates.** 'seclo' (178 orders) is unmapped while
     'Seclo' is mapped -- the same product counted as two labels.
  2. **~13% of orders unmapped**, including high-frequency brands (Avolac,
     Atova, Filmet, Alatrol, Fexo).
  3. **Inconsistent granularity.** Some categories are molecules
     ('Paracetamol', 'Naproxen'), some are classes ('NSAIDs', 'PPI'), and
     'Others' is a 117-brand dumping ground (1,925 orders) that is deliberately
     NOT folded here -- those brands are resolved individually instead.

Resolution order: curated table -> normalised rx_category -> canonical-name
fold -> local LLM -> unmapped. Every LLM mapping is constrained to a fixed
class vocabulary and validated against held-out curated brands; the measured
disagreement rate is reported rather than assumed.

    python -m src.phcrx.drugmap.normalize_drugs --llm
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter

import pandas as pd

from ..config import PROCESSED, RESULTS

OUT = RESULTS / "drugmap"
OUT.mkdir(parents=True, exist_ok=True)

# Fixed class vocabulary. The LLM must choose from this list, so its output is
# a classification rather than free text and can be validated mechanically.
# 'antispasmodic' was added to the original 47 after measurement: tiemonium
# methylsulphate and its relatives account for 667 orders (2.2%), and folding
# them into 'other' would recreate exactly the dumping ground this module
# exists to remove.
CLASSES = [
    "proton_pump_inhibitor", "h2_blocker", "antacid", "prokinetic",
    "antispasmodic",
    "analgesic_antipyretic", "nsaid", "opioid_analgesic", "muscle_relaxant",
    "antihypertensive_acei_arb", "beta_blocker", "calcium_channel_blocker",
    "diuretic", "statin", "antiplatelet", "cardiac_other",
    "antidiabetic_biguanide", "antidiabetic_sulfonylurea", "antidiabetic_other",
    "insulin", "thyroid",
    "antibiotic", "antifungal", "antiviral", "antiparasitic", "antitubercular",
    "antihistamine", "corticosteroid", "bronchodilator", "respiratory_other",
    "anxiolytic_benzodiazepine", "antidepressant", "antipsychotic",
    "antiepileptic", "neuro_other",
    "vitamin", "mineral_supplement", "calcium_vitamin_d", "iron_supplement",
    "laxative", "antidiarrhoeal", "antiemetic", "urological", "antigout",
    "gynaecological", "dermatological", "ophthalmic", "vaccine", "other",
]

# Curated from inspection of the corpus. These are Bangladeshi brands a general
# medical LLM gets wrong -- it read 'Napa' as naproxen in earlier testing when
# it is paracetamol -- so the highest-frequency products are pinned by hand.
CURATED: dict[str, tuple[str, str]] = {
    "napa": ("paracetamol", "analgesic_antipyretic"),
    "napa extend": ("paracetamol", "analgesic_antipyretic"),
    "ace": ("paracetamol", "analgesic_antipyretic"),
    "pyrenol": ("paracetamol", "analgesic_antipyretic"),
    "maxpro": ("esomeprazole", "proton_pump_inhibitor"),
    "seclo": ("omeprazole", "proton_pump_inhibitor"),
    "losectil": ("omeprazole", "proton_pump_inhibitor"),
    "esonix": ("esomeprazole", "proton_pump_inhibitor"),
    "finix": ("esomeprazole", "proton_pump_inhibitor"),
    "pantid": ("pantoprazole", "proton_pump_inhibitor"),
    "pantonix": ("pantoprazole", "proton_pump_inhibitor"),
    "ranitid": ("ranitidine", "h2_blocker"),
    "neoceptin-r": ("ranitidine", "h2_blocker"),
    "neoceptin r": ("ranitidine", "h2_blocker"),
    "entacyd plus": ("antacid_combination", "antacid"),
    "omidon": ("domperidone", "prokinetic"),
    "calbo d": ("calcium_vitamin_d", "calcium_vitamin_d"),
    "ostocal d": ("calcium_vitamin_d", "calcium_vitamin_d"),
    "neuro b": ("vitamin_b_complex", "vitamin"),
    "bextrum silver": ("multivitamin_mineral", "vitamin"),
    "multivit plus": ("multivitamin_mineral", "vitamin"),
    "proviten a-z": ("multivitamin_mineral", "vitamin"),
    "fillwel silver": ("multivitamin_mineral", "vitamin"),
    "ferocit": ("iron_folic_acid", "iron_supplement"),
    "comet": ("metformin", "antidiabetic_biguanide"),
    "comprid": ("glimepiride", "antidiabetic_sulfonylurea"),
    "amdocal": ("amlodipine", "calcium_channel_blocker"),
    "osartil": ("losartan", "antihypertensive_acei_arb"),
    "indever": ("propranolol", "beta_blocker"),
    "atova": ("atorvastatin", "statin"),
    "rivotril": ("clonazepam", "anxiolytic_benzodiazepine"),
    "laxyl": ("amitriptyline", "antidepressant"),
    "tryptin": ("amitriptyline", "antidepressant"),
    "frenxit": ("flupentixol_melitracen", "antidepressant"),
    "naprox": ("naproxen", "nsaid"),
    "flexi": ("aceclofenac", "nsaid"),
    "a- fenac": ("aceclofenac", "nsaid"),
    "a-fenac": ("aceclofenac", "nsaid"),
    # 'Algin' (Square) is tiemonium methylsulphate, NOT an NSAID -- corrected
    # after the corpus' own category label came back 'Trimonium' (= tiemonium)
    # for 306 of its 321 orders and 'Antispasmodics' for the remainder.
    "algin": ("tiemonium_methylsulphate", "antispasmodic"),
    "voltalin": ("diclofenac", "nsaid"),
    "reumacap": ("nsaid_combination", "nsaid"),
    "joinix plus": ("glucosamine", "other"),
    "neso": ("naproxen_esomeprazole", "nsaid"),
    "myolax": ("baclofen", "muscle_relaxant"),
    "beklo": ("baclofen", "muscle_relaxant"),
    "zif-ci": ("cefixime", "antibiotic"),
    "filmet": ("metronidazole", "antibiotic"),
    "alatrol": ("cetirizine", "antihistamine"),
    "fexo": ("fexofenadine", "antihistamine"),
    "deslor": ("desloratadine", "antihistamine"),
    "fenadin": ("fexofenadine", "antihistamine"),
    "avolac": ("lactulose", "laxative"),
    "febustat": ("febuxostat", "antigout"),
    "gastralfet": ("sucralfate", "antacid"),
}

# Second curation pass, added after the local LLM scored 27% on held-out
# curated brands and was therefore not trusted with the remainder (see
# docs/drug_normalization.md). Every entry here was checked against how the
# brand is actually used in THIS corpus -- dosage form, dose pattern, duration,
# co-prescribed drugs and the free-text symptoms of the encounters it appears
# in -- so the mapping is falsifiable from the data rather than recalled.
# An empty molecule means the therapeutic class is established by that usage
# evidence but the exact active ingredient is not; the class is still usable as
# a label, and inventing a molecule would be the same failure mode as the LLM.
CURATED_EVIDENCE: dict[str, tuple[str, str]] = {
    # --- resolved from usage evidence, molecule not established ---------
    # Renova (180): Tab, 1+1+1 or 1+0+1 after meal for 8-14d, 42% of its
    # encounters co-prescribe a PPI, symptoms are back/body/joint pain and no
    # other analgesic is co-ordered -- an NSAID under gastro-protection.
    "renova": ("", "nsaid"),
    # Relentus (68): co-prescribed WITH aceclofenac (Flexi) in 51% of its
    # encounters, so it is the adjunct rather than the analgesic; symptoms are
    # muscle stiffness / back and knee pain.
    "relentus": ("", "muscle_relaxant"),
    # Uromax (81): 0+0+1 nocte, >90d continuous, symptoms are frequency,
    # urgency, dribbling and burning micturition -- a BPH/LUTS agent.
    "uromax": ("", "urological"),
    # Viset (78): 1+1+1 for 4-7d, co-prescribed with ciprofloxacin and
    # nitrofurantoin, symptoms lower abdominal pain + burning micturition.
    "viset": ("", "urological"),
    "urodart": ("", "urological"),
    # Kilmax (31): 1+0+1 for 4-7d with metronidazole and vaginal preparations,
    # symptoms fever + burning micturition + lower abdominal pain.
    "kilmax": ("", "antibiotic"),
    # Fixocard (5/50) (61): >90d continuous, symptom text says "known
    # hypertensive"; the 5/50 strength notation is an ARB+CCB fixed dose.
    "fixocard (5/50)": ("", "antihypertensive_acei_arb"),
    # --- brand identified, corroborated by corpus usage -----------------
    # Norium (87): 0+0+1 nocte for 15-30d, symptoms headache + vertigo,
    # co-prescribed with Tufnil, propranolol and amitriptyline -- the classic
    # migraine-prophylaxis regimen. Norium-5 is the 5 mg strength.
    "norium": ("flunarizine", "neuro_other"),
    "norium-5": ("flunarizine", "neuro_other"),
    # Contine (89): 1/2+0+1/2, >90d continuous, co-prescribed with montelukast
    # and salbutamol inhalers, symptoms breathlessness/asthma -- theophylline
    # SR (the '-contine' suffix; cf. Unicontine in the same corpus).
    "contine": ("theophylline", "bronchodilator"),
    "unicontine": ("theophylline", "bronchodilator"),
    # Tufnil (125): headache in 81 of its encounters, co-ordered with
    # flunarizine/propranolol/amitriptyline -- tolfenamic acid for migraine.
    "tufnil": ("tolfenamic_acid", "nsaid"),
    "metro": ("metronidazole", "antibiotic"),      # with cipro/fluconazole, PID/UTI
    "cef-3": ("cefixime", "antibiotic"),           # Cap 1+0+1 x 5d, UTI/fever
    "zimax": ("azithromycin", "antibiotic"),       # once daily x 5d, fever/cough
    "azin": ("azithromycin", "antibiotic"),
    "nintoin": ("nitrofurantoin", "antibiotic"),
    "furocef": ("cefuroxime", "antibiotic"),
    "cefotil": ("cefuroxime", "antibiotic"),
    "fimoxyclav": ("amoxicillin_clavulanate", "antibiotic"),
    "flucloxin": ("flucloxacillin", "antibiotic"),
    "doxicab": ("doxycycline", "antibiotic"),
    "doxiva": ("doxycycline", "antibiotic"),
    "secnid-ds": ("secnidazole", "antibiotic"),
    "secnid ds": ("secnidazole", "antibiotic"),
    # Feofol CI (139): Cap, long duration, co-ordered with albendazole (the
    # standard anaemia + deworming pair), 'anaemia' in the symptom text.
    "feofol ci": ("iron_folic_acid", "iron_supplement"),
    "feofol": ("iron_folic_acid", "iron_supplement"),
    "ferocit z": ("iron_zinc_folic_acid", "iron_supplement"),
    # ORS (150): dispensed as 'Saline', no dose/duration, co-ordered with
    # multivitamins and iron for weakness/malaise. Classed as an electrolyte
    # supplement rather than ATC A07CA antidiarrhoeal because that is how this
    # corpus uses it -- diarrhoea is not among its symptoms.
    "ors": ("oral_rehydration_salts", "mineral_supplement"),
    "zinc": ("zinc", "mineral_supplement"),
    "zinc b": ("zinc", "mineral_supplement"),
    "xinc-b": ("zinc", "mineral_supplement"),
    # Pregaba (116): 1+0+1 for 15-30d, symptoms burning/tingling in the limbs,
    # co-ordered with B-vitamins and oral hypoglycaemics -- diabetic
    # neuropathy. Filed under antiepileptic with the other gabapentinoids.
    "pregaba": ("pregabalin", "antiepileptic"),
    "pegalin": ("pregabalin", "antiepileptic"),
    "pregalin": ("pregabalin", "antiepileptic"),
    "gaba-p": ("pregabalin", "antiepileptic"),
    "encorate": ("valproate", "antiepileptic"),
    "eracet": ("levetiracetam", "antiepileptic"),
    "dilentin": ("phenytoin", "antiepileptic"),
    "perkinil": ("trihexyphenidyl", "neuro_other"),
    "sedil": ("diazepam", "anxiolytic_benzodiazepine"),
    "dormicum": ("midazolam", "anxiolytic_benzodiazepine"),
    "prodep": ("fluoxetine", "antidepressant"),
    # Loratin (103) / Oradin (59): nocte dosing, symptoms itching, allergic
    # rhinitis, cough and cold.
    "loratin": ("loratadine", "antihistamine"),
    "oradin": ("loratadine", "antihistamine"),
    "tofen": ("ketotifen", "antihistamine"),
    # Cinaron (53) and vertina plus (68): vertigo is the dominant symptom.
    "cinaron": ("cinnarizine", "antiemetic"),
    "vertina plus": ("cinnarizine_dimenhydrinate", "antiemetic"),
    "stemetil": ("prochlorperazine", "antiemetic"),
    "vomitop": ("domperidone", "prokinetic"),
    # Ambrox (65): syrup, cough in 35 of its encounters.
    "ambrox": ("ambroxol", "respiratory_other"),
    "ambrox sr": ("ambroxol", "respiratory_other"),
    "mucolet": ("", "respiratory_other"),
    "mucolyte": ("", "respiratory_other"),
    "adovas": ("herbal_cough_syrup", "respiratory_other"),
    "bukof": ("", "respiratory_other"),
    "montene": ("montelukast", "respiratory_other"),
    "afrin": ("oxymetazoline", "respiratory_other"),
    "sultolin inhaler": ("salbutamol", "bronchodilator"),
    "sultolin": ("salbutamol", "bronchodilator"),
    "azmasol hfa inhaler": ("salbutamol", "bronchodilator"),
    "azmasol": ("salbutamol", "bronchodilator"),
    "ventolin": ("salbutamol", "bronchodilator"),
    "beclomin 100 inhaler": ("beclometasone", "corticosteroid"),
    "camlodin": ("amlodipine", "calcium_channel_blocker"),
    "losart 50 plus": ("losartan_hydrochlorothiazide", "antihypertensive_acei_arb"),
    "angilock 25 plus": ("losartan_hydrochlorothiazide", "antihypertensive_acei_arb"),
    "omlesan": ("olmesartan", "antihypertensive_acei_arb"),
    "omlesan 20 plus": ("olmesartan_hydrochlorothiazide", "antihypertensive_acei_arb"),
    "tenocab": ("atenolol", "beta_blocker"),
    "bisoprol": ("bisoprolol", "beta_blocker"),
    "nitrocard": ("glyceryl_trinitrate", "cardiac_other"),
    "catapress 0.1mg": ("clonidine", "cardiac_other"),
    "mixtard 30": ("insulin", "insulin"),
    "mixterd 30 hm 100": ("insulin", "insulin"),
    "thyrox": ("levothyroxine", "thyroid"),
    "rabeca": ("rabeprazole", "proton_pump_inhibitor"),
    "exium": ("esomeprazole", "proton_pump_inhibitor"),
    "pep 20": ("omeprazole", "proton_pump_inhibitor"),
    "marlox plus": ("antacid_combination", "antacid"),
    "flatameal ds": ("antacid_simethicone", "antacid"),
    # Magfin (51): syrup, constipation in half its encounters.
    "magfin": ("magnesium_hydroxide", "laxative"),
    "mom": ("magnesium_hydroxide", "laxative"),
    "milk of magnesia": ("magnesium_hydroxide", "laxative"),
    "laxena": ("", "laxative"),
    "meverin": ("mebeverine", "antispasmodic"),
    "mebeverin": ("mebeverine", "antispasmodic"),
    "tramal": ("tramadol", "opioid_analgesic"),
    "naprosyn plus": ("naproxen_esomeprazole", "nsaid"),
    "ultrafen": ("diclofenac", "nsaid"),
    "febus": ("febuxostat", "antigout"),
    "feburen": ("febuxostat", "antigout"),
    "jointec max": ("glucosamine", "other"),
    "joinix": ("glucosamine", "other"),
    "econate-vt": ("econazole", "antifungal"),
    "microral gel": ("miconazole", "antifungal"),
    "flugal ointment": ("fluconazole", "antifungal"),
    "omastin ointment": ("", "antifungal"),
    "nystat oral drop": ("nystatin", "antifungal"),
    "gynokmix vt": ("", "gynaecological"),
    "gynaepro vt": ("", "gynaecological"),
    "gynomix vt": ("", "gynaecological"),
    "ovestin": ("estriol", "gynaecological"),
    "clob": ("clobetasol", "dermatological"),
    "clobate ointment": ("clobetasol", "dermatological"),
    "dermovate": ("clobetasol", "dermatological"),
    "exovate": ("clobetasol", "dermatological"),
    "betnovate-cl": ("betamethasone_clioquinol", "dermatological"),
    "scabex": ("permethrin", "dermatological"),
    "permin 5%": ("permethrin", "dermatological"),
    "anustat": ("", "other"),
}
CURATED.update(CURATED_EVIDENCE)

# The corpus' own `rx_category` labels, folded onto the fixed vocabulary. Some
# labels are already molecule names, so a molecule is carried where the label
# supplies one. '' means "no usable signal": 'Others' and the blank label are
# passed on to the curated/LLM stages rather than folded into class 'other'.
CAT_MAP: dict[str, tuple[str | None, str]] = {
    # --- acid / GI -----------------------------------------------------
    "ppi": (None, "proton_pump_inhibitor"),
    "pantoprazol": ("pantoprazole", "proton_pump_inhibitor"),
    "esomeprazol": ("esomeprazole", "proton_pump_inhibitor"),
    "rabeprazol": ("rabeprazole", "proton_pump_inhibitor"),
    "rabeprazole": ("rabeprazole", "proton_pump_inhibitor"),
    "h2-blocker": (None, "h2_blocker"),
    "famotidine": ("famotidine", "h2_blocker"),
    "antacids": (None, "antacid"),
    "domperidne": ("domperidone", "prokinetic"),
    "ondansetron": ("ondansetron", "antiemetic"),
    "anti emetic": (None, "antiemetic"),
    "anti-vomiting": (None, "antiemetic"),
    "cinnarizine": ("cinnarizine", "antiemetic"),
    "cinnerizin+dimenhydrynate": ("cinnarizine_dimenhydrinate", "antiemetic"),
    "trimonium": ("tiemonium_methylsulphate", "antispasmodic"),
    "temorium methylsulphate": ("tiemonium_methylsulphate", "antispasmodic"),
    "antispasmodics": (None, "antispasmodic"),
    # --- analgesia -----------------------------------------------------
    "paracetamol": ("paracetamol", "analgesic_antipyretic"),
    "nsaids": (None, "nsaid"),
    "naproxen": ("naproxen", "nsaid"),
    "aceclofcnac": ("aceclofenac", "nsaid"),
    "aceclofenac": ("aceclofenac", "nsaid"),
    "diclofenac": ("diclofenac", "nsaid"),
    "ibuprofen": ("ibuprofen", "nsaid"),
    "esomeprazole + naproxen": ("naproxen_esomeprazole", "nsaid"),
    "tolperisone hydrochloride": ("tolperisone", "muscle_relaxant"),
    # --- cardiometabolic ------------------------------------------------
    "arb/acei": (None, "antihypertensive_acei_arb"),
    "combination antihypertenive": (None, "antihypertensive_acei_arb"),
    "anti-htm": (None, "antihypertensive_acei_arb"),
    "beta blocker": (None, "beta_blocker"),
    "ca blocker": (None, "calcium_channel_blocker"),
    "diuretics": (None, "diuretic"),
    "statin": (None, "statin"),
    "lipid lowering agents": (None, "statin"),
    "anti-platelet": (None, "antiplatelet"),
    "gtn": ("glyceryl_trinitrate", "cardiac_other"),
    "oral hypo glycenic drug": (None, "antidiabetic_other"),
    "hypoglycaemic agent": (None, "antidiabetic_other"),
    "insulin": ("insulin", "insulin"),
    # --- anti-infective -------------------------------------------------
    "anti-biotics": (None, "antibiotic"),
    "ciprofloxacin": ("ciprofloxacin", "antibiotic"),
    "levofloxacin": ("levofloxacin", "antibiotic"),
    "azithromycin": ("azithromycin", "antibiotic"),
    "cefixime": ("cefixime", "antibiotic"),
    "cefuroxime": ("cefuroxime", "antibiotic"),
    "ciprofloxacin+dexamethasone": ("ciprofloxacin_dexamethasone", "antibiotic"),
    "anti fungal": (None, "antifungal"),
    "clotrimazole": ("clotrimazole", "antifungal"),
    "acyclovir": ("aciclovir", "antiviral"),
    "albendazole": ("albendazole", "antiparasitic"),
    # --- respiratory / allergy -----------------------------------------
    "anti histacin": (None, "antihistamine"),
    "antihistamine": (None, "antihistamine"),
    "fexofenadin": ("fexofenadine", "antihistamine"),
    "loratadine": ("loratadine", "antihistamine"),
    "levocetrizin": ("levocetirizine", "antihistamine"),
    "rupatadine": ("rupatadine", "antihistamine"),
    "ketotifen": ("ketotifen", "antihistamine"),
    "monteleukast": ("montelukast", "respiratory_other"),
    "anti asthmatics": (None, "bronchodilator"),
    "xylometazoline": ("xylometazoline", "respiratory_other"),
    # --- CNS ------------------------------------------------------------
    "bromazepum": ("bromazepam", "anxiolytic_benzodiazepine"),
    "bromazepam": ("bromazepam", "anxiolytic_benzodiazepine"),
    "clonazepum": ("clonazepam", "anxiolytic_benzodiazepine"),
    "clonazepam": ("clonazepam", "anxiolytic_benzodiazepine"),
    "sedative": (None, "anxiolytic_benzodiazepine"),
    "tca/ssri": (None, "antidepressant"),
    "antidepresent": (None, "antidepressant"),
    "haloperidol": ("haloperidol", "antipsychotic"),
    "gabapentin": ("gabapentin", "antiepileptic"),
    "carbidopa + levodopa": ("levodopa_carbidopa", "neuro_other"),
    "trihexyphenidyl hydrochloride": ("trihexyphenidyl", "neuro_other"),
    "flunarizin": ("flunarizine", "neuro_other"),
    # --- supplements ----------------------------------------------------
    "vitamin": (None, "vitamin"),
    "vitamin+minaral": ("multivitamin_mineral", "vitamin"),
    "vitamin b1 + vitamin b6 + vitamin b12": ("vitamin_b_complex", "vitamin"),
    "mecobalamin": ("mecobalamin", "vitamin"),
    "calcium+vit+d": ("calcium_vitamin_d", "calcium_vitamin_d"),
    "calcium": ("calcium", "mineral_supplement"),
    "minerals": (None, "mineral_supplement"),
    "ferrous ascorbate": ("ferrous_ascorbate", "iron_supplement"),
    # --- other systems --------------------------------------------------
    "hormone": (None, "gynaecological"),
    "norethisterone": ("norethisterone", "gynaecological"),
    "allylestrenol": ("allylestrenol", "gynaecological"),
    "cream/ointment": (None, "dermatological"),
    "olopatadine": ("olopatadine", "ophthalmic"),
    "polyethylene glycol+propylene": ("artificial_tears", "ophthalmic"),
    "ear drop": (None, "other"),
    # --- deliberately not folded ---------------------------------------
    "others": (None, ""),
    "other": (None, ""),
    "cat_name": (None, ""),
}


def norm_name(s) -> str:
    """Canonical brand key: case, unicode, punctuation and spacing folded."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"[®™]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .-")


def normalise_category(cat: str) -> tuple[str | None, str]:
    """Fold an rx_category label onto (molecule|None, class|'')."""
    return CAT_MAP.get((cat or "").lower().strip(), (None, ""))


SCHEMA = {
    "type": "object",
    "properties": {
        "molecule": {"type": "string"},
        "drug_class": {"type": "string", "enum": CLASSES},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["molecule", "drug_class", "confidence"],
}

SYSTEM = ("You are a pharmacist in Bangladesh. You know local brand names of "
          "medicines marketed by Square, Beximco, Incepta, Renata, ACI and "
          "Eskayef. You identify the active molecule and therapeutic class of "
          "a brand.")


def llm_map(names: list[str], model: str, examples: list[str]) -> dict:
    from ..nlp.ollama_client import Ollama, OllamaError
    llm = Ollama(model=model, num_predict=200, timeout=180)
    shots = "\n".join(f'  "{n}" -> {{"molecule": "{CURATED[norm_name(n)][0]}", '
                      f'"drug_class": "{CURATED[norm_name(n)][1]}"}}'
                      for n in examples if norm_name(n) in CURATED)
    out = {}
    for i, nm in enumerate(names, 1):
        prompt = (f"Known examples from this same formulary:\n{shots}\n\n"
                  f'Brand: "{nm}"\n\n'
                  f"Identify its active molecule and therapeutic class.\n"
                  f"drug_class MUST be one of: {', '.join(CLASSES)}\n"
                  f"If you do not recognise the brand, use drug_class 'other' "
                  f"and confidence 'low'. Do not guess a plausible-sounding "
                  f"molecule for an unfamiliar brand.\n"
                  f'Return JSON: {{"molecule": "...", "drug_class": "...", '
                  f'"confidence": "high|medium|low"}}')
        try:
            r = llm.generate_json(prompt, SCHEMA, system=SYSTEM) or {}
        except OllamaError:
            r = {}
        cls = r.get("drug_class")
        conf = r.get("confidence")
        out[nm] = {
            "molecule": (r.get("molecule") or "").strip().lower() or None,
            "drug_class": cls if cls in CLASSES else None,
            "confidence": conf if conf in ("high", "medium", "low") else None,
        }
        if i % 25 == 0:
            print(f"   ... {i}/{len(names)}", flush=True)
    return out


def validate_llm(model: str, shots: list[str], n: int = 30) -> dict:
    """Re-map a random sample of CURATED brands blind and score agreement.

    Sampled across the whole curated table rather than its head: the earliest
    entries are the highest-volume, best-known products, so scoring only those
    would flatter the LLM relative to the long tail it is actually asked to map.
    """
    shot_keys = {norm_name(s) for s in shots}
    pool = sorted(k for k in CURATED if k not in shot_keys)
    val = random.Random(0).sample(pool, min(n, len(pool)))
    got = llm_map(val, model, shots)
    hits = [(k, CURATED[k][1], got[k]["drug_class"], got[k]["confidence"]) for k in val]
    agree = sum(1 for _, c, l, _ in hits if c == l)
    hi = [h for h in hits if h[3] == "high"]
    hi_agree = sum(1 for _, c, l, _ in hi if c == l)
    print(f"\nVALIDATION ({model}): LLM agreed with curated class on "
          f"{agree}/{len(val)} ({agree / len(val):.0%}) held-out brands")
    if hi:
        print(f"            high-confidence subset: {hi_agree}/{len(hi)} "
              f"({hi_agree / len(hi):.0%})")
    print(f"        {'brand':18s} {'curated':28s} llm (confidence)")
    for k, c, l, cf in hits:
        print(f"   {'ok  ' if c == l else 'MISS'} {k:18s} {c:28s} {l} ({cf})")
    return {"n": len(val), "agree": agree, "accuracy": agree / len(val),
            "n_high_conf": len(hi),
            "high_conf_accuracy": hi_agree / len(hi) if hi else None,
            "detail": [{"brand": k, "curated": c, "llm": l, "confidence": cf}
                       for k, c, l, cf in hits]}


def cross_era(orders: pd.DataFrame, enc: pd.DataFrame, m: pd.DataFrame) -> list[dict]:
    """Vocabulary churn between the <=2015 and >=2017 eras, at each granularity.

    The unseen-order rate is the number that matters: the share of later-era
    orders carrying a label absent from the earlier era, i.e. orders no model
    trained on the earlier era could ever emit. Orders whose brand resolves to
    nothing are kept in the denominator under an '<unmapped>' sentinel so the
    rate is not flattered by silently dropping them -- the original code
    dropped them, which is what made molecule/class look like a perfect 0.0%.
    The strict variant instead counts every unmapped later-era order as unseen,
    bounding the truth from above.
    """
    enc = enc.assign(year=pd.to_numeric(enc["year"], errors="coerce"))
    early = set(enc.loc[enc["year"] <= 2015, "prescription_id"])
    late = set(enc.loc[enc["year"] >= 2017, "prescription_id"])
    E = orders[orders["prescription_id"].isin(early)]
    L = orders[orders["prescription_id"].isin(late)]
    keymap = dict(zip(m["drug_id"], m["key"]))
    molmap = dict(zip(m["drug_id"], m["molecule"]))
    clsmap = dict(zip(m["drug_id"], m["drug_class"]))
    UNK = "<unmapped>"

    def labels(d: pd.DataFrame, lookup) -> list[str]:
        return [lookup(i) or UNK for i in d["drug_id"].dropna().astype(int)]

    levels = (
        ("brand (drug_id)", lambda i: f"id{i}"),
        ("brand (name-folded)", lambda i: keymap.get(i)),
        ("molecule", lambda i: molmap.get(i)),
        ("class", lambda i: clsmap.get(i)),
    )
    print(f"\n{'level':22s} {'|V|early':>9s} {'|V|late':>8s} {'Jaccard':>8s} "
          f"{'unseen':>8s} {'strict':>8s} {'top10':>7s}")
    print("-" * 78)
    rows = []
    for label, fn in levels:
        ka, kb = Counter(labels(E, fn)), Counter(labels(L, fn))
        sa, sb = set(ka) - {UNK}, set(kb) - {UNK}
        jac = len(sa & sb) / max(len(sa | sb), 1)
        tot = max(sum(kb.values()), 1)
        unseen = sum(c for k, c in kb.items() if k != UNK and k not in sa) / tot
        strict = unseen + kb.get(UNK, 0) / tot
        t = len({k for k, _ in ka.most_common(10)} & {k for k, _ in kb.most_common(10)})
        print(f"{label:22s} {len(sa):9d} {len(sb):8d} {jac:8.3f} {unseen:7.1%} "
              f"{strict:7.1%} {t:5d}/10")
        rows.append({"level": label, "v_early": len(sa), "v_late": len(sb),
                     "jaccard": jac, "unseen_order_rate": unseen,
                     "unseen_order_rate_strict": strict, "top10_shared": t})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="map leftovers with the local LLM")
    ap.add_argument("--model", default="medgemma:latest")
    ap.add_argument("--min-conf", default="low", choices=["low", "medium", "high"],
                    help="lowest LLM confidence accepted into the map")
    ap.add_argument("--validate-only", action="store_true",
                    help="score the LLM against held-out curated brands and stop")
    args = ap.parse_args()

    shots = ["Napa", "Maxpro", "Comet", "Alatrol", "Ferocit", "Amdocal"]
    if args.validate_only:
        validate_llm(args.model, shots)
        return

    orders = pd.read_parquet(PROCESSED / "rxgen_orders.parquet")
    V = json.loads((PROCESSED / "rxgen_vocab.json").read_text())
    d2c = {int(k): v for k, v in V["drug2cat"].items()}
    id2cat = {v: k for k, v in V["category"].items()}
    catname = V.get("category_names", {})

    drugs = (orders.dropna(subset=["drug_id"])
             .assign(drug_id=lambda d: d["drug_id"].astype(int))
             .groupby("drug_id")
             .agg(drug_name=("drug_name", "first"), n=("drug_id", "size"))
             .reset_index())
    drugs["key"] = drugs["drug_name"].map(norm_name)

    # --- 1. case/whitespace folding -------------------------------------
    named = drugs[drugs["key"] != ""]
    dupes = named.groupby("key")["drug_id"].nunique()
    n_dupe_keys = int((dupes > 1).sum())
    n_dupe_ids = int(dupes[dupes > 1].sum() - (dupes > 1).sum())
    print(f"brands: {len(drugs)} ids -> {named['key'].nunique()} canonical names "
          f"({n_dupe_keys} names carried {n_dupe_ids} redundant id(s)); "
          f"{int((drugs['key'] == '').sum())} ids have no name at all")

    # --- 2. resolve molecule/class --------------------------------------
    rows = []
    unresolved_cats = Counter()
    for _, r in drugs.iterrows():
        key, did = r["key"], int(r["drug_id"])
        mol = cls = None
        src = "unmapped"
        raw = catname.get(str(id2cat.get(d2c.get(did, 0), "")), "")
        if key in CURATED:
            mol, cls = CURATED[key]
            src = "curated"
        else:
            cmol, folded = normalise_category(raw)
            if folded:
                mol, cls, src = cmol, folded, "rx_category"
            elif raw and raw.lower() not in CAT_MAP:
                unresolved_cats[raw] += int(r["n"])
        rows.append({"drug_id": did, "drug_name": r["drug_name"], "key": key,
                     "n_orders": int(r["n"]), "rx_category": raw, "molecule": mol,
                     "drug_class": cls, "source": src, "llm_confidence": None})
    m = pd.DataFrame(rows)
    if unresolved_cats:
        print("WARNING: rx_category labels missing from CAT_MAP:",
              dict(unresolved_cats.most_common(10)))

    # Propagate a curated/category hit to every id sharing the canonical name.
    known = (m[m["drug_class"].notna() & (m["key"] != "")]
             .sort_values("n_orders", ascending=False)
             .drop_duplicates("key").set_index("key"))
    fill = m["drug_class"].isna() & (m["key"] != "") & m["key"].isin(known.index)
    m.loc[fill, "drug_class"] = m.loc[fill, "key"].map(known["drug_class"])
    m.loc[fill, "molecule"] = m.loc[fill, "key"].map(known["molecule"])
    m.loc[fill, "source"] = "name_fold"
    todo_orders = int(m.loc[m["drug_class"].isna(), "n_orders"].sum())
    print(f"after curation + category + name folding: "
          f"{m['drug_class'].notna().sum()}/{len(m)} brands, "
          f"{100 * (1 - todo_orders / len(orders)):.1f}% of orders; "
          f"{todo_orders} orders left for the LLM")

    # --- 3. LLM for the remainder ---------------------------------------
    validation = None
    if args.llm:
        # Validate FIRST: if the LLM cannot recover brands we already know,
        # there is no reason to trust it on the ones we do not.
        validation = validate_llm(args.model, shots)

        todo = (m[m["drug_class"].isna() & (m["key"] != "")]
                .sort_values("n_orders", ascending=False)["drug_name"]
                .dropna().unique().tolist())
        print(f"\nLLM mapping {len(todo)} unmapped brands with {args.model} ...")
        got = llm_map(todo, args.model, shots)
        (OUT / "llm_raw_mappings.json").write_text(
            json.dumps(got, indent=2), encoding="utf-8")
        rank = {"low": 0, "medium": 1, "high": 2}
        floor = rank[args.min_conf]
        kept = 0
        for nm, r in got.items():
            if r["drug_class"] and rank.get(r["confidence"] or "low", 0) >= floor:
                sel = m["drug_name"] == nm
                m.loc[sel, "drug_class"] = r["drug_class"]
                m.loc[sel, "molecule"] = r["molecule"]
                m.loc[sel, "source"] = "llm"
                m.loc[sel, "llm_confidence"] = r["confidence"]
                kept += 1
        print(f"accepted {kept}/{len(todo)} LLM mappings at min-conf={args.min_conf}")

    m.to_parquet(PROCESSED / "drug_normalization.parquet", index=False)

    cov_brand = m["drug_class"].notna().mean()
    ordmap = dict(zip(m["drug_id"], m["drug_class"]))
    molmap = dict(zip(m["drug_id"], m["molecule"]))
    ords = orders["drug_id"].dropna().astype(int)
    cov_order = sum(1 for x in ords if ordmap.get(x)) / len(ords)
    cov_mol = sum(1 for x in ords if molmap.get(x)) / len(ords)
    print(f"\ncoverage: {cov_brand:.1%} of brands, {cov_order:.1%} of orders at "
          f"class level ({cov_mol:.1%} at molecule level); was 86.8% of orders")
    print("sources:", dict(Counter(m["source"])))
    print("classes used:", m["drug_class"].nunique(),
          " molecules:", m["molecule"].nunique())
    print("largest still-unmapped:",
          (m[m["drug_class"].isna()].nlargest(8, "n_orders")[["drug_name", "n_orders"]]
           .to_records(index=False).tolist()))

    # --- 4. cross-era stability at each granularity ---------------------
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    rows = cross_era(orders, enc, m)

    (OUT / "drug_normalization.json").write_text(json.dumps(
        {"coverage_brands": cov_brand, "coverage_orders": cov_order,
         "coverage_orders_molecule": cov_mol,
         "sources": {k: int(v) for k, v in Counter(m["source"]).items()},
         "n_classes": int(m["drug_class"].nunique()),
         "n_molecules": int(m["molecule"].nunique()),
         "llm_model": args.model if args.llm else None,
         "llm_min_conf": args.min_conf if args.llm else None,
         "llm_validation": validation,
         "cross_era": rows}, indent=2, default=float), encoding="utf-8")
    print("\nwrote", PROCESSED / "drug_normalization.parquet")


if __name__ == "__main__":
    main()
