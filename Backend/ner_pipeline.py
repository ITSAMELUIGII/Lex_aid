import spacy
from spacy.pipeline import EntityRuler
import re

# Load standard English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Add a custom EntityRuler to the SpaCy pipeline before 'ner' 
# This handles the specific legal tagging that a pre-trained 
# OpenNyAI model would handle, mapping out Indian specific context.
ruler = nlp.add_pipe("entity_ruler", before="ner")

legal_patterns = [
    # Identify roles in scenarios identifying who is who
    {"label": "PETITIONER_ROLE", "pattern": [{"LOWER": "tenant"}]},
    {"label": "RESPONDENT_ROLE", "pattern": [{"LOWER": "landlord"}]},
    
    # Identify statutes
    {"label": "STATUTE", "pattern": [{"LOWER": "consumer"}, {"LOWER": "protection"}, {"LOWER": "act"}]},
    {"label": "STATUTE", "pattern": [{"LOWER": "indian"}, {"LOWER": "penal"}, {"LOWER": "code"}]},
    {"label": "STATUTE", "pattern": [{"LOWER": "model"}, {"LOWER": "tenancy"}, {"LOWER": "act"}]},
    
    # Identify Sections/Articles roughly
    {"label": "PROVISION", "pattern": [{"LOWER": {"IN": ["section", "sec.", "article", "art."]}}, {"IS_DIGIT": True}]},
    
    # Identify Court entities
    {"label": "COURT", "pattern": [{"LOWER": "supreme"}, {"LOWER": "court"}]},
    {"label": "COURT", "pattern": [{"LOWER": "high"}, {"LOWER": "court"}]},
    {"label": "COURT", "pattern": [{"LOWER": "national"}, {"LOWER": "commission"}]},
    {"label": "COURT", "pattern": [{"LOWER": "district"}, {"LOWER": "commission"}]},
]

ruler.add_patterns(legal_patterns)

def extract_legal_entities(text):
    """
    Parses a string using the hybrid spaCy NER pipeline and extracts specific structured legal entities.
    """
    doc = nlp(text)
    
    extracted = {
        "PETITIONERS_OR_RESPONDENTS": [],
        "MONEY_INVOLVED": [],
        "STATUTE": [],
        "PROVISION": [],
        "COURT": [],
        "OTHER_ENTITIES": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "PETITIONER_ROLE" or ent.label_ == "RESPONDENT_ROLE":
            extracted["PETITIONERS_OR_RESPONDENTS"].append((ent.text, ent.label_))
        elif ent.label_ == "MONEY":
            extracted["MONEY_INVOLVED"].append(ent.text)
        elif ent.label_ == "STATUTE":
            extracted["STATUTE"].append(ent.text)
        elif ent.label_ == "PROVISION":
            extracted["PROVISION"].append(ent.text)
        elif ent.label_ == "COURT":
            extracted["COURT"].append(ent.text)
        elif ent.label_ == "PERSON":
            extracted["OTHER_ENTITIES"].append((ent.text, "PERSON"))
        elif ent.label_ == "ORG" and "court" in ent.text.lower():
            extracted["COURT"].append(ent.text)

    # Some fallback Regex for money specifically formatted in Rupees (Rs. 50,000 or 50,000 rupee)
    rupee_pattern = r"((?:Rs\.?|₹|Rupees?)\s?[\d,]+|[\d,]+\s?(?:Rs\.?|₹|Rupees?))"
    for match in re.finditer(rupee_pattern, text, re.IGNORECASE):
        money_str = match.group(0)
        if money_str not in extracted["MONEY_INVOLVED"]:
            extracted["MONEY_INVOLVED"].append(money_str)

    return extracted

if __name__ == "__main__":
    print("\n--- Day 3 Legal NER Pipeline Tests ---\n")
    
    # 1. Core Feature Test: Messy User Complaint
    print("TEST 1: Messy User Complaint")
    complaint = "My landlord won't return my 50,000 rupee deposit and he told John to kick me out."
    print(f"Text: {complaint}")
    print("Extracted Info:", extract_legal_entities(complaint))
    
    # 2. Differentiate Petitioner and Respondent Test Cases
    print("\nTEST 2: Differentiate Roles")
    scenario = "The tenant filed a suit against the landlord because the landlord refused to fix the roof."
    print(f"Text: {scenario}")
    print("Extracted Info:", extract_legal_entities(scenario))
    
    # 3. Real Data Test: Consumer Protection Act 2019
    print("\nTEST 3: Real Data Usage - CPA 2019 text")
    cpa_text = "Whoever, by himself or by any other person on his behalf, manufactures for sale or stores or sells or distributes or imports any spurious goods shall be punished under Section 91 of the Consumer Protection Act, and cases might be taken up by the National Commission."
    print(f"Text: {cpa_text}")
    print("Extracted Info:", extract_legal_entities(cpa_text))
