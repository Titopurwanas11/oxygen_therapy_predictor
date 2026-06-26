"""
Clinical Feature Reference Database for OxyPredict.
Contains descriptions, normal ranges, and clinical importance for all 44 features.
"""

import pandas as pd

def get_clinical_features_data() -> list:
    """Get the full list of 44 clinical features with explanations."""
    return [
        {
            "Feature": "Age (months)",
            "Clinical Meaning": "Patient age in months.",
            "Normal Range": "0 - 216",
            "Clinical Importance": "Age affects respiratory physiology, organ development, and susceptibility to specific pathogens."
        },
        {
            "Feature": "Gender",
            "Clinical Meaning": "Biological sex of the patient.",
            "Normal Range": "Female / Male",
            "Clinical Importance": "Physiological baseline markers and susceptibility ratios vary between male and female children."
        },
        {
            "Feature": "Medical insurance",
            "Clinical Meaning": "Availability of healthcare insurance.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Socioeconomic indicator affecting access to early clinical consultation and medication."
        },
        {
            "Feature": "Number of persons living in house",
            "Clinical Meaning": "Total members living in the patient's household.",
            "Normal Range": "Integer (typically 2 - 15)",
            "Clinical Importance": "Crowded living conditions are a significant risk factor for transmission of viral pathogens."
        },
        {
            "Feature": "Number of siblings",
            "Clinical Meaning": "Number of brothers and sisters.",
            "Normal Range": "Integer (typically 0 - 10)",
            "Clinical Importance": "Indicates familial contact density and risk of sibling transmission."
        },
        {
            "Feature": "Number of rooms in house",
            "Clinical Meaning": "Count of habitable rooms in the residence.",
            "Normal Range": "Integer (typically 1 - 10)",
            "Clinical Importance": "Indicates physical spacing and house ventilation capacity."
        },
        {
            "Feature": "Smokers at home",
            "Clinical Meaning": "Exposure to passive cigarette smoke.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Secondhand smoke damages the pediatric respiratory epithelium, increasing infection risk and severity."
        },
        {
            "Feature": "Contact with a tuberculosis patient",
            "Clinical Meaning": "History of exposure to a TB patient.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Critically identifies risk of childhood TB, a serious pulmonary condition requiring specific therapy."
        },
        {
            "Feature": "Health history : Prior admission because of respiratory condition",
            "Clinical Meaning": "Previous hospitalization for breathing issues.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Indicates hyper-reactive airways or baseline chronic susceptibility."
        },
        {
            "Feature": "Health history : Prematurity",
            "Clinical Meaning": "Child born before 37 weeks of gestation.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Premature infants have structurally and functionally immature lungs, prone to severe infections."
        },
        {
            "Feature": "Breastfeeding",
            "Clinical Meaning": "History of breastfeeding.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Provides passive immunity (maternal antibodies) protecting infants from severe respiratory illnesses."
        },
        {
            "Feature": "Age-appropriate vaccinations",
            "Clinical Meaning": "Status of childhood immunization schedule.",
            "Normal Range": "No / Partially / Yes",
            "Clinical Importance": "Prevents serious bacterial (e.g. Hib, pneumococcal) and viral respiratory diseases."
        },
        {
            "Feature": "Previous history of antibiotic use in the 2 weeks leading up to the admission",
            "Clinical Meaning": "Recent antibiotic therapy before presenting.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Can mask underlying bacterial infections or indicate advanced pre-admission illness."
        },
        {
            "Feature": "Known asthmatic patient",
            "Clinical Meaning": "Prior clinical diagnosis of bronchial asthma.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Prone to severe bronchospasm and wheezing exacerbations requiring respiratory support."
        },
        {
            "Feature": "Patient with a diagnosed chronic condition",
            "Clinical Meaning": "Presence of long-term underlying illnesses (e.g., cardiac, genetic).",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Comorbidities significantly lower physiological reserves, increasing threat of respiratory failure."
        },
        {
            "Feature": "Duration of pain before consultation (days)",
            "Clinical Meaning": "Days since clinical symptoms started.",
            "Normal Range": "0 - 30 days",
            "Clinical Importance": "Reflects disease progression and delays in seeking medical attention."
        },
        {
            "Feature": "History of fever",
            "Clinical Meaning": "Anamnesis reporting elevated body temperature.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Suggests active systemic immune response to infection."
        },
        {
            "Feature": "Number of days with fever",
            "Clinical Meaning": "Duration of febrile symptoms.",
            "Normal Range": "0 - 30 days",
            "Clinical Importance": "Prolonged fever points to severe infective processes or atypical complications."
        },
        {
            "Feature": "History of vomiting",
            "Clinical Meaning": "Anamnesis reporting emesis.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Associated with severity, feeding difficulties, or systemic stress."
        },
        {
            "Feature": "Number of days of vomiting",
            "Clinical Meaning": "Duration of emesis.",
            "Normal Range": "0 - 30 days",
            "Clinical Importance": "Prolonged vomiting increases dehydration risk and limits oral intake, worsening distress."
        },
        {
            "Feature": "History of diarrhea",
            "Clinical Meaning": "Anamnesis reporting loose watery stools.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Suggests systemic/viral spread or concurrent infection, contributing to dehydration."
        },
        {
            "Feature": "History of cough",
            "Clinical Meaning": "Anamnesis reporting coughing.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Primary indicator of lower or upper respiratory tract irritation."
        },
        {
            "Feature": "History of rhinorrhea",
            "Clinical Meaning": "Anamnesis reporting runny nose.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Indicates upper respiratory congestion, typical in early viral infection phases."
        },
        {
            "Feature": "Weight (Kg)",
            "Clinical Meaning": "Patient weight.",
            "Normal Range": "1.5 - 60 Kg",
            "Clinical Importance": "Used to evaluate nutritional status and calculate medical dosages."
        },
        {
            "Feature": "Height (cm)",
            "Clinical Meaning": "Patient height.",
            "Normal Range": "45 - 150 cm",
            "Clinical Importance": "Somatic growth marker; useful for computing body surface area and BMI."
        },
        {
            "Feature": "Unusual sleepiness",
            "Clinical Meaning": "Lethargy or excessive somnolence.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Crucial red-flag sign of central nervous system distress, hypoxia, or severe exhaustion."
        },
        {
            "Feature": "Oxygen saturation (SaO2) at admission",
            "Clinical Meaning": "Arterial oxygen saturation measured via pulse oximetry.",
            "Normal Range": "95 - 100%",
            "Clinical Importance": "Values < 92-90% directly indicate hypoxemia, requiring oxygen supplementation."
        },
        {
            "Feature": "Axillary temperature (°C)",
            "Clinical Meaning": "Body temperature measured via axilla.",
            "Normal Range": "36.5 - 37.5°C",
            "Clinical Importance": "High values indicate fever/infection; low values (<36°C) can suggest neonatal sepsis."
        },
        {
            "Feature": "Respiratory rate",
            "Clinical Meaning": "Breaths per minute (bpm).",
            "Normal Range": "Age-dependent (typically 20 - 60)",
            "Clinical Importance": "Elevated rates (tachypnea) are a primary diagnostic sign of pediatric pneumonia."
        },
        {
            "Feature": "Heart rate",
            "Clinical Meaning": "Heart beats per minute (bpm).",
            "Normal Range": "Age-dependent (typically 80 - 160)",
            "Clinical Importance": "Tachycardia indicates systemic infection, fever, dehydration, or hypoxemia."
        },
        {
            "Feature": "Paleness",
            "Clinical Meaning": "Skin pallor.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Indicates anemia, poor perfusion, or systemic vasoconstriction in response to stress."
        },
        {
            "Feature": "Disorders of consciousness",
            "Clinical Meaning": "Altered mental status (stupor, coma).",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Critical emergency sign indicating severe cerebral hypoxia or sepsis."
        },
        {
            "Feature": "Dehydration signs",
            "Clinical Meaning": "Signs of fluid loss (e.g. dry mucosa, poor skin turgor).",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Worsens respiratory exhaustion and decreases cardiovascular output."
        },
        {
            "Feature": "Restlessness",
            "Clinical Meaning": "Agitation or irritability.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Early sign of cerebral hypoxemia and breathing distress in infants."
        },
        {
            "Feature": "Cyanosis",
            "Clinical Meaning": "Bluish discoloration of the skin or mucous membranes.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Severe clinical sign of central hypoxemia (low oxygen saturation)."
        },
        {
            "Feature": "Nasal flaring",
            "Clinical Meaning": "Widening of the nostrils during inhalation.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Key clinical sign of increased work of breathing and respiratory distress."
        },
        {
            "Feature": "Laryngeal stridor",
            "Clinical Meaning": "High-pitched, inspiratory sound from upper airway obstruction.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Indicates croup or upper airway swelling requiring prompt assessment."
        },
        {
            "Feature": "Rhonchi",
            "Clinical Meaning": "Low-pitched rattling breath sounds resembling snoring.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Suggests secretions or fluid blockage in the larger airways."
        },
        {
            "Feature": "Crackles",
            "Clinical Meaning": "Discontinuous clicking or rattling lung sounds.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Suggests fluid or exudates in the alveoli, indicative of pneumonia."
        },
        {
            "Feature": "Wheezing",
            "Clinical Meaning": "High-pitched whistling sound during exhalation.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Indicates narrowing of lower small airways (bronchiolitis, asthma)."
        },
        {
            "Feature": "Hypoventilation",
            "Clinical Meaning": "Abnormally slow or shallow breathing.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Dreaded sign of imminent respiratory muscle exhaustion and hypercapnia."
        },
        {
            "Feature": "Nasopharyngeal aspiration",
            "Clinical Meaning": "History of nasopharyngeal suctioning.",
            "Normal Range": "No / Yes",
            "Clinical Importance": "Identifies patients needing airway clearing for breathing ease."
        },
        {
            "Feature": "C-reactive protein",
            "Clinical Meaning": "Acute-phase inflammatory marker in serum.",
            "Normal Range": "< 10 mg/L",
            "Clinical Importance": "Elevated values suggest active systemic inflammation or bacterial infection."
        },
        {
            "Feature": "Procalcitonin",
            "Clinical Meaning": "Highly specific inflammatory biomarker.",
            "Normal Range": "< 0.5 ng/mL",
            "Clinical Importance": "Significant elevations highly correlate with invasive bacterial infections/sepsis."
        }
    ]

def get_clinical_reference_df() -> pd.DataFrame:
    """Get the full reference data as a pandas DataFrame."""
    return pd.DataFrame(get_clinical_features_data())
