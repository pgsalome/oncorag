#!/usr/bin/env python3
"""
Generate synthetic medical notes for testing the Oncorag2 extraction pipeline.

This module provides functions to create realistic synthetic patient data
in the form of markdown files structured to match the feature extraction requirements.
"""

import os
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta

# Sample patient data to use for generating synthetic records
PATIENT_DATA = {
    "first_name": "John",
    "last_name": "Anderson",
    "mrn": "76549821",
    "dob": "06/18/1958",
    "age": 66,
    "sex": "Male",
    "race": "White",
    "ethnicity": "Non-Hispanic",
    "diagnosis": "Lung Adenocarcinoma",
    "diagnosis_date": "01/22/2025",
    "smoking_status": "Former",
    "pack_years": 30,
    "ecog_status": 1,
    "height_cm": 178,
    "weight_kg": 82,
    "bmi": 25.9,
    "diabetes": True,
    "hypertension": True,
    "comorbidities": ["Hypertension", "Type 2 Diabetes Mellitus", "Hyperlipidemia"],
    "zipcode": "19103",
    "state": "Pennsylvania",
    "country": "United States",
    "residence_type": "Urban",
    "tumor_size_cm": 3.2
}

# Templates for different medical note types
DISCHARGE_SUMMARY_TEMPLATE = """# DISCHARGE SUMMARY

**Date**: {current_date}
**Patient**: {first_name} {last_name}
**MRN**: {mrn}
**DOB**: {dob}
**Age**: {age}
**Sex**: {sex}
**Race**: {race}
**Ethnicity**: {ethnicity}

**Date of Admission**: {admission_date}
**Date of Discharge**: {discharge_date}
**Primary Diagnosis**: Stage III {diagnosis}
**Date of Diagnosis**: {diagnosis_date}

## ATTENDING PHYSICIAN
Dr. Sarah Wilson, MD

## ADMITTING DIAGNOSIS
1. {diagnosis}, Stage IIIA (T2N2M0)
2. Pneumonia

## DISCHARGE DIAGNOSIS
1. {diagnosis}, Stage IIIA (T2N2M0)
2. Resolving pneumonia
3. Hypertension
4. Type 2 Diabetes Mellitus

## HISTORY OF PRESENT ILLNESS
{age}-year-old {sex_lowercase} with recently diagnosed {diagnosis} presented with fever, productive cough, and shortness of breath for 3 days. Patient was admitted for treatment of pneumonia and further workup of {his_her} {cancer_type}. {he_she} has completed 1 cycle of chemotherapy (carboplatin/paclitaxel) prior to this admission.

## PHYSICAL EXAMINATION
**Height**: {height_cm} cm
**Weight**: {weight_kg} kg
**BMI**: {bmi}
**Blood Pressure**: 148/92 mmHg
**Pulse**: 92 BPM
**Temperature**: 37.8°C
**Respiratory Rate**: 20/min
**O2 Saturation**: 94% on room air

**General**: Patient is a well-developed, well-nourished {sex_lowercase} in no acute distress.
**HEENT**: Normocephalic, atraumatic. Pupils equal, round, and reactive to light. Oropharynx clear.
**Neck**: Supple, no lymphadenopathy.
**Lungs**: Decreased breath sounds in right upper lobe. Scattered rhonchi and crackles throughout right lung field.
**Cardiovascular**: Regular rate and rhythm. No murmurs, gallops, or rubs.
**Abdomen**: Soft, non-tender, non-distended. Bowel sounds normal.
**Extremities**: No edema, cyanosis, or clubbing.
**Neurological**: Alert and oriented x3. Cranial nerves II-XII intact. Motor strength 5/5 in all extremities.

## ADDITIONAL INFORMATION
**Smoking Status**: {smoking_status} smoker ({smoking_quit_detail})
**Pack Years**: {pack_years}
**ECOG Performance Status**: {ecog_status}
**Comorbidities**: {comorbidities_list}
**Allergies**: None known
**ZIP Code**: {zipcode}
**State/Country**: {state}, {country}
**Residence**: {residence_type}
**Socioeconomic Status**: Middle income

---

Electronically signed by: Sarah Wilson, MD
Date: {discharge_date} 2:45 PM
"""

PATHOLOGY_REPORT_TEMPLATE = """# PATHOLOGY REPORT

**Report Number**: P-2025-37852
**Date of Procedure**: {procedure_date}
**Date of Report**: {report_date}

**Patient Name**: {first_name} {last_name}
**MRN**: {mrn}
**DOB**: {dob}
**Sex**: {sex}
**Race**: {race}
**Ethnicity**: {ethnicity}

**Attending Physician**: Emily Chen, MD
**Requesting Physician**: Robert Thompson, MD

## CLINICAL INFORMATION
{age}-year-old {sex_lowercase} with {pack_years} pack-year smoking history ({smoking_status} smoker, {smoking_quit_detail}) and {tumor_size_cm} cm right upper lobe lung mass identified on CT scan.

## SPECIMEN RECEIVED
A. Right upper lobe lung, CT-guided core needle biopsy (3 cores)

## DIAGNOSIS
Right upper lobe lung, CT-guided core needle biopsy (3 cores):
- Invasive adenocarcinoma, moderately differentiated (Grade 2)
- Histologic Pattern: Acinar (60%) and papillary (40%)
- Tumor Size (based on imaging): {tumor_size_cm} cm
- KRAS G12C mutation present
- PD-L1 expression: 55%

**Electronically signed**: {report_date} 11:23 AM

***This is the end of the pathology report***
"""

SURGICAL_REPORT_TEMPLATE = """# SURGICAL REPORT

**Date of Surgery**: {surgery_date}
**Time of Surgery**: 08:15 AM - 11:45 AM

**Patient Name**: {first_name} {last_name}
**MRN**: {mrn}
**DOB**: {dob}
**Age**: {age}
**Sex**: {sex}
**Race**: {race}
**Ethnicity**: {ethnicity}

**Surgeon**: Michael Rodriguez, MD
**Assistant Surgeon**: Jennifer Lee, MD

## PREOPERATIVE DIAGNOSIS
Right upper lobe lung adenocarcinoma, Stage IIIA (T2N2M0)

## POSTOPERATIVE DIAGNOSIS
Right upper lobe lung adenocarcinoma, Stage IIIA (T2N2M0)

## PROCEDURE PERFORMED
Right upper lobectomy with mediastinal lymph node dissection

## FINDINGS
- {tumor_size_cm_adjusted} cm firm mass in the right upper lobe with no direct invasion of the chest wall or mediastinum
- Enlarged lymph nodes at station 4R, measuring up to 1.5 cm
- No evidence of pleural metastases or malignant pleural effusion
- No tumor involvement of major vascular structures

---

Electronically signed by: Michael Rodriguez, MD
Date: {surgery_date} 3:15 PM
"""

IMAGING_REPORT_TEMPLATE = """# RADIOLOGY REPORT

**Exam Date**: {exam_date}
**Report Date**: {report_date}

**Patient Name**: {first_name} {last_name}
**MRN**: {mrn}
**DOB**: {dob}
**Age**: {age}
**Sex**: {sex}
**Race**: {race}
**Ethnicity**: {ethnicity}
**Weight**: {weight_kg} kg
**Height**: {height_cm} cm

**Ordering Physician**: Robert Thompson, MD
**Radiologist**: Jennifer Williams, MD

## EXAMINATION PERFORMED
CT Chest with IV Contrast

## CLINICAL INDICATION
{age}-year-old {sex_lowercase}, {smoking_status} smoker ({pack_years} pack-years, {smoking_quit_detail}) with persistent cough for 3 months. Recent chest X-ray showed a right upper lobe opacity.

## IMPRESSION
1. {tumor_size_cm} cm spiculated mass in the right upper lobe, highly suspicious for primary lung malignancy. Clinical stage T2aN2M0 (Stage IIIA) based on imaging findings.
2. Enlarged right paratracheal, subcarinal, and right hilar lymph nodes, concerning for nodal metastases.
3. Mild centrilobular emphysema, consistent with smoking history.
4. No evidence of distant metastatic disease in the visualized structures.

---

Electronically signed by: Jennifer Williams, MD
Board Certified Radiologist
Date: {report_date} 9:45 AM

***This is the end of the radiology report***
"""


def generate_dates():
    """Generate a set of realistic dates for medical records"""
    current_date = datetime.now().strftime("%m/%d/%Y")

    # Convert string dates to datetime objects for calculations
    diag_date = datetime.strptime(PATIENT_DATA["diagnosis_date"], "%m/%d/%Y")

    # Generate dates relative to diagnosis date
    imaging_date = (diag_date - timedelta(days=12)).strftime("%m/%d/%Y")
    imaging_report_date = (diag_date - timedelta(days=11)).strftime("%m/%d/%Y")
    procedure_date = (diag_date - timedelta(days=7)).strftime("%m/%d/%Y")
    pathology_report_date = (diag_date - timedelta(days=4)).strftime("%m/%d/%Y")
    surgery_date = (diag_date + timedelta(days=21)).strftime("%m/%d/%Y")
    admission_date = (diag_date + timedelta(days=45)).strftime("%m/%d/%Y")
    discharge_date = (diag_date + timedelta(days=52)).strftime("%m/%d/%Y")

    # Follow-up dates
    followup_date1 = (diag_date + timedelta(days=59)).strftime("%m/%d/%Y")
    followup_date2 = (diag_date + timedelta(days=66)).strftime("%m/%d/%Y")
    followup_date3 = (diag_date + timedelta(days=73)).strftime("%m/%d/%Y")

    # Include both keys for every date needed by templates
    return {
        "current_date": current_date,
        "exam_date": imaging_date,
        "imaging_date": imaging_date,
        "imaging_report_date": imaging_report_date,
        "procedure_date": procedure_date,
        "report_date": pathology_report_date,
        "pathology_report_date": pathology_report_date,
        "surgery_date": surgery_date,
        "admission_date": admission_date,
        "discharge_date": discharge_date,
        "followup_date1": followup_date1,
        "followup_date2": followup_date2,
        "followup_date3": followup_date3
    }


def download_example_pdf(output_dir):
    """Download an example pathology report PDF from a public source"""
    # Use a sample pathology report PDF that's publicly available
    url = "https://www.cancer.org/content/dam/CRC/PDF/Public/8580.00.pdf"
    output_path = Path(output_dir) / "pathology_report.pdf"

    # Only download if the file doesn't exist
    if not output_path.exists():
        print(f"Downloading sample pathology report from {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded sample report to {output_path}")
        except Exception as e:
            print(f"Error downloading sample report: {e}")
            # Create a small dummy PDF if download fails
            try:
                with open(output_path, 'w') as f:
                    f.write("Sample PDF content - dummy file")
                print(f"Created dummy PDF at {output_path}")
            except:
                print("Could not create dummy PDF")
            return None
    else:
        print(f"Using existing sample report at {output_path}")

    return str(output_path)


def create_synthetic_notes(output_dir, include_pdf=True):
    """
    Create synthetic medical notes and return the path where they were created.

    Args:
        output_dir (str): Directory to save the notes
        include_pdf (bool): Whether to include a PDF sample

    Returns:
        str: Path to the directory containing the generated data
    """
    try:
        output_dir = Path(output_dir)

        # Create the necessary directories
        (output_dir / "summary").mkdir(parents=True, exist_ok=True)
        (output_dir / "pathology").mkdir(parents=True, exist_ok=True)
        (output_dir / "surgery").mkdir(parents=True, exist_ok=True)
        (output_dir / "imaging").mkdir(parents=True, exist_ok=True)

        # Generate dates
        dates = generate_dates()

        # Prepare variables for templates
        data = PATIENT_DATA.copy()
        data.update(dates)

        # Additional formatting
        data["sex_lowercase"] = data["sex"].lower()
        data["his_her"] = "his" if data["sex"] == "Male" else "her"
        data["he_she"] = "he" if data["sex"] == "Male" else "she"
        data["He_She"] = "He" if data["sex"] == "Male" else "She"
        data["cancer_type"] = "lung cancer"
        data["comorbidities_list"] = ", ".join(data["comorbidities"])
        data["smoking_quit_detail"] = "quit 5 years ago" if data["smoking_status"] == "Former" else ""
        data["tumor_size_cm_width"] = round(data["tumor_size_cm"] - 0.4, 1)
        data["tumor_size_cm_height"] = round(data["tumor_size_cm"] - 0.2, 1)
        data["tumor_size_cm_adjusted"] = round(data["tumor_size_cm"] + 0.3,
                                               1)  # Slightly different measurement in surgery

        # Generate the notes from templates
        discharge_summary = DISCHARGE_SUMMARY_TEMPLATE.format(**data)
        pathology_report = PATHOLOGY_REPORT_TEMPLATE.format(**data)
        surgical_report = SURGICAL_REPORT_TEMPLATE.format(**data)
        imaging_report = IMAGING_REPORT_TEMPLATE.format(**data)

        # Write the notes to files
        (output_dir / "summary" / "summary.md").write_text(discharge_summary)
        (output_dir / "pathology" / "pathology.md").write_text(pathology_report)
        (output_dir / "surgery" / "surgery.md").write_text(surgical_report)
        (output_dir / "imaging" / "imaging.md").write_text(imaging_report)

        # Download example PDF pathology report if requested
        if not include_pdf:
            pdf_path = download_example_pdf(output_dir)

        print(f"Created synthetic medical notes in {output_dir}")

        # Return the path where data was created
        return str(output_dir)

    except Exception as e:
        print(f"Error generating synthetic data: {str(e)}")
        return None