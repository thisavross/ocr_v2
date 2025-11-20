import tempfile
from PIL import Image
import re
import numpy as np
import cv2
import os
import glob
import numpy as np
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
import re
from tqdm import tqdm
from rapidfuzz import process, fuzz
from typing import List, Tuple, Dict, Any

def preprocess_img_to_tempfile(img_path):
    """Preprocess image and save to temporary file for Doctr"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    # Blurriness check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 120:
        # Sharpen
        gaussian_blur = cv2.GaussianBlur(img, (5,5), sigmaX=2)
        img = cv2.addWeighted(img, 1.2, gaussian_blur, -0.5, 0)
        # Contrast CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert to PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Save to temporary file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_img.save(tmp_file.name)
    tmp_file.close()
    return tmp_file.name

def smart_ktp_ocr(image_path):
    """Enhanced KTP OCR using preprocessed temp file"""
    model = ocr_predictor(pretrained=True)

    # Preprocess and get temp file path
    temp_image_path = preprocess_img_to_tempfile(image_path)

    # Use DocumentFile.from_images with file path
    doc = DocumentFile.from_images([temp_image_path])

    result = model(doc)

    # Extract all text
    extracted_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                extracted_text.append(line_text)

    return extracted_text, result


#define key
EXPECTED_FIELDS = [
    "PROVINSI", "KOTA", "KABUPATEN", "NIK", "NAMA", "TEMPAT/TGL LAHIR",
    "JENIS KELAMIN","GOL DARAH", "ALAMAT", "RT/RW", "KEL/DESA", "KECAMATAN",
    "AGAMA", "STATUS PERKAWINAN", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU HINGGA"
]

#define value
CONTROLLED_VALUES = {
    "JENIS KELAMIN" : ["LAKILAKI", "PEREMPUAN"],
    "GOL_DARAH" : ["A", "B", "AB", "O"],
    "STATUS PERKAWINAN" : ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"],
    "AGAMA" : ["BUDDHA", "HINDU", "ISLAM", "KATOLIK", "KRISTEN", "KONGHUCU", "KEPERCAYAAN"]
}


def fuzzy_match_value(field, value, threshold=80):
    """
    If field has controlled values (like gender or blood type),
    do fuzzy matching to correct OCR errors.
    """
    if field in CONTROLLED_VALUES:
        # Compare OCR value to allowed list
        match = process.extractOne(value.upper(), [v.upper() for v in CONTROLLED_VALUES[field]], scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            # Return correctly cased value from CONTROLLED_VALUES
            idx = [v.upper() for v in CONTROLLED_VALUES[field]].index(match[0])
            return CONTROLLED_VALUES[field][idx]
    return value  # return original if no controlled list or no good match

# def fuzzy_match_field(text, threshold=70):
#     """Fuzzy match text to expected KTP field names"""
#     if not text.strip():
#         return None
#     match = process.extractOne(text.upper(), EXPECTED_FIELDS, scorer=fuzz.ratio)
#     if match and match[1] >= threshold:
#         return match[0]
#     return None

def fuzzy_match_field(text, threshold=70):
    """Fuzzy match text to expected KTP field names (robust version)"""
    if not text or not text.strip():
        return None
    
    # Clean and normalize text
    clean_text = re.sub(r'[^A-Z0-9\s/]', '', text.upper())
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Early return if text becomes empty after cleaning
    if not clean_text:
        return None
    
    # Use partial ratio for better substring matching
    match = process.extractOne(clean_text, EXPECTED_FIELDS, scorer=fuzz.partial_ratio)
    
    if match and match[1] >= threshold:
        return match[0]
    return None


def parse_ktp_lines(lines):
    """
    Parse OCR lines into structured key-value pairs,
    even when ':' is missing or OCR is fuzzy.
    """
    data = {}
    skip_next = False

    for i, raw_line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        line = raw_line.strip()
        if not line:
            continue

        # --- Case 1: Normal 'key: value' line ---
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().replace('/', '_').replace(' ', '_')
            data[key] = value.strip()
            continue

        # --- Case 2: Next line starts with ':' ---
        if i + 1 < len(lines) and lines[i+1].strip().startswith(':'):
            key = line.strip().replace('/', '_').replace(' ', '_')
            value = lines[i+1].strip(': ').strip()
            data[key] = value
            skip_next = True
            continue

        # --- Case 3: Line looks like "PROVINSI XXX" (no ':') ---
        # Try fuzzy match on the first token (potential key)
        parts = line.split(maxsplit=1)
        if len(parts) >= 2:
            possible_key = parts[0]
            matched_field = fuzzy_match_field(possible_key)
            if matched_field:
                data[matched_field.replace('/', '_').replace(' ', '_')] = parts[1].strip()
                continue

        # --- Case 4: Fuzzy match whole line to expected field ---
        matched_field = fuzzy_match_field(line)
        if matched_field:
            # If next line looks like a value, grab it
            if i + 1 < len(lines) and not fuzzy_match_field(lines[i+1]):
                data[matched_field.replace('/', '_').replace(' ', '_')] = lines[i+1].strip()
                skip_next = True
            else:
                data[matched_field.replace('/', '_').replace(' ', '_')] = ""
            continue

        # --- Case 5: Handle special numeric-only fields like NIK ---
        if re.search(r"\b\d{8,}\b", line):
            data["NIK"] = re.sub(r"\D", "", line)
            continue

    return data

# ========== ENHANCED BLOOD TYPE PARSING ==========

def normalize_blood_type(text):
    """Normalize blood type text before parsing"""
    if not text:
        return ""
    
    # Convert to uppercase and strip
    text = text.upper().strip()
    
    # Common OCR errors for blood types
    blood_type_corrections = {
        '0': 'O',  # Zero to letter O
        'O': 'O',  # Ensure consistent
        'A': 'A',
        'B': 'B',
        'AB': 'AB',
        'ABO': 'AB',  # Handle extra characters
        'BO': 'B',
        'AO': 'A'
    }
    
    # Extract only blood type characters
    blood_chars = re.sub(r'[^ABO0]', '', text)
    
    # Apply corrections
    if blood_chars in blood_type_corrections:
        return blood_type_corrections[blood_chars]
    
    return text

def jaro_winkler_similarity(s1, s2):
    """Calculate Jaro-Winkler similarity between two strings"""
    return JaroWinkler.similarity(s1, s2) * 100

def levenshtein_similarity(s1, s2):
    """Calculate Levenshtein similarity between two strings"""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 100
    distance = Levenshtein.distance(s1, s2)
    return (1 - distance / max_len) * 100

def parse_blood_type_character_based(text, threshold=80):
    """Use character-level distance metrics for blood type parsing"""
    if not text:
        return None
    
    normalized = normalize_blood_type(text)
    controlled_values = ["A", "B", "AB", "O"]
    
    # Try multiple character-based scorers
    scorers = [
        ("ratio", fuzz.ratio),
        ("partial_ratio", fuzz.partial_ratio),
        ("token_sort_ratio", fuzz.token_sort_ratio),
        ("jaro_winkler", jaro_winkler_similarity),
        ("levenshtein", levenshtein_similarity)
    ]
    
    best_overall_match = None
    best_overall_score = 0
    
    for scorer_name, scorer in scorers:
        try:
            best_match, score, _ = process.extractOne(
                normalized,
                controlled_values,
                scorer=scorer
            )
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall_match = best_match
                
        except Exception as e:
            continue
    
    # Special handling for single character matches
    if len(normalized) == 1:
        single_char_scores = {}
        for blood_type in controlled_values:
            # For single char input, check if it matches any character in blood type
            if normalized in blood_type:
                similarity = fuzz.ratio(normalized, blood_type)
                single_char_scores[blood_type] = similarity
        
        if single_char_scores:
            best_single_char = max(single_char_scores.items(), key=lambda x: x[1])
            if best_single_char[1] > best_overall_score:
                best_overall_match, best_overall_score = best_single_char
    
    return best_overall_match if best_overall_score >= threshold else None

def extract_blood_type_by_pattern(text):
    """Extract blood type using regex patterns"""
    patterns = [
        r'GOLDARAH\s*:?\s*([ABO]+)',        # Handle "GOLDARAH" stuck together
        r'GOL\.?\s*DARAH\s*:?\s*([ABO]+)',  # Standard format
        r'DARAH\s*:?\s*([ABO]+)',           # Without "GOL"
        r'([ABO]{1,2})\s*$',                # Standalone at end
        r':\s*([ABO]{1,2})\s*',             # After colon
    ]

    # First, check if the text contains only a dash or is empty after "GOL DARAH"
    if re.search(r'GOL\.?\s*DARAH\s*:?\s*-$', text.upper()) or \
       re.search(r'GOL\.?\s*DARAH\s*:?\s*$', text.upper()):
        return None  # Empty blood type
    
    for pattern in patterns:
        try:
            match = re.search(pattern, text.upper())
            if match:
                blood_type = match.group(1)
                # Skip if we matched just a dash or empty
                if blood_type.strip() in ['', '-']:
                    continue
                normalized = normalize_blood_type(blood_type)
                if normalized in ["A", "B", "AB", "O"]:
                    return normalized
        except re.error:
            # Skip invalid regex patterns
            continue
    
    return None

def parse_blood_type_comprehensive(text, threshold=70):
    """Multi-stage blood type parsing with character-level metrics"""
    if not text or text.strip() in ['', '-']:
        return None
    
    # Check for empty blood type patterns
    if re.search(r'GOL\.?\s*DARAH\s*:?\s*-$', text.upper()) or \
       re.search(r'GOL\.?\s*DARAH\s*:?\s*$', text.upper()):
        return None
    
    # Stage 1: Direct normalization and exact match
    normalized = normalize_blood_type(text)
    if normalized in ["A", "B", "AB", "O"]:
        return normalized
    
    # Stage 2: Pattern extraction
    pattern_match = extract_blood_type_by_pattern(text)
    if pattern_match:
        return pattern_match
    
    # Stage 3: Character-based fuzzy matching
    char_based_match = parse_blood_type_character_based(text, threshold)
    if char_based_match:
        return char_based_match
    
    # Stage 4: Direct character search as last resort
    for char in text.upper():
        normalized_char = normalize_blood_type(char)
        if normalized_char in ["A", "B", "AB", "O"]:
            return normalized_char
    
    return None

# --- Fuzzy matching function ---

def fuzzy_match_value(key, val):
    """
    Fuzzy-correct value based on controlled list for the given key.
    Only returns the corrected string, not a tuple.
    """
    controlled_list = CONTROLLED_VALUES.get(key, [])
    
    # 1. If exact match exists, use it
    if val in controlled_list:
        return val

    # 2. Fuzzy match fallback
    if controlled_list:
        match, score, *_ = process.extractOne(val, controlled_list, score_cutoff=70) or (val, 0)
        return match
    
    # 3. If no controlled list or no match, return original
    return val

def normalize_key(key):
    key = key.strip().replace("/", "_").replace(" ", "_").upper()
    # Consistent key mapping
    KEY_MAP = {
        "PROVINSI" :"PROVINSI",
        "KOTA" : "KOTA",
        "KABUPATEN" : "KABUPATEN",
        "NIK": "NIK",
        "NAMA": "Nama", 
        "TEMPAT_TGL_LAHIR": "Tempat_Tgl_Lahir",
        "JENIS_KELAMIN": "Jenis_Kelamin",
        "GOL_DARAH": "Gol_Darah", 
        "ALAMAT": "Alamat",
        "RT_RW": "RT_RW",
        "KEL_DESA": "Kel_Desa", 
        "KECAMATAN": "Kecamatan",
        "AGAMA": "Agama",
        "STATUS_PERKAWINAN": "Status_Perkawinan",
        "PEKERJAAN": "Pekerjaan",
        "KEWARGANEGARAAN": "Kewarganegaraan",
        "BERLAKU_HINGGA": "Berlaku_Hingga"
    }
    return KEY_MAP.get(key, key)


# --- Helper functions ---
def clean_text(text):
    return text.strip()

def extract_value(line):
    """Extract value after ':' and clean it"""
    if ':' in line:
        parts = line.split(':', 1)
        value = parts[1].strip()
        # Remove any leading colon that might remain
        if value.startswith(':'):
            value = value[1:].strip()
        return value
    return ''

def is_date(text):
    # Simple date check (DD/MM/YYYY)
    return bool(re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text))

def is_value_line(line):
    return bool(line.strip())

def extract_birth_date(tempat_tgl_lahir):
    """Extract birth date from Tempat_Tgl_Lahir field"""
    if not tempat_tgl_lahir:
        return None
    dates = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', str(tempat_tgl_lahir))
    return dates[0] if dates else None

def correct_nik_ocr(text):
    """Correct common OCR misreadings in NIK"""
    if not text:
        return None
    
    # Common OCR misreadings (letter -> number)
    ocr_corrections = {
        'O': '0', 'o': '0',
        'I': '1', 'i': '1', 'l': '1', 'L': '1',
        'S': '5', 's': '5',
        'B': '8', 'b': '8',
        'Z': '2', 'z': '2',
        'G': '6', 'g': '6',
        'T': '7', 't': '7',
        'Q': '9', 'q': '9',
        'E': '3', 'e': '3',
        'A': '4', 'a': '4'
    }
    
    # Apply corrections
    corrected = ''
    for char in text:
        if char in ocr_corrections:
            corrected += ocr_corrections[char]
        elif char.isdigit():
            corrected += char
        # Ignore non-alphanumeric characters
    
    return corrected

def extract_nik_with_fuzzy(text_lines):
    """Extract and correct NIK using fuzzy matching"""
    nik_patterns = ["NIK", "NIK:", "NIK :", "NOMOR INDUK KEPENDUDUKAN"]
    
    for i, line in enumerate(text_lines):
        line_clean = line.upper().strip()
        
        # Fuzzy match for NIK label
        for pattern in nik_patterns:
            if fuzz.partial_ratio(line_clean, pattern) >= 80:
                # Look for NIK value in current or next line
                if i + 1 < len(text_lines):
                    potential_nik = text_lines[i + 1].strip()
                    corrected_nik = correct_nik_ocr(potential_nik)
                    
                    # Validate NIK (16 digits)
                    if corrected_nik and len(corrected_nik) == 16 and corrected_nik.isdigit():
                        return corrected_nik
                
                # Also check current line (if NIK and value are on same line)
                parts = line.split()
                for part in parts:
                    if part.isdigit() or any(c.isalpha() for c in part):
                        corrected = correct_nik_ocr(part)
                        if corrected and len(corrected) == 16 and corrected.isdigit():
                            return corrected
    
    return None

def enhanced_nik_extraction(text_lines):
    """Extract NIK using multiple fuzzy strategies"""
    
    def find_nik_line_index(text_lines):
        """Find the line containing NIK using fuzzy matching"""
        for i, line in enumerate(text_lines):
            line_upper = line.upper().strip()
            
            # Fuzzy match with common NIK patterns
            nik_scores = [
                fuzz.partial_ratio(line_upper, "NIK"),
                fuzz.partial_ratio(line_upper, "NIK:"),
            ]
            
            if max(nik_scores) >= 75:
                return i
        return -1
    
    nik_line_idx = find_nik_line_index(text_lines)
    
    if nik_line_idx == -1:
        return None
    
    # Strategy 1: Check next line (most common format)
    if nik_line_idx + 1 < len(text_lines):
        next_line = text_lines[nik_line_idx + 1].strip()
        corrected = correct_nik_ocr(next_line)
        if validate_nik(corrected):
            return corrected
    
    # Strategy 2: Check current line (NIK and value might be together)
    current_line = text_lines[nik_line_idx]
    parts = re.split(r'[\s:]', current_line)
    
    for part in parts:
        if len(part) >= 16:  # Potential NIK value
            corrected = correct_nik_ocr(part)
            if validate_nik(corrected):
                return corrected
    
    # Strategy 3: Extract all potential 16-character sequences
    all_text = ' '.join(text_lines)
    potential_niks = re.findall(r'[A-Z0-9]{16,}', all_text)
    
    for potential in potential_niks:
        corrected = correct_nik_ocr(potential)
        if validate_nik(corrected):
            return corrected
    
    return None

def validate_nik(nik):
    """Validate NIK format"""
    if not nik or len(nik) != 16:
        return False
    return nik.isdigit()

# --- Main parser ---
def parse_ktp_fields(text_lines):
    ktp_data = {}
    lines = [clean_text(l) for l in text_lines if l.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        upper = line.upper()
        
        # --- NIK ---
        if fuzzy_match_field(upper) == "NIK":
            val = extract_value(line) 
            if not val and i + 1 < len(lines):
                # Use the enhanced NIK correction instead of simple digit extraction
                val = correct_nik_ocr(lines[i + 1])
            
            # If still no value, try to find NIK in nearby lines
            if not val:
                val = enhanced_nik_extraction(text_lines)
            
            if val:
                ktp_data["NIK"] = val

        # --- Provinsi ---
        elif fuzzy_match_field(upper) == "PROVINSI":
            val = re.sub(r'^PROVINSI\s*', '', line, flags=re.IGNORECASE).strip()
            
            # If value is empty, check next line (unlikely but safe)
            if not val and i + 1 < len(lines) and not fuzzy_match_field(lines[i + 1].upper()):
                val = lines[i + 1].strip()
                i += 1
                
            if val:
                ktp_data["Provinsi"] = val

        # --- Kabupaten ---  
        elif fuzzy_match_field(upper) == "KABUPATEN":
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            else:
                val = line.replace('KABUPATEN', '').strip()
            ktp_data["Kabupaten"] = val

        # --- Nama ---
        elif fuzzy_match_field(upper) == "NAMA":
            val = extract_value(line) 
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            # Clean leading colon if present
            if val and val.startswith(':'):
                val = val[1:].strip()
            ktp_data["Nama"] = val
            
        # --- Tempat/Tgl Lahir ---
        elif any(keyword in upper for keyword in ["TEMPAT", "TGL", "LAHIR", "TGI"]):
            # Handle various OCR variations
            if ':' in line:
                # Extract everything after the colon
                val = line.split(':', 1)[1].strip()
            else:
                # Try to remove common label patterns
                patterns = [
                    r'TEMPAT\s*[/\\]?\s*TGL?\s*LAHIR\s*',
                    r'TEMPAT\s*[/\\]?\s*TGI\s*LAHIR\s*',  # Handle "Tgi" OCR error
                    r'.*LAHIR\s*'
                ]
                for pattern in patterns:
                    val = re.sub(pattern, '', line, flags=re.IGNORECASE).strip()
                    if val != line:  # If substitution happened
                        break
                else:
                    val = line.strip()  # Use whole line if no pattern matched
            
            # If still no meaningful value, check next line
            if (not val or val.upper() in ["TEMPAT", "TGL", "LAHIR"]) and i + 1 < len(lines):
                next_line = lines[i + 1]
                if not fuzzy_match_field(next_line.upper()):
                    val = next_line.strip()
                    i += 1  # Skip next line
            
            if val:
                ktp_data["Tempat_Tgl_Lahir"] = val

        # --- Jenis Kelamin ---
        elif fuzzy_match_field(upper) == "JENIS KELAMIN":
            val = extract_value(line) 
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            if val:
                # Clean leading colon
                if val.startswith(':'):
                    val = val[1:].strip()
                val = fuzzy_match_value("JENIS KELAMIN", val)
                ktp_data["Jenis_Kelamin"] = val

        # --- Golongan Darah ---
        elif fuzzy_match_field(upper) == "GOL DARAH":
            val = extract_value(line) 
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            
            # Check for empty blood type (just dash or empty)
            if val and val.strip() not in ['', '-']:
                if val.startswith(':'):
                    val = val[1:].strip()
                blood_type = parse_blood_type_comprehensive(val)
                if blood_type:
                    ktp_data["Gol_Darah"] = blood_type
                else:
                    # Empty blood type
                    ktp_data["Gol_Darah"] = None

        # --- Alamat ---
        elif fuzzy_match_field(upper) == "ALAMAT":
            val = extract_value(line)
            if val and val.startswith(':'):
                val = val[1:].strip()
                
            addr_parts = [val] if val else []

            # Collect address lines until we hit another field
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                next_upper = next_line.upper()
                if (fuzzy_match_field(next_upper) or 
                    any(stop in next_upper for stop in ["RT/RW", "RTRW", "RT_RW", "RTIRW", "KEL/DESA", "KECAMATAN"])):
                    break
                addr_parts.append(next_line.strip())
                j += 1
            
            ktp_data["Alamat"] = " ".join(addr_parts).strip()

        # --- RT/RW ---
        elif fuzzy_match_field(upper) in ["RT/RW", "RT_RW", "RTIRW"]:
            val = extract_value(line)
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            # Clean up OCR artifacts
            if val:
                if val.startswith(':'):
                    val = val[1:].strip()
                val = val.replace('I', '/').replace('|', '/').replace('-', ' ').strip()
                ktp_data["RT_RW"] = val

        # --- Kel/Desa ---
        elif fuzzy_match_field(upper) == "KEL/DESA":
            val = extract_value(line) 
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            if val:
                if val.startswith(':'):
                    val = val[1:].strip()
                ktp_data["Kel_Desa"] = val

        # --- Kecamatan - IMPROVED ---
        if "KECAMATAN" in upper and "Kecamatan" not in ktp_data:
            if ':' in line:
                val = line.split(':', 1)[1].strip()
                if val and not fuzzy_match_field(val.upper()):
                    ktp_data["Kecamatan"] = val
            elif i + 1 < len(lines) and not fuzzy_match_field(lines[i + 1].upper()):
                ktp_data["Kecamatan"] = lines[i + 1].strip()

        # --- Agama ---
        elif fuzzy_match_field(upper) == "AGAMA":
            val = extract_value(line) 
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            if val:
                if val.startswith(':'):
                    val = val[1:].strip()
                val = fuzzy_match_value("AGAMA", val)
                ktp_data["Agama"] = val

        # --- Status Perkawinan - IMPROVED ---
        elif "STATUS" in upper and "PERKAWINAN" in upper:
            # Handle "Status Perkawinan: BELUM KAWIN" format
            if ':' in line:
                parts = line.split(':', 1)
                val = parts[1].strip()
                if val.startswith(':'):
                    val = val[1:].strip()
            else:
                # No colon - extract after keywords
                val = re.sub(r'.*STATUS\s*PERKAWINAN\s*', '', line, flags=re.IGNORECASE).strip()
            
            # Check next line if needed
            if not val and i + 1 < len(lines) and not fuzzy_match_field(lines[i + 1].upper()):
                val = lines[i + 1].strip()
                i += 1
                
            if val:
                val = fuzzy_match_value("STATUS PERKAWINAN", val)
                ktp_data["Status_Perkawinan"] = val

        # --- Pekerjaan ---
        elif fuzzy_match_field(upper) == "PEKERJAAN":
            val = extract_value(line) 
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            if val:
                if val.startswith(':'):
                    val = val[1:].strip()
                ktp_data["Pekerjaan"] = val

        # --- Kewarganegaraan ---
        elif fuzzy_match_field(upper) == "KEWARGANEGARAAN":
            val = extract_value(line) 
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            if val:
                if val.startswith(':'):
                    val = val[1:].strip()
                ktp_data["Kewarganegaraan"] = val

        # --- Berlaku Hingga ---
        elif fuzzy_match_field(upper) == "BERLAKU HINGGA":
            val = extract_value(line) 
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            if val:
                if val.startswith(':'):
                    val = val[1:].strip()
                val = val.upper()
                if "SEUMUR" in val or "HIDUP" in val:
                    ktp_data["Berlaku_Hingga"] = "Seumur Hidup"
                elif is_date(val):
                    ktp_data["Berlaku_Hingga"] = val
                else:
                    cleaned = re.findall(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", val)
                    ktp_data["Berlaku_Hingga"] = cleaned[0] if cleaned else val

        i += 1

    # === SMART DATE ASSIGNMENT LOGIC ===
    # Extract all dates from the entire document
    all_dates = []
    for line in lines:
        dates = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', line)
        all_dates.extend(dates)
    
    # Remove duplicates while preserving order
    unique_dates = []
    for date in all_dates:
        if date not in unique_dates:
            unique_dates.append(date)
    
    # Extract birth date for exclusion
    birth_date = extract_birth_date(ktp_data.get("Tempat_Tgl_Lahir", ""))
    
    # Remove birth date from consideration
    non_birth_dates = [date for date in unique_dates if date != birth_date]
    
    # Check for SEUMUR HIDUP
    has_seumur_hidup = any(
        "SEUMUR" in line.upper() and "HIDUP" in line.upper()
        for line in lines
    )
    
    # Smart date assignment (only set if not already set by field parsing)
    if has_seumur_hidup and "Berlaku_Hingga" not in ktp_data:
        ktp_data["Berlaku_Hingga"] = "SEUMUR HIDUP"
    
    # Assign Tanggal Terbit and Berlaku Hingga based on sequential order
    if non_birth_dates:
        if len(non_birth_dates) >= 3:
            # Typical case: [terbit_date, other_date, berlaku_date]
            if "Tanggal_Terbit" not in ktp_data:
                ktp_data["Tanggal_Terbit"] = non_birth_dates[0]
            if "Berlaku_Hingga" not in ktp_data and not has_seumur_hidup:
                ktp_data["Berlaku_Hingga"] = non_birth_dates[2]
        
        elif len(non_birth_dates) == 2:
            # Two dates found: [terbit_date, berlaku_date]
            if "Tanggal_Terbit" not in ktp_data:
                ktp_data["Tanggal_Terbit"] = non_birth_dates[0]
            if "Berlaku_Hingga" not in ktp_data and not has_seumur_hidup:
                ktp_data["Berlaku_Hingga"] = non_birth_dates[1]
        
        elif len(non_birth_dates) == 1:
            # Only one date found - assume it's Tanggal Terbit
            if "Tanggal_Terbit" not in ktp_data:
                ktp_data["Tanggal_Terbit"] = non_birth_dates[0]
    
    # Final fallback: if no dates found, use empty strings
    if "Tanggal_Terbit" not in ktp_data:
        ktp_data["Tanggal_Terbit"] = ""
    if "Berlaku_Hingga" not in ktp_data:
        ktp_data["Berlaku_Hingga"] = ""

    return ktp_data


# def visualize_ocr_results(image, result):
#     """Visualize OCR results with bounding boxes"""
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.imshow(image)
#     ax.axis('off')
    
#     for page in result.pages:
#         for block in page.blocks:
#             for line in block.lines:
#                 # Get bounding box coordinates
#                 (x1, y1), (x2, y2) = line.geometry
#                 h, w = image.shape[:2]
#                 x1, x2 = int(x1 * w), int(x2 * w)
#                 y1, y2 = int(y1 * h), int(y2 * h)

                
#                 # Draw bounding box
#                 rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
#                                    fill=False, color='red', linewidth=1)
#                 ax.add_patch(rect)
                
#                 # Add text
#                 line_text = " ".join([word.value for word in line.words])
#                 ax.text(x1, y1-5, line_text, fontsize=8, color='red', 
#                        bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", alpha=0.7))
    
#     plt.title('OCR Results Visualization')
#     plt.tight_layout()
#     plt.show()

def enhanced_ktp_processing(image_path: str, visualize: bool = False) -> Tuple[List[str], Dict[str, Any], str]:
    """
    Full KTP OCR pipeline:
    1. Run OCR using Doctr
    2. Parse OCR lines into structured fields
    3. Normalize keys and fuzzy-correct controlled values
    4. Optional visualization
    5. Return raw text lines, structured dict, and JSON string

    Args:
        image_path (str): Path to the KTP image
        visualize (bool, optional): Whether to display OCR visualization. Defaults to False.

    Returns:
        Tuple[List[str], Dict[str, Any], str]: raw OCR lines, structured dictionary, JSON string
    """

    # Step 1: Run OCR
    text_lines, ocr_result = smart_ktp_ocr(image_path)
    if not text_lines:
        print(f"Warning: No text extracted from {image_path}")


    # Step 2: Parse OCR lines into structured KTP fields
    ktp_data = parse_ktp_fields(text_lines)

    # Step 3: Normalize keys and fuzzy-correct controlled values
    normalized_data = {}
    for key, val in ktp_data.items():
        clean_key = key.strip().replace(" ", "_").replace("/", "_").title()
        if clean_key in normalized_data:
            continue  # skip duplicate keys
        if isinstance(val, str):
            val = val.strip()
        normalized_val = fuzzy_match_value(clean_key.upper(), val)
        normalized_data[clean_key] = normalized_val

    # Step 4: Convert to JSON output
    output_json = json.dumps({"structured_data": normalized_data}, indent=4, ensure_ascii=False)

    return text_lines, normalized_data, output_json


if __name__ == "__main__":
    # Folder containing your images
    image_folder = os.path.join(os.path.dirname(__file__), "ktp_image")
    
    # Create result directory
    result_dir = os.path.join(image_folder, "result")
    os.makedirs(result_dir, exist_ok=True)

    # Find all images
    extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, f"*.{ext}")))

    if not image_files:
        print("❌ No images found in folder:", image_folder)
    else:
        print(f"🔍 Found {len(image_files)} image(s) in {image_folder}\n")

        for image_path in image_files:
            print(f"🖼️ Processing: {os.path.basename(image_path)} ...")

            try:
                text_lines, normalized_data, output_json = enhanced_ktp_processing(image_path, visualize=True)

                print("\n=== OCR RAW TEXT ===")
                for line in text_lines:
                    print(line)

                print("\n=== STRUCTURED JSON OUTPUT ===")
                print(output_json)

                # Save JSON to file (same name as image but with .json extension)
                output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
                output_path = os.path.join(result_dir, output_filename)  # Fixed this line
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output_json)

                print(f"✅ Saved result to {output_path}\n")

            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}\n")