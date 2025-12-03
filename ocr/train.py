import functools
from multiprocessing import Process
import os
import subprocess
import sys
import tempfile
import re
from PIL import Image, ImageFilter
import Levenshtein
import numpy as np
import cv2
import json
from typing import List, Tuple, Dict, Any
import logging
import threading
import time
import select
from contextlib import contextmanager
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
import concurrent.futures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

#  timeout decorator using concurrent.futures
def timeout(seconds):
    """Timeout decorator using concurrent.futures"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    raise TimeoutException(f"Operation timed out after {seconds} seconds")
        return wrapper
    return decorator

# FIXED: Context manager for timeouts
@contextmanager
def time_limit(seconds):
    """Context manager for timeouts using threads"""
    def timeout_handler():
        time.sleep(seconds)
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    # Start a timer thread
    timer = threading.Thread(target=timeout_handler)
    timer.daemon = True
    timer.start()
    
    try:
        yield
    except TimeoutException:
        raise
    finally:
        # Try to stop the timer (though it's daemonized)
        pass

    
# Pre-compile frequently used regex patterns (SPEED BOOST)
BLOOD_TYPE_PATTERNS = [
    re.compile(r'GOLDARAH\s*:?\s*([ABO]+)', re.IGNORECASE),
    re.compile(r'GOL\.?\s*DARAH\s*:?\s*([ABO]+)', re.IGNORECASE),
    re.compile(r'DARAH\s*:?\s*([ABO]+)', re.IGNORECASE),
    re.compile(r'([ABO]{1,2})\s*$', re.IGNORECASE),
    re.compile(r':\s*([ABO]{1,2})\s*', re.IGNORECASE),
]

EMPTY_BLOOD_PATTERNS = [
    re.compile(r'GOL\.?\s*DARAH\s*:?\s*-$', re.IGNORECASE),
    re.compile(r'GOL\.?\s*DARAH\s*:?\s*$', re.IGNORECASE)
]

DATE_PATTERN = re.compile(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}')
NIK_PATTERN = re.compile(r'[A-Z0-9]{16,}')
DIGIT_PATTERN = re.compile(r'\d{8,}')


# Pre-compute uppercase versions for faster matching (SPEED BOOST)
EXPECTED_FIELDS_UPPER = [
    "PROVINSI", "KOTA", "KABUPATEN", "NIK", "NAMA", "TEMPAT/TGL LAHIR",
    "JENIS KELAMIN","GOL DARAH", "ALAMAT", "RT/RW", "KEL/DESA", "KECAMATAN",
    "AGAMA", "STATUS PERKAWINAN", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU HINGGA"
]

CONTROLLED_VALUES = {
    "JENIS KELAMIN" : ["LAKILAKI", "PEREMPUAN"],
    "GOL_DARAH" : ["A", "B", "AB", "O"],
    "STATUS PERKAWINAN" : ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"],
    "AGAMA" : ["BUDDHA", "HINDU", "ISLAM", "KATOLIK", "KRISTEN", "KONGHUCU", "KEPERCAYAAN"]
}

# Pre-compute uppercase controlled values (SPEED BOOST)
CONTROLLED_VALUES_UPPER = {
    key: [v.upper() for v in values] 
    for key, values in CONTROLLED_VALUES.items()
}

# Create a set for O(1) lookups of perfect field matches
EXPECTED_FIELDS_SET = set(EXPECTED_FIELDS_UPPER)
# OPTIMIZED FUZZY MATCHING WITH PERFECT MATCH SKIPPING 
def fuzzy_match_value(field, value, threshold=60):
    """
    Optimized fuzzy matching with perfect match skipping.
    If field has controlled values, do fuzzy matching to correct OCR errors.
    """
    if field not in CONTROLLED_VALUES:
        return value  # No controlled list for this field
    
    # Skip fuzzy matching for perfect matches 
    value_upper = value.upper()
    controlled_list_upper = CONTROLLED_VALUES_UPPER[field]
    
    # 1. Check for exact match first 
    if value_upper in controlled_list_upper:
        idx = controlled_list_upper.index(value_upper)
        return CONTROLLED_VALUES[field][idx]
    
    # 2. Check for common OCR variations 
    corrected = quick_ocr_correction(value_upper, field)
    if corrected:
        return corrected
    
    # 3. Only then use fuzzy matching 
    match = process.extractOne(value_upper, controlled_list_upper, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        idx = controlled_list_upper.index(match[0])
        return CONTROLLED_VALUES[field][idx]
    
    return value  # return original if no good match

def quick_ocr_correction(value_upper, field):
    """Quick corrections for common OCR errors without fuzzy matching"""
    if field == "JENIS KELAMIN":
        # Common OCR errors for gender
        corrections = {
            "LAKILAKI": ["LAKILAKI", "LAKI-LAKI", "LAKI LAKI", "LAKILAK"],
            "PEREMPUAN": ["PEREMPUAN", "PEREMPUAN", "PEREMPUA", "PEREMPWN"]
        }
        for correct, variants in corrections.items():
            if value_upper in variants:
                return correct
    
    elif field == "AGAMA":
        # Common OCR errors for religion
        corrections = {
            "ISLAM": ["ISLAM", "ISIAM", "ISLAM", "ISLAM"],
            "KRISTEN": ["KRISTEN", "KRISTEN", "KRISTEN"],
            "KATOLIK": ["KATOLIK", "KATOLIK", "KATOLLK"],
            "HINDU": ["HINDU", "HINDU", "HINDU"],
            "BUDDHA": ["BUDDHA", "BUDHA", "BUDDHA"],
            "KONGHUCU": ["KONGHUCU", "KONGHUCU", "KONGHUCU"]
        }
        for correct, variants in corrections.items():
            if value_upper in variants:
                return correct
    
    elif field == "STATUS PERKAWINAN":
        # Common OCR errors for marital status
        corrections = {
            "BELUM KAWIN": ["BELUM KAWIN", "BELUMKAWIN", "BELUM KAWIN", "BELOM KAWIN"],
            "KAWIN": ["KAWIN", "KAWIN", "KAWIN"],
            "CERAI HIDUP": ["CERAI HIDUP", "CERAIHIDUP", "CERAI HIDUP"],
            "CERAI MATI": ["CERAI MATI", "CERAIMATI", "CERAI MATI"]
        }
        for correct, variants in corrections.items():
            if value_upper in variants:
                return correct
    
    return None

def fuzzy_match_field(text, threshold=70):
    """
    Optimized fuzzy field matching with perfect match skipping.
    Fuzzy match text to expected KTP field names.
    """
    if not text or not text.strip():
        return None
    
    # Clean and normalize text
    clean_text = re.sub(r'[^A-Z0-9\s/]', '', text.upper())
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Early return if text becomes empty after cleaning
    if not clean_text:
        return None
    
    # 1. Check for perfect match first (FAST PATH - O(1))
    if clean_text in EXPECTED_FIELDS_SET:
        return clean_text
    
    # 2. Check for common field variations (FAST PATH)
    field_variations = {
        "PROVINSI": ["PROVINSI", "PROV", "PROVINS"],
        "KABUPATEN": ["KABUPATEN", "KAB", "KABUPATEN"],
        "TEMPAT/TGL LAHIR": ["TEMPAT TGL LAHIR", "TEMPAT/TGLLAHIR"],
        "JENIS KELAMIN": ["JENIS KELAMIN", "JENISKELAMIN"],
        "GOL DARAH": ["GOL. DARAH", "GOLDARAH"],
        "RT/RW": ["RT RW", "RTRW"],
        "KEL/DESA": ["KEL DESA", "KELDESA"],
        "STATUS PERKAWINAN": ["STATUSPERKAWINAN"],
        "BERLAKU HINGGA": ["BERLAKUHINGGA"]
    }
    
    for expected_field, variations in field_variations.items():
        if clean_text in variations:
            return expected_field
    
    # 3. Only then use fuzzy matching (SLOW PATH)
    match = process.extractOne(clean_text, EXPECTED_FIELDS_UPPER, scorer=fuzz.partial_ratio)
    
    if match and match[1] >= threshold:
        return match[0]
    
    return None

def exact_field_match(text):
    """
    Fast exact field matching without fuzzy logic.
    Returns the field name if it's a perfect match or common variation.
    """
    text_upper = text.upper().strip()
    
    # Direct set lookup (fastest)
    if text_upper in EXPECTED_FIELDS_SET:
        return text_upper
    
    # Common variations lookup
    field_variations = {
        "PROVINSI": ["PROVINSI"],
        "KOTA": ["KOTA"], 
        "KABUPATEN": ["KABUPATEN", "KAB"],
        "NIK": ["NIK", "NIK:"],
        "NAMA": ["NAMA", "NAMA:"],
        "TEMPAT/TGL LAHIR": ["TEMPAT", "TGL", "LAHIR", "TEMPAT/TGL", "TEMPAT TGL"],
        "JENIS KELAMIN": ["JENIS", "KELAMIN", "JENISKELAMIN"],
        "GOL DARAH": ["GOL", "DARAH", "GOL.DARAH", "GOLDARAH"],
        "ALAMAT": ["ALAMAT"],
        "RT/RW": ["RT/RW", "RTRW", "RT", "RW"],
        "KEL/DESA": ["KEL/DESA", "KEL", "DESA", "KELDESA"],
        "KECAMATAN": ["KECAMATAN", "KEC"],
        "AGAMA": ["AGAMA"],
        "STATUS PERKAWINAN": ["STATUS", "PERKAWINAN", "STATUSPERKAWINAN"],
        "PEKERJAAN": ["PEKERJAAN"],
        "KEWARGANEGARAAN": ["KEWARGANEGARAAN"],
        "BERLAKU HINGGA": ["BERLAKU", "HINGGA", "BERLAKUHINGGA"]
    }
    
    for expected_field, variations in field_variations.items():
        if text_upper in variations:
            return expected_field
    
    return None


def extract_potential_nik_with_ocr_correction(text):
    """Extract and correct potential NIK from text with OCR error handling"""
    # Common OCR misreadings (number -> letter)
    ocr_corrections = {
        'O': '0', 'o': '0', 'Q': '0',
        'I': '1', 'i': '1', 'l': '1', 'L': '1', '|': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5', 
        'G': '6', 'g': '6',
        'T': '7', 't': '7',
        'B': '8', 'b': '8',
        'A': '4', 'a': '4',
        'E': '3', 'e': '3'
    }
    
    # Look for sequences that could be NIK (16 chars with mix of digits/letters)
    potential_matches = re.findall(r'[A-Z0-9]{16,}', text.upper())
    
    for match in potential_matches:
        if len(match) >= 16:
            # Apply OCR corrections
            corrected = ''
            for char in match:
                if char in ocr_corrections:
                    corrected += ocr_corrections[char]
                else:
                    corrected += char
            
            # Take only the first 16 characters
            corrected = corrected[:16]
            
            # Check if it's a valid NIK after correction
            if validate_nik(corrected):
                return corrected
    
    return None

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

#KTP LINE PARSING
def parse_ktp_lines(lines):
    """
    Optimized KTP line parsing with early returns for perfect matches.
    Parse OCR lines into structured key-value pairs.
    """
    data = {}
    skip_next = False
    processed_fields = set()  # Track processed fields to avoid duplicates

    for i, raw_line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        line = raw_line.strip()
        if not line:
            continue

        # --- Case 1: Normal 'key: value' line (FAST PATH) ---
        if ':' in line:
            key, value = line.split(':', 1)
            key_clean = key.strip().replace('/', '_').replace(' ', '_')
            value_clean = value.strip()
            
            # Skip if we've already processed this field
            if key_clean in processed_fields:
                continue
                
            # Try exact field matching first
            field_match = exact_field_match(key.strip())
            if field_match:
                data[field_match] = value_clean
                processed_fields.add(field_match)
                continue
            else:
                # Fall back to original logic
                data[key_clean] = value_clean
                continue

        # --- Case 2: Next line starts with ':' ---
        if i + 1 < len(lines) and lines[i+1].strip().startswith(':'):
            key_clean = line.strip().replace('/', '_').replace(' ', '_')
            value = lines[i+1].strip(': ').strip()
            
            if key_clean not in processed_fields:
                data[key_clean] = value
                processed_fields.add(key_clean)
                skip_next = True
                continue

        # --- Case 3: Line looks like "PROVINSI XXX" (no ':') ---
        parts = line.split(maxsplit=1)
        if len(parts) >= 2:
            possible_key = parts[0]
            
            # Try exact match first
            field_match = exact_field_match(possible_key)
            if field_match and field_match not in processed_fields:
                data[field_match] = parts[1].strip()
                processed_fields.add(field_match)
                continue
            
            # Then try fuzzy match
            matched_field = fuzzy_match_field(possible_key)
            if matched_field and matched_field not in processed_fields:
                field_key = matched_field.replace('/', '_').replace(' ', '_')
                data[field_key] = parts[1].strip()
                processed_fields.add(field_key)
                continue

        # --- Case 4: Fuzzy match whole line to expected field ---
        matched_field = fuzzy_match_field(line)
        if matched_field and matched_field not in processed_fields:
            field_key = matched_field.replace('/', '_').replace(' ', '_')
            
            # If next line looks like a value, grab it
            if i + 1 < len(lines) and not fuzzy_match_field(lines[i+1]):
                data[field_key] = lines[i+1].strip()
                skip_next = True
            else:
                data[field_key] = ""
            processed_fields.add(field_key)
            continue

        # --- Case 5: Handle special numeric-only fields like NIK ---
        if "NIK" not in processed_fields:
            # Strategy 1: Quick digit-only check (fastest path for clean OCR)
            digit_only_nik = re.search(r'\b(\d{16})\b', line)
            if digit_only_nik:
                data["NIK"] = digit_only_nik.group(1)
                processed_fields.add("NIK")
                continue
            
            # Strategy 2: OCR-corrected extraction (handles B→8, O→0, etc.)
            potential_nik = extract_potential_nik_with_ocr_correction(line)
            if potential_nik:
                data["NIK"] = potential_nik
                processed_fields.add("NIK")
                continue

    return data

# ========== OPTIMIZED BLOOD TYPE PARSING ==========

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

def parse_blood_type_comprehensive(text, threshold=70):
    """Optimized blood type parsing with early returns"""
    if not text or text.strip() in ['', '-']:
        return None
    
    text_upper = text.upper()
    
    # Check for empty blood type patterns using pre-compiled patterns
    for pattern in EMPTY_BLOOD_PATTERNS:
        if pattern.search(text_upper):
            return None
    
    # Direct normalization and exact match (FAST PATH)
    normalized = normalize_blood_type(text)
    if normalized in ["A", "B", "AB", "O"]:
        return normalized
    
    # Stage 2: Quick pattern check for common formats
    quick_patterns = [
        (r'GOL\.?DARAH\s*:?\s*([ABO])', 1),  # "GOLDARAH:A"
        (r'DARAH\s*:?\s*([ABO])', 1),        # "DARAH:B"  
        (r'^([ABO])$', 1),                   # Standalone "A"
        (r':\s*([ABO])\s*$', 1),             # ": A"
    ]
    
    for pattern, group in quick_patterns:
        match = re.search(pattern, text_upper)
        if match:
            blood_type = match.group(group)
            normalized = normalize_blood_type(blood_type)
            if normalized in ["A", "B", "AB", "O"]:
                return normalized
    
    #  Full pattern extraction with pre-compiled patterns
    for pattern in BLOOD_TYPE_PATTERNS:
        match = pattern.search(text_upper)
        if match:
            blood_type = match.group(1)
            # Skip if we matched just a dash or empty
            if blood_type.strip() in ['', '-']:
                continue
            normalized = normalize_blood_type(blood_type)
            if normalized in ["A", "B", "AB", "O"]:
                return normalized
    
    # Character-based fuzzy matching 
    char_based_match = parse_blood_type_character_based(text, threshold)
    if char_based_match:
        return char_based_match
    
    # Stage 5: Direct character search
    for char in text_upper:
        normalized_char = normalize_blood_type(char)
        if normalized_char in ["A", "B", "AB", "O"]:
            return normalized_char
    
    return None

# blood type helper functions 
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


def clean_text(text):
    return text.strip()

# def extract_value(line):
#     """Extract value after ':' and clean it"""
#     if ':' in line:
#         parts = line.split(':', 1)
#         value = parts[1].strip()
#         # Remove any leading colon that might remain
#         if value.startswith(':'):
#             value = value[1:].strip()
#         return value
#     return ''

def is_field_name(text, threshold=60):
    """
    Universal check if text is a field name/key.
    Returns True if text matches any field name (even fuzzily).
    """
    if not text or not text.strip():
        return False
    
    text_upper = text.upper().strip()
    
    # Quick exact matches
    if text_upper in EXPECTED_FIELDS_SET:
        return True
    
    # Check for common field patterns
    field_patterns = [
        r'^(PROVINSI|KOTA|KABUPATEN|NIK|NAMA|TEMPAT|TGL|LAHIR|JENIS|KELAMIN|'
        r'GOL|DARAH|ALAMAT|RT|RW|KEL|DESA|KECAMATAN|AGAMA|STATUS|PERKAWINAN|'
        r'PEKERJAAN|KEWARGANEGARAAN|BERLAKU|HINGGA)$',
        r'^[A-Z]{4,}$',  # All caps words of 4+ letters (often field names)
    ]
    
    for pattern in field_patterns:
        if re.match(pattern, text_upper):
            return True
    
    # Fuzzy match check
    matched_field = fuzzy_match_field(text_upper, threshold)
    return matched_field is not None

def extract_value(line, context_field=None):
    """Extract value after ':' and filter out field names"""
    if ':' not in line:
        return ''
    
    parts = line.split(':', 1)
    key_part = parts[0].strip()
    value = parts[1].strip()
    
    # Remove any leading colon
    if value.startswith(':'):
        value = value[1:].strip()
    
    # CRITICAL: If value is empty or just symbols, return empty
    if not value or value in [':', '-', '']:
        return ''
    
    # UNIVERSAL FIX: If the value looks like a field name, reject it
    if is_field_name(value):
        return ''  # Not a valid value - it's a field name
    
    # Additional check: if value contains the key itself
    if context_field:
        field_words = context_field.upper().split()
        value_upper = value.upper()
        
        # If value is just the key words (common OCR error)
        for word in field_words:
            if word == value_upper:
                return ''
    
    return value

def is_date(text):
    # Simple date check (DD/MM/YYYY)
    return bool(DATE_PATTERN.match(text))

def is_value_line(line):
    return bool(line.strip())

def extract_birth_date(tempat_tgl_lahir):
    """Extract birth date from Tempat_Tgl_Lahir field"""
    if not tempat_tgl_lahir:
        return None
    dates = DATE_PATTERN.findall(str(tempat_tgl_lahir))
    return dates[0] if dates else None

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

def parse_ktp_fields(text_lines):
    """
    Improved KTP field parsing with better field tracking and date assignment
    """
    ktp_data = {}
    lines = [clean_text(l) for l in text_lines if l.strip()]
    
    # Track processed fields
    processed_fields = set()
    # Store all dates found in the document
    all_dates_in_doc = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
            
        # Extract dates from current line for later use
        dates_in_line = DATE_PATTERN.findall(line)
        all_dates_in_doc.extend(dates_in_line)
        
        upper_line = line.upper()
        
        # === NIK ===
        if "NIK" not in processed_fields and "NIK" in upper_line:
            val = None
            # Check current line for NIK value
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            elif i + 1 < len(lines):
                # Check next line
                next_line = lines[i + 1]
                if next_line and next_line.strip():
                    # Check if next line contains digits (likely NIK)
                    if re.search(r'\d{16}', next_line):
                        val = next_line.strip()
                        i += 1
            
            if not val:
                # Extract from current line pattern
                nik_match = re.search(r'(\d{16})', line)
                if nik_match:
                    val = nik_match.group(1)
            
            if val:
                ktp_data["NIK"] = val
                processed_fields.add("NIK")
                i += 1
                continue
        
        # === NAMA ===
        elif "NAMA" not in processed_fields and "NAMA" in upper_line:
            val = None
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            
            # If no value after colon, check next lines for name
            if not val or val in [":", ""]:
                name_parts = []
                j = i + 1
                # Collect name parts until we hit another field
                while j < len(lines) and j - i < 5:  # Limit to 5 lines
                    next_line_upper = lines[j].upper()
                    # Stop if we encounter another field indicator
                    if (any(field in next_line_upper for field in ["TEMPAT", "TGL", "LAHIR", "JENIS", "KELAMIN"]) or
                        DATE_PATTERN.search(lines[j])):
                        break
                    name_parts.append(lines[j].strip())
                    j += 1
                
                if name_parts:
                    val = " ".join(name_parts).strip()
                    i = j - 1
            
            if val and val not in ["NAMA", ":", ""]:
                ktp_data["Nama"] = val
                processed_fields.add("NAMA")
                i += 1
                continue
        
        # === TEMPAT/TGL LAHIR ===
        elif "TEMPAT_TGL_LAHIR" not in processed_fields and any(keyword in upper_line for keyword in ["TEMPAT", "TGL", "LAHIR"]):
            # Try to find the birth info pattern
            j = i
            birth_info = []
            found_date = False
            
            while j < len(lines) and j - i < 4:  # Look ahead 4 lines max
                current = lines[j]
                current_upper = current.upper()
                
                # Stop if we hit the next major field
                if j > i and any(field in current_upper for field in ["JENIS", "KELAMIN", "GOL", "DARAH", "ALAMAT"]):
                    break
                
                birth_info.append(current.strip())
                
                # Check if we found a date in this line
                if DATE_PATTERN.search(current):
                    found_date = True
                
                j += 1
            
            if birth_info:
                # Join birth info and clean it up
                birth_text = " ".join(birth_info)
                # Remove "Tempat/Tgl Lahir" or similar prefixes
                birth_text = re.sub(r'^.*?(TEMPAT|TGL|LAHIR)[:\s]*', '', birth_text, flags=re.IGNORECASE).strip()
                
                if birth_text:
                    ktp_data["Tempat_Tgl_Lahir"] = birth_text
                    processed_fields.add("TEMPAT_TGL_LAHIR")
                    i = j - 1
                    i += 1
                    continue
        
        # === JENIS KELAMIN ===
        elif "JENIS_KELAMIN" not in processed_fields and ("JENIS" in upper_line or "KELAMIN" in upper_line):
            val = None
            # Try to find the gender value
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            
            # Look for gender in current or next line
            if not val or val in ["JENIS", "KELAMIN", ":", ""]:
                # Check current line for gender keywords
                if "LAKI" in upper_line or "PEREMPUAN" in upper_line:
                    val = line
                elif i + 1 < len(lines):
                    next_line_upper = lines[i + 1].upper()
                    if "LAKI" in next_line_upper or "PEREMPUAN" in next_line_upper:
                        val = lines[i + 1].strip()
                        i += 1
            
            if val:
                # Clean the value - remove field names
                val_clean = re.sub(r'.*(JENIS|KELAMIN)[:\s]*', '', val, flags=re.IGNORECASE).strip()
                if val_clean:
                    # Apply fuzzy matching for controlled value
                    matched_val = fuzzy_match_value("JENIS KELAMIN", val_clean)
                    ktp_data["Jenis_Kelamin"] = matched_val if matched_val else val_clean
                    processed_fields.add("JENIS_KELAMIN")
                    i += 1
                    continue
        
        # === GOLONGAN DARAH ===
        elif "GOL_DARAH" not in processed_fields and ("GOL" in upper_line or "DARAH" in upper_line):
            val = None
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            
            if not val or val in ["GOL", "DARAH", ":", ""]:
                # Check current and next line for blood type
                if any(bt in upper_line for bt in ["A", "B", "AB", "O"]):
                    val = line
                elif i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if any(bt in next_line.upper() for bt in ["A", "B", "AB", "O"]):
                        val = next_line.strip()
                        i += 1
            
            if val:
                # Parse blood type
                blood_type = parse_blood_type_comprehensive(val)
                if blood_type:
                    ktp_data["Gol_Darah"] = blood_type
                    processed_fields.add("GOL_DARAH")
                    i += 1
                    continue
        
        # === ALAMAT ===
        elif "ALAMAT" not in processed_fields and "ALAMAT" in upper_line:
            addr_parts = []
            if ':' in line:
                val = line.split(':', 1)[1].strip()
                if val and val not in [":", ""]:
                    addr_parts.append(val)
            
            # Collect address lines until next field
            j = i + 1
            while j < len(lines) and j - i < 10:  # Limit address to 10 lines
                next_line = lines[j]
                next_upper = next_line.upper()
                
                # Stop conditions for address
                stop_conditions = [
                    "RT/RW" in next_upper, "RTRW" in next_upper,
                    "KEL/DESA" in next_upper, "KEL" in next_upper and "DESA" in next_upper,
                    "KECAMATAN" in next_upper, "KEC" in next_upper,
                    "AGAMA" in next_upper, "STATUS" in next_upper,
                    "PEKERJAAN" in next_upper
                ]
                
                if any(stop_conditions):
                    break
                
                addr_parts.append(next_line.strip())
                j += 1
            
            if addr_parts:
                ktp_data["Alamat"] = " ".join(addr_parts).strip()
                processed_fields.add("ALAMAT")
                i = j - 1
                i += 1
                continue
        
        # === RT/RW ===
        elif "RT_RW" not in processed_fields and ("RT/RW" in upper_line or "RTRW" in upper_line or "RT" in upper_line and "RW" in upper_line):
            val = None
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            
            if not val or val in [":", ""]:
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if re.search(r'\d{2,3}\s*[/\\]\s*\d{2,3}', next_line):
                        val = next_line.strip()
                        i += 1
            
            if val:
                # Clean RT/RW format
                val_clean = re.sub(r'[^0-9/]', '', val)
                ktp_data["RT_RW"] = val_clean
                processed_fields.add("RT_RW")
                i += 1
                continue
        
        # === KEL/DESA ===
        elif "KEL_DESA" not in processed_fields and ("KEL" in upper_line or "DESA" in upper_line):
            val = None
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            
            if not val or val in ["KEL", "DESA", ":", ""]:
                if i + 1 < len(lines) and not fuzzy_match_field(lines[i + 1].upper()):
                    val = lines[i + 1].strip()
                    i += 1
            
            if val and val not in ["KEL", "DESA", ":", ""]:
                ktp_data["Kel_Desa"] = val
                processed_fields.add("KEL_DESA")
                i += 1
                continue
        
        # === AGAMA ===
        elif "AGAMA" not in processed_fields and "AGAMA" in upper_line:
            val = None
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            
            if not val or val in ["AGAMA", ":", ""]:
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # Check if next line looks like a religion
                    if any(religion in next_line.upper() for religion in ["ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "KONGHUCU"]):
                        val = next_line
                        i += 1
            
            if val:
                # Apply fuzzy matching for religion
                matched_val = fuzzy_match_value("AGAMA", val)
                ktp_data["Agama"] = matched_val if matched_val else val
                processed_fields.add("AGAMA")
                i += 1
                continue
        
        # === PEKERJAAN ===
        elif "PEKERJAAN" not in processed_fields and "PEKERJAAN" in upper_line:
            val = None
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            
            if not val or val in ["PEKERJAAN", ":", ""]:
                if i + 1 < len(lines) and not fuzzy_match_field(lines[i + 1].upper()):
                    val = lines[i + 1].strip()
                    i += 1
            
            if val and val not in ["PEKERJAAN", ":", ""]:
                ktp_data["Pekerjaan"] = val
                processed_fields.add("PEKERJAAN")
                i += 1
                continue
        
        # === KEWARGANEGARAAN ===
        elif "KEWARGANEGARAAN" not in processed_fields and ("KEWARGANEGARAAN" in upper_line or "WNI" in upper_line or "WNA" in upper_line):
            val = None
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            
            if not val or val in ["KEWARGANEGARAAN", ":", ""]:
                # Check current line for WNI/WNA
                if "WNI" in upper_line:
                    val = "WNI"
                elif "WNA" in upper_line:
                    val = "WNA"
                elif i + 1 < len(lines):
                    next_line_upper = lines[i + 1].upper()
                    if "WNI" in next_line_upper:
                        val = "WNI"
                        i += 1
                    elif "WNA" in next_line_upper:
                        val = "WNA"
                        i += 1
            
            if val and val not in ["KEWARGANEGARAAN", ":", ""]:
                ktp_data["Kewarganegaraan"] = val
                processed_fields.add("KEWARGANEGARAAN")
                i += 1
                continue
        
        # === PROVINSI ===
        elif "PROVINSI" not in processed_fields and "PROVINSI" in upper_line:
            val = None
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            else:
                # Remove "PROVINSI" prefix
                val = re.sub(r'^PROVINSI\s*', '', line, flags=re.IGNORECASE).strip()
            
            if not val and i + 1 < len(lines):
                next_line = lines[i + 1]
                if not fuzzy_match_field(next_line.upper()):
                    val = next_line.strip()
                    i += 1
            
            if val:
                ktp_data["Provinsi"] = val
                processed_fields.add("PROVINSI")
                i += 1
                continue
        
        i += 1
    
    # === SMART DATE PROCESSING ===
    # Remove duplicates while preserving order
    unique_dates = []
    for date in all_dates_in_doc:
        if date not in unique_dates:
            unique_dates.append(date)
    
    # Extract birth date if available
    birth_date = None
    if "Tempat_Tgl_Lahir" in ktp_data:
        birth_dates = DATE_PATTERN.findall(ktp_data["Tempat_Tgl_Lahir"])
        if birth_dates:
            birth_date = birth_dates[0]
    
    # Filter out birth date
    other_dates = [date for date in unique_dates if date != birth_date]
    
    # Check for SEUMUR HIDUP
    has_seumur_hidup = False
    for line in text_lines:
        line_upper = line.upper()
        if "SEUMUR" in line_upper and "HIDUP" in line_upper:
            has_seumur_hidup = True
            break
    
    # Assign dates smartly
    if has_seumur_hidup:
        ktp_data["Berlaku_Hingga"] = "SEUMUR HIDUP"
    
    if other_dates:
        if len(other_dates) >= 2:
            # Usually first non-birth date is issue date, second is expiry
            ktp_data["Tanggal_Terbit"] = other_dates[0]
            if not has_seumur_hidup:
                ktp_data["Berlaku_Hingga"] = other_dates[1]
        elif len(other_dates) == 1:
            # Single date - likely issue date
            ktp_data["Tanggal_Terbit"] = other_dates[0]
    
    # Ensure all required fields exist
    required_fields = ["NIK", "Nama", "Tempat_Tgl_Lahir", "Jenis_Kelamin", 
                      "Alamat", "RT_RW", "Kel_Desa", "Agama", "Pekerjaan",
                      "Kewarganegaraan", "Tanggal_Terbit", "Berlaku_Hingga"]
    
    for field in required_fields:
        if field not in ktp_data:
            ktp_data[field] = ""
    
    return ktp_data

# OCR Worker Management
_worker_process = None
_worker_lock = threading.Lock()

def cleanup_worker():
    """Force cleanup of worker process"""
    global _worker_process
    if _worker_process:
        try:
            _worker_process.terminate()
            try:
                _worker_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _worker_process.kill()
                _worker_process.wait()
        except:
            pass
        finally:
            _worker_process = None

def start_worker_process():
    """Start worker process"""
    global _worker_process
    
    try:
        logger.info("Starting OCR worker process...")
        
        # Cleanup any existing process
        cleanup_worker()
        
        worker_script = os.path.join(BASE_DIR, "workers.py")

        _worker_process = subprocess.Popen(
            [sys.executable, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Wait for worker to initialize
        logger.info("Waiting for worker to load models...")
        time.sleep(5)
        
        # Test if worker is responsive
        try:
            _worker_process.stdin.write("test\n")
            _worker_process.stdin.flush()
            
            # Read with timeout
            start_time = time.time()
            while time.time() - start_time < 10:
                if _worker_process.poll() is not None:
                    raise RuntimeError("Worker process died")
                
                if sys.platform == "win32":
                    line = _worker_process.stdout.readline()
                    if line:
                        response = json.loads(line.strip())
                        if response.get("raw_text") == ["test_success"]:
                            logger.info("Worker process started and responsive")
                            return True
                else:
                    ready, _, _ = select.select([_worker_process.stdout], [], [], 0.1)
                    if ready:
                        line = _worker_process.stdout.readline().strip()
                        if line:
                            response = json.loads(line)
                            if response.get("raw_text") == ["test_success"]:
                                logger.info("Worker process started and responsive")
                                return True
                time.sleep(0.1)
            
            raise RuntimeError("Worker not responsive")
            
        except Exception as e:
            logger.error(f"Worker test failed: {e}")
            cleanup_worker()
            return False
        
    except Exception as e:
        logger.error(f"Failed to start worker process: {e}")
        cleanup_worker()
        return False

def get_worker_process():
    """Get or create worker process"""
    global _worker_process
    
    with _worker_lock:
        if _worker_process is None or _worker_process.poll() is not None:
            if not start_worker_process():
                raise RuntimeError("Failed to start OCR worker")
        
        return _worker_process

@timeout(30)  # Use the decorator for timeout
def safe_ocr_internal(image_path):
    """Internal OCR function with timeout decorator"""
    process = get_worker_process()
    
    # Send image path
    process.stdin.write(image_path + "\n")
    process.stdin.flush()
    
    # Read response
    start_time = time.time()
    while time.time() - start_time < 30:
        if process.poll() is not None:
            raise RuntimeError("Worker process died")
        
        if sys.platform == "win32":
            line = process.stdout.readline()
            if line:
                return json.loads(line.strip())
        else:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                line = process.stdout.readline().strip()
                if line:
                    return json.loads(line)
        
        time.sleep(0.1)
    
    raise TimeoutException("Timeout waiting for worker response")

def safe_ocr(image_path, timeout=30):
    """Send image to worker process"""
    try:
        return safe_ocr_internal(image_path)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise

def preprocess_img_to_tempfile(img_path, long_side=1024):
    """Simple PIL-based preprocessing for doctr"""
    try:
        with time_limit(30):  # Use the fixed context manager
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            w, h = img.size
            if max(h, w) > long_side:
                scale = long_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(tmp_file.name, 'JPEG', quality=95, optimize=True)
            tmp_file.close()
            return tmp_file.name
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

@timeout(60)  # Overall timeout for processing
def enhanced_ktp_processing(image_path: str):
    """Main KTP processing function with timeout"""
    try:
        # Validate input
        if not os.path.exists(image_path):
            return [], {}, json.dumps({
                "raw_text": [],
                "extracted_data": {},
                "confidence_info": {"error": f"Image file not found: {image_path}"}
            })
        
        # Get OCR results
        try:
            # Preprocess image
            temp_image_path = preprocess_img_to_tempfile(image_path, long_side=1024)
            worker_output = safe_ocr(temp_image_path, timeout=30)
            
            # Clean up temp file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
                
        except TimeoutException as e:
            logger.error(f"OCR timeout: {e}")
            return [], {}, json.dumps({
                "raw_text": [],
                "extracted_data": {},
                "confidence_info": {"error": f"OCR timeout: {str(e)}"}
            })
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return [], {}, json.dumps({
                "raw_text": [],
                "extracted_data": {},
                "confidence_info": {"error": f"OCR failed: {str(e)}"}
            })
        
        text_lines = worker_output.get("raw_text", [])
        confidence_info = {
            "mean_confidence": worker_output.get("mean_confidence", 0),
            "min_confidence": worker_output.get("min_confidence", 0),
            "max_confidence": worker_output.get("max_confidence", 0)
        }
        
        if worker_output.get("error"):
            confidence_info["error"] = worker_output["error"]
        
        # Parse KTP fields
        ktp_data = {}
        if text_lines:
            try:
                ktp_data = parse_ktp_fields(text_lines)
            except Exception as e:
                logger.error(f"Parsing failed: {e}")
                ktp_data = {}
        
        # Normalize data
        normalized_data = {}
        for key, val in ktp_data.items():
            clean_key = key.strip().replace(" ", "_").replace("/", "_").title()
            if clean_key in normalized_data:
                continue
            if isinstance(val, str):
                val = val.strip()
            normalized_val = fuzzy_match_value(clean_key.upper(), val)
            normalized_data[clean_key] = normalized_val
        
        # Build output
        output_json = json.dumps({
            "raw_text": text_lines,
            "extracted_data": normalized_data,
            "confidence_info": confidence_info
        }, indent=2, ensure_ascii=False)
        
        return text_lines, normalized_data, output_json
        
    except TimeoutException as e:
        logger.error(f"KTP processing timeout: {e}")
        return [], {}, json.dumps({
            "raw_text": [],
            "extracted_data": {},
            "confidence_info": {"error": f"Processing timeout: {str(e)}"}
        })
    except Exception as e:
        logger.error(f"KTP processing failed: {e}")
        return [], {}, json.dumps({
            "raw_text": [],
            "extracted_data": {},
            "confidence_info": {"error": f"Processing failed: {str(e)}"}
        })

# Global cleanup
import atexit
def global_cleanup():
    """Cleanup resources on exit"""
    logger.info("Performing global cleanup...")
    cleanup_worker()

atexit.register(global_cleanup)
