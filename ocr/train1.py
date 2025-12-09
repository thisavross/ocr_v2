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
from typing import List, Optional, Tuple, Dict, Any
import logging
import threading
import time
import select
from contextlib import contextmanager
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from functools import lru_cache
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

# timeout decorator using concurrent.futures
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

# FIXED: Context manager for timeouts with better thread management
@contextmanager
def time_limit(seconds):
    """Context manager for timeouts using threads"""
    stop_event = threading.Event()
    
    def timeout_handler():
        time.sleep(seconds)
        if not stop_event.is_set():
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
        stop_event.set()
        # Wait briefly for thread to notice stop event
        if timer.is_alive():
            timer.join(0.1)

# Pre-compile ALL regex patterns at module level
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

# Unified pattern for all common KTP patterns
KTP_PATTERNS = {
    'date': re.compile(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'),
    'nik': re.compile(r'\d{16}'),
    'rt_rw': re.compile(r'(\d{2,3})\s*[/\s]\s*(\d{2,3})'),
    'blood_type': re.compile(r'[ABO]{1,2}', re.IGNORECASE),
}


# Pre-processed expected fields for faster matching
EXPECTED_FIELDS = [
    "PROVINSI", "KOTA", "KABUPATEN", "NIK", "NAMA", "TEMPAT/TGL LAHIR",
    "JENIS KELAMIN", "GOL DARAH", "ALAMAT", "RT/RW", "KEL/DESA", "KECAMATAN",
    "AGAMA", "STATUS PERKAWINAN", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU HINGGA"
]

# Create multiple lookup structures for O(1) access
EXPECTED_FIELDS_SET = set(EXPECTED_FIELDS)
EXPECTED_FIELDS_WORDS_SET = set()
for field in EXPECTED_FIELDS:
    for word in field.replace('/', ' ').split():
        EXPECTED_FIELDS_WORDS_SET.add(word)

FIELD_VARIATIONS = {
    "PROVINSI": ["PROVINSI", "PROV", "PROVINS"],
    "KOTA": ["KOTA"],
    "KABUPATEN": ["KABUPATEN", "KAB"],
    "NIK": ["NIK", "NIK:"],
    "NAMA": ["NAMA", "NAMA:"],
    "TEMPAT/TGL LAHIR": ["TEMPAT", "TGL", "LAHIR", "TEMPAT/TGL", "TEMPAT TGL", "TEMPATTGL"],
    "JENIS KELAMIN": ["JENIS", "KELAMIN", "JENISKELAMIN", "JENIS KELAMIN"],
    "GOL DARAH": ["GOL", "DARAH", "GOL.DARAH", "GOLDARAH", "GOL DARAH"],
    "ALAMAT": ["ALAMAT"],
    "RT/RW": ["RT/RW", "RTRW", "RT", "RW", "AT/RW", "AT RW", "ATRW"],
    "KEL/DESA": ["KEL/DESA", "KEL", "DESA", "KELDESA"],
    "KECAMATAN": ["KECAMATAN", "KEC"],
    "AGAMA": ["AGAMA"],
    "STATUS PERKAWINAN": ["STATUS", "PERKAWINAN", "STATUSPERKAWINAN"],
    "PEKERJAAN": ["PEKERJAAN"],
    "KEWARGANEGARAAN": ["KEWARGANEGARAAN", "WNI", "WNA"],
    "BERLAKU HINGGA": ["BERLAKU", "HINGGA", "BERLAKUHINGGA"]
}

# Reverse lookup for variations to standard field names
VARIATION_TO_FIELD = {}
for field, variations in FIELD_VARIATIONS.items():
    for var in variations:
        VARIATION_TO_FIELD[var.upper()] = field

# Controlled values with pre-processed uppercase versions
CONTROLLED_VALUES = {
    "JENIS KELAMIN": ["LAKILAKI", "PEREMPUAN"],
    "GOL_DARAH": ["A", "B", "AB", "O"],
    "STATUS PERKAWINAN": ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"],
    "AGAMA": ["BUDDHA", "HINDU", "ISLAM", "KATOLIK", "KRISTEN", "KONGHUCU", "KEPERCAYAAN"]
}

# Pre-compute all controlled values in uppercase
CONTROLLED_VALUES_UPPER = {
    key: [v.upper() for v in values]
    for key, values in CONTROLLED_VALUES.items()
}

# Create a flat set of all controlled values for quick lookup
ALL_CONTROLLED_VALUES_SET = set()
for values in CONTROLLED_VALUES.values():
    ALL_CONTROLLED_VALUES_SET.update(v.upper() for v in values)

# OCR correction mapping
OCR_CORRECTIONS = {
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

# ========== LRU CACHED FUNCTIONS ==========

@functools.lru_cache(maxsize=128)
def is_field_name_fast_cached(text: str) -> Optional[str]:
    if not text:
        return None
    
    text_upper = text.upper().strip()
    
    # Clean common OCR artifacts
    text_clean = text_upper
    for char in [':', '-', '.', ';']:
        text_clean = text_clean.replace(char, '')
    text_clean = text_clean.strip()
    
    # 1. Direct set lookup (fastest)
    if text_clean in EXPECTED_FIELDS_SET:
        return text_clean
    
    # 2. Check field variations
    if text_clean in VARIATION_TO_FIELD:
        return VARIATION_TO_FIELD[text_clean]
    
    # 3. Check if it's a single word from a field name
    if text_clean in EXPECTED_FIELDS_WORDS_SET:
        # Map single word back to full field name
        for field in EXPECTED_FIELDS:
            field_words = field.replace('/', ' ').upper().split()
            if text_clean in field_words:
                return field
    
    # 4. Fuzzy matching for OCR errors
    for field in EXPECTED_FIELDS:
        field_simple = field.replace('/', '').replace(' ', '')
        text_simple = text_clean.replace(' ', '').replace('/', '')
        if field_simple in text_simple or text_simple in field_simple:
            return field
    
    return None

@lru_cache(maxsize=256)
def fuzzy_match_controlled_cached(field: str, value: str, threshold: int = 75) -> Optional[str]:
    """
    Cached fuzzy matching for controlled values.
    Larger cache since this is called frequently with similar inputs.
    """
    value_upper = value.upper()
    
    # Fast path: exact match
    if field in CONTROLLED_VALUES_UPPER:
        if value_upper in CONTROLLED_VALUES_UPPER[field]:
            idx = CONTROLLED_VALUES_UPPER[field].index(value_upper)
            return CONTROLLED_VALUES[field][idx]

    # Fuzzy matching fallback
    if field in CONTROLLED_VALUES:
        match = process.extractOne(value_upper, CONTROLLED_VALUES_UPPER[field], scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            idx = CONTROLLED_VALUES_UPPER[field].index(match[0])
            return CONTROLLED_VALUES[field][idx]
    
    return None       

@lru_cache(maxsize=128)
def parse_blood_type_fast_cached(text: str) -> Optional[str]:
    """
    Cached blood type parsing.
    Blood types are limited (A, B, AB, O), so cache works well.
    """
    if not text:
        return None
    
    text_upper = text.upper()
    
    # Check for empty patterns first
    for pattern in EMPTY_BLOOD_PATTERNS:
        if pattern.search(text_upper):
            return None
        
    # Direct character search (fastest)
    for char in text_upper:
        if char in ['A', 'B', 'O']:
            # Check for AB
            if char == 'A' and 'B' in text_upper:
                return 'AB'
            return char
    
    # Regex fallback
    for pattern in BLOOD_TYPE_PATTERNS:
        match = pattern.search(text_upper)
        if match:
            blood_type = match.group(1).upper()
            if blood_type in ['A', 'B', 'AB', 'O']:
                return blood_type
    
    return None

@lru_cache(maxsize=64)
def correct_nik_ocr_cached(text: str) -> str:
    """
    Cached NIK OCR correction.
    NIKs are unique but often have similar OCR errors.
    """
    if not text:
        return ""
    
    corrected = ''
    for char in text:
        if char in OCR_CORRECTIONS:
            corrected += OCR_CORRECTIONS[char]
        elif char.isdigit():
            corrected += char
    
    return corrected

@lru_cache(maxsize=128)
def extract_value_smart_cached(line: str, current_field: str = None) -> str:
    """
    FIXED: Better value extraction that handles field contamination
    """
    if ':' not in line:
        # Try pattern like "Gol. Darah O" (no colon)
        if current_field and ' ' in line:
            # Check if line contains a value after field name
            field_lower = current_field.lower()
            line_lower = line.lower()
            if field_lower in line_lower:
                # Extract everything after the field name
                idx = line_lower.find(field_lower)
                value_part = line[idx + len(field_lower):].strip()
                # Clean up common separators
                for sep in [':', '-', '.', ';']:
                    if value_part.startswith(sep):
                        value_part = value_part[1:].strip()
                return value_part if value_part else ''
        return ''
    
    parts = line.split(':', 1)
    value = parts[1].strip()
    
    # Remove any leading colon or other separators
    for sep in [':', '-', '.', ';']:
        if value.startswith(sep):
            value = value[1:].strip()
    
    # Quick empty check
    if not value or value in [':', '-', '.', '']:
        return ''
    
    value_upper = value.upper()
    
    # FIXED: More comprehensive check for field name contamination
    # Split value into words and check each one
    value_words = value_upper.split()
    for word in value_words:
        # Check if word is part of any field name
        if word in EXPECTED_FIELDS_WORDS_SET:
            # Remove this word from value
            value = ' '.join([w for w in value.split() if w.upper() != word])
    
    # Also check if entire value is a field variation
    if value_upper in VARIATION_TO_FIELD or value_upper in EXPECTED_FIELDS_SET:
        return ''
    
    # Additional check for current field contamination
    if current_field:
        field_words = current_field.upper().replace('/', ' ').split()
        for word in field_words:
            if word == value_upper:
                return ''
    
    return value.strip()

# ========== WRAPPER FUNCTIONS WITH CACHING ==========

def is_field_name_fast(text: str) -> Optional[str]:
    """Wrapper for cached field detection"""
    return is_field_name_fast_cached(text) if text else None

def fuzzy_match_controlled(field: str, value: str, threshold: int = 75) -> Optional[str]:
    """Wrapper for cached fuzzy matching"""
    return fuzzy_match_controlled_cached(field, value, threshold) if value else None

def parse_blood_type_fast(text: str) -> Optional[str]:
    """Wrapper for cached blood type parsing"""
    return parse_blood_type_fast_cached(text) if text else None

def extract_value_smart(line: str, current_field: str = None) -> str:
    """Wrapper for cached value extraction"""
    return extract_value_smart_cached(line, current_field) if line else ''

# ========== CACHED COMPLEX FUNCTIONS ==========

@lru_cache(maxsize=32)
@lru_cache(maxsize=32)
def process_ktp_chunk(lines_tuple: tuple) -> Dict[str, str]:
    """
    FIXED: More reliable KTP parsing with better value extraction
    """
    text_lines = list(lines_tuple)
    ktp_data = {}
    processed_fields = set()
    all_dates = []
    
    i = 0
    while i < len(text_lines):
        line = text_lines[i].strip()
        if not line:
            i += 1
            continue
        
        line_upper = line.upper()
        
        # Extract dates for later processing
        dates_in_line = KTP_PATTERNS['date'].findall(line)
        all_dates.extend(dates_in_line)
        
        # Handle NIK detection separately (it's often on its own line)
        if KTP_PATTERNS['nik'].search(line):
            nik_match = KTP_PATTERNS['nik'].search(line)
            if nik_match:
                nik_value = nik_match.group()
                corrected_nik = correct_nik_ocr_cached(nik_value)
                if KTP_PATTERNS['nik'].fullmatch(corrected_nik):
                    ktp_data["NIK"] = corrected_nik
                    processed_fields.add("NIK")
                    i += 1
                    continue
        
        # Try to find field-value pairs
        field_name = None
        value = ""
        
        # Pattern 1: Colon-separated (e.g., "Nama: HANDOKO")
        if ':' in line:
            # Split only on first colon
            parts = line.split(':', 1)
            key_part = parts[0].strip()
            val_part = parts[1].strip()
            
            # Check if key part contains a field name
            field_name = is_field_name_fast(key_part)
            
            if field_name and field_name not in processed_fields:
                # Extract value, handling cases where value might be on next line
                value = extract_value_smart_cached(line, field_name)
                
                # If value is empty or looks incomplete, check next line
                if not value or len(value) < 2:
                    if i + 1 < len(text_lines):
                        next_line = text_lines[i + 1].strip()
                        if not (':' in next_line or is_field_name_fast(next_line)):
                            value = next_line
                            i += 1  # Skip next line since we used it
        
        # Pattern 2: Field and value on same line without colon (e.g., "NIK 2171101212749021")
        if not field_name:
            words = line.split()
            if words:
                # Try first word as field
                possible_field = is_field_name_fast(words[0])
                if possible_field and possible_field not in processed_fields:
                    field_name = possible_field
                    # Rest of line is value
                    value = ' '.join(words[1:]).strip()
                    # Clean up separators
                    for sep in [':', '-', '.']:
                        if value.startswith(sep):
                            value = value[1:].strip()
        
        # Pattern 3: Field might be in the middle of line (OCR errors)
        if not field_name:
            for field in EXPECTED_FIELDS:
                if field.upper() in line_upper:
                    # Try to extract value after field name
                    field_idx = line_upper.find(field.upper())
                    if field_idx >= 0:
                        value_start = field_idx + len(field)
                        value = line[value_start:].strip()
                        # Clean up
                        for sep in [':', '-', '.', ';']:
                            if value.startswith(sep):
                                value = value[1:].strip()
                        if value:
                            field_name = field
                            break
        
        # Process the field if found
        if field_name and field_name not in processed_fields:
            standardized_key = field_name.replace(' ', '_').replace('/', '_')
            
            # Clean the value
            value = value.strip()
            
            # Skip if value is obviously bad
            if value and value not in ["", ":", "-", ".", ";"]:
                # Special handling for controlled values
                if field_name in ["JENIS KELAMIN", "AGAMA", "STATUS PERKAWINAN"]:
                    matched_val = fuzzy_match_controlled(field_name, value)
                    if matched_val:
                        ktp_data[standardized_key] = matched_val
                        processed_fields.add(field_name)
                    elif value and not is_field_name_fast(value):
                        # Store raw value if fuzzy match fails but it's not a field name
                        ktp_data[standardized_key] = value
                        processed_fields.add(field_name)
                
                # Special handling for blood type
                elif field_name == "GOL DARAH":
                    blood_type = parse_blood_type_fast(value)
                    if blood_type:
                        ktp_data["Gol_Darah"] = blood_type
                        processed_fields.add(field_name)
                
                # Special handling for RT/RW
                elif field_name == "RT/RW":
                    rt_rw_match = KTP_PATTERNS['rt_rw'].search(value)
                    if rt_rw_match:
                        ktp_data["RT_RW"] = f"{rt_rw_match.group(1)}/{rt_rw_match.group(2)}"
                        processed_fields.add(field_name)
                    elif value and '/' in value:
                        # Try simple split
                        parts = value.split('/')
                        if len(parts) == 2:
                            ktp_data["RT_RW"] = f"{parts[0].strip()}/{parts[1].strip()}"
                            processed_fields.add(field_name)
                
                # For other fields
                else:
                    # Additional check to ensure value is not a field name
                    if not is_field_name_fast(value):
                        ktp_data[standardized_key] = value
                        processed_fields.add(field_name)
        
        i += 1
    
    # Post-processing - DON'T overwrite extracted data
    ktp_data = post_process_ktp_data(ktp_data, all_dates, text_lines)
    
    return ktp_data

# ========== UPDATED MAIN PARSING FUNCTION ==========

def parse_ktp_fields_efficient(text_lines: List[str]) -> Dict[str, str]:
    """
    Single-pass efficient KTP parsing with caching.
    """
    # Convert list to tuple for caching
    lines_tuple = tuple(line.strip() for line in text_lines if line.strip())
    
    if not lines_tuple:
        return {}
    
    # Use cached processing for common KTP patterns
    result = process_ktp_chunk(lines_tuple)
    
    # DEBUG: Log what was extracted
    logger.info(f"Extracted fields: {list(result.keys())}")
    
    return result

# ========== CACHE MANAGEMENT FUNCTIONS ==========

def clear_all_caches():
    """Clear all LRU caches (useful for memory management)"""
    is_field_name_fast_cached.cache_clear()
    fuzzy_match_controlled_cached.cache_clear()
    parse_blood_type_fast_cached.cache_clear()
    correct_nik_ocr_cached.cache_clear()
    extract_value_smart_cached.cache_clear()
    process_ktp_chunk.cache_clear()
    logger.info("All LRU caches cleared")

def get_cache_info():
    """Get cache statistics for monitoring"""
    return {
        "is_field_name_fast": is_field_name_fast_cached.cache_info(),
        "fuzzy_match_controlled": fuzzy_match_controlled_cached.cache_info(),
        "parse_blood_type_fast": parse_blood_type_fast_cached.cache_info(),
        "correct_nik_ocr": correct_nik_ocr_cached.cache_info(),
        "extract_value_smart": extract_value_smart_cached.cache_info(),
        "process_ktp_chunk": process_ktp_chunk.cache_info()
    }

# ========== OPTIMIZED POST-PROCESSING ==========

@lru_cache(maxsize=64)
def extract_dates_from_text_cached(text: str) -> tuple:
    """Extract dates from text with caching"""
    return tuple(KTP_PATTERNS['date'].findall(text))

def post_process_ktp_data(ktp_data: Dict[str, str], all_dates: List[str], text_lines: List[str]) -> Dict[str, str]:
    """
    FIXED: Post-processing that doesn't overwrite extracted data
    """
    # Extract dates from all text with caching
    unique_dates = []
    seen = set()
    
    for line in text_lines:
        dates_in_line = extract_dates_from_text_cached(line)
        for date in dates_in_line:
            if date not in seen:
                seen.add(date)
                unique_dates.append(date)
    
    # Also add dates found during initial parsing
    for date in all_dates:
        if date not in seen:
            seen.add(date)
            unique_dates.append(date)
    
    # Extract birth date if in Tempat_Tgl_Lahir
    birth_date = None
    if "Tempat_Tgl_Lahir" in ktp_data:
        birth_match = KTP_PATTERNS['date'].search(ktp_data["Tempat_Tgl_Lahir"])
        if birth_match:
            birth_date = birth_match.group()
    
    # Filter out birth date from other dates
    other_dates = [date for date in unique_dates if date != birth_date]
    
    # Check for SEUMUR HIDUP
    has_seumur_hidup = any(
        "SEUMUR" in line.upper() and "HIDUP" in line.upper() 
        for line in text_lines
    )
    
    # Assign dates ONLY if not already extracted
    if has_seumur_hidup and "Berlaku_Hingga" not in ktp_data:
        ktp_data["Berlaku_Hingga"] = "SEUMUR HIDUP"
    elif other_dates:
        if len(other_dates) >= 2 and "Tanggal_Terbit" not in ktp_data:
            ktp_data["Tanggal_Terbit"] = other_dates[0]
            if "Berlaku_Hingga" not in ktp_data and not has_seumur_hidup:
                ktp_data["Berlaku_Hingga"] = other_dates[1]
        elif len(other_dates) == 1 and "Tanggal_Terbit" not in ktp_data:
            ktp_data["Tanggal_Terbit"] = other_dates[0]
    
    # Ensure required fields exist - but don't overwrite existing ones
    required_fields = [
        "NIK", "Nama", "Tempat_Tgl_Lahir", "Jenis_Kelamin",
        "Alamat", "RT_RW", "Kel_Desa", "Agama", "Pekerjaan",
        "Kewarganegaraan", "Tanggal_Terbit", "Berlaku_Hingga"
    ]
    
    for field in required_fields:
        if field not in ktp_data:
            ktp_data[field] = ""
    
    return ktp_data

# ========== MAIN ENTRY POINT ==========

def parse_ktp_fields(text_lines: List[str]) -> Dict[str, str]:
    """
    Main entry point for KTP parsing - uses the efficient single-pass parser with caching.
    """
    # Clean and filter lines
    clean_lines = [line.strip() for line in text_lines if line.strip()]
    
    if not clean_lines:
        return {}
    
    return parse_ktp_fields_efficient(clean_lines)

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
                response = json.loads(line.strip())
                if "raw_text" in response:
                    return response
        else:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                line = process.stdout.readline().strip()
                if line:
                    response = json.loads(line)
                    return response
        
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

# ========== ENHANCED KTP PROCESSING WITH CACHE MONITORING ==========

@timeout(60)
def enhanced_ktp_processing(image_path: str, enable_caching: bool = True):
    """Main KTP processing function with timeout and caching"""
    try:
        # Clear caches if disabled
        if not enable_caching:
            clear_all_caches()
        
        # Validate input
        if not os.path.exists(image_path):
            return [], {}, json.dumps({
                "raw_text": [],
                "extracted_data": {},
                "confidence_info": {"error": f"Image file not found: {image_path}"}
            })
        
        # Get OCR results
        try:
            temp_image_path = preprocess_img_to_tempfile(image_path, long_side=1024)
            worker_output = safe_ocr(temp_image_path, timeout=30)
            
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
                
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return [], {}, json.dumps({
                "raw_text": [],
                "extracted_data": {},
                "confidence_info": {"error": f"OCR failed: {str(e)}"}
            })
        
        text_lines = worker_output.get("raw_text", [])
        
        # Parse KTP fields using efficient parser with caching
        ktp_data = parse_ktp_fields(text_lines) if text_lines else {}
        
        # SIMPLIFIED: Get cache stats safely without recursion
        cache_hits = 0
        cache_misses = 0
        try:
            cache_info = process_ktp_chunk.cache_info()
            cache_hits = cache_info.hits
            cache_misses = cache_info.misses
        except:
            pass  # If cache info not available, use defaults
        
        # Build output
        output_json = json.dumps({
            "raw_text": text_lines,
            "extracted_data": ktp_data,
            "confidence_info": {
                "mean_confidence": worker_output.get("mean_confidence", 0),
                "line_count": len(text_lines),
                "error": worker_output.get("error", ""),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses
            }
        }, indent=2, ensure_ascii=False)
        
        return text_lines, ktp_data, output_json
        
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

# ========== GLOBAL CLEANUP WITH CACHE CLEARING ==========

def global_cleanup():
    """Cleanup resources on exit including clearing caches"""
    logger.info("Performing global cleanup...")
    cleanup_worker()
    clear_all_caches()
    logger.info("Caches cleared during cleanup")

import atexit
atexit.register(global_cleanup)