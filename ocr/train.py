import functools
from multiprocessing import Process
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import re
import traceback
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
import psutil
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
        import sys
        if not timer_done[0]:
            raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    timer_done = [False]
    timer = threading.Thread(target=timeout_handler)
    timer.daemon = True
    timer.start()
    
    try:
        yield
        timer_done[0] = True
    except TimeoutException:
        raise
    finally:
        timer_done[0] = True


# ========== COMPILED REGEX PATTERNS (FASTER) ==========
# SIMPLIFY these patterns - they're too broad
DATE_PATTERN = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')  
RT_RW_PATTERN = re.compile(r'\b(\d{2,3})\s*[/|]\s*(\d{2,3})\b') 


# Use compiled patterns for better performance
PROVINCE_PATTERN = re.compile(r'^PROVINSI\s*(.*)', re.IGNORECASE)
NIK_PATTERN = re.compile(r'(\d{16}|\d{10,20}|[0-9OIlS]{16})')
DATE_PATTERN = re.compile(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}')
BLOOD_TYPE_PATTERN = re.compile(
    r'\b([ABO]|0)[\+\-]?\b',
    re.IGNORECASE
)
RT_RW_PATTERN = re.compile(r'(\d{2,3})\s*[/|]\s*(\d{2,3})')

# Pre-compiled field patterns
FIELD_PATTERNS = {
    "PROVINSI": re.compile(r'PROVINSI', re.IGNORECASE),
    "KABUPATEN": re.compile(r'KABUPATEN', re.IGNORECASE),
    "KOTA": re.compile(r'KOTA', re.IGNORECASE),  
    "KECAMATAN": re.compile(r'KECAMATAN', re.IGNORECASE),  
    "NIK": re.compile(r'NIK', re.IGNORECASE),
    "NAMA": re.compile(r'NAMA', re.IGNORECASE),
    "TEMPAT_TGL_LAHIR": re.compile(r'TEMPAT\s*[/|]?\s*T?GL?\s*LAHIR', re.IGNORECASE),
    "JENIS_KELAMIN": re.compile(r'JENIS\s*KELAMIN', re.IGNORECASE),
    "GOL_DARAH": re.compile(r'GOL\.?\s*DARAH', re.IGNORECASE),
    "ALAMAT": re.compile(r'ALAMAT', re.IGNORECASE),
    "RT_RW": re.compile(r'R?T\s*[/|]?\s*R?W', re.IGNORECASE),
    "KEL_DESA": re.compile(r'KEL\s*[/|]?\s*DESA', re.IGNORECASE),
    "AGAMA": re.compile(r'AGAMA', re.IGNORECASE),
    "PEKERJAAN": re.compile(r'PEKERJAAN', re.IGNORECASE),
    "KEWARGANEGARAAN": re.compile(r'KEWARGANEGARAAN', re.IGNORECASE),
    "BERLAKU_HINGGA": re.compile(r'BERLAKU\s*HINGGA', re.IGNORECASE),
}

# Fix FIELD_MAPPING - lists should not be used as values
FIELD_MAPPING = {
    "Provinsi": "Provinsi",
    "Kota_Kabupaten": "Kota_Kabupaten", 
    "NIK": "NIK",
    "Nama": "Nama",
    "Tempat_Tgl_Lahir": "Tempat_Tgl_Lahir",
    "Jenis_Kelamin": "Jenis_Kelamin",
    "Gol_Darah": "Gol_Darah",
    "Alamat": "Alamat",
    "Rt_Rw": "Rt_Rw",
    "Kel_Desa": "Kel_Desa",
    "Kecamatan": "Kecamatan",
    "Agama": "Agama",
    "Pekerjaan": "Pekerjaan",
    "Kewarganegaraan": "Kewarganegaraan",
    "Tanggal_Terbit": "Tanggal_Terbit",
    "Berlaku_Hingga": "Berlaku_Hingga"
}


REQUIRED_FIELDS = [
    "NIK", "Nama", "Tempat_Tgl_Lahir", "Jenis_Kelamin",
    "Alamat", "Kel_Desa", "Pekerjaan",
    "Kewarganegaraan", "Tanggal_Terbit", "Berlaku_Hingga"
]
# Controlled values with corrections
CONTROLLED_VALUES = {
    "Jenis_Kelamin": {
        "LAKI-LAKI": ["LAKILAKI", "LAKILAK", "LAKILAKT", "LAKILAKI", "LAKI LAKI", "MALE"],
        "PEREMPUAN": ["PEREMPUAN", "PEREMPUA", "PEREMPUAN","FEMALES"]
    },
    "Agama": {
        "ISLAM": ["ISLAM", "ISLAM", "ISLAM"],
        "KRISTEN": ["KRISTEN", "KRISTEN", "KAISTEN", "CHRISTIAN"],
        "KATOLIK": ["KATOLIK", "KATHOLIK"],
        "BUDDHA": ["BUDDHA", "BUDHA"],
        "HINDU": ["HINDU"],
        "KONGHUCU": ["KONGHUCU"]
    },
}

def fuzzy_match_field(text, threshold=75):
    """Fuzzy match text to expected KTP field names with aggressive cleanup"""
    if not text.strip():
        return None

    # remove all non-alphabetic characters (colon, dots, commas, OCR noise)
    cleaned = re.sub(r'[^A-Z]', '', text.upper())
    
    # Use keys from FIELD_MAPPING for matching
    field_names = list(FIELD_MAPPING.keys())
    
    match = process.extractOne(cleaned, field_names, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        # Return the actual field name to use (the value from FIELD_MAPPING)
        return FIELD_MAPPING[match[0]]
    return None


def fuzzy_match_value(key, val, threshold=75):
    """
    Fuzzy-correct value based on controlled list for the given key.
    """
    if not val:
        return val

    val_up = val.upper().replace(" ", "")
    
    # CONTROLLED_VALUES has nested structure, get the correct sub-dictionary
    controlled_dict = CONTROLLED_VALUES.get(key, {})
    
    if not controlled_dict:
        return val  # No controlled values → return as is

    # Normalize controlled values
    controlled_list = []
    controlled_to_original = {}
    
    for correct_value, variations in controlled_dict.items():
        controlled_list.append(correct_value.upper().replace(" ", ""))
        controlled_to_original[correct_value.upper().replace(" ", "")] = correct_value
        
        for variation in variations:
            normalized_variation = variation.upper().replace(" ", "")
            controlled_list.append(normalized_variation)
            controlled_to_original[normalized_variation] = correct_value

    # 1️ Exact match
    if val_up in controlled_to_original:
        return controlled_to_original[val_up]

    # 2️ Fuzzy match
    match = process.extractOne(val_up, controlled_list, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return controlled_to_original.get(match[0], val)

    return val

# ========== OPTIMIZED PARSING FUNCTIONS ==========
def extract_field_value(line, field_name):
    """Extract value from a line that contains a field name"""
    if ':' in line:
        parts = line.split(':', 1)
        if len(parts) == 2:
            return parts[1].strip()
    
    # Try to extract after field name
    field_pattern = FIELD_PATTERNS.get(field_name)
    if field_pattern:
        match = field_pattern.search(line)
        if match:
            # Get text after the field name
            value = line[match.end():].strip()
            # Clean up common prefixes
            value = re.sub(r'^[:\-\s\.]+', '', value)
            return value
    
    return line.strip()


def fix_tempat_tgl_lahir(value):
    """Fix common issues in Tempat_Tgl_Lahir field"""
    if not value:
        return ""
    
    # Remove common OCR artifacts
    value = re.sub(r'^[/:\-\s\.]+', '', value)  # Remove leading symbols
    value = re.sub(r'[Tt]gl\s*[Ll]ahir\s*[\.:]?\s*', '', value)  # Remove "Tgl Lahir" text
    value = re.sub(r'Tempat\s*[/|]?\s*', '', value, flags=re.IGNORECASE)  # Remove "Tempat/"
    
    # Fix common OCR errors
    corrections = {
        'ahr ': '', 'ahir ': '', '/Tal ': '', '/Tg ': ''
    }
    for wrong, correct in corrections.items():
        value = value.replace(wrong, correct)
    
    # Ensure proper format
    if ', ' not in value and re.search(r'\d', value):
        # Add comma if missing before date
        value = re.sub(r'(\D)(\d{1,2}-\d{1,2}-\d{4})', r'\1, \2', value)
    
    return value.strip()

def normalize_controlled_value(field, value):
    """Normalize controlled values using direct mapping"""
    if not value:
        return value
    
    value_upper = value.upper().strip()
    
    if field in CONTROLLED_VALUES:
        for correct_value, variations in CONTROLLED_VALUES[field].items():
            if value_upper in variations or value_upper == correct_value.upper():
                return correct_value
        
        # Try fuzzy matching as fallback
        best_match = None
        best_score = 0
        
        for correct_value in CONTROLLED_VALUES[field].keys():
            score = fuzz.ratio(value_upper, correct_value.upper())
            if score > best_score and score > 80:
                best_score = score
                best_match = correct_value
        
        if best_match:
            return best_match
    
    return value

def extract_nik_smart(lines):
    """Smart NIK extraction with OCR correction"""
    # Define replacements at the top of the function
    replacements = {
        'O': '0', 'I': '1', 'L': '1', 'S': '5',
        'U': '0', 'Z': '2', 'B': '8', 'G': '6',
        'H': '', 'E': '', 'D': '0', 'T': '7',
        ' ': '', ':': '', '-': '', '/': ''
    }
    
    for i, line in enumerate(lines):
        line_upper = line.upper()
        
        # Look for NIK field
        if "NIK" in line_upper:
            # Try to extract from current line
            value = extract_field_value(line, "NIK")
            if value:
                # Clean NIK value - replace common OCR errors
                value = value.upper()
                
                for wrong, correct in replacements.items():
                    value = value.replace(wrong, correct)
                
                # Extract only digits
                digits = re.findall(r'\d', value)
                if len(digits) >= 16:
                    return ''.join(digits)[:16]
            
            # Check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Clean next line similarly
                next_line_clean = next_line.upper()
                for wrong, correct in replacements.items():
                    next_line_clean = next_line_clean.replace(wrong, correct)
                
                digits = re.findall(r'\d', next_line_clean)
                if len(digits) >= 16:
                    return ''.join(digits)[:16]
    
    # Fallback: search for 16-digit pattern in any line
    for line in lines:
        # Clean the line
        clean_line = line.upper()
        for wrong, correct in replacements.items():
            clean_line = clean_line.replace(wrong, correct)
        
        # Look for 16 consecutive digits
        digits = re.findall(r'\d', clean_line)
        if len(digits) >= 16:
            return ''.join(digits)[:16]
    
    return ""

def clean_nik_value(value):
    """Clean NIK value by fixing OCR errors"""
    if not value:
        return ""
    
    # Convert to uppercase
    value = value.upper()
    
    # Common OCR replacements for NIK - define it here
    replacements = {
        'O': '0', 'I': '1', 'L': '1', 'S': '5',
        'U': '0', 'Z': '2', 'B': '8', 'G': '6',
        'H': '', 'E': '', 'D': '0', 'T': '7',
        'A': '4', ' ': '', ':': '', '-': '',
        '/': '', '\\': '', '|': ''
    }
    
    for wrong, correct in replacements.items():
        value = value.replace(wrong, correct)
    
    # Extract only digits
    digits = re.findall(r'\d', value)
    
    if len(digits) >= 16:
        return ''.join(digits)[:16]
    elif len(digits) > 0:
        # Return what we have if less than 16
        return ''.join(digits)
    
    return value

def parse_ktp_fields(raw_text):
    """
    Optimized KTP parser with better field detection and consistency
    """
    if not raw_text:
        return {field: "" for field in FIELD_MAPPING.values()}
    
    # Clean and prepare lines
    lines = [line.strip() for line in raw_text if line.strip()]
    result = {field: "" for field in FIELD_MAPPING.values()}
    
    # Extract NIK first 
    nik_raw = extract_nik_smart(lines)
    result["NIK"] = clean_nik_value(nik_raw)
    
    # Find all dates in the document
    all_dates = []
    for line in lines:
        dates = DATE_PATTERN.findall(line)
        all_dates.extend(dates)
    
    # Remove duplicates while preserving order
    unique_dates = []
    for date in all_dates:
        if date not in unique_dates:
            unique_dates.append(date)
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_upper = line.upper()
        
        # === PROVINSI ===
        if not result["Provinsi"] and "PROVINSI" in line_upper:
            value = extract_field_value(line, "PROVINSI")
            if value:
                result["Provinsi"] = value
        
        # === KOTA / KABUPATEN ===
        elif not result["Kota_Kabupaten"] and (
            "KOTA" in line_upper or "KABUPATEN" in line_upper or "KAB " in line_upper or "KAB." in line_upper
        ):
            # Try extracting after the field label
            value = extract_field_value(line, "KOTA")  # Will match "KOTA"
            if not value:
                value = extract_field_value(line, "KABUPATEN")
            if not value:
                value = extract_field_value(line, "KAB")  # fallback for "KAB."

            if value:
                result["Kota_Kabupaten"] = value.strip()
            else:
                # Often the actual name is on the next line
                if i + 1 < len(lines):
                    result["Kota_Kabupaten"] = lines[i + 1].strip()
                    i += 1

        # === NAMA ===
        elif not result["Nama"] and "NAMA" in line_upper.replace(" ", ""):
            # Try same-line extraction first
            value = extract_field_value(line, "NAMA")

            if value and value not in ["NAMA", ":", "", ".", "-"]:
                # Clean same-line value
                cleaned = re.sub(r'^[:\.\-\s]+', '', value).strip()
                if cleaned:
                    result["Nama"] = cleaned
            else:
                # Fallback: Read from next line
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()

                    # Reject if next line is another field
                    if not any(
                        key in nxt.upper().replace(" ", "")
                        for key in FIELD_PATTERNS):
                        result["Nama"] = nxt
                        i += 1  # consume next line

        # === TEMPAT/TGL LAHIR ===
        elif not result["Tempat_Tgl_Lahir"] and any(
            keyword in line_upper for keyword in ["TEMPAT", "TGL", "LAHIR"]
        ):
            birth_info = []
            j = i

            while j < len(lines) and j - i < 3:
                current_line = lines[j].strip()
                upper_current = current_line.upper().replace(" ", "")

                # === STOP IF THIS LINE IS ANOTHER FIELD ===
                if any(key in upper_current for key in FIELD_PATTERNS) and j != i:
                    break

                birth_info.append(current_line)

                # === STOP EARLY IF DATE FOUND ===
                if DATE_PATTERN.search(current_line):
                    break

                j += 1
            
            if birth_info:
                # Join and clean birth info
                birth_text = " ".join(birth_info)
                # Extract just the birth location and date
                birth_text = re.sub(r'.*?(TEMPAT\s*[/|]?\s*T?GL?\s*LAHIR\s*[\.:]?\s*)', '', 
                                  birth_text, flags=re.IGNORECASE)
                result["Tempat_Tgl_Lahir"] = fix_tempat_tgl_lahir(birth_text)
                i = j - 1

        # === JENIS KELAMIN ===
        elif not result["Jenis_Kelamin"] and any(
            keyword in line_upper for keyword in ["JENIS", "KELAMIN"]
        ):
            value = extract_field_value(line, "JENIS_KELAMIN")
            if value:
                result["Jenis_Kelamin"] = normalize_controlled_value("Jenis_Kelamin", value)
            elif i + 1 < len(lines):
                next_line = lines[i + 1].upper()
                if "LAKI" in next_line or "PEREMPUAN" in next_line:
                    result["Jenis_Kelamin"] = normalize_controlled_value("Jenis_Kelamin", lines[i + 1])
                    i += 1
        
        # === GOLONGAN DARAH ===
        elif not result["Gol_Darah"] and any(
            keyword in line_upper for keyword in ["GOL", "DARAH"]
        ):
            value = extract_field_value(line, "GOL_DARAH")
            if value:
                # Extract blood type
                blood_match = BLOOD_TYPE_PATTERN.search(value)
                if blood_match:
                    result["Gol_Darah"] = blood_match.group(1).upper()
        
        # === ALAMAT ===
        elif not result["Alamat"] and "ALAMAT" in line_upper:
            value = extract_field_value(line, "ALAMAT")
            if value:
                result["Alamat"] = value
            else:
                # Collect multi-line address
                addr_parts = []
                j = i + 1
                while j < len(lines) and j - i < 5:  # Limit to 5 lines
                    next_line_upper = lines[j].upper()
                    # Stop at next field
                    if any(pattern.search(next_line_upper) for pattern in [
                        FIELD_PATTERNS["RT_RW"], FIELD_PATTERNS["KEL_DESA"]
                    ]):
                        break
                    addr_parts.append(lines[j])
                    j += 1
                
                if addr_parts:
                    result["Alamat"] = " ".join(addr_parts).strip()
                    i = j - 1
        
        # === RT/RW ===
        elif not result["Rt_Rw"] and FIELD_PATTERNS["RT_RW"].search(line_upper):
            value = extract_field_value(line, "RT_RW")
            if value:
                # Extract RT/RW numbers
                rt_rw_match = RT_RW_PATTERN.search(value)
                if rt_rw_match:
                    result["Rt_Rw"] = f"{rt_rw_match.group(1)}/{rt_rw_match.group(2)}"
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                rt_rw_match = RT_RW_PATTERN.search(next_line)
                if rt_rw_match:
                    result["Rt_Rw"] = f"{rt_rw_match.group(1)}/{rt_rw_match.group(2)}"
                    i += 1
        
        # === KEL/DESA ===
        elif not result["Kel_Desa"] and FIELD_PATTERNS["KEL_DESA"].search(line_upper):
            value = extract_field_value(line, "KEL_DESA")
            if value:
                result["Kel_Desa"] = value
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                if not any(pattern.search(next_line.upper()) for pattern in [
                    FIELD_PATTERNS["KECAMATAN"], FIELD_PATTERNS["AGAMA"]
                ]):
                    result["Kel_Desa"] = next_line.strip()
                    i += 1
        
        # === KECAMATAN ===
        elif not result["Kecamatan"] and any(
            fuzz.ratio(k, line_upper) >= 70
            for k in ["KECAMATAN", "KECMATAN", "KCAMATAN", "KECAMATAM"]
        ):
            # remove corrupted label
            text = re.sub(r'KEC[A-Z]*[^A-Z0-9]*', '', line_upper).strip()

            # if empty, maybe next line is value
            if not text and i+1 < len(lines):
                nxt = lines[i+1].strip()
                # stop if next line is a new field
                if not any(p.search(nxt.upper()) for p in [
                    FIELD_PATTERNS["AGAMA"], FIELD_PATTERNS["PEKERJAAN"], FIELD_PATTERNS["KEL_DESA"]
                ]):
                    text = nxt

            result["Kecamatan"] = text.title().strip()

        # === AGAMA ===
        elif not result["Agama"] and "AGAMA" in line_upper:
            value = extract_field_value(line, "AGAMA")
            if value:
                result["Agama"] = normalize_controlled_value("Agama", value)
            elif i+1 < len(lines):
                nxt = lines[i+1].strip()
                # stop if next line is a new field
                if not any(pattern.search(nxt.upper()) for pattern in FIELD_PATTERNS.values()):
                    value = nxt
                    result["Agama"] = normalize_controlled_value("Agama", value)

        # === PEKERJAAN ===
        elif not result["Pekerjaan"] and "PEKERJAAN" in line_upper:
            value = extract_field_value(line, "PEKERJAAN")
            if value:
                result["Pekerjaan"] = value
            # stop if next line is a new field
                if not any(p.search(nxt.upper()) for p in FIELD_PATTERNS):
                    value = nxt
                result["Pekerjaan"] = normalize_controlled_value("Pekerjaan", value)
            else:
                # Fallback: Read from next line
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()

                    # Reject if next line is another field
                    if not any(
                        key in nxt.upper().replace(" ", "")
                        for key in FIELD_PATTERNS):
                        result["Pekerjaan"] = nxt
                        i += 1  # consume next line
        
        # === KEWARGANEGARAAN ===
        elif not result["Kewarganegaraan"] and any(
            keyword in line_upper for keyword in ["KEWARGANEGARAAN", "WNI", "WNA"]
        ):
            value = extract_field_value(line, "KEWARGANEGARAAN")
            if value:
                result["Kewarganegaraan"] = normalize_controlled_value("Kewarganegaraan", value)
        
        # === BERLAKU HINGGA ===
        elif not result["Berlaku_Hingga"] and "BERLAKU HINGGA" in line_upper:
            value = extract_field_value(line, "BERLAKU_HINGGA")
            if value:
                result["Berlaku_Hingga"] = value
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                if DATE_PATTERN.search(next_line) or "SEUMUR" in next_line.upper():
                    result["Berlaku_Hingga"] = next_line.strip()
                    i += 1
        
        i += 1
    
    # === SMART DATE ASSIGNMENT ===
    # Check for SEUMUR HIDUP
    has_seumur_hidup = any("SEUMUR" in line.upper() and "HIDUP" in line.upper() 
                          for line in lines)
    
    if has_seumur_hidup:
        result["Berlaku_Hingga"] = "SEUMUR HIDUP"
    
    # Extract birth date for filtering
    birth_date = None
    if result["Tempat_Tgl_Lahir"]:
        birth_dates = DATE_PATTERN.findall(result["Tempat_Tgl_Lahir"])
        if birth_dates:
            birth_date = birth_dates[0]
    
    # Filter out birth date from unique dates
    other_dates = [date for date in unique_dates if date != birth_date]
    
    # Assign dates
    if other_dates:
        if len(other_dates) >= 2:
            # Usually: issue date, expiry date
            result["Tanggal_Terbit"] = other_dates[0]
            if not has_seumur_hidup:
                result["Berlaku_Hingga"] = other_dates[1]
        elif len(other_dates) == 1:
            result["Tanggal_Terbit"] = other_dates[0]
    
    return result




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
    """Get or create worker process with proper initialization"""
    global _worker_process
    
    with _worker_lock:
        if _worker_process is None or _worker_process.poll() is not None:
            cleanup_worker()  # Clean up first
            
            logger.info("Starting OCR worker process...")
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
            
            # Simple initialization check
            time.sleep(2)  # Reduced from 5 seconds
            if _worker_process.poll() is not None:
                raise RuntimeError("Worker process failed to start")
            
            logger.info("Worker process started")
        
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

@timeout(60)  # Overall timeout for processing
def enhanced_ktp_processing(image_path: str) -> Tuple[List[str], Dict[str, str], str]:
    """Simplified KTP processing function"""
    try:
        # Simple preprocessing
        temp_image_path = preprocess_img_to_tempfile(image_path, long_side=1024)
        
        # OCR with timeout
        worker_output = safe_ocr(temp_image_path, timeout=20)  # Reduced timeout
        
        # Clean up temp file
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        
        text_lines = worker_output.get("raw_text", [])
        
        # Parse fields
        parsed_data = parse_ktp_fields(text_lines)
        
        # Prepare output
        output_data = {
            "raw_text": text_lines,
            "extracted_data": parsed_data,
            "confidence_info": worker_output.get("confidence_info", {})
        }
        
        return text_lines, parsed_data, json.dumps(output_data, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return [], {}, json.dumps({
            "raw_text": [],
            "extracted_data": {},
            "confidence_info": {"error": str(e)}
        }, ensure_ascii=False)

def get_current_memory_mb(process=None):
    """Get current memory usage in MB"""
    try:
        if process is None:
            process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0

#Simple memory tracking decorator
def track_memory_usage(func):
    """Decorator to track memory usage of a function"""
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        
        logger.debug(f"{func.__name__} - Time: {(end_time-start_time)*1000:.1f}ms, Memory: {end_memory-start_memory:.1f}MB")
        
        return result
    return wrapper

# Global cleanup
import atexit
def global_cleanup():
    """Cleanup resources on exit"""
    logger.info("Performing global cleanup...")
    cleanup_worker()

atexit.register(global_cleanup)