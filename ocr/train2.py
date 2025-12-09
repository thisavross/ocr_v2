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

#define key
EXPECTED_FIELDS = [
    "PROVINSI", "KOTA", "KABUPATEN", "NIK", "NAMA", "TEMPAT/TGL LAHIR",
    "JENIS KELAMIN","GOL DARAH", "ALAMAT", "RT/RW", "KEL/DESA", "KECAMATAN",
    "AGAMA", "STATUS PERKAWINAN", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU HINGGA"
]

#define value
CONTROLLED_VALUES = {
    "JENIS KELAMIN" : ["LAKILAKI", "PEREMPUAN"],
    "GOL DARAH" : ["A", "B", "AB", "O"],
    "STATUS PERKAWINAN" : ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"],
    "AGAMA" : ["BUDDHA", "HINDU", "ISLAM", "KATOLIK", "KRISTEN", "KONGHUCU", "KEPERCAYAAN"]
}


def fuzzy_match_field(text, threshold=75):
    """Fuzzy match text to expected KTP field names with aggressive cleanup"""
    if not text.strip():
        return None

    # remove all non-alphabetic characters (colon, dots, commas, OCR noise)
    cleaned = re.sub(r'[^A-Z]', '', text.upper())

    match = process.extractOne(cleaned, EXPECTED_FIELDS, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return match[0]
    return None


def fuzzy_match_value(key, val, threshold=75):
    """
    Fuzzy-correct value based on controlled list for the given key.
    """
    if not val:
        return val

    val_up = val.upper().replace(" ", "")
    controlled_list = CONTROLLED_VALUES.get(key, [])

    if not controlled_list:
        return val  # No controlled values → return as is

    # Normalize controlled list
    controlled_up = [v.upper().replace(" ", "") for v in controlled_list]

    # 1️ Exact match
    if val_up in controlled_up:
        return controlled_list[controlled_up.index(val_up)]

    # 2️ Fuzzy match
    match = process.extractOne(val_up, controlled_up, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return controlled_list[controlled_up.index(match[0])]

    return val  # fallback

# --- Fuzzy matching function ---

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
            field_match = normalize_key(key.strip())
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
            field_match = normalize_key(possible_key)
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
            
            # # Strategy 2: OCR-corrected extraction (handles B→8, O→0, etc.)
            # potential_nik = extract_potential_nik_with_ocr_correction(line)
            # if potential_nik:
            #     data["NIK"] = potential_nik
            #     processed_fields.add("NIK")
            #     continue

    return data


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

def detect_controlled_value(text):
    """
    Detect if the line contains a valid controlled value
    like 'ISLAM', 'HINDU', 'O', 'AB', 'LAKILAKI'.
    If found → return (key, corrected_value).
    """
    clean = text.upper().replace(" ", "")

    for key, values in CONTROLLED_VALUES.items():
        for v in values:
            if clean == v.replace(" ", ""):  # exact normalized match
                return key, v

    # fuzzy fallback
    for key, values in CONTROLLED_VALUES.items():
        normalized = [v.replace(" ", "") for v in values]
        match = process.extractOne(clean, normalized, scorer=fuzz.ratio)
        if match and match[1] >= 80:
            idx = normalized.index(match[0])
            return key, values[idx]

    return None, None

def extract_field_value(line, keyword):
    """Extract value after keyword, handling both colon and non-colon formats."""
    keyword = EXPECTED_FIELDS
    # Try colon format first
    if ':' in line:
        # Look for pattern like "KABUPATEN : VALUE" or "KABUPATEN: VALUE"
        pattern = rf'{keyword}\s*:\s*(.+)$'
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no colon or colon format didn't match, try direct keyword removal
    val = re.sub(rf'^{keyword}\s*', '', line, flags=re.IGNORECASE).strip()
    return val

def fix_ocr_nik(text):
    """
    Convert OCR-mangled NIK text into a valid 16 digit numeric string.
    Handles letter→digit corrections and removes noise.
    """

    mapping = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', '|': '1',
        'Z': '2',
        'S': '5', 's': '5',
        'B': '8',
        'G': '9', 'g': '9', 'q': '9',
        "h": "6"
    }

    cleaned = ""
    for ch in text:
        if ch.isdigit():
            cleaned += ch
        elif ch in mapping:
            cleaned += mapping[ch]

    # Limit to 16 digits (NIK is always 16)
    if len(cleaned) >= 16:
        return cleaned[:16]

    return cleaned  # allow partial if OCR too bad

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
            raw_val = extract_value(line)

            # If same-line value empty, try next line
            if not raw_val and i + 1 < len(lines):
                raw_val = lines[i + 1]

            if raw_val:
                raw_val = raw_val.strip()

                # Extract only digits first
                digits_only = re.sub(r"\D", "", raw_val)

                if len(digits_only) == 16:
                    # Already valid → no correction
                    ktp_data["NIK"] = digits_only
                else:
                    # Not valid → run OCR correction
                    corrected = fix_ocr_nik(raw_val)

                    # If OCR correction still invalid, leave as-is (your fix_ocr_nik returns partial)
                    ktp_data["NIK"] = corrected


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
        elif any(k in upper.replace(" ", "") for k in ["TEMPAT", "TGL", "LAHIR"]):

            raw = line

            # 1. Remove all label variants: TEMPAT, TGL, TGLANI, TGL/TGL, TGI, LAHIR, etc.
            label_pattern = r'(TEMPAT|TMPAT|TMPT)?\s*[/\\]?\s*(TGL|TGI|TGLANI|TGLAN|TGLNI)?\s*(LAHIR|LAIIR|LAHIRI)?'
            cleaned = re.sub(label_pattern, '', raw, flags=re.IGNORECASE).strip()

            # 2. If cleaned has a date → use whole string
            date_found = re.findall(r'\d{2}[-/]\d{2}[-/]\d{4}', raw)
            if date_found:
                # Extract CITY + DATE pattern
                m = re.search(r'([A-Z ]+),\s*\d{2}[-/]\d{2}[-/]\d{4}', raw.upper())
                if m:
                    cleaned = m.group(0).title()
                else:
                    cleaned = cleaned

            # 3. If still empty or still looks like label → take next line
            if (cleaned == "" or cleaned.upper() in ["TEMPAT", "TGL", "LAHIR"]) and i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if not fuzzy_match_field(next_line.upper()):
                    cleaned = next_line
                    i += 1

            # 4. Final cleanup: remove stray noise like 'a', '.', '-', etc.
            cleaned = re.sub(r'^[^A-Za-z0-9]+', '', cleaned).strip()

            if cleaned:
                ktp_data["Tempat_Tgl_Lahir"] = cleaned


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
            if val:
                if val.startswith(':'):
                    val = val[1:].strip()
                ktp_data["Gol_Darah"] = val

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
                    any(stop in next_upper for stop in ["RT/RW", "RT_RW", "RTIRW", "KEL/DESA", "KECAMATAN"])):
                    break
                addr_parts.append(next_line.strip())
                j += 1
            
            ktp_data["Alamat"] = " ".join(addr_parts).strip()

        # --- RT/RW ---
        elif fuzzy_match_field(upper) in ["RT/RW", "RT_RW", "RTIRW","BT/RW", "ATRW"]:
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
@timeout(60)  # Overall timeout for processing
def enhanced_ktp_processing(image_path: str):
    """Main KTP processing function with correct low-confidence handling."""
    try:
        # Validate image path
        if not os.path.exists(image_path):
            return [], {}, json.dumps({
                "raw_text": [],
                "extracted_data": {},
                "mean_confidence": 0,
                "line_count": 0,
                "status": "rejected",
                "error": f"Image file not found: {image_path}"
            })

        # === STEP 1: OCR PROCESSING ===
        try:
            temp_image_path = preprocess_img_to_tempfile(image_path, long_side=1024)
            worker_output = safe_ocr(temp_image_path, timeout=30)

            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

        except TimeoutException as e:
            logger.error(f"OCR timeout: {e}")
            return [], {}, json.dumps({
                "raw_text": [],
                "extracted_data": {},
                "mean_confidence": 0,
                "line_count": 0,
                "status": "rejected",
                "error": f"OCR timeout: {str(e)}"
            })

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return [], {}, json.dumps({
                "raw_text": [],
                "extracted_data": {},
                "mean_confidence": 0,
                "line_count": 0,
                "status": "rejected",
                "error": f"OCR failed: {str(e)}"
            })

        # === STEP 2: IMMEDIATE REJECTION FROM WORKER ===
        if worker_output.get("status") == "rejected":
            # MATCH WORKER FORMAT EXACTLY (OPTION C)
            return (
                worker_output.get("raw_text", []),
                {},
                json.dumps({
                    "raw_text": worker_output.get("raw_text", []),
                    "extracted_data": {},
                    "mean_confidence": worker_output.get("mean_confidence", 0),
                    "line_count": worker_output.get("line_count", 0),
                    "status": "rejected",
                    "error": worker_output.get("error", "low_confidence")
                })
            )

        # === STEP 3: NORMAL OCR RESULT ===
        text_lines = worker_output.get("raw_text", [])
        mean_conf = worker_output.get("mean_confidence", 0)
        line_count = worker_output.get("line_count", len(text_lines))

        logger.info(f"OCR returned {len(text_lines)} lines")
        for i, line in enumerate(text_lines):
            logger.info(f"Line {i+1}: {line}")

        # === STEP 4: PARSE FIELDS ===
        try:
            ktp_data = parse_ktp_fields(text_lines)
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            ktp_data = {}

        # === STEP 5: NORMALIZATION ===
        normalized_data = {}
        for key, val in ktp_data.items():
            clean_key = key.strip().replace(" ", "_").replace("/", "_").title()
            if isinstance(val, str):
                val = val.strip()

            normalized_val = fuzzy_match_value(clean_key.upper(), val)
            normalized_data[clean_key] = normalized_val

        # === STEP 6: BUILD FINAL OUTPUT ===
        output_json = json.dumps({
            "raw_text": text_lines,
            "extracted_data": normalized_data,
            "mean_confidence": mean_conf,
            "line_count": line_count,
            "status": "success",
            "error": None
        }, ensure_ascii=False, indent=2)

        return text_lines, normalized_data, output_json

    except Exception as e:
        logger.error(f"Fatal processing error: {e}")
        return [], {}, json.dumps({
            "raw_text": [],
            "extracted_data": {},
            "mean_confidence": 0,
            "line_count": 0,
            "status": "rejected",
            "error": f"Processing failed: {str(e)}"
        })


# ========== GLOBAL CLEANUP WITH CACHE CLEARING ==========

def global_cleanup():
    """Cleanup resources on exit including clearing caches"""
    logger.info("Performing global cleanup...")
    cleanup_worker()
    logger.info("Caches cleared during cleanup")

import atexit
atexit.register(global_cleanup)