import io
import re
import csv
import base64
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

import streamlit as st
import fitz  # PyMuPDF

# OCR (facultatif)
USE_OCR_DEFAULT = True
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------------------------
# Config UI
# ---------------------------

st.set_page_config(page_title="Contrôle retours vs facture", layout="wide")

st.title("🔎 Contrôle des numéros de retours vs facture fournisseur")
st.caption("Upload : Demandes de retours (PDF multiples) + Facture fournisseur (PDF unique). "
           "Extraction → Comparaison → Rapport & exports CSV.")

DEFAULT_REGEX_LINES = [
    r"\bRMA[-\s:#]?([A-Z0-9\-]{5,})\b",
    r"\bRET(?:OUR)?[-\s:#]?([A-Z0-9\-]{4,})\b",
    r"\b(?:Retour|Return)[\s:#]*([A-Z0-9\-]{4,})\b",
    # Fallback générique si les docs listent des IDs alphanum >= 5
    r"\b([A-Z0-9]{5,})\b",
]

# ---------------------------
# Utils
# ---------------------------

def normalize_id(raw: str) -> str:
    """Uppercase + remove non-alphanum to compare proprement."""
    return re.sub(r"[^A-Z0-9]", "", raw.upper())

def compile_patterns(lines: List[str]) -> List[re.Pattern]:
    patterns = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            patterns.append(re.compile(ln, flags=re.I))
        except re.error as e:
            st.warning(f"Regex invalide ignorée: {ln} ({e})")
    return patterns

def extract_text_with_pymupdf(pdf_bytes: bytes) -> str:
    text_chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text_chunks.append(page.get_text("text"))
    return "\n".join(text_chunks)

def extract_text_with_ocr(pdf_bytes: bytes, dpi: int = 250, lang: str = "eng+fra") -> str:
    if not OCR_AVAILABLE:
        return ""
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    pieces = []
    for img in pages:
        try:
            pieces.append(pytesseract.image_to_string(img, lang=lang))
        except Exception:
            # Si Tesseract pas dispo/configuré : ignorer silencieusement
            pass
    return "\n".join(pieces)

def extract_ids_from_text(text: str, patterns: List[re.Pattern]) -> Tuple[List[str], Set[str]]:
    """
    Retourne (liste_raw_ids, set_norm_ids).
    On prend group(1) si présent, sinon group(0).
    """
    raw_ids = []
    for pat in patterns:
        for m in pat.finditer(text):
            grp = m.group(1) if m.lastindex else m.group(0)
            raw_ids.append(grp.strip())
    norm_ids = {normalize_id(x) for x in raw_ids if normalize_id(x)}
    return raw_ids, norm_ids

@dataclass
class ExtractionResult:
    file_name: str
    raw_ids: List[str]
    norm_ids: Set[str]
    used_ocr: bool
    text_len: int

# ---------------------------
# Extraction orchestrator
# ---------------------------

def extract_ids_from_pdf(pdf_bytes: bytes, file_name: str, patterns: List[re.Pattern],
                         try_ocr: bool, text_min_chars_for_ok: int = 80) -> ExtractionResult:
    text = extract_text_with_pymupdf(pdf_bytes)
    used_ocr = False

    # Heuristique: si le texte est trop court (scanné), tenter OCR
    if try_ocr and len(text.strip()) < text_min_chars_for_ok:
        ocr_text = extract_text_with_ocr(pdf_bytes)
        if len(ocr_text.strip()) > len(text.strip()):
            text = ocr_text
            used_ocr = True

    raw_ids, norm_ids = extract_ids_from_text(text, patterns)
    return ExtractionResult(
        file_name=file_name,
        raw_ids=raw_ids,
        norm_ids=norm_ids,
        used_ocr=used_ocr,
        text_len=len(text),
    )

def build_download_button(df_rows: List[Dict[str, str]], filename: str, label: str):
    if not df_rows:
        st.button(label, disabled=True)
        return
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(df_rows[0].keys()))
    writer.writeheader()
    writer.writerows(df_rows)
    b64 = base64.b64encode(output.getvalue().encode()).decode()
    href = f'<a download="{filename}" href="data:text/csv;base64,{b64}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------------------------
# UI
# ---------------------------

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Upload des PDF")
    returns_files = st.file_uploader(
        "Demandes de retours (PDF multiples)",
        type=["pdf"], accept_multiple_files=True
    )
    invoice_file = st.file_uploader(
        "Facture fournisseur (PDF unique)",
        type=["pdf"], accept_multiple_files=False
    )

with right:
    st.subheader("2) Paramètres")
    regex_text = st.text_area(
        "Regex (une par ligne, group(1) capturant l'ID si possible)",
        value="\n".join(DEFAULT_REGEX_LINES),
        height=150,
        help="Tu peux ajuster pour chaque fournisseur (RMA, RET, format interne, etc.)."
    )
    try_ocr = st.checkbox(
        "Activer OCR (fallback) si PDF scanné",
        value=USE_OCR_DEFAULT and OCR_AVAILABLE,
        help="Nécessite Tesseract installé côté machine."
    )
    text_len_threshold = st.slider(
        "Seuil de texte minimal avant OCR (caractères)",
        min_value=20, max_value=400, value=80, step=10
    )
    compiled = compile_patterns(regex_text.splitlines())

st.divider()
go = st.button("🚀 Lancer la comparaison", type="primary", disabled=not(returns_files and invoice_file))

if go:
    if not compiled:
        st.error("Aucune regex valide. Corrige les expressions puis relance.")
        st.stop()

    # 1) Extraction retours
    st.subheader("Résultats d'extraction : Demandes de retours")
    returns_map: Dict[str, List[str]] = {}   # norm_id -> fichiers sources
    duplicates: Dict[str, int] = {}          # norm_id -> occurrences

    logs = []
    for f in returns_files:
        pdf_bytes = f.read()
        res = extract_ids_from_pdf(pdf_bytes, f.name, compiled, try_ocr, text_len_threshold)
        logs.append(res)
        for nid in res.norm_ids:
            returns_map.setdefault(nid, []).append(f.name)
        # Doublons RAW par fichier (informationnel)
        for rid in res.raw_ids:
            key = normalize_id(rid)
            if key:
                duplicates[key] = duplicates.get(key, 0) + 1

    required_ids: Set[str] = set(returns_map.keys())

    # 2) Extraction facture
    st.subheader("Résultats d'extraction : Facture fournisseur")
    inv_bytes = invoice_file.read()
    inv_res = extract_ids_from_pdf(inv_bytes, invoice_file.name, compiled, try_ocr, text_len_threshold)
    invoice_ids = inv_res.norm_ids

    # 3) Comparaison
    st.subheader("Comparaison")
    missing = sorted(required_ids - invoice_ids)
    extra = sorted(invoice_ids - required_ids)
    covered = sorted(required_ids & invoice_ids)

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Nombre d'IDs attendus (retours)", len(required_ids))
    colB.metric("Présents sur la facture", len(covered))
    colC.metric("Manquants sur la facture", len(missing))
    colD.metric("En trop sur la facture", len(extra))

    if required_ids:
        coverage = 100.0 * len(covered) / len(required_ids)
        st.progress(min(1.0, coverage / 100.0), text=f"Taux de couverture : {coverage:.1f}%")

    st.markdown("### ✅ Présents")
    if covered:
        present_rows = [{"id_normalisé": x, "fichiers_retour": ", ".join(sorted(set(returns_map.get(x, []))))} for x in covered]
        st.dataframe(present_rows, use_container_width=True, hide_index=True)
        build_download_button(present_rows, "ids_presents.csv", "⬇️ Export CSV (présents)")
    else:
        st.info("Aucun ID couvert.")

    st.markdown("### ❌ Manquants sur la facture")
    if missing:
        miss_rows = [{"id_normalisé": x, "fichiers_retour": ", ".join(sorted(set(returns_map.get(x, []))))} for x in missing]
        st.dataframe(miss_rows, use_container_width=True, hide_index=True)
        build_download_button(miss_rows, "ids_manquants.csv", "⬇️ Export CSV (manquants)")
    else:
        st.success("Aucun manquant 👌")

    st.markdown("### ⚠️ En trop sur la facture (non attendus)")
    if extra:
        extra_rows = [{"id_normalisé": x} for x in extra]
        st.dataframe(extra_rows, use_container_width=True, hide_index=True)
        build_download_button(extra_rows, "ids_en_trop.csv", "⬇️ Export CSV (en trop)")
    else:
        st.info("Aucun ID en trop.")

    # 4) Doublons info
    st.markdown("### 🔁 Doublons détectés (compte brut)")
    dups_rows = [{"id_normalisé": k, "occurrences": v} for k, v in sorted(duplicates.items(), key=lambda x: -x[1]) if v > 1]
    if dups_rows:
        st.dataframe(dups_rows, use_container_width=True, hide_index=True)
        build_download_button(dups_rows, "doublons_retours.csv", "⬇️ Export CSV (doublons)")
    else:
        st.caption("Pas de doublons significatifs repérés dans les extractions brutes.")

    # 5) Logs techniques
    with st.expander("🧾 Détails d'extraction (logs)"):
        st.write(f"Facture `{inv_res.file_name}` : {len(inv_res.norm_ids)} IDs normalisés, "
                 f"text_len={inv_res.text_len}, OCR={'oui' if inv_res.used_ocr else 'non'}")
        for res in logs:
            st.write(f"Retour `{res.file_name}` : {len(res.norm_ids)} IDs normalisés, "
                     f"text_len={res.text_len}, OCR={'oui' if res.used_ocr else 'non'}")

