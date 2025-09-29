# app.py
# -*- coding: utf-8 -*-
"""
App Streamlit : contrôle des demandes de retour (RetourenPrint) vs factures (FakturenPrint)

Tâches :
1) RetourenPrint.pdf : extraire "N° de retour fournisseur : <num>" et compter le nombre de lignes
   = plus longue suite consécutive 1..k trouvée dans le tableau après "Lig. no .".
2) FakturenPrint.pdf : compter les lignes contenant "Retournummer : <num>" (par numéro).
3) Tableau :
   n° retour | fichier demande | nbre lignes (demande) | fichier(s) facture | nbre lignes sur facture | représentation (x / y) | statut

Dépendances :
  streamlit==1.37.0
  pymupdf==1.24.8
  pdf2image==1.17.0
  pytesseract==0.3.10
  Pillow==10.4.0

Exécution :
  streamlit run app.py
"""

from __future__ import annotations

import io
import re
import csv
from dataclasses import dataclass
from typing import Dict, List

import streamlit as st

# PDF / OCR
import fitz  # PyMuPDF
from PIL import Image

# OCR (fallback si PDF scanné)
OCR_AVAILABLE = True
try:
    from pdf2image import convert_from_bytes
    import pytesseract
except Exception:
    OCR_AVAILABLE = False
    convert_from_bytes = None  # type: ignore
    pytesseract = None  # type: ignore

# =========================
# Regex / constantes
# =========================
SUPPLIER_RET_NO_RE = re.compile(
    r"""N[°º]\s*de\s*retour\s*fournisseur\s*:\s*(\d{5,})""", re.IGNORECASE
)
INVOICE_RET_NO_RE = re.compile(
    r"""Retournummer\s*:\s*(\d{5,})""", re.IGNORECASE
)
LIG_HEADER_RE = re.compile(r"""Lig\.\s*no\s*\.""", re.IGNORECASE)

# Numéro de ligne au début d’une ligne (table), ex: "  12  "
LEADING_ROW_NUM_RE = re.compile(r"""^\s*(\d{1,6})\s""", re.MULTILINE)


@dataclass
class RetourRequest:
    file_name: str
    supplier_return_no: str
    lines_count: int
    debug_numbers_found: List[int]  # pour vérifier ce qui a été capté


@dataclass
class InvoiceScan:
    file_name: str
    counts_by_return_no: Dict[str, int]


# =========================
# Utilitaires
# =========================
@st.cache_data(show_spinner=False)
def read_pdf_text(file_bytes: bytes, use_ocr_if_needed: bool = True) -> str:
    """Lit le texte d'un PDF (PyMuPDF d'abord ; fallback OCR si page quasi vide)."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    all_text_parts: List[str] = []

    # Chemin Tesseract optionnel depuis la sidebar
    tesseract_cmd = st.session_state.get("__tesseract_cmd__", None)
    if OCR_AVAILABLE and tesseract_cmd:
        import pytesseract as _pt
        _pt.pytesseract.tesseract_cmd = tesseract_cmd

    for page in doc:
        text = page.get_text("text") or ""
        # Si très peu de texte et OCR dispo, tenter OCR
        if use_ocr_if_needed and OCR_AVAILABLE and len(text.strip()) < 20:
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                try:
                    import pytesseract as _pt
                    text = _pt.image_to_string(img, lang="fra")
                except Exception:
                    pass
            except Exception:
                # Tentative via pdf2image sur la page courante
                if convert_from_bytes is not None:
                    try:
                        images = convert_from_bytes(
                            file_bytes, dpi=300,
                            first_page=page.number + 1,
                            last_page=page.number + 1
                        )
                        if images:
                            try:
                                import pytesseract as _pt
                                text = _pt.image_to_string(images[0], lang="fra")
                            except Exception:
                                pass
                    except Exception:
                        pass

        all_text_parts.append(text)

    return "\n".join(all_text_parts)


def longest_consecutive_from_one(nums: List[int]) -> int:
    """
    Retourne le plus grand k tel que {1,2,...,k} ⊆ nums.
    Robuste si d'autres nombres (hors suite) sont présents dans le texte.
    """
    s = set(n for n in nums if 1 <= n <= 1_000_000)
    k = 0
    while (k + 1) in s:
        k += 1
    return k


def parse_retour_request(text: str, file_name: str) -> RetourRequest:
    """Extrait le n° de retour fournisseur et le nombre de lignes depuis un 'RetourenPrint'."""
    m = SUPPLIER_RET_NO_RE.search(text)
    if not m:
        raise ValueError(f"{file_name}: N° de retour fournisseur introuvable.")
    ret_no = m.group(1)

    # On cible la zone du tableau après l'en-tête 'Lig. no .'
    hm = LIG_HEADER_RE.search(text)
    sub = text if not hm else text[hm.end():]

    numbers_found = [int(n) for n in LEADING_ROW_NUM_RE.findall(sub)]
    lines_count = longest_consecutive_from_one(numbers_found)

    return RetourRequest(
        file_name=file_name,
        supplier_return_no=ret_no,
        lines_count=lines_count,
        debug_numbers_found=numbers_found[:50],  # pour inspection
    )


def scan_invoice_counts(text: str, file_name: str) -> InvoiceScan:
    """Parcourt un 'FakturenPrint' et compte les 'Retournummer : <no>' (par numéro)."""
    numbers = INVOICE_RET_NO_RE.findall(text)
    counts: Dict[str, int] = {}
    for n in numbers:
        counts[n] = counts.get(n, 0) + 1
    return InvoiceScan(file_name=file_name, counts_by_return_no=counts)


def rows_to_csv(rows: List[dict], fieldnames: List[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8")


# =========================
# UI
# =========================
st.set_page_config(page_title="Contrôle retours vs factures", layout="wide")
st.title("🔎 Contrôle des retours vs factures fournisseurs")
st.caption("Chargez plusieurs **demandes de retour** (RetourenPrint*.pdf) et plusieurs **factures** (FakturenPrint*.pdf).")

with st.sidebar:
    st.subheader("Options")
    use_ocr = st.checkbox("Activer le fallback OCR si nécessaire", value=True, help="Utile pour les PDF scannés.")
    if OCR_AVAILABLE:
        st.text_input(
            "Chemin de l'exécutable Tesseract (optionnel)",
            key="__tesseract_cmd__",
            help="Laissez vide si Tesseract est déjà dans le PATH.",
        )
    else:
        st.info("OCR indisponible (pytesseract/pdf2image non importés).")

col1, col2 = st.columns(2)
with col1:
    retour_files = st.file_uploader(
        "📥 Fichiers 'RetourenPrint' (demandes de retour)",
        type=["pdf"], accept_multiple_files=True, key="retour_files",
        help="Chaque fichier doit contenir 'N° de retour fournisseur : <num>'."
    )
with col2:
    invoice_files = st.file_uploader(
        "📥 Fichiers 'FakturenPrint' (factures)",
        type=["pdf"], accept_multiple_files=True, key="invoice_files",
        help="Les lignes contiennent 'Retournummer : <num>'."
    )

if retour_files and invoice_files:
    st.divider()
    st.subheader("Résultats")

    # 1) Parse demandes de retour
    retour_list: List[RetourRequest] = []
    retour_errors: List[str] = []
    for up in retour_files:
        try:
            data = up.read()
            text = read_pdf_text(data, use_ocr_if_needed=use_ocr)
            retour = parse_retour_request(text, up.name)
            retour_list.append(retour)
        except Exception as e:
            retour_errors.append(f"{up.name} ➜ {e}")

    # 2) Parse factures (compteurs par 'Retournummer')
    invoice_scans: List[InvoiceScan] = []
    for up in invoice_files:
        try:
            data = up.read()
            text = read_pdf_text(data, use_ocr_if_needed=use_ocr)
            scan = scan_invoice_counts(text, up.name)
            invoice_scans.append(scan)
        except Exception as e:
            st.error(f"{up.name} ➜ Erreur lecture facture: {e}")

    # Indexation: ret_no -> {file_name: count}
    index_counts: Dict[str, Dict[str, int]] = {}
    for scan in invoice_scans:
        for ret_no, cnt in scan.counts_by_return_no.items():
            by_file = index_counts.setdefault(ret_no, {})
            by_file[scan.file_name] = by_file.get(scan.file_name, 0) + cnt

    # 3) Récapitulatif
    rows: List[dict] = []
    for r in retour_list:
        by_file = index_counts.get(r.supplier_return_no, {})
        files_with_counts = [f"{fn} ({cnt})" for fn, cnt in sorted(by_file.items()) if cnt > 0]
        total_on_invoices = sum(by_file.values()) if by_file else 0

        # Représentation du ratio : "x / y"
        display_ratio = f"{total_on_invoices} / {r.lines_count}"

        # Statut lisible
        if r.lines_count == 0:
            badge = "❌ 0/0"
        elif total_on_invoices == r.lines_count:
            badge = "✅ match"
        elif total_on_invoices == 0:
            badge = "❌ aucun"
        else:
            badge = "⚠️ partiel"

        rows.append({
            "N° retour": r.supplier_return_no,
            "Fichier demande": r.file_name,
            "Nbre lignes (demande)": r.lines_count,
            "Fichier(s) facture correspondant(s)": ", ".join(files_with_counts) if files_with_counts else "—",
            "Nbre lignes sur facture": total_on_invoices,
            "Représentation": display_ratio,  # <= "x / y"
            "Statut": badge,
        })

    # Affichage
    if rows:
        column_order = [
            "N° retour",
            "Fichier demande",
            "Nbre lignes (demande)",
            "Fichier(s) facture correspondant(s)",
            "Nbre lignes sur facture",
            "Représentation",
            "Statut",
        ]
        st.data_editor(
            rows,
            column_order=column_order,
            hide_index=True,
            use_container_width=True,
            disabled=True,
        )

        # Export CSV
        csv_bytes = rows_to_csv(rows, fieldnames=column_order)
        st.download_button(
            "💾 Télécharger le CSV",
            data=csv_bytes,
            file_name="resultat_retours_vs_factures.csv",
            mime="text/csv",
        )
    else:
        st.info("Aucune correspondance trouvée.")

    # Debug
    with st.expander("🔍 Détails demandes de retour (debug)", expanded=False):
        if retour_list:
            for r in retour_list:
                st.write(
                    f"**{r.file_name}** — N° retour fournisseur: `{r.supplier_return_no}`, "
                    f"Lignes détectées: **{r.lines_count}** — premiers numéros captés: {r.debug_numbers_found}"
                )
        if retour_errors:
            st.warning("\n".join(retour_errors))

    with st.expander("🧾 Détails factures (compteurs par 'Retournummer')", expanded=False):
        if invoice_scans:
            for scan in invoice_scans:
                st.write(f"**{scan.file_name}**")
                if scan.counts_by_return_no:
                    st.json(scan.counts_by_return_no)
                else:
                    st.write("(Aucun 'Retournummer' détecté)")
else:
    st.info("Chargez au moins un fichier de demande et un fichier de facture pour lancer l'analyse.")
