import io
import re
import csv
import base64
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from collections import Counter

import streamlit as st
import fitz  # PyMuPDF

# OCR (facultatif, fallback si PDF scanné)
USE_OCR_DEFAULT = True
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# =========================
# Config UI
# =========================

st.set_page_config(page_title="Contrôle retours vs factures", layout="wide")

st.title("🔎 Contrôle des retours vs factures fournisseurs (multi-factures + ratio par fichier)")
st.caption(
    "Uploads : Demandes de retours (PDF multiples) avec `N° du retour :` et "
    "`N° de retour fournisseur :` + Factures fournisseur (PDF **multiples**) avec `Retournummer :`."
)

# =========================
# Regex spécifiques métier
# =========================
# - Demandes de retours :
#   1) Interne         :  'N° du retour : <ID>'
#   2) Fournisseur     :  'N° de retour fournisseur : <ID>'
# - Factures :
#   - Retournummer     :  'Retournummer : <ID>'

RET_INTERNAL_PAT = re.compile(
    r"(?i)\bN[\u00B0ºo]\s*du\s*retour\s*:?\s*([A-Z0-9][A-Z0-9_\-\/]{2,})"
)
RET_SUPPLIER_PAT = re.compile(
    r"(?i)\bN[\u00B0ºo]\s*de\s*retour\s*fournisseur\s*:?\s*([A-Z0-9][A-Z0-9_\-\/]{2,})"
)
INVOICE_RET_PAT = re.compile(
    r"(?i)\bRetournummer\s*:?\s*([A-Z0-9][A-Z0-9_\-\/]{2,})"
)

# Langues OCR (on couvre fr/en/de/nl)
OCR_LANG = "fra+eng+deu+nld"

# =========================
# Utils
# =========================

def normalize_id(raw: str) -> str:
    """Uppercase + suppression des non-alphanum pour matcher robustement."""
    return re.sub(r"[^A-Z0-9]", "", raw.upper())

def build_download_button(rows: List[Dict[str, str]], filename: str, label: str):
    """Export CSV téléchargeable."""
    if not rows:
        st.button(label, disabled=True)
        return
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    b64 = base64.b64encode(buf.getvalue().encode()).decode()
    href = f'<a download="{filename}" href="data:text/csv;base64,{b64}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def extract_ids_by_page_with_pymupdf(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Retourne [(page_index, text)] pour corrélation par page."""
    out = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            out.append((i, page.get_text("text")))
    return out

def extract_ids_with_ocr_pages(pdf_bytes: bytes, dpi=250, lang: str = OCR_LANG) -> List[Tuple[int, str]]:
    """OCR page par page (fallback pour PDF scannés)."""
    if not OCR_AVAILABLE:
        return []
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    out = []
    for i, img in enumerate(pages):
        try:
            txt = pytesseract.image_to_string(img, lang=lang)
        except Exception:
            txt = ""
        out.append((i, txt))
    return out

# =========================
# Data classes résultats
# =========================

@dataclass
class ReturnExtraction:
    file_name: str
    used_ocr: bool
    internal_ids: Set[str]                 # uniques
    supplier_ids: Set[str]                 # uniques
    supplier_counts: Dict[str, int]        # occurrences (lignes)
    page_pairs: List[Tuple[str, str, int]] # (internal, supplier, page_index)
    text_len: int

@dataclass
class InvoiceExtraction:
    file_name: str
    used_ocr: bool
    retournummer_ids: Set[str]             # uniques
    retournummer_counts: Dict[str, int]    # occurrences (lignes)
    text_len: int

# =========================
# Extraction (Demandes retours)
# =========================

def extract_returns_pdf(pdf_bytes: bytes, file_name: str, try_ocr: bool, text_min: int = 80) -> ReturnExtraction:
    # 1) Texte par page via PyMuPDF
    pages = extract_ids_by_page_with_pymupdf(pdf_bytes)
    flat_text = "\n".join(text for _, text in pages)
    used_ocr = False

    # 2) Heuristique : si peu de texte, tenter OCR
    if try_ocr and len(flat_text.strip()) < text_min:
        ocr_pages = extract_ids_with_ocr_pages(pdf_bytes)
        if sum(len(t.strip()) for _, t in ocr_pages) > len(flat_text.strip()):
            pages = ocr_pages
            flat_text = "\n".join(t for _, t in pages)
            used_ocr = True

    # 3) Extraction des 2 types d'IDs + mapping par page
    internal_list = []
    supplier_list = []
    page_pairs: List[Tuple[str, str, int]] = []

    for i, txt in pages:
        internals = [normalize_id(m.group(1)) for m in RET_INTERNAL_PAT.finditer(txt)]
        suppliers = [normalize_id(m.group(1)) for m in RET_SUPPLIER_PAT.finditer(txt)]

        # Occurrences (pour notion de "lignes")
        internal_list.extend([x for x in internals if x])
        supplier_list.extend([x for x in suppliers if x])

        # Mapping si les deux présents sur la même page
        for a in internals:
            for b in suppliers:
                if a and b:
                    page_pairs.append((a, b, i))

    internal_ids = set(internal_list)
    supplier_ids = set(supplier_list)
    supplier_counts = dict(Counter(supplier_list))

    return ReturnExtraction(
        file_name=file_name,
        used_ocr=used_ocr,
        internal_ids=internal_ids,
        supplier_ids=supplier_ids,
        supplier_counts=supplier_counts,
        page_pairs=page_pairs,
        text_len=len(flat_text),
    )

# =========================
# Extraction (Factures)
# =========================

def extract_invoice_pdf(pdf_bytes: bytes, file_name: str, try_ocr: bool, text_min: int = 80) -> InvoiceExtraction:
    pages = extract_ids_by_page_with_pymupdf(pdf_bytes)
    flat_text = "\n".join(t for _, t in pages)
    used_ocr = False

    if try_ocr and len(flat_text.strip()) < text_min:
        ocr_pages = extract_ids_with_ocr_pages(pdf_bytes)
        if sum(len(t.strip()) for _, t in ocr_pages) > len(flat_text.strip()):
            pages = ocr_pages
            flat_text = "\n".join(t for _, t in pages)
            used_ocr = True

    retour_list = []
    for _, txt in pages:
        for m in INVOICE_RET_PAT.finditer(txt):
            nid = normalize_id(m.group(1))
            if nid:
                retour_list.append(nid)

    retour_ids = set(retour_list)
    retour_counts = dict(Counter(retour_list))

    return InvoiceExtraction(
        file_name=file_name,
        used_ocr=used_ocr,
        retournummer_ids=retour_ids,
        retournummer_counts=retour_counts,
        text_len=len(flat_text),
    )

# =========================
# UI
# =========================

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Upload des PDF")
    returns_files = st.file_uploader(
        "Demandes de retours (PDF multiples)",
        type=["pdf"], accept_multiple_files=True
    )
    invoice_files = st.file_uploader(
        "Factures fournisseur (PDF **multiples**)",
        type=["pdf"], accept_multiple_files=True
    )

with right:
    st.subheader("2) Paramètres")
    try_ocr = st.checkbox(
        "Activer OCR (fallback) si PDF scanné",
        value=USE_OCR_DEFAULT and OCR_AVAILABLE,
        help="Nécessite Tesseract installé (sinon ignoré automatiquement)."
    )
    text_len_threshold = st.slider(
        "Seuil de texte minimal avant OCR (caractères)",
        min_value=20, max_value=400, value=80, step=10
    )

st.divider()
go = st.button(
    "🚀 Lancer la comparaison",
    type="primary",
    disabled=not(returns_files and invoice_files)
)

if go:
    # ================
    # 1) Extraction retours (par fichier)
    # ================
    st.subheader("Résultats d'extraction : Demandes de retours")
    ret_logs: List[ReturnExtraction] = []

    all_supplier_expected_set: Set[str] = set()          # uniques attendus (pour comparaison set)
    all_internal_ids_set: Set[str] = set()               # info
    all_supplier_counts_total: Counter = Counter()       # occurrences globales (lignes attendues)

    sources_by_supplier: Dict[str, Set[str]] = {}        # dans quels fichiers on a vu cet ID (info)
    pairs_rows: List[Dict[str, str]] = []                # correspondances Interne ↔ Fournisseur détectées par page

    for f in returns_files:
        pdf_bytes = f.read()
        res = extract_returns_pdf(pdf_bytes, f.name, try_ocr, text_len_threshold)
        ret_logs.append(res)

        all_internal_ids_set |= res.internal_ids
        all_supplier_expected_set |= res.supplier_ids
        all_supplier_counts_total.update(res.supplier_counts)

        for sid in res.supplier_ids:
            sources_by_supplier.setdefault(sid, set()).add(f.name)

        # Correspondances
        seen = set()
        for a, b, page_idx in res.page_pairs:
            key = (a, b, page_idx)
            if key in seen:
                continue
            seen.add(key)
            pairs_rows.append({
                "id_interne": a,
                "id_fournisseur": b,
                "fichier": res.file_name,
                "page": page_idx + 1,
            })

    # Doublons info côté retours (occurrences > 1 sur l'ensemble)
    duplicates_returns_rows = [
        {"id_normalise": k, "occurrences": v}
        for k, v in sorted(all_supplier_counts_total.items(), key=lambda x: -x[1]) if v > 1
    ]

    # ================
    # 2) Extraction factures (plusieurs fichiers)
    # ================
    st.subheader("Résultats d'extraction : Factures fournisseurs (toutes)")
    inv_logs: List[InvoiceExtraction] = []
    invoice_ids_set_global: Set[str] = set()
    invoice_counts_global: Counter = Counter()

    for inv in invoice_files:
        pdf_bytes = inv.read()
        inv_res = extract_invoice_pdf(pdf_bytes, inv.name, try_ocr, text_len_threshold)
        inv_logs.append(inv_res)
        invoice_ids_set_global |= inv_res.retournummer_ids
        invoice_counts_global.update(inv_res.retournummer_counts)

    # ================
    # 3) Comparaison set (IDs uniques) — vue globale
    # ================
    st.subheader("Comparaison **IDs uniques** (Fournisseur retours ↔ Retournummer factures)")
    missing_ids = sorted(all_supplier_expected_set - invoice_ids_set_global)
    extra_ids   = sorted(invoice_ids_set_global - all_supplier_expected_set)
    covered_ids = sorted(all_supplier_expected_set & invoice_ids_set_global)

    colA, colB, colC, colD = st.columns(4)
    colA.metric("IDs fournisseur attendus (uniques)", len(all_supplier_expected_set))
    colB.metric("Présents sur factures (uniques)", len(covered_ids))
    colC.metric("Manquants (uniques)", len(missing_ids))
    colD.metric("En trop (uniques)", len(extra_ids))

    if all_supplier_expected_set:
        coverage_unique = 100.0 * len(covered_ids) / len(all_supplier_expected_set)
        st.progress(min(1.0, coverage_unique / 100.0), text=f"Couverture (uniques) : {coverage_unique:.1f}%")

    st.markdown("### ✅ Présents (uniques)")
    if covered_ids:
        present_rows = [{"id_fournisseur": x, "fichiers_retour": ", ".join(sorted(sources_by_supplier.get(x, [])))} for x in covered_ids]
        st.dataframe(present_rows, use_container_width=True, hide_index=True)
        build_download_button(present_rows, "fournisseur_presents_uniques.csv", "⬇️ Export CSV (présents uniques)")
    else:
        st.info("Aucun ID présent (unique).")

    st.markdown("### ❌ Manquants (uniques)")
    if missing_ids:
        miss_rows = [{"id_fournisseur": x, "fichiers_retour": ", ".join(sorted(sources_by_supplier.get(x, [])))} for x in missing_ids]
        st.dataframe(miss_rows, use_container_width=True, hide_index=True)
        build_download_button(miss_rows, "fournisseur_manquants_uniques.csv", "⬇️ Export CSV (manquants uniques)")
    else:
        st.success("Aucun manquant (unique) 👌")

    st.markdown("### ⚠️ En trop (uniques)")
    if extra_ids:
        extra_rows = [{"retournummer": x} for x in extra_ids]
        st.dataframe(extra_rows, use_container_width=True, hide_index=True)
        build_download_button(extra_rows, "retournummer_en_trop_uniques.csv", "⬇️ Export CSV (en trop uniques)")
    else:
        st.info("Aucun ID en trop (unique).")

    # ================
    # 4) Comparaison par **lignes** (occurrences) — vue par fichier
    # ================
    st.subheader("Comparaison **par lignes** (occurrences) — par fichier de retour")
    file_summary_rows: List[Dict[str, str]] = []
    total_expected_lines = 0
    total_found_lines = 0

    for r in ret_logs:
        expected_lines = sum(r.supplier_counts.values())
        found_lines = 0
        missing_detail_parts = []

        for sid, cnt in r.supplier_counts.items():
            inv_cnt = invoice_counts_global.get(sid, 0)
            matched = min(cnt, inv_cnt)
            found_lines += matched
            if inv_cnt < cnt:
                # détail : id : trouvés/sur
                missing_detail_parts.append(f"{sid}:{inv_cnt}/{cnt}")

        total_expected_lines += expected_lines
        total_found_lines += found_lines
        pct = (100.0 * found_lines / expected_lines) if expected_lines else 100.0

        file_summary_rows.append({
            "fichier_retour": r.file_name,
            "lignes_retour": expected_lines,
            "lignes_trouvees_factures": found_lines,
            "ratio": f"{found_lines}/{expected_lines}",
            "couverture_%": f"{pct:.1f}",
            "manquants_detail": "; ".join(missing_detail_parts) if missing_detail_parts else "",
        })

    if file_summary_rows:
        st.dataframe(file_summary_rows, use_container_width=True, hide_index=True)
        build_download_button(file_summary_rows, "resume_par_fichier.csv", "⬇️ Export CSV (résumé par fichier)")
    else:
        st.info("Aucune ligne détectée dans les demandes de retours.")

    # Vue globale par **lignes** (tous fichiers de retours confondus)
    st.markdown("### Vue globale par lignes (tous retours)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total lignes retours", total_expected_lines)
    col2.metric("Total lignes trouvées sur factures", total_found_lines)
    ratio_txt = f"{total_found_lines}/{total_expected_lines}" if total_expected_lines else "0/0"
    global_pct = (100.0 * total_found_lines / total_expected_lines) if total_expected_lines else 100.0
    col3.metric("Couverture globale", f"{global_pct:.1f}%")
    st.caption(f"Ratio global (lignes) : **{ratio_txt}**")

    # ================
    # 5) Correspondance Interne ↔ Fournisseur (détectée par page)
    # ================
    st.markdown("### 🔗 Correspondance Interne ↔ Fournisseur (détectée par page)")
    if pairs_rows:
        st.dataframe(pairs_rows, use_container_width=True, hide_index=True)
        build_download_button(pairs_rows, "correspondance_interne_fournisseur.csv", "⬇️ Export CSV (correspondance)")
    else:
        st.caption("Aucune page n'avait les deux champs simultanément — correspondance non inférée.")

    # ================
    # 6) Doublons côté retours (info)
    # ================
    st.markdown("### 🔁 Doublons détectés côté retours (occurrences > 1)")
    if duplicates_returns_rows:
        st.dataframe(duplicates_returns_rows, use_container_width=True, hide_index=True)
        build_download_button(duplicates_returns_rows, "doublons_retours.csv", "⬇️ Export CSV (doublons retours)")
    else:
        st.caption("Pas de doublons significatifs repérés côté retours.")

    # ================
    # 7) Logs techniques
    # ================
    with st.expander("🧾 Détails d'extraction (logs)"):
        for inv in inv_logs:
            st.write(f"Facture `{inv.file_name}` : uniques={len(inv.retournummer_ids)}, "
                     f"occurrences={sum(inv.retournummer_counts.values())}, "
                     f"text_len={inv.text_len}, OCR={'oui' if inv.used_ocr else 'non'}")
        for r in ret_logs:
            st.write(f"Retour `{r.file_name}` : interne_uniques={len(r.internal_ids)}, "
                     f"fournisseur_uniques={len(r.supplier_ids)}, "
                     f"occurrences_fournisseur={sum(r.supplier_counts.values())}, "
                     f"text_len={r.text_len}, OCR={'oui' if r.used_ocr else 'non'}")
