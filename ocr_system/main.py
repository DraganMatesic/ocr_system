import os
import io
import zipfile
import tempfile
import pdfplumber
from enum import Enum
from typing import List, Optional, Iterable
from dataclasses import dataclass, field
from datetime import datetime

# ---------- Types ----------

class Extensions(Enum):
    PDF = '.pdf'

@dataclass
class ExtractData:
    text_data: str = None
    info: List[str] = field(default_factory=list)
    warning: List[str] = field(default_factory=list)
    error: List[str] = field(default_factory=list)

@dataclass
class ZipMember:
    # Basic meta
    name: str
    size: int
    compress_size: int
    modified: datetime
    crc: int

    # Integrity & diagnostics
    crc_ok: bool = False
    error: Optional[str] = None

    # Always use a temp file (seekable path for downstream tools)
    temp_path: Optional[str] = None


@dataclass
class PdfPageStats:
    index: int
    width: float
    height: float
    image_area_ratio: float              # 0..1 of page area covered by images
    text_chars: int                      # len(page.chars)
    text_words: int                      # len(page.extract_words())
    text_area_ratio: float               # sum of word boxes area / page area
    has_text: bool
    has_images: bool
    image_dominant: bool                 # image_area_ratio >= IMG_RATIO_SCANNED
    text_dominant: bool                  # text_area_ratio >= TEXT_AREA_MIN_RATIO and not image_dominant

@dataclass
class PdfLayoutProfile:
    file_name: Optional[str]
    readable: bool
    pages: int
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # Aggregates
    pages_image_dominant: int = 0
    pages_text_dominant: int = 0
    pages_mixed: int = 0

    # Recommendation for processing
    recommended: str = "hybrid"          # one of: 'ocr', 'pdfplumber', 'hybrid'
    rationale: str = ""

    # Per-page details (optional for logging/QA)
    page_stats: List[PdfPageStats] = field(default_factory=list)


@dataclass
class ZipData:
    ok_files: List[str] = field(default_factory=list)
    bad_files: List[str] = field(default_factory=list)
    ok_files_count: int = 0
    bad_files_count: int = 0
    errors: List[str] = field(default_factory=list)

    # Rich results for downstream processing
    ok_members: List[ZipMember] = field(default_factory=list)
    bad_members: List[ZipMember] = field(default_factory=list)


# ---------- Extractor ----------
# Page is considered "scanned/image-based" if image coverage dominates
IMG_RATIO_SCANNED = 0.75       # ≥75% of page area is image

# Minimal text criteria (to guard against stamps/single words)
CHAR_MIN_FOR_TEXT = 50         # at least 50 glyphs
TEXT_AREA_MIN_RATIO = 0.02     # ≥2% of page area covered by word boxes

class TextExtractor:
    def __init__(self, extensions: Optional[Iterable[Extensions]] = None, temp_dir: Optional[str] = None):
        """
        temp_dir: where to place temp files; defaults to system temp if None.
        """
        self.valid_extensions = {e.value.lower() for e in (extensions or {Extensions.PDF})}
        self.zip_password: Optional[bytes] = None
        self.temp_dir = temp_dir

    def set_zip_password(self, zip_password):
        if isinstance(zip_password, str):
            zip_password = zip_password.encode('utf-8')
        self.zip_password = zip_password

    def extract(self, file_path: str) -> ExtractData:
        zip_data = ZipData()
        try:
            extract_data = ExtractData()

            if not os.path.exists(file_path):
                extract_data.error.append(f"could not find file on path {file_path}")
                return extract_data

            extension = self._get_extension(file_path)

            if extension == '.zip':
                zip_data = self.get_zip_members(file_path)
                # Example: act on ok members (call pdfplumber or OCR)
                for zip_member in zip_data.ok_members:
                    prof = self._profile_pdf_layout(path=zip_member.temp_path, original_name=zip_member.name)
                    print(prof)
                    # Route based on prof.recommended:
                    #   'pdfplumber' -> extract text directly
                    #   'ocr'        -> run OCR
                    #   'hybrid'     -> per-page: OCR pages where s.image_dominant else pdfplumber
                    # if prof.recommended == "hybrid":
                    #     image_pages = [s.index for s in prof.page_stats if s.image_dominant]
                    #     text_pages = [s.index for s in prof.page_stats if s.text_dominant]
                        # OCR image_pages; pdfplumber text_pages

                return extract_data

            elif not self._is_valid_member(file_path):
                extract_data.error.append(
                    f"file doesn't have valid extension. Got {extension} > expected {self.valid_extensions}"
                )
                return extract_data

            else:
                # Standalone PDF: you can call pdfplumber/ocr directly on file_path
                return extract_data
        finally:
            # After processing, clean up temp files:
            self.cleanup_zip_members(zip_data)

    def _get_extension(self, file_path: str) -> str:
        return os.path.splitext(file_path)[1].lower()

    def _is_valid_member(self, name: str) -> bool:
        return self._get_extension(name) in self.valid_extensions

    def get_zip_members(self, zip_path: str) -> ZipData:
        """
        Streams each valid member to a temp file (always).
        - CRC/encryption issues -> member marked bad; any partial temp file is removed.
        - On success, member.temp_path points to a seekable file for downstream tools.
        """
        zd = ZipData()
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if self.zip_password:
                    zf.setpassword(self.zip_password)

                entries = [i for i in zf.infolist() if not i.is_dir()]
                if not entries:
                    return zd

                candidates = [i for i in entries if self._is_valid_member(i.filename)]
                if not candidates:
                    return zd  # nothing to process; not an error

                for info in candidates:
                    member = ZipMember(
                        name=info.filename,
                        size=info.file_size,
                        compress_size=info.compress_size,
                        modified=datetime(*info.date_time),
                        crc=info.CRC
                    )

                    tmp_path = None
                    try:
                        # Read+write in chunks; reading entire stream triggers CRC verification.
                        with zf.open(info, "r") as fh:
                            suffix = self._get_extension(info.filename) or ".bin"
                            with tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir, suffix=suffix) as tmp:
                                tmp_path = tmp.name
                                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                                    tmp.write(chunk)

                        # If we reached here, full read succeeded -> CRC OK.
                        member.crc_ok = True
                        member.temp_path = tmp_path
                        zd.ok_members.append(member)
                        zd.ok_files.append(member.name)

                    except (zipfile.BadZipFile, RuntimeError, OSError, ValueError) as e:
                        member.error = str(e)
                        zd.bad_members.append(member)
                        zd.bad_files.append(member.name)

                        # Encrypted/wrong password is a common RuntimeError case
                        if "password" in str(e).lower() or "decrypt" in str(e).lower():
                            zd.errors.append(f"{member.name}: Encrypted entry (wrong/missing password).")
                        else:
                            zd.errors.append(f"{member.name}: {e}")

                        # Remove any partial temp file
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass

        except zipfile.BadZipFile:
            zd.errors.append("Not a ZIP file or file is severely corrupted.")
        except zipfile.LargeZipFile as e:
            zd.errors.append(f"ZIP too large or ZIP64 issue: {e}")
        except Exception as e:
            zd.errors.append(f"Unexpected error: {e}")

        zd.ok_files_count = len(zd.ok_files)
        zd.bad_files_count = len(zd.bad_files)
        return zd

    def cleanup_zip_members(self, zip_data: ZipData) -> None:
        """
        Delete all temp files created for members (ok and bad).
        Call after you're done with pdfplumber/OCR.
        """
        for m in zip_data.ok_members + zip_data.bad_members:
            if m.temp_path and os.path.exists(m.temp_path):
                try:
                    os.remove(m.temp_path)
                except Exception:
                    # Intentionally swallow; you may want to log this
                    pass

    def _profile_pdf_layout(self, path: Optional[str], original_name: Optional[str] = None) -> PdfLayoutProfile:
        """
        Classifies a PDF by page:
          - image_area_ratio (0..1)
          - text density (chars, words, text_area_ratio)
        And recommends: 'ocr' | 'pdfplumber' | 'hybrid'

        Relies on pdfplumber. Returns PdfLayoutProfile with per-page stats.
        """
        name = original_name or (os.path.basename(path) if path else None)
        if not path:
            return PdfLayoutProfile(file_name=name, readable=False, pages=0, error="no input provided")

        try:
            with pdfplumber.open(path) as pdf:
                meta = getattr(pdf, "metadata", {}) or {}
                n_pages = len(pdf.pages)
                profile = PdfLayoutProfile(file_name=name, readable=True, pages=n_pages, metadata=meta)

                for idx, pg in enumerate(pdf.pages):
                    W, H = float(pg.width), float(pg.height)
                    page_area = max(W * H, 1.0)

                    # ----- image area ratio -----
                    imgs = getattr(pg, "images", []) or []
                    img_area = 0.0
                    for im in imgs:
                        # pdfplumber images have x0, y0, x1, y1
                        x0, x1 = float(im.get("x0", 0)), float(im.get("x1", 0))
                        y0, y1 = float(im.get("y0", 0)), float(im.get("y1", 0))
                        w = max(x1 - x0, 0.0)
                        h = max(y1 - y0, 0.0)
                        img_area += (w * h)
                    image_area_ratio = min(img_area / page_area, 1.0)

                    # ----- text density -----
                    # glyph-level count
                    chars = getattr(pg, "chars", []) or []
                    text_chars = len(chars)

                    # word boxes (sum area to approximate text coverage)
                    try:
                        words = pg.extract_words() or []
                    except Exception:
                        words = []
                    word_area = 0.0
                    for wdict in words:
                        x0, x1 = float(wdict.get("x0", 0)), float(wdict.get("x1", 0))
                        top = float(wdict.get("top", 0))
                        bottom = float(wdict.get("bottom", top))
                        w = max(x1 - x0, 0.0)
                        h = max(bottom - top, 0.0)
                        word_area += (w * h)
                    text_area_ratio = min(word_area / page_area, 1.0)

                    has_images = image_area_ratio > 0.0
                    has_text = (text_chars >= CHAR_MIN_FOR_TEXT) or (text_area_ratio >= TEXT_AREA_MIN_RATIO)

                    image_dominant = image_area_ratio >= IMG_RATIO_SCANNED
                    text_dominant = (text_area_ratio >= TEXT_AREA_MIN_RATIO and not image_dominant)

                    profile.page_stats.append(
                        PdfPageStats(
                            index=idx,
                            width=W,
                            height=H,
                            image_area_ratio=image_area_ratio,
                            text_chars=text_chars,
                            text_words=len(words),
                            text_area_ratio=text_area_ratio,
                            has_text=has_text,
                            has_images=has_images,
                            image_dominant=image_dominant,
                            text_dominant=text_dominant,
                        )
                    )

                # ----- aggregates & recommendation -----
                img_dom = sum(1 for s in profile.page_stats if s.image_dominant)
                txt_dom = sum(1 for s in profile.page_stats if s.text_dominant)
                mixed = n_pages - (img_dom + txt_dom)

                profile.pages_image_dominant = img_dom
                profile.pages_text_dominant = txt_dom
                profile.pages_mixed = mixed

                # Recommend pipeline:
                # - If most pages are image-dominant -> OCR
                # - If most pages are text-dominant  -> pdfplumber
                # - Otherwise                        -> hybrid (OCR image-dominant pages; parse text pages)
                if n_pages > 0:
                    frac_img = img_dom / n_pages
                    frac_txt = txt_dom / n_pages

                    if frac_img >= 0.60:
                        profile.recommended = "ocr"
                        profile.rationale = f"{img_dom}/{n_pages} pages are image-dominant (≥{int(IMG_RATIO_SCANNED * 100)}% image area)."
                    elif frac_txt >= 0.90:
                        profile.recommended = "pdfplumber"
                        profile.rationale = f"{txt_dom}/{n_pages} pages have sufficient text density."
                    else:
                        profile.recommended = "hybrid"
                        profile.rationale = (
                            f"Mixed content: image-dominant={img_dom}, text-dominant={txt_dom}, mixed={mixed}."
                        )

                # Extra hint: if Producer indicates scanner, bias toward OCR if not already
                producer = (profile.metadata.get("Producer") or "").lower()
                if "scan" in producer and profile.recommended == "pdfplumber":
                    profile.recommended = "hybrid"
                    profile.rationale += " Producer suggests a scanner; using hybrid for safety."

                return profile

        except Exception as e:
            return PdfLayoutProfile(file_name=name, readable=False, pages=0, error=str(e))

# ---------- Example usage ----------
if __name__ == '__main__':
    api = TextExtractor()
    test_file = r"D:\Downloads\oglas.722031c2-2de4-446d-ac86-19ec0b8da90b.zip"
    api.extract(test_file)

    # zd = api.get_zip_members(test_file)
    #
    # # Example: try pdfplumber on OK members
    # # for m in zd.ok_members:
    # #     r, has_imgs, pages, err = api._probe_pdfplumber(path=m.temp_path)
    # #     print(m.name, r, has_imgs, pages, err)
    #
    # print(f"OK: {zd.ok_files_count}, BAD: {zd.bad_files_count}")
    # # After processing, clean up temp files:
    # api.cleanup_zip_members(zd)
