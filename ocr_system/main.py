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
class PdfProbeResult:
    readable: bool
    has_images: bool
    pages: int
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    encrypted: Optional[bool] = None
    has_text: Optional[bool] = None


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
        extract_data = ExtractData()

        if not os.path.exists(file_path):
            extract_data.error.append(f"could not find file on path {file_path}")
            return extract_data

        extension = self._get_extension(file_path)

        if extension == '.zip':
            zip_data = self.get_zip_members(file_path)
            # Example: act on ok members (call pdfplumber or OCR)
            # for m in zip_data.ok_members:
            #     readable, has_images, pages, err = self._probe_pdfplumber(path=m.temp_path)
            #     ...
            return extract_data

        elif not self._is_valid_member(file_path):
            extract_data.error.append(
                f"file doesn't have valid extension. Got {extension} > expected {self.valid_extensions}"
            )
            return extract_data

        else:
            # Standalone PDF: you can call pdfplumber/ocr directly on file_path
            return extract_data

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

    # ---- Optional: probe with pdfplumber on a path ----
    def _probe_pdfplumber(self, path: Optional[str] = None):
        """
        Returns (readable, has_images, pages, error_msg)
        """
        if not path:
            return False, False, 0, "no input provided"

        try:
            with pdfplumber.open(path) as pdf:
                pages = len(pdf.pages)
                has_images = any(getattr(pg, "images", []) for pg in pdf.pages)
                return True, has_images, pages, ""
        except Exception as e:
            return False, False, 0, str(e)


# ---------- Example usage ----------
if __name__ == '__main__':
    api = TextExtractor()
    test_file = r"D:\Downloads\oglas.0c02154b-243a-403c-8c00-3467bfe8bb64.zip"
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
    # # api.cleanup_zip_members(zd)
