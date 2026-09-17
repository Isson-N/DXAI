from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CR_STORAGE = "1.2.840.10008.5.1.4.1.1.1"


def make_dicom(path: Path, rows: int, cols: int, photometric: str = "MONOCHROME2",
               study_uid: str | None = None, pixels: np.ndarray | None = None) -> Path:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CR_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CR_STORAGE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.Modality = "CR"
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    arr = pixels if pixels is not None else (np.arange(rows * cols) % 256).astype(np.uint8).reshape(rows, cols)
    ds.PixelData = arr.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def study_dir(tmp_path: Path) -> Path:
    """Папка с типичными и проблемными файлами."""
    root = tmp_path / "in"
    uid = generate_uid()
    make_dicom(root / "study1" / "spine.dcm", 290, 300, study_uid=uid)
    make_dicom(root / "study1" / "hip.dcm", 260, 280, study_uid=uid)
    # копия того же изображения с другим SOPInstanceUID
    make_dicom(root / "study1" / "spine_copy.dcm", 290, 300, study_uid=uid)
    make_dicom(root / "study2" / "inverted.dcm", 260, 280, photometric="MONOCHROME1")
    (root / "study2" / "broken.dcm").write_bytes(b"\x00" * 128 + b"DICM" + b"garbage")
    (root / "study2" / "notes.txt").write_text("не DICOM")
    no_pixels = Dataset()
    no_pixels.file_meta = FileMetaDataset()
    no_pixels.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.1.3.10"
    no_pixels.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    no_pixels.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    no_pixels.SOPClassUID = "1.2.840.10008.1.3.10"
    no_pixels.SOPInstanceUID = no_pixels.file_meta.MediaStorageSOPInstanceUID
    no_pixels.save_as(root / "DICOMDIR.dcm", enforce_file_format=True)
    return root
