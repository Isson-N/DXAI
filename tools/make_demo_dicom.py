"""Create two synthetic DICOM images for an API smoke test, without patient data."""
from __future__ import annotations

import argparse
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ComputedRadiographyImageStorage, ExplicitVRLittleEndian


def uid(name: str) -> str:
    return f"2.25.{uuid5(NAMESPACE_DNS, 'dxaqc-demo-' + name).int}"


def synthetic_pixels(region: str) -> np.ndarray:
    height, width = 300, 300 if region == "spine" else 280
    y, x = np.indices((height, width))
    image = np.full((height, width), 8, dtype=np.uint8)
    if region == "spine":
        for center in (55, 100, 145, 190, 235):
            body = ((x - 150) / 37) ** 2 + ((y - center) / 17) ** 2 < 1
            image[body] = 180
        image[((x - 73) / 42) ** 2 + ((y - 270) / 22) ** 2 < 1] = 120
        image[((x - 227) / 42) ** 2 + ((y - 270) / 22) ** 2 < 1] = 120
    else:
        head = ((x - 93) / 32) ** 2 + ((y - 92) / 32) ** 2 < 1
        shaft = (abs(x - (154 + 0.11 * (y - 145))) < 20) & (y > 137)
        neck = (abs(y - (0.7 * x + 25)) < 15) & (x > 95) & (x < 158)
        image[head | shaft | neck] = 180
    return image


def write_dicom(path: Path, region: str) -> None:
    pixels = synthetic_pixels(region)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    meta.MediaStorageSOPInstanceUID = uid(region + "-image")
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = uid(region + "-study")
    ds.SeriesInstanceUID = uid(region + "-series")
    ds.PatientName = "SYNTHETIC^DEMO"
    ds.PatientID = "SYNTHETIC"
    ds.Modality = "CR"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for region in ("spine", "hip"):
        write_dicom(args.output / f"synthetic_{region}.dcm", region)


if __name__ == "__main__":
    main()
