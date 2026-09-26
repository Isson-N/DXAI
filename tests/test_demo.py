"""Synthetic demo inputs remain valid DICOM and contain no patient data."""
import tempfile
import unittest
from pathlib import Path

import pydicom

from tools.make_demo_dicom import write_dicom


class DemoDicomTest(unittest.TestCase):
    def test_synthetic_images_are_decodable(self):
        with tempfile.TemporaryDirectory() as directory:
            for region, columns in (("spine", 300), ("hip", 280)):
                with self.subTest(region=region):
                    path = Path(directory) / f"{region}.dcm"
                    write_dicom(path, region)
                    dataset = pydicom.dcmread(path)
                    self.assertEqual(dataset.Columns, columns)
                    self.assertEqual(dataset.pixel_array.shape, (300, columns))
                    self.assertEqual(dataset.PatientID, "SYNTHETIC")
                    self.assertTrue(str(dataset.StudyInstanceUID).startswith("2.25."))


if __name__ == "__main__":
    unittest.main()
