"""Tests for the mounted-input batch API without network or model weights."""
import csv
import http.client
import json
import tempfile
import threading
import unittest
import zipfile
from http.server import HTTPServer
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from dxaqc.api import RequestError, make_handler, predict_batch
from dxaqc.cli import main
from dxaqc.io import extract_if_archive
from dxaqc.model import StubModel
from dxaqc.pipeline import run
from dxaqc.report import COLUMNS


def make_dicom(path: Path) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "CR"
    ds.Rows, ds.Columns = 8, 300
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((8, 300), dtype=np.uint8).tobytes()
    ds.save_as(path, enforce_file_format=True)


class BatchApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.input_root = base / "input"
        self.output_root = base / "output"
        self.input_root.mkdir()
        make_dicom(self.input_root / "sample.dcm")
        self.model = StubModel()

    def test_batch_writes_organizer_format(self):
        response = predict_batch({"input": ".", "format": "csv"}, self.model,
                                 self.input_root, self.output_root)
        self.assertEqual((response["processed"], response["failed"]), (1, 0))
        self.assertEqual(response["rows"][0]["processing_status"], "Success")
        with Path(response["results_file"]).open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, COLUMNS)
            self.assertEqual(len(list(reader)), 1)
        self.assertTrue(Path(response["errors_file"]).exists())

    def test_zip_and_xlsx_input(self):
        archive = self.input_root / "batch.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.write(self.input_root / "sample.dcm", "Исследование/sample.dcm")
        response = predict_batch({"input": "batch.zip", "format": "xlsx"}, self.model,
                                 self.input_root, self.output_root)
        self.assertEqual(response["processed"], 1)
        self.assertTrue(Path(response["results_file"]).exists())

    def test_bad_paths_and_format_rejected(self):
        for payload in ({"input": "../outside"}, {"input": "/etc/passwd"},
                        {"input": ".", "format": "pdf"}, {"input": "sample.dcm"}):
            with self.subTest(payload=payload), self.assertRaises(RequestError):
                predict_batch(payload, self.model, self.input_root, self.output_root)

    def test_api_requires_all_weights(self):
        self.assertEqual(main(["serve", "--input-root", str(self.input_root),
                               "--output-root", str(self.output_root),
                               "--models", str(self.input_root / "missing")]), 2)

    def test_cli_predict_and_synchronization_hook(self):
        destination = self.output_root / "cli.csv"
        self.assertEqual(main(["predict", "--stub", "--input", str(self.input_root),
                               "--output", str(destination)]), 0)
        self.assertTrue(destination.exists())
        calls = []
        rows, _ = run(self.input_root, self.model, synchronize=lambda: calls.append(True))
        self.assertEqual(len(calls), len(rows))

    def test_zip_cannot_escape_extraction_directory(self):
        archive = self.input_root / "escape.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../outside.dcm", b"outside")
        workdir = Path(self.tmp.name) / "extract"
        extract_if_archive(archive, workdir)
        self.assertFalse((workdir / "outside.dcm").exists())

    def test_http_health_and_predict(self):
        server = HTTPServer(("127.0.0.1", 0), make_handler(
            self.model, self.input_root, self.output_root))
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        self.addCleanup(connection.close)
        connection.request("GET", "/health")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.load(response)["status"], "ok")
        connection.request("POST", "/predict", json.dumps({"input": "."}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.load(response)["processed"], 1)
        connection.request("POST", "/predict", json.dumps({"input": "../outside"}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        self.assertIn("error", json.load(response))


if __name__ == "__main__":
    unittest.main()
