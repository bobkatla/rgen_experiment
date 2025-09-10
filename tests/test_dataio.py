from __future__ import annotations
import shutil
import tempfile
import os
from rgen.dataio.cifar_export import export_cifar10_as_images

def test_cifar_export_structure_and_counts():
    tmp = tempfile.mkdtemp()
    try:
        summary = export_cifar10_as_images(tmp, splits=("train","test"), overwrite=False)
        # Expected CIFAR-10 counts
        assert summary["train"] == 50000
        assert summary["test"] == 10000

        # Check folder structure exists
        for split, expected in [("train", 50000), ("test", 10000)]:
            split_dir = os.path.join(tmp, "cifar10", split)
            assert os.path.isdir(split_dir)
            classes = os.listdir(split_dir)
            assert sorted(classes) == sorted(
                ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
            )
            # spot-check at least one file exists
            any_class_dir = os.path.join(split_dir, classes[0])
            files = os.listdir(any_class_dir)
            assert len(files) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
