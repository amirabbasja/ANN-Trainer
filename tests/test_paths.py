"""Application files remain independent of the shell's working directory."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from LRAEP_trainer import paths


class ApplicationPathTests(unittest.TestCase):
    def test_shared_checkout_config_is_found_independently_of_current_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "trainer"
            source = project / "src" / "LRAEP_trainer" / "paths.py"
            source.parent.mkdir(parents=True)
            source.touch()
            (project / "ann_trainer.py").touch()
            with patch.object(paths, "__file__", str(source)), \
                    patch.object(Path, "cwd", return_value=Path(temporary) / "elsewhere"):
                self.assertEqual(paths.default_config_path(), project / "config.json")

    def test_installed_package_uses_the_current_directory_shared_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "installed" / "site-packages" / "LRAEP_trainer" / "paths.py"
            source.parent.mkdir(parents=True)
            source.touch()
            directory = Path(temporary) / "working"
            with patch.object(paths, "__file__", str(source)), patch.object(Path, "cwd", return_value=directory):
                self.assertEqual(paths.default_config_path(), directory / "config.json")

    def test_helpers_are_located_beside_the_installed_package(self):
        expected = Path(paths.__file__).resolve().parents[1] / "LRAEP_helpers"
        self.assertEqual(paths.DEFAULT_SOURCE, expected)

    def test_default_generated_paths_use_the_home_directory(self):
        with patch.dict("os.environ", {}, clear=True), \
                patch.object(Path, "home", return_value=Path("/tmp/lraep-home")):
            self.assertEqual(paths.cache_dir(), Path("/tmp/lraep-home/.cache/LRAEP"))
            self.assertEqual(paths.artifacts_dir(), Path("/tmp/lraep-home/.local/share/LRAEP/artifacts"))

    def test_xdg_directories_are_used_without_app_overrides(self):
        with patch.dict("os.environ", {
            "XDG_CACHE_HOME": "/tmp/lraep-xdg-cache",
            "XDG_DATA_HOME": "/tmp/lraep-xdg-data",
        }, clear=True):
            self.assertEqual(paths.cache_dir(), Path("/tmp/lraep-xdg-cache/LRAEP"))
            self.assertEqual(paths.artifacts_dir(), Path("/tmp/lraep-xdg-data/LRAEP/artifacts"))

    def test_app_overrides_take_priority_and_expand_home(self):
        with patch.dict("os.environ", {
            "HOME": "/tmp/lraep-home",
            "LRAEP_CACHE_DIR": "~/kernel-cache",
            "LRAEP_DATA_DIR": "~/mining-data",
            "XDG_CACHE_HOME": "/tmp/unused-cache",
            "XDG_DATA_HOME": "/tmp/unused-data",
        }, clear=True):
            self.assertEqual(paths.cache_dir(), Path("/tmp/lraep-home/kernel-cache"))
            self.assertEqual(paths.artifacts_dir(), Path("/tmp/lraep-home/mining-data/artifacts"))

    def test_empty_environment_values_use_the_default_locations(self):
        with patch.dict("os.environ", {
            "LRAEP_CACHE_DIR": "", "LRAEP_DATA_DIR": "",
            "XDG_CACHE_HOME": "", "XDG_DATA_HOME": "",
        }, clear=True), patch.object(Path, "home", return_value=Path("/tmp/lraep-home")):
            self.assertEqual(paths.cache_dir(), Path("/tmp/lraep-home/.cache/LRAEP"))
            self.assertEqual(paths.artifacts_dir(), Path("/tmp/lraep-home/.local/share/LRAEP/artifacts"))


if __name__ == "__main__":
    unittest.main()
