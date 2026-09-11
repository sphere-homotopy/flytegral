from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WindowsBootstrapTests(unittest.TestCase):
    def test_powershell_bootstrap_is_documented_and_uses_windows_paths(self):
        script_path = ROOT / 'scripts' / 'bootstrap-malecns.ps1'
        self.assertTrue(script_path.exists(), 'PowerShell bootstrap script is missing')
        script = script_path.read_text(encoding='utf-8')
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        package = (ROOT / 'package.json').read_text(encoding='utf-8')

        self.assertIn('71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33', script)
        self.assertIn('py -3.11', script)
        self.assertIn('Scripts\\python.exe', script)
        self.assertIn('.\\scripts\\bootstrap-malecns.ps1', readme)
        self.assertIn('.\\.venv-malecns\\Scripts\\python.exe -m brain_runtime.train_readout', readme)
        self.assertIn('.\\.venv-malecns\\Scripts\\python.exe -m brain_runtime.server', readme)
        self.assertIn('brain_runtime.test_windows_bootstrap', package)


if __name__ == '__main__':
    unittest.main()
