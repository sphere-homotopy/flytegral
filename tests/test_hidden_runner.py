from pathlib import Path
import shutil
import subprocess
import sys
import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / 'scripts' / 'run_fly_v2_hidden.vbs'

@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Script Host integration')
@pytest.mark.parametrize('live,code', [(False, 0), (True, 7)])
def test_hidden_runner_preserves_arguments_completion_and_exit(tmp_path, live, code):
    folder = tmp_path / 'runner with spaces'
    folder.mkdir()
    shutil.copyfile(LAUNCHER, folder / LAUNCHER.name)
    marker = folder / 'completed.txt'
    (folder / 'run_fly_v2_native.ps1').write_text(
        "param([switch]$AllowLive)\nStart-Sleep -Milliseconds 300\n"
        "[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'completed.txt'), [string]$AllowLive)\n"
        f'exit {code}\n', encoding='utf-8')
    result = subprocess.run(['wscript.exe', '//B', '//Nologo', str(folder / LAUNCHER.name)]
                            + (['-AllowLive'] if live else []), timeout=20)
    assert result.returncode == code
    assert marker.read_text() == str(live)
