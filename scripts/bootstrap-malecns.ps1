$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Vendor = Join-Path $Root '.vendor\doomfly'
$Venv = Join-Path $Root '.venv-malecns'
$Python = Join-Path $Venv 'Scripts\python.exe'
$UpstreamUrl = 'https://github.com/nftechie/doomfly.git'
$UpstreamCommit = '71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33'

function Assert-LastExitCode([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is required. Install Git for Windows and reopen PowerShell.'
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python launcher (py.exe) is required. Install 64-bit Python 3.11 and reopen PowerShell.'
}

& py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11); print(sys.executable)"
Assert-LastExitCode 'Python 3.11 check'

$VendorParent = Split-Path $Vendor -Parent
New-Item -ItemType Directory -Force -Path $VendorParent | Out-Null

if (-not (Test-Path (Join-Path $Vendor '.git'))) {
    & git clone $UpstreamUrl $Vendor
    Assert-LastExitCode 'DOOMFLY clone'
}

& git -C $Vendor fetch origin $UpstreamCommit
Assert-LastExitCode 'DOOMFLY fetch'
& git -C $Vendor checkout --detach $UpstreamCommit
Assert-LastExitCode 'DOOMFLY checkout'

if (-not (Test-Path $Python)) {
    & py -3.11 -m venv $Venv
    Assert-LastExitCode 'virtual environment creation'
}

& $Python -m pip install --upgrade pip
Assert-LastExitCode 'pip upgrade'
& $Python -m pip install -r (Join-Path $Vendor 'requirements-neural.txt') -r (Join-Path $Vendor 'doom\requirements.txt') --build-constraint (Join-Path $Vendor 'neural-build-constraints.txt')
Assert-LastExitCode 'MaleCNS dependency installation'

$DownloadScript = @'
from pathlib import Path
import hashlib
import json
import urllib.request

name = 'malecns_v1'
registry = json.loads(Path('doom/datasets.json').read_text())['datasets'][name]
locked = json.loads(Path(f'data-provenance/{name}/source.lock.json').read_text())
root = Path('connectome_data') / name
root.mkdir(parents=True, exist_ok=True)

for filename, url in registry['files'].items():
    target = root / filename
    if not target.exists():
        print(f'downloading {filename} ...', flush=True)
        partial = target.with_suffix(target.suffix + '.download')
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    with target.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != locked[filename]['sha256']:
        raise RuntimeError(f'Source checksum mismatch: {filename}')

(root / 'source.lock.json').write_text(json.dumps(locked, indent=2) + '\n')
'@

Push-Location $Vendor
try {
    $DownloadScript | & $Python -
    Assert-LastExitCode 'MaleCNS data download/checksum verification'
    & $Python -m doom.connectome malecns_v1
    Assert-LastExitCode 'MaleCNS connectome import'
    & $Python -m doom.prepare
    Assert-LastExitCode 'MaleCNS graph preparation'
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host 'MaleCNS graph prepared at:'
Write-Host "  $(Join-Path $Vendor 'outputs\doom\malecns_v1\graph.npz')"
Write-Host 'Next:'
Write-Host '  .\.venv-malecns\Scripts\python.exe -m brain_runtime.train_readout --train 128 --test 64'
