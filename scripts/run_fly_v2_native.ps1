param([switch]$AllowLive)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$state = Join-Path $env:LOCALAPPDATA 'flytegral\fly-tweets\state'
$queue = Join-Path $state 'v2-native-x-queue.json'
$rows = Join-Path $state 'rows.json'
$owner = Join-Path $repo '.private\fly_tweets_owner_math_en.jsonl'
$profile = 'C:\tools\x-stats\x-browser-profile'
$log = Join-Path $state 'v2-native-x-run.log'
function RunOrFail([string]$label,[scriptblock]$body) {
  & $body
  if ($LASTEXITCODE -ne 0) { throw "$label failed with exit $LASTEXITCODE" }
}
& (Join-Path $PSScriptRoot 'with_fly_state_lock.ps1') -Body {
  New-Item -ItemType Directory -Force -Path $state | Out-Null
  try {
    if (-not (Test-Path $rows)) { throw "Fly runtime rows.json missing" }
    if (-not (Test-Path $owner)) { throw "Owner corpus missing: $owner" }
    if (-not (Test-Path (Join-Path $repo 'collector\node_modules\playwright'))) { throw "Install collector/npm dependencies first" }
    Set-Location $repo
    if (Test-Path $queue) {
      RunOrFail 'reconcile before scheduling' { & py -3.12 scripts\reconcile_native_x_rows.py --rows-json $rows --queue-json $queue }
    }
    RunOrFail 'prepare v2 native queue' { & py -3.12 scripts\queue_fly_v2_live.py --rows-json $rows --owner-corpus $owner --queue-json $queue }
    $mode = if ($AllowLive) { '--allow-live' } else { '--dry-run' }
    RunOrFail 'X native publish' { & node collector\x-native-publish.js --queue-json $queue --profile-dir $profile --expected-handle fly_topology $mode }
    RunOrFail 'reconcile after delivery' { & py -3.12 scripts\reconcile_native_x_rows.py --rows-json $rows --queue-json $queue }
    "$([DateTimeOffset]::UtcNow.ToString('o')) SUCCESS live=$AllowLive" | Out-File -FilePath $log -Append -Encoding utf8
  } catch {
    "$([DateTimeOffset]::UtcNow.ToString('o')) ERROR $($_.Exception.Message)" | Out-File -FilePath $log -Append -Encoding utf8
    throw
  }
}
