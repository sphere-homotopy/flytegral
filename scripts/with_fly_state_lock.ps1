param(
  [Parameter(Mandatory = $true)][scriptblock]$Body,
  [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = 'Stop'
if ($TimeoutSeconds -lt 0) { throw 'TimeoutSeconds must be nonnegative.' }
$mutex = [System.Threading.Mutex]::new($false, 'Local\FlyTweets-DurableState')
$acquired = $false
try {
  try {
    $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds($TimeoutSeconds))
  } catch [System.Threading.AbandonedMutexException] {
    # Windows grants ownership to the waiter when the previous owner exits.
    $acquired = $true
  }
  if (-not $acquired) {
    throw "Timed out waiting for Fly Tweets durable state after $TimeoutSeconds seconds."
  }
  # Preserve the body's output and LASTEXITCODE. Callers check native failures
  # inside Body so no later command can conceal a failed transaction.
  & $Body
} finally {
  try {
    if ($acquired) { $mutex.ReleaseMutex() }
  } finally {
    $mutex.Dispose()
  }
}
