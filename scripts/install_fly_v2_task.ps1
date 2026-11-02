$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'run_fly_v2_native.ps1'
if (-not (Test-Path $script)) { throw "Fly native runner missing" }
$name = 'FlyTweets-V2-Native-X'
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$script`" -AllowLive"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 364)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 8) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
$task = Get-ScheduledTask -TaskName $name
$info = Get-ScheduledTaskInfo -TaskName $name
Write-Output "TASK=$name"
Write-Output "STATE=$($task.State)"
Write-Output "NEXT=$($info.NextRunTime.ToString('o'))"
Write-Output "USER=$user"
Write-Output 'AUTOMATION=WINDOWS_INTERACTIVE_SESSION'
