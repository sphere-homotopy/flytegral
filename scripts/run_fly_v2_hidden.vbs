Option Explicit
Dim shell, fso, script, command, i, argument
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
script = fso.BuildPath(fso.GetParentFolderName(WScript.ScriptFullName), "run_fly_v2_native.ps1")
If Not fso.FileExists(script) Then WScript.Quit 2
command = """" & shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\WindowsPowerShell\v1.0\powershell.exe"" -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & script & """"
For i = 0 To WScript.Arguments.Count - 1
    argument = WScript.Arguments(i)
    If argument <> "-AllowLive" Then WScript.Quit 2
    command = command & " " & argument
Next
' Hide the console before creation; wait so Scheduler retains overlap protection
' and receives the real runner exit code. Child console programs inherit it.
WScript.Quit shell.Run(command, 0, True)
