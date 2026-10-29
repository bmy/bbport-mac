# Standing test run on Windows: a copy of the saves, scripted pad (BB_PAD_FILE), enter the game
# (cross at the title, cross to continue), stand, quit. Extra BB_* as "NAME=value" arguments.
# Only a game this script started is closed; with another bb-probe running it does nothing.
param([int]$TitleWait = 40, [int]$Stand = 90, [string[]]$Env = @(), [string]$Shot = '', [int]$ShotAt = 40)
$ErrorActionPreference = 'Stop'
$pkg = 'E:\bloodborne\bbport-windows'
$user = 'E:\bloodborne\test_user'
$pad = 'E:\bloodborne\test_pad.txt'
if (Get-Process bb-probe -ErrorAction SilentlyContinue) { Write-Output 'standrun: a game is running - left alone'; exit 1 }
if (-not (Test-Path $user)) { Copy-Item 'E:\bloodborne\saves' $user -Recurse }
Set-Content $pad '' -NoNewline
# The launcher's settings (bbport-run.exe --play) with the saves' copy: its own APPDATA, so the
# player's settings.json (user_dir = their saves) stays as it is.
$appdata = 'E:\bloodborne\test_appdata'
New-Item -ItemType Directory -Force "$appdata\bbport-launcher" | Out-Null
$settings = Get-Content "$env:APPDATA\bbport-launcher\settings.json" -Raw -Encoding utf8 | ConvertFrom-Json
$settings.user_dir = $user
$settings | ConvertTo-Json | Set-Content "$appdata\bbport-launcher\settings.json" -Encoding utf8
$env:APPDATA = $appdata
$env:BB_PROBE = 'E:\bloodborne\bbport\out\win-native\bb-probe.exe'
$env:BB_USER_DIR = $user
$env:BB_PAD_FILE = $pad
$env:BB_SKIP_NETWORK_CHOICE = '1'
$env:SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS = '1'
foreach ($kv in $Env) { $k, $v = $kv -split '=', 2; Set-Item "env:$k" $v; Write-Output "env $k=$v" }
$before = Get-ChildItem "$pkg\logs\*.log" | ForEach-Object Name
$run = Start-Process "$pkg\bbport-run.exe" -ArgumentList '--play' -WorkingDirectory $pkg -WindowStyle Hidden -PassThru
$probe = $null
for ($i = 0; $i -lt 120 -and -not $probe; ++$i) { Start-Sleep 1; $probe = Get-Process bb-probe -ErrorAction SilentlyContinue }
if (-not $probe) { Write-Output 'standrun: no bb-probe'; exit 1 }
Start-Sleep $TitleWait
function Press($what) { Set-Content $pad $what -NoNewline; Start-Sleep -Milliseconds 300; Set-Content $pad '' -NoNewline }
Press 'cross' # the port skips the online choice: one cross is Continue (tools/restart.sh)
if ($Shot) {
    # A screenshot $ShotAt s into the stand (the HUD, the scene), the rest of the stand after it.
    Start-Sleep ([math]::Min($ShotAt, $Stand))
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    $s = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bmp = New-Object System.Drawing.Bitmap $s.Width, $s.Height
    [System.Drawing.Graphics]::FromImage($bmp).CopyFromScreen($s.Location, [System.Drawing.Point]::Empty, $s.Size)
    $bmp.Save($Shot)
    Start-Sleep ([math]::Max(0, $Stand - $ShotAt))
} else {
    Start-Sleep $Stand
}
Stop-Process -Id $probe.Id -ErrorAction SilentlyContinue
Start-Sleep 3
$log = Get-ChildItem "$pkg\logs\*.log" | Where-Object { $before -notcontains $_.Name } | Sort-Object LastWriteTime | Select-Object -Last 1
Write-Output "standrun: log $($log.FullName)"
