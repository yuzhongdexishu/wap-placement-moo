$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $root "result\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$out = Join-Path $logDir "_services_console2.log"
$err = Join-Path $logDir "_services_console2.err.log"
$p = Start-Process -FilePath "python" -ArgumentList "run_all.py" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
Set-Content -Path (Join-Path $logDir "_services2_pid.txt") -Value $p.Id
Write-Output ("started services pid=" + $p.Id)
