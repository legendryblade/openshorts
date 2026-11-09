# OpenShorts server watchdog: keeps the UI alive and makes every crash visible.
#
# Why this exists: uvicorn has died twice after job completion (oct-2026) with
# no evidence captured - Start-Process -WindowStyle Hidden throws the server's
# output away. This wrapper (a) redirects ALL server output to log files so the
# next death is diagnosable, and (b) restarts the server within 15s of it
# going away, logging each restart.
#
# To run the app from a different directory, set OPENSHORTS_APP_DIR first:
#   $env:OPENSHORTS_APP_DIR = 'D:\openshorts'

$ErrorActionPreference = 'Stop'
$appDir = if ($env:OPENSHORTS_APP_DIR) { $env:OPENSHORTS_APP_DIR } else { 'D:\openshorts' }
if (-not (Test-Path $appDir)) {
    Write-Error "App directory not found: $appDir (set OPENSHORTS_APP_DIR)"
    exit 1
}
$python = Join-Path $appDir '.venv311\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "Python not found: $python - the web UI needs the repo venv. Use portable\main.exe for a no-Python install."
    exit 1
}
$watchLog = Join-Path $appDir 'watchdog.log'

function Now { Get-Date -Format 'yyyy-MM-dd HH:mm:ss' }

function Test-Up {
    try {
        $r = Invoke-WebRequest -Uri 'http://localhost:8001/health' -TimeoutSec 5 -UseBasicParsing
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Start-Server {
    # Rotate logs before each start: -RedirectStandardOutput TRUNCATES, which
    # is how the evidence for a server death (or an unexpected job purge) got
    # lost. Keep the last 5 runs.
    foreach ($name in 'server.out.log', 'server.err.log') {
        $path = Join-Path $appDir $name
        if (Test-Path $path) {
            $stamp = (Get-Item $path).LastWriteTime.ToString('yyyyMMdd-HHmmss')
            Move-Item $path "$path.$stamp" -Force
            Get-ChildItem "$path.*" | Sort-Object Name -Descending |
                Select-Object -Skip 5 | Remove-Item -Force
        }
    }
    # Server output goes to log files: the crash evidence we never had.
    $p = Start-Process -FilePath $python `
        -ArgumentList '-m','uvicorn','app:app','--host','0.0.0.0','--port','8001','--timeout-keep-alive','65' `
        -WorkingDirectory $appDir `
        -RedirectStandardOutput (Join-Path $appDir 'server.out.log') `
        -RedirectStandardError (Join-Path $appDir 'server.err.log') `
        -WindowStyle Hidden -PassThru
    Add-Content $watchLog "$(Now) server started (pid $($p.Id)) from $appDir"
    return $p
}


# Tunnel watch: quick-tunnel URLs rotate on restart (read the fresh one from
# the tunnel log). If cloudflared dies, bring it back so the public link
# only breaks once, not forever.
$cfExe = "$env:USERPROFILE\AppData\Local\Microsoft\WinGet\Links\cloudflared.exe"
$cfLog = Join-Path $appDir 'cloudflared.log'
function Test-TunnelProcess {
    return [bool](Get-Process -Name cloudflared -ErrorAction SilentlyContinue)
}
function Start-Tunnel {
    if (Test-Path $cfLog) {
        $stamp = (Get-Item $cfLog).LastWriteTime.ToString('yyyyMMdd-HHmmss')
        Move-Item $cfLog "$cfLog.$stamp" -Force
        Get-ChildItem "$cfLog.*" | Sort-Object Name -Descending | Select-Object -Skip 5 | Remove-Item -Force
    }
    Start-Process -FilePath $cfExe `
        -ArgumentList 'tunnel','--protocol','http2','--url','http://localhost:8001',"--logfile",$cfLog `
        -WindowStyle Hidden
    # Write the routable URL to tunnel-url.txt so START.bat and the user
    # always see the CURRENT tunnel, not a stale one.
    for ($i = 0; $i -lt 12; $i++) {
        Start-Sleep -Seconds 5
        $u = (Get-Content $cfLog -ErrorAction SilentlyContinue |
            Select-String -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" |
            Select-Object -Last 1 -ExpandProperty Matches).Value
        if ($u) {
            try {
                $r = Invoke-WebRequest "$u/health" -TimeoutSec 8 -UseBasicParsing
                if ($r.StatusCode -eq 200) {
                    Set-Content (Join-Path $appDir 'tunnel-url.txt') $u
                    Add-Content $watchLog "$(Now) tunnel URL recorded: $u"
                    break
                }
            } catch { }
        }
    }
    Add-Content $watchLog "$(Now) cloudflared restarted (URL in $(Join-Path $appDir 'tunnel-url.txt'))"
}

Add-Content $watchLog "$(Now) tunnel watch armed"
if (-not (Test-TunnelProcess)) { Start-Tunnel }

Add-Content $watchLog "$(Now) watchdog up"
if (-not (Test-Up)) { Start-Server | Out-Null }

while ($true) {
    Start-Sleep -Seconds 15
    if (-not (Test-TunnelProcess)) {
    Add-Content $watchLog "$(Now) tunnel process gone - restarting cloudflared"
    Start-Tunnel
  }
  if (-not (Test-Up)) {
        # Kill any half-dead listener first so the port is free for the restart.
        $lines = netstat -ano | Select-String ':8001.*LISTENING'
        foreach ($l in $lines) {
            $procId = ($l -split '\s+')[-1]
            if ($procId -match '^\d+$') {
                try { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue } catch {}
            }
        }
        Add-Content $watchLog "$(Now) health FAILED - restarting server"
        Start-Server | Out-Null
        Start-Sleep -Seconds 20
    }
}
