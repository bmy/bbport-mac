# A look at a running game from outside (read only): CPU of the process and its busiest threads,
# GPU engine load and VRAM, RAM, and the game's own frame statistics from its newest log.
param([int]$Seconds = 5)
$p = Get-Process bb-probe -ErrorAction SilentlyContinue
if (-not $p) { Write-Output 'watch: no game running'; exit 1 }
$cores = [Environment]::ProcessorCount

# Per-thread CPU over the window (TotalProcessorTime deltas).
$before = @{}
foreach ($t in $p.Threads) { try { $before[$t.Id] = $t.TotalProcessorTime.TotalMilliseconds } catch {} }
$cpu0 = $p.TotalProcessorTime.TotalMilliseconds
$gpuSamples = Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples $Seconds -ErrorAction SilentlyContinue
$p.Refresh()
$cpu1 = $p.TotalProcessorTime.TotalMilliseconds
$wall = $Seconds * 1000.0
$threads = foreach ($t in $p.Threads) {
    try {
        $ms = $t.TotalProcessorTime.TotalMilliseconds - ($before[$t.Id] | ForEach-Object { $_ })
        [pscustomobject]@{ Id = $t.Id; Pct = [math]::Round(100.0 * $ms / $wall, 1); State = $t.ThreadState; Wait = $t.WaitReason }
    } catch {}
}
"process CPU: {0:N0}% of one core ({1:N1}% of all {2} threads); RAM working set {3:N1} GB, private {4:N1} GB" -f (100.0 * ($cpu1 - $cpu0) / $wall), (100.0 * ($cpu1 - $cpu0) / $wall / $cores), $cores, ($p.WorkingSet64 / 1GB), ($p.PrivateMemorySize64 / 1GB)
"threads: {0}; busiest (% of one core):" -f $p.Threads.Count
$threads | Sort-Object Pct -Descending | Select-Object -First 10 | ForEach-Object { "  tid {0,6}: {1,5:N1}%  {2}" -f $_.Id, $_.Pct, $_.State }

# GPU: engine utilization summed per engine type for this process, averaged over the samples.
if ($gpuSamples) {
    $byType = @{}
    foreach ($set in $gpuSamples.CounterSamples | Where-Object { $_.InstanceName -match "pid_$($p.Id)_" }) {
        $type = if ($set.InstanceName -match 'engtype_(.+)$') { $matches[1] } else { '?' }
        $byType[$type] += $set.CookedValue / $Seconds
    }
    "GPU (this process, avg): " + (($byType.GetEnumerator() | Sort-Object Value -Descending | Where-Object { $_.Value -ge 0.5 } | ForEach-Object { "{0} {1:N0}%" -f $_.Key, $_.Value }) -join ', ')
}
$mem = Get-Counter -Counter "\GPU Process Memory(pid_$($p.Id)*)\Dedicated Usage", "\GPU Process Memory(pid_$($p.Id)*)\Shared Usage" -ErrorAction SilentlyContinue
if ($mem) {
    $ded = ($mem.CounterSamples | Where-Object { $_.Path -match 'dedicated' } | Measure-Object CookedValue -Sum).Sum
    $sha = ($mem.CounterSamples | Where-Object { $_.Path -match 'shared' } | Measure-Object CookedValue -Sum).Sum
    "VRAM: dedicated {0:N2} GB, shared (system RAM used by the GPU) {1:N2} GB" -f ($ded / 1GB), ($sha / 1GB)
}
$os = Get-CimInstance Win32_OperatingSystem
"system RAM: {0:N1} GB free of {1:N1} GB" -f ($os.FreePhysicalMemory / 1MB), ($os.TotalVisibleMemorySize / 1MB)

# The game's view: its last frame statistics.
$log = Get-ChildItem 'E:\bloodborne\bbport-windows\logs\*.log' | Sort-Object LastWriteTime | Select-Object -Last 1
"log: $($log.Name)"
Select-String -Path $log.FullName -Pattern '^Frame stats:|^Draw pipe:|^Frame pacing:|^Guest waits per frame|^Stall:' | Select-Object -Last 6 |
    ForEach-Object { '  ' + $_.Line.Substring(0, [math]::Min(330, $_.Line.Length)) }
