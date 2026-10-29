# Stall blocks of a game log, condensed: frame time, the layer's trap time, the write trap
# statistics (all threads), VirtualProtect calls, the GPU thread's user/kernel time.
param([string]$Log, [int]$Max = 80)
$c = Get-Content $Log -Encoding utf8
$out = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $c.Count; $i++) {
    if ($c[$i] -notmatch '^Stall: ([\d.]+) ms') { continue }
    $ms = $matches[1]
    $blk = $c[$i..([math]::Min($i + 8, $c.Count - 1))]
    $line = "$ms ms"
    foreach ($b in $blk) {
        if ($b -match 'traps set/cleared ([\d.]+) \((\d+) mprotect\), to VRAM ([\d.]+)') { $line += " | layer traps $($matches[1]) ms/$($matches[2]), to VRAM $($matches[3])" }
        elseif ($b -match 'write traps \(all threads\): (\d+) calls, lock wait ([\d.]+) ms, protecting ([\d.]+) ms, ([\d.]+) MiB') { $line += " | trap calls $($matches[1]) wait $($matches[2]) prot $($matches[3]) ms $($matches[4]) MiB" }
        elseif ($b -match 'Windows: (\d+) VirtualProtect') { $line += " | VP $($matches[1])" }
        elseif ($b -match 'user ([\d.]+) ms, kernel ([\d.]+) ms') { $line += " | user $($matches[1]) kern $($matches[2])" }
        elseif ($b -match '(\d+) images registered') { $line += " | img $($matches[1])" }
    }
    $out.Add($line)
}
"stalls: $($out.Count)"
$out | Select-Object -First $Max
