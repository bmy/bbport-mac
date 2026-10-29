# GPU profile blocks of a log (BB_GPU_PROFILE=1), averaged: ms per frame by category (the game's
# passes, the port's upscaler / frame generation / AO / copies, idle gaps) and the heaviest labels.
param([string]$Log, [int]$Skip = 2, [int]$Top = 15)
$c = Get-Content $Log -Encoding utf8
$blocks = New-Object System.Collections.Generic.List[object]
for ($i = 0; $i -lt $c.Count; $i++) {
    if ($c[$i] -notmatch '^GPU profile: ([\d.]+) ms/frame') { continue }
    $blockTotal = [double]$matches[1]
    $labels = @{}
    for ($j = $i + 1; $j -lt $c.Count -and $c[$j] -match '^\s+([\d.]+) ms/frame\s+[\d.]+/frame\s+(.*)$'; $j++) {
        $labels[$matches[2].Trim()] = [double]$matches[1]
    }
    $blocks.Add([pscustomobject]@{ Total = $blockTotal; Labels = $labels })
}
$use = $blocks | Select-Object -Skip $Skip
"profile blocks: $($blocks.Count), averaged: $(@($use).Count) (first $Skip skipped: title, loading)"
function Category([string]$label) {
    if ($label -match '^upscaler') { return 'port: upscaler' }
    if ($label -match '^frame generation') { return 'port: frame generation' }
    if ($label -match '(?i)vbao') { return 'port: VBAO' }
    if ($label -match '(?i)smaa|fxaa|sharpen|rcas') { return 'port: AA/sharpen' }
    if ($label -match 'GPU idle|presenter') { return 'GPU idle (gaps)' }
    if ($label -match '^dispatch cs 00000000fefebf9f') { return 'game: copy shader (translated)' }
    if ($label -match '^dispatch') { return 'game: compute dispatches' }
    if ($label -match '^pass') { return 'game: render passes' }
    return 'other: ' + $label.Substring(0, [math]::Min(40, $label.Length))
}
$sum = @{}; $labelSum = @{}; $total = 0.0
foreach ($b in $use) {
    $total += $b.Total
    foreach ($kv in $b.Labels.GetEnumerator()) {
        $sum[(Category $kv.Key)] += $kv.Value
        $labelSum[$kv.Key] += $kv.Value
    }
}
$n = [math]::Max(1, @($use).Count)
"GPU time per frame (avg): {0:N2} ms; listed labels {1:N2} ms (the rest: labels below the top list)" -f ($total / $n), (($sum.Values | Measure-Object -Sum).Sum / $n)
$sum.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object { "  {0,-34} {1,6:N2} ms  {2,5:N1}%" -f $_.Key, ($_.Value / $n), (100 * $_.Value / $total) }
"heaviest labels:"
$labelSum.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First $Top | ForEach-Object { "  {0,6:N2} ms  {1}" -f ($_.Value / $n), $_.Key.Substring(0, [math]::Min(150, $_.Key.Length)) }
