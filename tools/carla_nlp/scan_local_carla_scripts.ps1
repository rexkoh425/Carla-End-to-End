# Scan already-cloned repos for CARLA entrypoint scripts (no network).
# Looks for Python files containing: import carla + __main__ + argparse.
#
# Usage examples:
#   powershell -File .\tools\carla_nlp\scan_local_carla_scripts.ps1 `
#     -Root "D:\Datasets\carla_python_script_dataset"
#
#   powershell -File .\tools\carla_nlp\scan_local_carla_scripts.ps1 `
#     -Root "D:\Datasets\carla_python_script_dataset" `
#     -RepoListPath .\tools\carla_nlp\carla_repos_p1_p6.txt

[CmdletBinding()]
param(
    [string]$Root = "D:\Datasets\carla_python_script_dataset",
    [string]$RepoListPath,
    [string]$OutputCsv = (Join-Path $PSScriptRoot "carla_local_scan.csv"),
    [string]$OutputJson = (Join-Path $PSScriptRoot "carla_local_scan.json")
)

function Read-RepoAllowlist {
    param($Path)
    if (-not $Path) { return $null }
    $names = Get-Content $Path -ErrorAction Stop |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and ($_ -match '^[^/]+/[^/]+$') } |
        ForEach-Object { $_ -replace '/', '__' }
    $set = New-Object 'System.Collections.Generic.HashSet[string]'
    $names | ForEach-Object { [void]$set.Add($_) }
    return $set
}

function Get-ArgumentFlags {
    param($Content)
    $matches = [regex]::Matches($Content, 'add_argument\([^\)]*?(-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)')
    ($matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
}

function Get-ShortDesc {
    param($Content)
    $m = [regex]::Match($Content, '"""(?s)(.*?)"""')
    if ($m.Success) { return ($m.Groups[1].Value.Trim() -replace '\s+', ' ') }
    $firstComment = ($Content -split "`n" | Where-Object { $_ -match '^\s*#' } | Select-Object -First 1)
    if ($firstComment) { return ($firstComment -replace '^\s*#\s*', '').Trim() }
    return ""
}

try {
    if (-not (Test-Path $Root)) { throw "Root path not found: $Root" }

    $allow = Read-RepoAllowlist -Path $RepoListPath

    # Use ripgrep to pre-filter files containing all three markers.
    $rgPattern = '(?s)(?=.*\bimport\s+carla\b)(?=.*__main__)(?=.*argparse)'
    $rgCmd = "rg --pcre2 --files-with-matches --iglob *.py ""$rgPattern"" ""$Root"""
    $paths = & $env:ComSpec /c $rgCmd 2>$null
    if (-not $paths) { $paths = @() }
    if ($allow) {
        $paths = $paths | Where-Object {
            $relTop = $_.Substring($Root.Length).TrimStart('\').Split('\')[0]
            $allow.Contains($relTop)
        }
    }

    $hits = @()
    foreach ($path in $paths) {
        try {
            $content = Get-Content -Raw -LiteralPath $path -ErrorAction Stop
        } catch {
            Write-Warning "skip (read): $path -> $_"
            continue
        }
        if ($content -notmatch 'import\s+carla') { continue }
        if ($content -notmatch '__main__') { continue }
        if ($content -notmatch 'argparse') { continue }

        $args = Get-ArgumentFlags -Content $content
        $desc = Get-ShortDesc -Content $content

        $rel = $path.Substring($Root.Length).TrimStart('\')
        $repoFolder = $rel.Split('\')[0]

        $hits += [pscustomobject]@{
            repoFolder   = $repoFolder
            relativePath = $rel
            suggestedCmd = "python $($rel -replace '^[^\\]+\\', '')"
            argFlags     = ($args -join ' ')
            description  = $desc
        }
    }

    $hits | Sort-Object repoFolder, relativePath -Unique | Export-Csv -NoTypeInformation -Path $OutputCsv
    $hits | Sort-Object repoFolder, relativePath -Unique | ConvertTo-Json -Depth 4 | Set-Content $OutputJson

    Write-Host "[done] scripts found: $($hits.Count)"
    Write-Host "CSV : $OutputCsv"
    Write-Host "JSON: $OutputJson"
}
catch {
    Write-Error $_
    exit 1
}
