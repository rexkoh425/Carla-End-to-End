# Clone CARLA-related repos from a list and summarize Python scripts that take CLI arguments.
# Usage:
#   powershell -File .\tools\carla_nlp\clone_carla_repos.ps1 -OutputRoot "D:\Datasets\carla_python_script_dataset"

[CmdletBinding()]
param(
    [string]$RepoListPath = (Join-Path $PSScriptRoot "carla_repos_p1_p6.txt"),
    [string]$OutputRoot   = "D:\Datasets\carla_python_script_dataset",
    [switch]$SkipClone    # set to skip cloning and only rescan existing checkouts
)

function Read-RepoList {
    param($Path)
    Get-Content $Path -ErrorAction Stop |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and ($_ -match '^[^/]+/[^/]+$') }
}

function Ensure-Dir {
    param($Path)
    if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Path $Path | Out-Null }
}

function Clone-Repos {
    param($Repos, $DestRoot)
    foreach ($repo in $Repos) {
        $destName = $repo -replace '/', '__'   # avoid nested dirs
        $destPath = Join-Path $DestRoot $destName
        if (Test-Path $destPath) {
            Write-Host "[skip] $repo (exists)"
            continue
        }
        $url = "https://github.com/$repo"
        Write-Host "[clone] $repo -> $destPath"
        git clone --depth 1 --quiet $url $destPath 2>$null
    }
}

function Get-ArgumentFlags {
    param($Content)
    # Capture flags from argparse add_argument calls
    # Use single backslash before '(' so regex sees a literal '(' (not an opening group).
    $matches = [regex]::Matches($Content, 'add_argument\([^\)]*?(-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)')
    ($matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
}

function Get-ShortDesc {
    param($Content)
    # Grab top-level triple-quoted docstring or first comment line
    $m = [regex]::Match($Content, '"""(?s)(.*?)"""')
    if ($m.Success) { return ($m.Groups[1].Value.Trim() -replace '\s+', ' ') }
    $firstComment = ($Content -split "`n" | Where-Object { $_ -match '^\s*#' } | Select-Object -First 1)
    if ($firstComment) { return ($firstComment -replace '^\s*#\s*', '').Trim() }
    return ""
}

function Summarize-PythonScripts {
    param($Root, $Repos)
    $summary = @()
    foreach ($repo in $Repos) {
        $destName = $repo -replace '/', '__'
        $repoPath = Join-Path $Root $destName
        if (-not (Test-Path $repoPath)) {
            Write-Warning "[missing] $repoPath"
            continue
        }
        $pyFiles = Get-ChildItem -Path $repoPath -Filter *.py -Recurse -ErrorAction SilentlyContinue
        foreach ($file in $pyFiles) {
            $content = Get-Content -Raw -Path $file.FullName
            if ($content -notmatch 'import\s+carla') { continue } # keep CARLA-related
            $args = Get-ArgumentFlags -Content $content
            if (-not $args) { continue } # only keep scripts that take CLI args
            $desc = Get-ShortDesc -Content $content
            $summary += [pscustomobject]@{
                repo         = $repo
                file         = $file.FullName
                relativePath = $file.FullName.Substring($Root.Length).TrimStart('\')
                hasMain      = [bool]($content -match '__main__')
                argFlags     = ($args -join ' ')
                description  = $desc
            }
        }
    }
    return $summary
}

try {
    $repos = Read-RepoList -Path $RepoListPath
    if (-not $repos) { throw "No repos found in $RepoListPath" }

    Ensure-Dir -Path $OutputRoot

    if (-not $SkipClone) {
        Clone-Repos -Repos $repos -DestRoot $OutputRoot
    }

    Write-Host "[scan] extracting CARLA python scripts with CLI args..."
    $summary = Summarize-PythonScripts -Root $OutputRoot -Repos $repos
    $summaryPathJson = Join-Path $OutputRoot "carla_scripts_summary.json"
    $summaryPathCsv  = Join-Path $OutputRoot "carla_scripts_summary.csv"
    $summary | ConvertTo-Json -Depth 4 | Set-Content $summaryPathJson
    $summary | Export-Csv -NoTypeInformation -Path $summaryPathCsv

    Write-Host "[done] found $($summary.Count) scripts"
    Write-Host "JSON: $summaryPathJson"
    Write-Host "CSV : $summaryPathCsv"
}
catch {
    Write-Error $_
    exit 1
}
