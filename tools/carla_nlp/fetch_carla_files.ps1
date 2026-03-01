# Download only CARLA-related Python scripts (no full clone).
# Criteria: file ends with .py AND contains "import carla" AND "__main__" AND "argparse".
# Uses GitHub API via `gh`; counts against core API (not code-search) limits.
#
# Usage example:
#   powershell -File .\tools\carla_nlp\fetch_carla_files.ps1 `
#     -OutputRoot "D:\Datasets\carla_python_script_dataset"
#
# Notes:
# - Destination layout: D:\Datasets\carla_python_script_dataset\<owner__repo>\<original path>
# - Skips files already downloaded unless -Force is set.

[CmdletBinding()]
param(
    [string]$RepoListPath = (Join-Path $PSScriptRoot "carla_repos_p1_p6.txt"),
    [string]$OutputRoot   = "D:\Datasets\carla_python_script_dataset",
    [switch]$Force,
    [int]$SleepMsBetweenRepos = 500  # mild throttle to avoid abuse detection
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

function Get-DefaultBranch {
    param($Repo)
    $info = gh api "repos/$Repo" | ConvertFrom-Json
    return $info.default_branch
}

function Get-PythonPaths {
    param($Repo, $Branch)
    $tree = gh api "repos/$Repo/git/trees/$Branch" -f recursive=1 | ConvertFrom-Json
    $tree.tree | Where-Object { $_.type -eq "blob" -and $_.path -like "*.py" } | Select-Object -ExpandProperty path
}

function Download-And-Filter {
    param($Repo, $Branch, $Paths, $OutRoot, $ForceDownload)
    $saved = @()
    $repoFolder = $Repo -replace '/', '__'
    foreach ($relPath in $Paths) {
        $destPath = Join-Path (Join-Path $OutRoot $repoFolder) $relPath
        if (-not $ForceDownload -and (Test-Path $destPath)) { continue }
        $content = gh api "repos/$Repo/contents/$relPath" -f ref=$Branch -H "Accept: application/vnd.github.raw"
        if ($content -notmatch 'import\s+carla') { continue }
        if ($content -notmatch '__main__') { continue }
        if ($content -notmatch 'argparse') { continue }
        Ensure-Dir -Path (Split-Path $destPath -Parent)
        Set-Content -Path $destPath -Value $content -Encoding UTF8
        $saved += [pscustomobject]@{
            repo  = $Repo
            file  = $relPath
            path  = $destPath
        }
    }
    return $saved
}

try {
    $repos = Read-RepoList -Path $RepoListPath
    if (-not $repos) { throw "No repos found in $RepoListPath" }

    Ensure-Dir -Path $OutputRoot
    $summary = @()

    foreach ($repo in $repos) {
        Write-Host "[repo] $repo"
        try {
            $branch = Get-DefaultBranch -Repo $repo
            $paths  = Get-PythonPaths -Repo $repo -Branch $branch
            $saved  = Download-And-Filter -Repo $repo -Branch $branch -Paths $paths -OutRoot $OutputRoot -ForceDownload:$Force
            if ($saved.Count -gt 0) {
                Write-Host "  saved $($saved.Count) files"
                $summary += $saved
            } else {
                Write-Host "  no matching files"
            }
        } catch {
            Write-Warning "  failed: $_"
        }
        Start-Sleep -Milliseconds $SleepMsBetweenRepos
    }

    $jsonPath = Join-Path $OutputRoot "carla_downloaded_files.json"
    $csvPath  = Join-Path $OutputRoot "carla_downloaded_files.csv"
    $summary | ConvertTo-Json -Depth 4 | Set-Content $jsonPath
    $summary | Export-Csv -NoTypeInformation -Path $csvPath
    Write-Host "[done] saved $($summary.Count) files"
    Write-Host "JSON: $jsonPath"
    Write-Host "CSV : $csvPath"
}
catch {
    Write-Error $_
    exit 1
}
