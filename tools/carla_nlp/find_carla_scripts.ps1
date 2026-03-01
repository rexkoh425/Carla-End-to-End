# Search repos for CARLA entrypoint scripts and output paths/commands (no cloning required).
# It uses GitHub code search scoped per repo: import carla + __main__ + argparse in Python files.
#
# Usage:
#   powershell -File .\tools\carla_nlp\find_carla_scripts.ps1
#
# If you also want the files downloaded (without full clone), add:
#     -DownloadDir "D:\Datasets\carla_python_script_dataset_snippets"

[CmdletBinding()]
param(
    [string]$RepoListPath = (Join-Path $PSScriptRoot "carla_repos_p1_p6.txt"),
    [string]$OutputCsv    = (Join-Path $PSScriptRoot "carla_script_hits.csv"),
    [string]$OutputJson   = (Join-Path $PSScriptRoot "carla_script_hits.json"),
    [string]$DownloadDir,
    [int]$SleepMsBetweenRepos = 500,
    [switch]$VerboseLogging
)

function Read-RepoList {
    param($Path)
    Get-Content $Path -ErrorAction Stop |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and ($_ -match '^[^/]+/[^/]+$') }
}

function Search-Repo {
    param($Repo)
    # GitHub code search supports paging up to 1000 results; we fetch up to 100 per repo.
    $results = gh api search/code `
        -f q="repo:$Repo import carla __main__ argparse language:python" `
        -F per_page=100 |
        ConvertFrom-Json
    return $results.items
}

function Download-File {
    param($Repo, $Path, $Sha, $OutDir)
    $safeRepo = $Repo -replace '/', '__'
    $destPath = Join-Path (Join-Path $OutDir $safeRepo) $Path
    $destDir  = Split-Path $destPath -Parent
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
    $raw = gh api "repos/$Repo/contents/$Path" -H "Accept: application/vnd.github.raw"
    Set-Content -Path $destPath -Value $raw -Encoding UTF8
    return $destPath
}

try {
    $repos = Read-RepoList -Path $RepoListPath
    if (-not $repos) { throw "No repos found in $RepoListPath" }

    $hits = @()

    foreach ($repo in $repos) {
        if ($VerboseLogging) { Write-Host "[repo] $repo" }
        try {
            $items = Search-Repo -Repo $repo
            foreach ($item in $items) {
                $hit = [pscustomobject]@{
                    repo        = $repo
                    file        = $item.path
                    html_url    = $item.html_url
                    raw_url     = $item.html_url -replace "/blob/", "/raw/"
                    suggested_cmd = "python $($item.path)"
                    downloaded  = ""
                }
                if ($DownloadDir) {
                    $saved = Download-File -Repo $repo -Path $item.path -Sha $item.sha -OutDir $DownloadDir
                    $hit.downloaded = $saved
                }
                $hits += $hit
            }
        } catch {
            Write-Warning "[fail] $repo : $_"
        }
        Start-Sleep -Milliseconds $SleepMsBetweenRepos
    }

    $hits | Sort-Object repo,file -Unique | Export-Csv -NoTypeInformation -Path $OutputCsv
    $hits | Sort-Object repo,file -Unique | ConvertTo-Json -Depth 4 | Set-Content $OutputJson

    Write-Host "[done] hits: $($hits.Count)"
    Write-Host "CSV : $OutputCsv"
    Write-Host "JSON: $OutputJson"
    if ($DownloadDir) { Write-Host "Files saved under: $DownloadDir" }
}
catch {
    Write-Error $_
    exit 1
}
