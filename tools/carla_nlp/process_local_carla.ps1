# Process pre-generated file list to produce CARLA script summary.
# Expects: tmp_carla1.txt (paths of files containing "import carla")
# Filters for "__main__" and "argparse", extracts add_argument flags and docstring/comment.
[CmdletBinding()]
param(
    [string]$Root = "D:\Datasets\carla_python_script_dataset",
    [string]$ImportList = (Join-Path $PSScriptRoot "tmp_carla1.txt"),
    [string]$OutputJson = "D:\Datasets\carla_python_script_dataset\carla_commands_full.json"
)

function Parse-Arguments {
    param($Content)
    $required = @()
    $optional = @()
    $matches = [regex]::Matches($Content, 'add_argument\s*\(([^)]*)\)', 'Singleline')
    foreach ($m in $matches) {
        $inside = $m.Groups[1].Value
        $opts = [regex]::Matches($inside, '(-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)') | ForEach-Object { $_.Value } | Sort-Object -Unique
        if (-not $opts) {
            $pos = [regex]::Match($inside, '^\s*([\"'']?)([A-Za-z0-9_]+)\1')
            if ($pos.Success) { $opts = @($pos.Groups[2].Value) }
        }
        $reqFlag = $inside -match 'required\s*=\s*True'
        $choices = @()
        $cm = [regex]::Match($inside, 'choices\s*=\s*\[([^\]]*)\]')
        if ($cm.Success) {
            $choices = $cm.Groups[1].Value -split ',' | ForEach-Object { $_.Trim().Trim('"').Trim('''') } | Where-Object { $_ }
        }
        $name = $null
        if ($opts) {
            $long = $opts | Where-Object { $_ -like '--*' }
            if ($long) { $name = $long[0] } else { $name = $opts[0] }
        }
        $entry = [pscustomobject]@{
            name           = $name
            option_strings = $opts
            choices        = $choices
            default        = $null
            help           = $null
        }
        $isPositional = $opts -and ($opts | Where-Object { $_ -notlike '-*' })
        if ($reqFlag -or $isPositional) { $required += $entry } else { $optional += $entry }
    }
    return @($required, $optional)
}

function Get-Description {
    param($Content)
    $m = [regex]::Match($Content,'(?s)"""(.*?)"""')
    if ($m.Success) { return ($m.Groups[1].Value.Trim() -replace '\s+',' ') }
    $first = ($Content -split "`n" | Where-Object { $_ -match '^\s*#' } | Select-Object -First 1)
    if ($first) { return ($first -replace '^\s*#\s*','').Trim() }
    return ""
}

if (-not (Test-Path $ImportList)) { Write-Error "Missing list: $ImportList"; exit 1 }
$paths = Get-Content $ImportList
$results = @()
foreach ($p in $paths) {
    try { $c = Get-Content -Raw -LiteralPath $p -ErrorAction Stop } catch { continue }
    if ($c -notmatch '__main__') { continue }
    if ($c -notmatch 'argparse') { continue }
    $rel = $p.Substring($Root.Length).TrimStart('\').Replace('\','/')
    $repo = $rel.Split('/')[0]
    $args = Parse-Arguments -Content $c
    $desc = Get-Description -Content $c
    $results += [pscustomobject]@{
        repo           = $repo
        relative_path  = $rel
        command        = "python $rel"
        required_args  = $args[0]
        optional_args  = $args[1]
        description    = $desc
    }
}
$results | Sort-Object repo, relative_path -Unique | ConvertTo-Json -Depth 8 | Set-Content $OutputJson -Encoding UTF8
Write-Host "records $($results.Count) -> $OutputJson"
