# Pack Greenline.Core into the shared local feed ..\packages (host apps pick it up via their nuget.config).
# NuGet caches each version in ~/.nuget/packages: bump <Version> in Greenline.Core.csproj before packing a change.
$ErrorActionPreference = 'Stop'
$feed = Join-Path $PSScriptRoot '..\packages'
dotnet pack "$PSScriptRoot\Greenline.Core" -c Release -o $feed
