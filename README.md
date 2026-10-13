# greenline

[Русская версия](README.ru.md)

NVIDIA GPU tuning for Windows through the official NVML and private NVAPI functions, with no third-party
dependencies:

- core clock cap;
- core and memory clock offsets;
- per-point VF curve with a voltage cap (undervolt);
- stability test for each curve point: a self-checking CUDA kernel (through `nvcuda.dll`, the driver compiles the PTX,
  so no CUDA SDK is needed).

Works with GPU 0 only. Every change lasts until a reboot or a driver reset; nothing is stored in the GPU or the driver.

| Project | What |
|---|---|
| `Greenline.Core/` | library (`net8.0-windows`, namespace `Greenline`): `Gpu` (NVML/NVAPI), `GpuStress` (CUDA stress test), `GpuCli` (command line front end, reusable by host apps) |
| `greenline-cli/` | `greenline.exe`, a thin wrapper around `GpuCli.Run` |

## Requirements

- Windows 10/11 x64, NVIDIA GPU with a recent driver (`nvmlDeviceSetClockOffsets` appeared in R555; tested on 616.56).
- Reading works as any user; every write needs an elevated (Administrator) terminal.
- Building: .NET 8 SDK. Running: .NET 8 Runtime.
- Tested on a GeForce RTX 4050 Laptop GPU. Other GPUs should work, but the VF curve layout and the offset ranges come
  from the driver and may differ.
- Other tuners (MSI Afterburner, UXTU and similar) write the same settings and overwrite each other: run one at a time.

## Build

```
dotnet build greenline-cli -c Release
greenline-cli\bin\Release\net8.0-windows\greenline.exe help
```

## Usage

```
greenline                                        status: clocks, offsets, max voltage
greenline maxclock 2100                          cap the core clock at 2100 MHz (off = remove the cap)
greenline maxvoltage 900                         flat curve above 900 mV (undervolt; off = remove)
greenline core 150                               core +150 MHz (whole curve shift)
greenline mem 1000                               memory +1000 MHz
greenline curve                                  VF curve: voltage -> clock and per-point offset
greenline curve set 700:+300 850:+250            per-point core offsets
greenline curve reset                            all point offsets back to 0, max voltage removed
greenline stress 700-900 30                      test each curve point from 700 to 900 mV, 30 s each
```

Writes ask for confirmation; `--yes` (`-y`) skips it. Example output:

```
> greenline
NVIDIA GeForce RTX 4050 Laptop GPU: core 210 MHz (max 3105), memory 810 MHz (max 8001)
Offsets: core 0 MHz (-1000..1000), memory +1200 MHz (-1000..3000)
Core: core offset 0 MHz, no max voltage
```

### VF curve

`curve set` takes anchors `<mV>:<MHz offset>`: offsets between anchors are interpolated by voltage and held flat
outside them. `core` rewrites every point with one value and replaces a per-point curve.

`maxvoltage <mV>` flattens the curve above that voltage at the clock it reaches there, so the GPU never goes above
that voltage. The flat top only exists as per-point offsets, so it is written together with the core offsets: `core`
and `curve set` keep the current max voltage, `maxvoltage` keeps the current core offset or curve.

### Stress test

`stress [<mV>-<mV>] [<seconds>] [<step mV>]` pins the core clock to each curve point in the range (default
600 mV..top, 50 mV step, 30 s each) and runs a CUDA kernel whose every result is compared bit-for-bit with a CPU
reference, so silent arithmetic errors are caught too. The clock cap and the VF curve are restored afterwards, also
after a driver reset. A markdown report is appended to `gpu-stress.md` in the current folder.

Pure compute does not load RT/tensor/texture units, and the power limit keeps the top of the curve from holding the
pinned clock: a pass is a good sign, not a guarantee. Final validation is still a real workload.

## Using the library

Reference `Greenline.Core/Greenline.Core.csproj` with a `ProjectReference`, or pack it into a local NuGet feed:

```
dotnet pack Greenline.Core -c Release -o <feed folder>
```

(`pack.ps1` does this into `..\packages`.) Bump `<Version>` in `Greenline.Core.csproj` on every change: NuGet caches
a version forever. The package embeds the pdb and the sources, so a debugger can step into the library.

Host settings (static, set once at startup):

| Property | Default | Purpose |
|---|---|---|
| `GpuCli.Prog` | `greenline` | how commands are spelled in help, prompts and the stress report, e.g. `myapp gpu` |
| `Gpu.StateDir` | `%ProgramData%\greenline` | `gpucurve.txt` / `gpucurve.sig`: the last written curve and its read-back signature (`Gpu.CurveIsLast`) |
| `GpuStress.ReportFile` | `gpu-stress.md` | markdown report of every run (relative to the current folder), `null` = no report |

`GpuStress.BusyMutex` (`Global\greenline-stress`) is held while a stress test runs: a host that periodically
re-applies its settings should leave the GPU alone then. For a GUI: `GpuStress.Execute(request, ISink, CancellationToken)`.

## License and risks

[MIT](LICENSE). Use at your own risk: greenline changes GPU clocks and voltages, partly through undocumented NVAPI
functions (function IDs and structure layouts from [nvapi-rs](https://github.com/arcnmx/nvapi-rs), verified on an
RTX 4050 Laptop GPU). Bad values can cause driver resets, artifacts, crashes and overheating. The author takes no
responsibility for the consequences.
