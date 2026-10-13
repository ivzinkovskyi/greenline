using System.Runtime.InteropServices;

namespace Greenline;

// NVIDIA dGPU tuning through NVML (nvml.dll from the driver, System32), GPU index 0:
//   - max graphics clock: nvmlDeviceSetGpuLockedClocks(min, max) / nvmlDeviceResetGpuLockedClocks.
//     Works on the GeForce RTX 4050 Laptop (verified: lock 1200..1200 -> 1200 MHz, 2000 -> 2010). NVML has no
//     readback for the lock, so status can only show live/max clocks.
//   - core/memory clock offsets (VF curve shift, P0): nvmlDeviceGetClockOffsets / nvmlDeviceSetClockOffsets
//     (nvmlClockOffset_v1). The driver reports the allowed range (4050 Laptop: core -1000..1000, memory -2000..6000 raw).
//     NVML takes the memory offset in double units (offset 1200 -> clock 8001 -> 8601 MHz); this class converts, so
//     public memory offsets/ranges are in the clock MHz that NVML, nvidia-smi and Afterburner show (range -1000..3000).
//     The older nvmlDevice{Gpc,Mem}ClkVfOffset calls return NOT_SUPPORTED on this driver.
// Everything is volatile (lost on reboot / driver reset) and other tuners (UXTU, Afterburner) overwrite it.
public static class Gpu
{
    const int ClockGraphics = 0, ClockMem = 2;
    static IntPtr _dev;

    [StructLayout(LayoutKind.Sequential)]
    struct ClockOffset { public uint Version; public int Type, Pstate, OffsetMhz, MinMhz, MaxMhz; }

    // Initializes NVML once; returns an error text or null. Locked: a host may sample sensors from its own thread.
    static string? Init()
    {
        lock (typeof(Gpu))
            return InitLocked();
    }

    static string? InitLocked()
    {
        if (_dev != IntPtr.Zero)
            return null;
        if (!NativeLibrary.TryLoad("nvml.dll", typeof(Gpu).Assembly, DllImportSearchPath.System32, out _))
            return "nvml.dll not found (no NVIDIA driver)";
        int rc = nvmlInit_v2();
        if (rc != 0)
            return $"nvmlInit failed ({rc})";
        rc = nvmlDeviceGetHandleByIndex_v2(0, out var d);
        if (rc != 0)
            return $"no NVIDIA GPU ({rc})";
        _dev = d;
        return null;
    }

    public record Info(string Name, uint Clock, uint MemClock, uint MaxClock, uint MaxMemClock, uint MinSupported,
                       (int Cur, int Min, int Max)? CoreOffset, (int Cur, int Min, int Max)? MemOffset);

    public static Info? Read(out string? error)
    {
        if ((error = Init()) is not null)
            return null;
        var name = new byte[96];
        nvmlDeviceGetName(_dev, name, (uint)name.Length);
        return new Info(System.Text.Encoding.ASCII.GetString(name).TrimEnd('\0'),
            ClockInfo(nvmlDeviceGetClockInfo, ClockGraphics), ClockInfo(nvmlDeviceGetClockInfo, ClockMem),
            ClockInfo(nvmlDeviceGetMaxClockInfo, ClockGraphics), ClockInfo(nvmlDeviceGetMaxClockInfo, ClockMem),
            MinSupportedClock(), Offset(ClockGraphics), Offset(ClockMem));
    }

    static uint ClockInfo(Func<IntPtr, int, uint[], int> f, int type)
    {
        var v = new uint[1];
        return f(_dev, type, v) == 0 ? v[0] : 0;
    }

    // Lowest supported graphics clock at the highest memory clock (210 MHz on the 4050 Laptop)
    static uint MinSupportedClock()
    {
        var mem = new uint[32];
        uint n = (uint)mem.Length;
        if (nvmlDeviceGetSupportedMemoryClocks(_dev, ref n, mem) != 0 || n == 0)
            return 0;
        var gfx = new uint[512];
        uint m = (uint)gfx.Length;
        return nvmlDeviceGetSupportedGraphicsClocks(_dev, mem[0], ref m, gfx) == 0 && m > 0 ? gfx.Take((int)m).Min() : 0;
    }

    static (int, int, int)? Offset(int type)
    {
        var o = new ClockOffset { Version = (uint)Marshal.SizeOf<ClockOffset>() | 1u << 24, Type = type };
        int k = type == ClockMem ? 2 : 1;
        return nvmlDeviceGetClockOffsets(_dev, ref o) == 0 ? (o.OffsetMhz / k, o.MinMhz / k, o.MaxMhz / k) : null;
    }

    // Graphics clock cap in MHz (0 = remove the cap); returns the NVML status (0 = OK)
    public static int SetMaxClock(uint mhz, uint minClock)
    {
        if (Init() is not null)
            return -1;
        return mhz == 0 ? nvmlDeviceResetGpuLockedClocks(_dev) : nvmlDeviceSetGpuLockedClocks(_dev, minClock, mhz);
    }

    // Pins the graphics clock (min = max) for the stress test; returns the NVML status
    public static int LockClock(uint mhz) => Init() is not null ? -1 : nvmlDeviceSetGpuLockedClocks(_dev, mhz, mhz);

    // Sensors for the stress test. Reasons = nvmlDeviceGetCurrentClocksThrottleReasons bits
    // (0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake)
    public record Sample(uint Mhz, int Mv, uint TempC, double Watts, ulong Reasons);

    public static Sample? ReadSample()
    {
        if (Init() is not null)
            return null;
        var c = new uint[1];
        nvmlDeviceGetClockInfo(_dev, ClockGraphics, c);
        nvmlDeviceGetTemperature(_dev, 0, out uint t);
        nvmlDeviceGetPowerUsage(_dev, out uint mw);
        nvmlDeviceGetCurrentClocksThrottleReasons(_dev, out ulong r);
        int mv = 0;
        try
        {
            if (NvApiInit() is null && NvQuery(0x465F9BCF, 76) is { } v)
                mv = BitConverter.ToInt32(v, 40) / 1000;
        }
        catch (Exception e) when (e is InvalidOperationException or DllNotFoundException or EntryPointNotFoundException) { }
        return new Sample(c[0], mv, t, mw / 1000.0, r);
    }

    // Light sensor set for periodic polling (e.g. an overlay): load %, graphics clock, board power, temperature
    public record Load(uint Percent, uint Mhz, double Watts, uint TempC);

    [StructLayout(LayoutKind.Sequential)]
    struct Utilization { public uint Gpu, Memory; }

    public static Load? ReadLoad()
    {
        if (Init() is not null)
            return null;
        int rc = nvmlDeviceGetUtilizationRates(_dev, out var u);
        if (rc != 0)
        {
            if (rc == 15) // GPU_IS_LOST: driver reset, re-initialize next time
                ResetHandles();
            return null;
        }
        var c = new uint[1];
        nvmlDeviceGetClockInfo(_dev, ClockGraphics, c);
        nvmlDeviceGetTemperature(_dev, 0, out uint t);
        nvmlDeviceGetPowerUsage(_dev, out uint mw);
        return new Load(u.Gpu, c[0], mw / 1000.0, t);
    }

    // Raw VF control buffer (every point offset), to put the curve back after a driver reset wiped it
    public static byte[]? SaveVf()
    {
        try { return QueryCurve(out _)?.Ctl; }
        catch (Exception e) when (e is InvalidOperationException or DllNotFoundException or EntryPointNotFoundException) { return null; }
    }

    public static bool RestoreVf(byte[] ctl)
    {
        try { return NvApiInit() is null && NvFn<NvGet>(0x0733E009)(_nvGpu, (byte[])ctl.Clone()) == 0; }
        catch (Exception e) when (e is InvalidOperationException or DllNotFoundException or EntryPointNotFoundException) { return false; }
    }

    // Drops the cached NVML/NVAPI handles so the next call re-initializes after a driver reset (TDR)
    public static void ResetHandles() => (_dev, _nvGpu) = (IntPtr.Zero, IntPtr.Zero);

    // Core and memory offsets in clock MHz (P0; null = leave that one alone); returns the first nonzero NVML status or 0.
    // Verified: a memory-only nvmlDeviceSetClockOffsets also re-applies the last NVML core offset to every VF point,
    // wiping a per-point curve, so with core null the VF control buffer is saved first and written back.
    public static int SetOffsets(int? core, int? mem)
    {
        if (Init() is not null)
            return -1;
        var vf = core is null ? SaveVf() : null;
        try { return SetOffsetsRaw(core, mem); }
        finally { if (vf is not null) RestoreVf(vf); }
    }

    static int SetOffsetsRaw(int? core, int? mem)
    {
        var writes = new List<(int Type, int Mhz)>();
        if (core is { } c)
            writes.Add((ClockGraphics, c));
        if (mem is { } m)
            writes.Add((ClockMem, m * 2));
        foreach (var (type, mhz) in writes)
        {
            var o = new ClockOffset { Version = (uint)Marshal.SizeOf<ClockOffset>() | 1u << 24, Type = type, OffsetMhz = mhz };
            int rc = nvmlDeviceSetClockOffsets(_dev, ref o);
            if (rc != 0)
                return rc;
        }
        return 0;
    }

    // Validates offsets against the ranges reported by the driver; returns an error text or null
    public static string? CheckOffsets(Info i, int core, int mem) =>
        i.CoreOffset is not var (_, cmin, cmax) || i.MemOffset is not var (_, mmin, mmax) ? "Driver does not report clock offsets."
        : core < cmin || core > cmax ? $"Core offset must be within {cmin}..{cmax} MHz."
        : mem < mmin || mem > mmax ? $"Memory offset must be within {mmin}..{mmax} MHz."
        : null;

    public static string Describe(Info i) =>
        $"{i.Name}: core {i.Clock} MHz (max {i.MaxClock}{(ReadMaxClockLock() is { } cap ? $", capped at {cap}" : "")}), " +
        $"memory {i.MemClock} MHz (max {i.MaxMemClock})\n" +
        $"Offsets: core {Fmt(i.CoreOffset)}, memory {Fmt(i.MemOffset)}";

    static string Fmt((int Cur, int Min, int Max)? o) => o is var (c, mn, mx) ? $"{c:+0;-0;0} MHz ({mn}..{mx})" : "n/a";

    // VF curve through private NVAPI calls (undocumented, what Afterburner's curve editor uses; IDs and layouts from
    // nvapi-rs, verified on the 4050 Laptop). Every request must carry the point mask from ClkVfPointsGetInfo
    // (bytes 4..19), otherwise the driver returns an empty curve. Entries start at 0x34 (info) / 0x44 (status, control):
    //   GetInfo 0x507B4B59 (0x182C, 24-byte entries, dword 5 = 1 for core points, dword 4 = 1 for memory points),
    //   GetStatus 0x21537AD4 (0x1C28, 28-byte entries {type, kHz, uV, ...}) - the effective curve, offsets included,
    //   GetControl/SetControl 0x23F1B133/0x0733E009 (0x2420, 36-byte entries, dword 5 = per-point offset in kHz).
    // 4050 Laptop: 127 core points 450..1240 mV, then 4 memory points. ClientVoltRailsGetStatus 0x465F9BCF (76 bytes,
    // dword 10 = core voltage in uV).
    // Observed: the driver keeps the curve non-decreasing (a point whose clock + offset falls below the previous point
    // is raised to it), the stock curve is flat above ~1095 mV, and nvmlDeviceSetClockOffsets overwrites every point
    // with one offset while nvmlDeviceGetClockOffsets keeps reporting the last NVML value after per-point writes.
    public record VfPoint(int Mv, int Mhz, int OffsetMhz);

    // Per-point offsets: anchors (mV, MHz offset) interpolated linearly by voltage and held flat outside them
    // (no anchors = 0); CapMv flattens the curve above that voltage at the clock it reaches there (undervolt).
    public record CurveSpec((int Mv, int Mhz)[] Anchors, int? CapMv)
    {
        public int OffsetAt(int mv)
        {
            var a = Anchors;
            if (a.Length == 0)
                return 0;
            if (mv <= a[0].Mv)
                return a[0].Mhz;
            for (int i = 1; i < a.Length; i++)
                if (mv <= a[i].Mv)
                    return (int)Math.Round(a[i - 1].Mhz + (double)(a[i].Mhz - a[i - 1].Mhz) * (mv - a[i - 1].Mv) / (a[i].Mv - a[i - 1].Mv));
            return a[^1].Mhz;
        }

        // Tokens: "<mV>:<MHz>" anchors and "cap <mV>", e.g. 700:+300 850:+250 cap 900
        public static CurveSpec? Parse(IList<string> t, out string? error)
        {
            var anchors = new List<(int, int)>();
            int? cap = null;
            for (int i = 0; i < t.Count; i++)
            {
                if (t[i] == "cap" && i + 1 < t.Count && int.TryParse(t[i + 1], out int c) && c is >= 450 and <= 1250)
                    cap = c;
                else if (t[i].Split(':') is [var v, var o] && int.TryParse(v, out int mv) && mv is >= 400 and <= 1300 &&
                         int.TryParse(o, out int mhz))
                {
                    anchors.Add((mv, mhz));
                    continue;
                }
                else
                {
                    error = $"bad curve token \"{t[i]}\" (expected <mV>:<MHz offset> or cap <mV 450..1250>)";
                    return null;
                }
                i++;
            }
            anchors.Sort();
            if (anchors.Select(x => x.Item1).Distinct().Count() != anchors.Count)
            {
                error = "duplicate anchor voltage";
                return null;
            }
            error = anchors.Count == 0 && cap is null ? "give at least one <mV>:<MHz> anchor or cap <mV>" : null;
            return error is null ? new CurveSpec([.. anchors], cap) : null;
        }

        public override string ToString() =>
            string.Join(" ", Anchors.Select(x => $"{x.Mv}:{x.Mhz:+0;-0;0}").Concat(CapMv is { } c ? ["cap", $"{c}"] : []));
    }

    delegate int NvInit();
    delegate int NvEnum([Out] IntPtr[] handles, out int count);
    delegate int NvGet(IntPtr gpu, [In, Out] byte[] data);
    static IntPtr _nvGpu;

    static T NvFn<T>(uint id) where T : Delegate =>
        nvapi_QueryInterface(id) is var p && p != IntPtr.Zero ? Marshal.GetDelegateForFunctionPointer<T>(p)
        : throw new InvalidOperationException($"NVAPI function {id:X8} not available");

    static string? NvApiInit()
    {
        if (_nvGpu != IntPtr.Zero)
            return null;
        if (!NativeLibrary.TryLoad("nvapi64.dll", typeof(Gpu).Assembly, DllImportSearchPath.System32, out _))
            return "nvapi64.dll not found (no NVIDIA driver)";
        if (NvFn<NvInit>(0x0150E828)() is var rc and not 0)
            return $"NvAPI_Initialize failed ({rc})";
        var h = new IntPtr[64];
        if (NvFn<NvEnum>(0xE5AC921F)(h, out int n) != 0 || n == 0)
            return "no NVIDIA GPU (NVAPI)";
        _nvGpu = h[0];
        return null;
    }

    // Versioned NVAPI query; mask (16 bytes) is copied into the request when given
    static byte[]? NvQuery(uint id, int size, byte[]? mask = null)
    {
        var b = new byte[size];
        BitConverter.GetBytes((uint)size | 1u << 16).CopyTo(b, 0);
        mask?.CopyTo(b, 4);
        return NvFn<NvGet>(id)(_nvGpu, b) == 0 ? b : null;
    }

    // Current status and control buffers plus the number of core points (the leading masked entries flagged as core)
    static (byte[] St, byte[] Ctl, int Count)? QueryCurve(out string? error)
    {
        if ((error = NvApiInit()) is not null)
            return null;
        var info = NvQuery(0x507B4B59, 0x182C);
        var mask = info?[4..20];
        var st = NvQuery(0x21537AD4, 0x1C28, mask);
        var ctl = NvQuery(0x23F1B133, 0x2420, mask);
        if (info is null || st is null || ctl is null)
        {
            error = "driver rejected the VF curve query";
            return null;
        }
        int n = 0;
        while (n < 128 && (mask![n / 8] >> (n % 8) & 1) != 0 && BitConverter.ToInt32(info, 0x34 + n * 24 + 20) == 1)
            n++;
        return (st, ctl, n);
    }

    static int StMv(byte[] st, int i) => BitConverter.ToInt32(st, 0x44 + i * 28 + 8) / 1000;
    static int StKhz(byte[] st, int i) => BitConverter.ToInt32(st, 0x44 + i * 28 + 4);
    static int CtlKhz(byte[] ctl, int i) => BitConverter.ToInt32(ctl, 0x44 + i * 36 + 20);

    public static (List<VfPoint> Points, int CurMv)? ReadCurve(out string? error)
    {
        try
        {
            if (QueryCurve(out error) is not var (st, ctl, n))
                return null;
            var pts = Enumerable.Range(0, n).Select(i => new VfPoint(StMv(st, i), StKhz(st, i) / 1000, CtlKhz(ctl, i) / 1000)).ToList();
            var volt = NvQuery(0x465F9BCF, 76);
            return (pts, volt is null ? 0 : BitConverter.ToInt32(volt, 40) / 1000);
        }
        catch (Exception e) when (e is InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            error = e.Message;
            return null;
        }
    }

    // Writes per-point core offsets (spec null = all 0; memory points are written back unchanged). With a cap the
    // anchors go first, then points above the cap get clock(cap) - their own base clock (status - offset, which can
    // only overestimate the base, so the driver's non-decreasing rule lifts them to exactly clock(cap)), clamped to
    // minMhz. Returns an error text or null.
    public static string? WriteCurve(CurveSpec? spec, int minMhz, int maxMhz)
    {
        string? error = WriteCurvePoints(spec, minMhz, maxMhz);
        if (error is null)
            SaveCurveSig(ReadCurve(out _)?.Points.Select(p => p.OffsetMhz));
        return error;
    }

    static string? WriteCurvePoints(CurveSpec? spec, int minMhz, int maxMhz)
    {
        try
        {
            if (QueryCurve(out var error) is not var (st, ctl, n))
                return error;
            for (int i = 0; i < n; i++)
                BitConverter.GetBytes(Math.Clamp(spec?.OffsetAt(StMv(st, i)) ?? 0, minMhz, maxMhz) * 1000).CopyTo(ctl, 0x44 + i * 36 + 20);
            if (NvFn<NvGet>(0x0733E009)(_nvGpu, ctl) is var rc and not 0)
                return $"driver rejected the curve (status {rc})";
            SaveLastCurve(spec);
            if (spec?.CapMv is not { } cap)
                return null;
            if (QueryCurve(out error) is not var (st2, ctl2, n2))
                return error;
            int ci = Enumerable.Range(0, n2).LastOrDefault(i => StMv(st2, i) <= cap);
            int capKhz = StKhz(st2, ci);
            for (int i = ci + 1; i < n2; i++)
                BitConverter.GetBytes(Math.Clamp(capKhz - (StKhz(st2, i) - CtlKhz(ctl2, i)), minMhz * 1000, CtlKhz(ctl2, i)))
                    .CopyTo(ctl2, 0x44 + i * 36 + 20);
            return NvFn<NvGet>(0x0733E009)(_nvGpu, ctl2) is var rc2 and not 0 ? $"driver rejected the cap (status {rc2})" : null;
        }
        catch (Exception e) when (e is InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            return e.Message;
        }
    }

    // Max clock lock (nvmlDeviceSetGpuLockedClocks has no NVML getter) read back from NvAPI_GPU_PerfClientLimitsGetStatus
    // 0xE440B867 (0x30C: version, flags, count, then 24-byte entries {id, ?, mode, ?, kHz, ?} from 0x0C; mode 2 = locked,
    // entry 0 = max, entry 1 = min). Verified: cap 2065 -> entry 0 = 2065000 kHz, entry 1 = 210000; cap removed -> mode 0.
    public static int? ReadMaxClockLock()
    {
        try
        {
            if (NvApiInit() is not null || NvQuery(0xE440B867, 0x30C) is not { } b)
                return null;
            for (int k = 0, n = Math.Min(BitConverter.ToInt32(b, 8), 32); k < n; k++)
            {
                int o = 0x0C + k * 24;
                if (BitConverter.ToInt32(b, o) == 0 && BitConverter.ToInt32(b, o + 8) == 2)
                    return BitConverter.ToInt32(b, o + 16) / 1000;
            }
            return null;
        }
        catch (Exception e) when (e is InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            return null;
        }
    }

    // Expected curve after WriteCurve(spec): base clock (effective - offset) + new offset, kept non-decreasing like the
    // driver does, flat above the cap
    public static int[] Predict(List<VfPoint> pts, CurveSpec spec, int minMhz = -1000, int maxMhz = 1000)
    {
        var r = new int[pts.Count];
        int ci = spec.CapMv is { } cap ? pts.FindLastIndex(p => p.Mv <= cap) : -1;
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = pts[i].Mhz - pts[i].OffsetMhz + Math.Clamp(spec.OffsetAt(pts[i].Mv), minMhz, maxMhz);
            if (i > 0)
                r[i] = ci >= 0 && i > ci ? r[ci] : Math.Max(r[i], r[i - 1]);
        }
        return r;
    }

    // Where the last written curve is kept (gpucurve.txt, gpucurve.sig); the host app may point it at its own folder
    public static string StateDir { get; set; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "greenline");

    // Last curve written through WriteCurve (so a GUI can show the anchors again)
    static string LastCurveFile => Path.Combine(StateDir, "gpucurve.txt");

    public static CurveSpec? LoadLastCurve()
    {
        try { return CurveSpec.Parse(File.ReadAllText(LastCurveFile).Split(' ', StringSplitOptions.RemoveEmptyEntries), out _); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return null; }
    }

    static void SaveLastCurve(CurveSpec? spec)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(LastCurveFile)!);
            if (spec is null || spec.Anchors.Length == 0 && spec.CapMv is null)
                File.Delete(LastCurveFile);
            else
                File.WriteAllText(LastCurveFile, spec.ToString());
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    // Per-point offsets read back right after the last WriteCurve (gpucurve.sig): a later read that differs means
    // something else rewrote the curve (NVML offset, driver reset, another tuner)
    static string CurveSigFile => Path.ChangeExtension(LastCurveFile, ".sig");

    static void SaveCurveSig(IEnumerable<int>? offs)
    {
        try
        {
            if (offs is null)
                File.Delete(CurveSigFile);
            else
                File.WriteAllText(CurveSigFile, string.Join(',', offs));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    // True when the curve last written through WriteCurve is this spec and the GPU still holds exactly those offsets
    public static bool CurveIsLast(CurveSpec spec, int[] offs)
    {
        try
        {
            return LoadLastCurve()?.ToString() == spec.ToString() && File.Exists(CurveSigFile) &&
                   File.ReadAllText(CurveSigFile) == string.Join(',', offs);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return false; }
    }

    // Validates anchors against the core offset range reported by NVML; returns an error text or null
    public static string? CheckCurve(Info i, CurveSpec s) =>
        i.CoreOffset is not var (_, min, max) ? "Driver does not report the core offset range."
        : s.Anchors.FirstOrDefault(a => a.Mhz < min || a.Mhz > max) is { Mv: > 0 } bad ? $"Offset {bad.Mhz} at {bad.Mv} mV is outside {min}..{max} MHz."
        : null;

    // Collapses runs of points with the same clock and offset: "450..470 mV  465 MHz  +260"
    public static string DescribeCurve(List<VfPoint> pts, int curMv, uint curMhz)
    {
        var sb = new System.Text.StringBuilder($"VF curve: {pts.Count} points, now {curMv} mV / {curMhz} MHz\n" +
                                               "  voltage, mV    clock, MHz  offset\n");
        for (int i = 0, j; i < pts.Count; i = j)
        {
            for (j = i + 1; j < pts.Count && pts[j].Mhz == pts[i].Mhz && pts[j].OffsetMhz == pts[i].OffsetMhz; j++) { }
            var mv = j - i == 1 ? $"{pts[i].Mv}" : $"{pts[i].Mv}..{pts[j - 1].Mv}";
            bool cur = curMv >= pts[i].Mv - 2 && curMv <= pts[j - 1].Mv + 2;
            sb.Append($"{(cur ? '>' : ' ')} {mv,-14} {pts[i].Mhz,6}  {pts[i].OffsetMhz,6:+0;-0;0}\n");
        }
        return sb.ToString().TrimEnd();
    }

    [DllImport("nvapi64")] static extern IntPtr nvapi_QueryInterface(uint id);
    [DllImport("nvml")] static extern int nvmlInit_v2();
    [DllImport("nvml")] static extern int nvmlDeviceGetHandleByIndex_v2(uint index, out IntPtr dev);
    [DllImport("nvml")] static extern int nvmlDeviceGetName(IntPtr dev, byte[] name, uint len);
    [DllImport("nvml")] static extern int nvmlDeviceGetClockInfo(IntPtr dev, int type, [Out] uint[] mhz);
    [DllImport("nvml")] static extern int nvmlDeviceGetMaxClockInfo(IntPtr dev, int type, [Out] uint[] mhz);
    [DllImport("nvml")] static extern int nvmlDeviceGetSupportedMemoryClocks(IntPtr dev, ref uint count, [Out] uint[] mhz);
    [DllImport("nvml")] static extern int nvmlDeviceGetSupportedGraphicsClocks(IntPtr dev, uint memMhz, ref uint count, [Out] uint[] mhz);
    [DllImport("nvml")] static extern int nvmlDeviceSetGpuLockedClocks(IntPtr dev, uint minMhz, uint maxMhz);
    [DllImport("nvml")] static extern int nvmlDeviceResetGpuLockedClocks(IntPtr dev);
    [DllImport("nvml")] static extern int nvmlDeviceGetClockOffsets(IntPtr dev, ref ClockOffset info);
    [DllImport("nvml")] static extern int nvmlDeviceSetClockOffsets(IntPtr dev, ref ClockOffset info);
    [DllImport("nvml")] static extern int nvmlDeviceGetTemperature(IntPtr dev, int sensor, out uint celsius);
    [DllImport("nvml")] static extern int nvmlDeviceGetPowerUsage(IntPtr dev, out uint milliwatts);
    [DllImport("nvml")] static extern int nvmlDeviceGetUtilizationRates(IntPtr dev, out Utilization u);
    [DllImport("nvml")] static extern int nvmlDeviceGetCurrentClocksThrottleReasons(IntPtr dev, out ulong reasons);
}
