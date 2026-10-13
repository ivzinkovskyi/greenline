using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Greenline;

// GPU stability test per VF curve point. The graphics clock is pinned (NVML min = max) to the clock of a chosen
// curve point, so the GPU runs at that point's voltage, and a CUDA kernel (driver API through nvcuda.dll; the PTX
// below is JIT-compiled by the driver, so no CUDA SDK is needed) runs there for a while. Every thread iterates the
// logistic map x = 3.9 * (x - x*x) built from a fused multiply-add, in four chains; the map is chaotic, so a single
// wrong bit anywhere grows into a completely different result within ~50 iterations. Each launch is compared
// bit-for-bit with a CPU reference computed with MathF.FusedMultiplyAdd (the same single rounding as fma.rn.f32),
// so silent arithmetic errors that never crash the GPU are caught too. A driver reset (TDR) shows up as a CUDA
// error, is reported, and the pre-test VF curve, offsets and clock cap are restored. Every launch is a few
// milliseconds, far below the TDR timeout.
//
// Limits this test cannot cover: pure compute never touches the RT/RT-core, tensor or texture units, and the
// laptop power limit means the top of the curve will not hold the pinned clock. An "OK" here does not guarantee
// that a game will not crash; final validation is still a real workload.
public static class GpuStress
{
    internal const int Threads = 256, Unroll = 8, Iters = 1 << 15; // Iters must be a multiple of Unroll
    internal const float R = 3.9f, SeedScale = 0.5f / 65536;
    internal static readonly float[] Bases = [0.1f, 0.2f, 0.3f, 0.4f];

    const int DefaultSeconds = 30, DefaultStepMv = 50;

    // Held while a test runs, so a host's periodic re-apply can leave the GPU alone: the test pins clocks and edits the
    // curve on purpose (and restores them itself)
    public const string BusyMutex = @"Global\greenline-stress";

    // Markdown report of every run is appended here (relative to the current directory); null = no report
    public static string? ReportFile { get; set; } = "gpu-stress.md";

    public record Step(int TargetMv, uint TargetMhz);

    public record Result(Step Step, bool Skipped, string? Fault, int Runs, long Errors, double AvgMhz, uint MinMhz,
                         int MaxMv, uint MaxTempC, double MaxWatts, ulong Reasons, bool Held);

    // What the CLI and the GUI ask for (null range = the whole working range)
    public sealed record Request(int? FromMv, int? ToMv, int Sec, int StepMv);

    public sealed record Summary(bool Available, bool Aborted, int Passed, int Errored, int Faulted, int Skipped,
                                 string RestoreNote, IReadOnlyList<Result> Results);

    // Progress/result callbacks so the CLI prints and the GUI updates live
    public interface ISink
    {
        void Line(string text);
        void Error(string text);
        void Progress(int index, int total, double fraction); // index 1-based, fraction within the step
        void StepDone(int index, int total, Result result);
        void Finished(Summary summary);
    }

    // CLI entry: <from>-<to> <sec> <step>, confirm, then run with console output
    public static int Run(string[] a, bool yes)
    {
        if (Parse(a, out var req, out var perr))
        {
            Console.Error.WriteLine(perr);
            return 1;
        }
        Console.Write("Run the GPU stress test? The clock cap and the VF curve are restored afterwards. [y/N] ");
        if (!yes && !string.Equals(Console.ReadLine()?.Trim(), "y", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine("Aborted.");
            return 1;
        }
        var s = Execute(req, new ConsoleSink(), CancellationToken.None);
        return s.Available && !s.Aborted && s.Errored == 0 && s.Faulted == 0 && s.Results.Count > 0 ? 0 : 3;
    }

    static bool Parse(string[] a, out Request request, out string? error)
    {
        request = new Request(null, null, DefaultSeconds, DefaultStepMv);
        error = null;
        int? from = null, to = null, seconds = null, stepMv = null;
        foreach (var t in a)
        {
            if (t.Split('-', StringSplitOptions.RemoveEmptyEntries) is [var s1, var s2] &&
                int.TryParse(s1, out int v1) && int.TryParse(s2, out int v2))
                (from, to) = (v1, v2);
            else if (int.TryParse(t, out int v))
            {
                if (seconds is null)
                    seconds = v;
                else
                    stepMv = v;
            }
            else
            {
                error = $"gpu stress: bad argument \"{t}\".";
                return true;
            }
        }
        int sec = seconds ?? DefaultSeconds, step = stepMv ?? DefaultStepMv;
        if (sec < 1)
        {
            error = "Seconds per step must be at least 1.";
            return true;
        }
        if (step is < 10 or > 200)
        {
            error = "Voltage step must be within 10..200 mV.";
            return true;
        }
        request = new Request(from, to, sec, step);
        return false;
    }

    // Shared core: pins the clock to each curve point of the range and checks every launch against the CPU
    // reference. Restores the clock cap and the VF curve at the end and after a reset; never touches the stale
    // NVML global offset.
    public static Summary Execute(Request req, ISink sink, CancellationToken ct)
    {
        var none = new Summary(false, false, 0, 0, 0, 0, "", []);
        using var busy = new Mutex(false, BusyMutex);
        if (Gpu.Read(out var err) is not { } info)
        {
            sink.Error($"GPU unavailable: {err}.");
            return none;
        }
        if (Gpu.ReadCurve(out var cerr) is not var (pts, curMv))
        {
            sink.Error($"VF curve unavailable: {cerr}.");
            return none;
        }
        int lo = req.FromMv ?? Math.Max(600, pts[0].Mv), hi = req.ToMv ?? pts[^1].Mv;
        if (lo < pts[0].Mv || hi > pts[^1].Mv || lo >= hi)
        {
            sink.Error($"Range must be within {pts[0].Mv}..{pts[^1].Mv} mV and not empty.");
            return none;
        }

        // One step per distinct clock of the curve inside the range: the clock pins the GPU to that segment's voltage
        var steps = new List<Step>();
        uint prev = 0;
        bool first = true;
        for (int mv = lo; mv <= hi; mv += req.StepMv)
        {
            uint mhz = (uint)ClockAt(pts, mv);
            if (mhz == 0 || (!first && mhz == prev))
                continue;
            steps.Add(new Step(mv, mhz));
            (prev, first) = (mhz, false);
        }
        if (!first && (uint)ClockAt(pts, hi) != prev)
            steps.Add(new Step(hi, (uint)ClockAt(pts, hi)));
        if (steps.Count == 0)
        {
            sink.Error("No curve points in that range.");
            return none;
        }

        int sec = req.Sec, step = req.StepMv;
        sink.Line($"GPU stress: {steps.Count} steps, {sec} s each; the graphics clock is pinned (min = max) to each");
        sink.Line($"curve point ({lo}..{hi} mV, {step} mV step -> {steps.Count} distinct clocks), every launch is compared");
        sink.Line($"bit-for-bit with the CPU reference. Now: {curMv} mV / {info.Clock} MHz.");

        // Pre-test state to restore: the clock cap and the raw VF control buffer (a driver reset wipes both).
        // The NVML core/memory offsets are deliberately not used: after a per-point curve write NVML keeps a stale
        // value (it keeps reporting the last offset set through NVML), so the only faithful snapshot is Gpu.SaveVf().
        int? savedCap = Gpu.ReadMaxClockLock();
        byte[]? savedVf = Gpu.SaveVf();
        int[] beforeOffsets = pts.Select(p => p.OffsetMhz).ToArray();

        var results = new List<Result>();
        var samples = new List<(int Step, Gpu.Sample S)>();
        var gate = new object();
        var cur = new int[1] { -1 };
        var running = new bool[1] { true };
        var monitor = new Thread(() =>
        {
            while (running[0])
            {
                if (Gpu.ReadSample() is { } s)
                    lock (gate)
                        samples.Add((cur[0], s));
                Thread.Sleep(100);
            }
        })
        { IsBackground = true, Name = "gpu-stress-monitor" };
        monitor.Start();

        Engine engine;
        sink.Line("Building the CUDA kernel and the CPU reference...");
        try
        {
            engine = new Engine();
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or InvalidOperationException)
        {
            running[0] = false;
            monitor.Join(500);
            sink.Error($"CUDA недоступна ({e.Message}). Настройки GPU не менялись.");
            return none;
        }

        // Warm-up before touching anything: if CUDA cannot run at all (destroyed context, launch error) or the very
        // first result already disagrees with the reference, stop here. A first-launch error is not a driver reset
        // during the test, so it must not be treated as one.
        var (warmBad, warmFault) = engine.Iterate();
        if (warmFault is not null || warmBad > 0)
        {
            running[0] = false;
            monitor.Join(500);
            engine.Dispose();
            sink.Error(warmFault is not null
                ? $"CUDA недоступна ({warmFault}). Настройки GPU не менялись."
                : $"CUDA self-check failed ({warmBad} mismatches at idle). Настройки GPU не менялись.");
            return none;
        }

        var report = new StringBuilder();
        report.Append($"\n### Стресс-тест GPU по точкам VF-кривой ({DateTime.Now:yyyy-MM-dd HH:mm}, CUDA-ядро, сверка с эталоном)\n\n");
        report.Append($"`{GpuCli.Prog} stress {lo}-{hi} {sec}` — частота ядра зафиксирована (NVML min = max) на частоте каждой точки\n" +
                      $"кривой в диапазоне {lo}..{hi} мВ (шаг {step} мВ, {steps.Count} различных частот), на каждой точке {sec} с.\n" +
                      "Ядро считает 4 цепочки логистического отображения `x = 3.9*(x - x*x)` на FMA; каждое значение сверяется\n" +
                      "бит-в-бит с эталоном на CPU, поэтому «тихие» ошибки арифметики видны как расхождение. Троттлинг — биты NVML\n" +
                      "(SW power cap / HW slowdown / SW/HW thermal). Столбец «держит» — средняя частота в пределах 3% от заданной.\n\n");
        report.Append("| мВ (цель) | МГц (цель) | ср. МГц | мин. МГц | мВ (факт) | T max, °C | W max | ошибки | троттлинг | держит |\n");
        report.Append("|---|---|---|---|---|---|---|---|---|---|\n");

        int passed = 0, errored = 0, faulted = 0, skipped = 0, recovered = 0;
        bool aborted = false;
        try
        {
            for (int i = 0; i < steps.Count && !ct.IsCancellationRequested; i++)
            {
                var stepSpec = steps[i];
                cur[0] = i;
                int rc = Gpu.LockClock(stepSpec.TargetMhz);
                if (rc != 0)
                {
                    var skip = new Result(stepSpec, true, null, 0, 0, 0, 0, 0, 0, 0, 0, false);
                    results.Add(skip);
                    sink.StepDone(i + 1, steps.Count, skip);
                    skipped++;
                    report.Append($"| {stepSpec.TargetMv} | {stepSpec.TargetMhz} | — | — | — | — | — | — | не поддержано | — |\n");
                    continue;
                }

                Thread.Sleep(400); // let the clock and the rail settle
                long errors = 0;
                string? fault = null;
                int runs = 0;
                var sw = Stopwatch.StartNew();
                var lastReport = TimeSpan.Zero;
                while (sw.Elapsed.TotalSeconds < sec && !ct.IsCancellationRequested)
                {
                    var (bad, f) = engine.Iterate();
                    runs++;
                    if (f is not null)
                    {
                        fault = f;
                        break;
                    }
                    errors += bad;
                    if (bad > 0)
                        break; // deterministic: same inputs would fail the next launch too
                    if (sw.Elapsed - lastReport >= TimeSpan.FromMilliseconds(100))
                    {
                        lastReport = sw.Elapsed;
                        sink.Progress(i + 1, steps.Count, Math.Min(1, sw.Elapsed.TotalSeconds / sec));
                    }
                }
                sw.Stop();

                List<Gpu.Sample> ss;
                lock (gate)
                    ss = samples.Where(x => x.Step == i && x.S.Mhz > 0).Select(x => x.S).ToList();
                double avg = ss.Count == 0 ? 0 : ss.Average(s => (double)s.Mhz);
                uint minMhz = ss.Count == 0 ? 0 : ss.Min(s => s.Mhz);
                int maxMv = ss.Count == 0 ? 0 : ss.Max(s => s.Mv);
                uint maxT = ss.Count == 0 ? 0 : ss.Max(s => s.TempC);
                double maxW = ss.Count == 0 ? 0 : ss.Max(s => s.Watts);
                ulong reasons = ss.Aggregate(0UL, (acc, s) => acc | s.Reasons);
                bool held = avg >= stepSpec.TargetMhz * 0.97;

                var result = new Result(stepSpec, false, fault, runs, errors, avg, minMhz, maxMv, maxT, maxW, reasons, held);
                results.Add(result);
                sink.Progress(i + 1, steps.Count, 1);
                sink.StepDone(i + 1, steps.Count, result);
                report.Append($"| {stepSpec.TargetMv} | {stepSpec.TargetMhz} | {avg:0} | {minMhz} | {maxMv} | {maxT} | " +
                              $"{maxW:0.0} | {errors} | {ThrottleText(reasons)} | {(held ? "да" : "нет")} |\n");

                if (fault is not null)
                {
                    faulted++;
                    report.Append($"| ^ | | | | | | | | | сброс драйвера: {fault} |\n");
                    sink.Line("  driver reset: restoring the VF curve and rebuilding the CUDA context...");
                    Gpu.ResetHandles();
                    if (savedVf is not null)
                        Gpu.RestoreVf(savedVf);
                    Gpu.LockClock(stepSpec.TargetMhz);
                    if (engine.TryRecover())
                    {
                        recovered++;
                        sink.Line("  recovered, continuing.");
                    }
                    else
                    {
                        sink.Error("  recovery failed; stopping.");
                        aborted = true;
                        break;
                    }
                }
                else if (errors > 0)
                {
                    errored++;
                }
                else
                {
                    passed++;
                }
            }
            if (ct.IsCancellationRequested)
                aborted = true;
        }
        finally
        {
            running[0] = false;
            monitor.Join(500);
            Gpu.SetMaxClock(savedCap is { } c ? (uint)c : 0, info.MinSupported);
            if (savedVf is not null)
                Gpu.RestoreVf(savedVf);
            engine.Dispose();
        }

        // Verify the curve came back: compare per-point offsets before and after, and shout if they differ
        string restoreNote;
        if (Gpu.ReadCurve(out var aerr) is var (afterPts, _))
        {
            int changed = Math.Abs(beforeOffsets.Length - afterPts.Count);
            for (int i = 0; i < Math.Min(beforeOffsets.Length, afterPts.Count); i++)
                if (afterPts[i].OffsetMhz != beforeOffsets[i])
                    changed++;
            restoreNote = changed == 0
                ? $"VF curve restored from the saved buffer ({beforeOffsets.Length} point offsets unchanged)."
                : $"!!! КРИВАЯ ИЗМЕНИЛАСЬ: {changed} из {beforeOffsets.Length} offset'ов точек отличаются до/после — " +
                  $"перепримените вашу кривую ({GpuCli.Prog} curve set ... / GUI) !!!";
        }
        else
        {
            restoreNote = $"!!! кривую не удалось перечитать после теста ({aerr}) — проверьте настройки вручную !!!";
        }

        int back = Gpu.ReadMaxClockLock() ?? 0;
        sink.Line("");
        sink.Line($"Done: {passed} ok, {errored} with errors, {faulted} driver resets" +
                  (skipped > 0 ? $", {skipped} skipped" : "") + (aborted ? " (aborted)" : "") + ".");
        sink.Line($"Clock cap restored: {(savedCap is { } sc ? $"{sc} MHz" : "removed")} (now {back} MHz)" +
                  (recovered > 0 ? $"; re-applied settings after {recovered} reset(s)." : "."));
        if (restoreNote.StartsWith("!!!"))
            sink.Error(restoreNote);
        else
            sink.Line(restoreNote);

        report.Append($"\nИтог: {passed} ok, {errored} с ошибками, {faulted} со сбросом драйвера" +
                      (skipped > 0 ? $", {skipped} пропущено" : "") + ".\n");
        report.Append(restoreNote.StartsWith("!!!") ? $"\n**{restoreNote}**\n" : $"Восстановление кривой: {restoreNote}\n");
        if (AppendReport(report.ToString()) is { } note)
            sink.Line(note);

        var summary = new Summary(true, aborted, passed, errored, faulted, skipped, restoreNote, results);
        sink.Finished(summary);
        return summary;
    }

    // One console line per step (also used as the GUI status line)
    public static string StepLine(int index, int total, Result r)
    {
        var s = r.Step;
        if (r.Skipped)
            return $"[{index,2}/{total}] {s.TargetMv,4} mV -> {s.TargetMhz,4} MHz: clock not supported, skipped";
        string status = r.Fault is not null ? $"СБОЙ ДРАЙВЕРА ({r.Fault})"
            : r.Errors > 0 ? $"ОШИБКИ: {r.Errors}"
            : r.Held ? "ok" : "частота не держится";
        return $"[{index,2}/{total}] {s.TargetMv,4} mV -> {s.TargetMhz,4} MHz: " +
               $"avg {r.AvgMhz:0} / min {r.MinMhz} MHz @ {r.MaxMv} mV, {r.MaxTempC} °C, {r.MaxWatts:0.0} W, " +
               $"err {r.Errors}, {ThrottleText(r.Reasons)}, {status}";
    }

    // Console sink for the CLI
    sealed class ConsoleSink : ISink
    {
        public void Line(string text) => Console.WriteLine(text);
        public void Error(string text) => Console.Error.WriteLine(text);
        public void Progress(int index, int total, double fraction) { }
        public void StepDone(int index, int total, Result result) => Console.WriteLine(StepLine(index, total, result));
        public void Finished(Summary summary) { }
    }

    // Effective clock of the curve at a voltage (step function, points are sorted by voltage)
    static int ClockAt(List<Gpu.VfPoint> pts, int mv)
    {
        int mhz = pts.Count > 0 ? pts[0].Mhz : 0;
        foreach (var p in pts)
        {
            if (p.Mv > mv)
                break;
            mhz = p.Mhz;
        }
        return mhz;
    }

    // nvmlDeviceGetCurrentClocksThrottleReasons bits
    public static string ThrottleText(ulong r)
    {
        var names = new List<string>();
        foreach (var (bit, name) in new (ulong, string)[]
                 {
                     (0x1, "IDLE"), (0x4, "SW power"), (0x8, "HW slowdown"), (0x10, "sync boost"),
                     (0x20, "SW thermal"), (0x40, "HW thermal"), (0x80, "power brake"),
                 })
            if ((r & bit) != 0)
                names.Add(name);
        return names.Count == 0 ? "-" : string.Join("+", names);
    }

    static string? AppendReport(string text)
    {
        if (ReportFile is null)
            return null;
        var path = Path.GetFullPath(ReportFile);
        try
        {
            File.AppendAllText(path, text);
            return $"Log appended to {path}.";
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return $"Could not append to {path}: {e.Message}";
        }
    }

    // PTX source for the stress kernel. It reads four seeds per thread, runs Iters steps of the logistic map with a
    // fused multiply-add (x - x*x) and multiplies by R, then writes four floats. Iterations run Unroll steps per
    // loop pass. fma.rn.f32 + mul.rn.f32 round exactly like MathF.FusedMultiplyAdd + float multiply on the CPU.
    internal static string Ptx(string version, string target)
    {
        string rf = $"0f{BitConverter.SingleToInt32Bits(R):X8}";
        var sb = new StringBuilder();
        sb.Append($".version {version}\n.target {target}\n.address_size 64\n\n");
        sb.Append(".visible .entry stress(.param .u64 p_in, .param .u64 p_out, .param .u32 p_iters)\n{\n");
        sb.Append("  .reg .pred %p<2>;\n");
        sb.Append("  .reg .b32  %r<8>;\n");
        sb.Append("  .reg .b64  %rd<6>;\n");
        sb.Append("  .reg .f32  %x<4>;\n");
        sb.Append("  .reg .f32  %n<4>;\n");
        sb.Append("  ld.param.u64 %rd1, [p_in];\n");
        sb.Append("  ld.param.u64 %rd2, [p_out];\n");
        sb.Append("  ld.param.u32 %r1, [p_iters];\n");
        sb.Append("  mov.u32 %r6, %r1;\n");
        sb.Append("  mov.u32 %r2, %ntid.x;\n");
        sb.Append("  mov.u32 %r3, %ctaid.x;\n");
        sb.Append("  mov.u32 %r4, %tid.x;\n");
        sb.Append("  mad.lo.u32 %r5, %r3, %r2, %r4;\n");
        sb.Append("  cvt.u64.u32 %rd3, %r5;\n");
        sb.Append("  shl.b64 %rd3, %rd3, 4;\n");
        sb.Append("  add.u64 %rd4, %rd1, %rd3;\n");
        sb.Append("  ld.global.f32 %x0, [%rd4+0];\n");
        sb.Append("  ld.global.f32 %x1, [%rd4+4];\n");
        sb.Append("  ld.global.f32 %x2, [%rd4+8];\n");
        sb.Append("  ld.global.f32 %x3, [%rd4+12];\n");
        sb.Append("$L_loop:\n");
        for (int u = 0; u < Unroll; u++)
            for (int k = 0; k < 4; k++)
            {
                sb.Append($"  neg.f32 %n{k}, %x{k};\n");
                sb.Append($"  fma.rn.f32 %x{k}, %n{k}, %x{k}, %x{k};\n");
                sb.Append($"  mul.rn.f32 %x{k}, %x{k}, {rf};\n");
            }
        sb.Append($"  sub.u32 %r6, %r6, {Unroll};\n");
        sb.Append("  setp.ne.u32 %p0, %r6, 0;\n");
        sb.Append("  @%p0 bra $L_loop;\n");
        sb.Append("  add.u64 %rd5, %rd2, %rd3;\n");
        sb.Append("  st.global.f32 [%rd5+0], %x0;\n");
        sb.Append("  st.global.f32 [%rd5+4], %x1;\n");
        sb.Append("  st.global.f32 [%rd5+8], %x2;\n");
        sb.Append("  st.global.f32 [%rd5+12], %x3;\n");
        sb.Append("  ret;\n}\n");
        return sb.ToString();
    }
}

// Resolves CUDA driver API entry points from nvcuda.dll by their exact export name. The plain and _v2 symbols are
// NOT interchangeable: cuMemAlloc_v2 works while the plain cuMemAlloc fails with 201, and this driver's
// cuCtxSynchronize_v2 always returns 709 (CUDA_ERROR_CONTEXT_IS_DESTROYED) while the plain cuCtxSynchronize works,
// so each call site names the exact symbol it needs.
static class Cu
{
    static readonly IntPtr Lib = Load();

    static IntPtr Load()
    {
        if (NativeLibrary.TryLoad("nvcuda.dll", typeof(GpuStress).Assembly, DllImportSearchPath.System32, out var h))
            return h;
        if (NativeLibrary.TryLoad("nvcuda.dll", out h))
            return h;
        throw new DllNotFoundException("nvcuda.dll not found (no NVIDIA driver)");
    }

    public static T Fn<T>(string name) where T : Delegate =>
        NativeLibrary.TryGetExport(Lib, name, out var p)
            ? Marshal.GetDelegateForFunctionPointer<T>(p)
            : throw new EntryPointNotFoundException($"nvcuda.dll: {name} not found");

    // First existing name, for symbols whose spelling differs between driver versions
    public static T FnAny<T>(params string[] names) where T : Delegate
    {
        foreach (var n in names)
            if (NativeLibrary.TryGetExport(Lib, n, out var p))
                return Marshal.GetDelegateForFunctionPointer<T>(p);
        throw new EntryPointNotFoundException($"nvcuda.dll: {string.Join("/", names)} not found");
    }
}

// One CUDA context, the JIT-compiled stress kernel and its buffers. The host-side seeds and CPU reference are built
// once and survive a driver reset; only the device side is recreated by TryRecover.
sealed class Engine : IDisposable
{
    delegate int D_Init(uint flags);
    delegate int D_DeviceGet(out int dev, int ordinal);
    delegate int D_Attr(out int value, int attrib, int dev);
    delegate int D_PrimaryRetain(out IntPtr ctx, int dev);
    delegate int D_PrimaryRelease(int dev);
    delegate int D_PrimaryReset(int dev);
    delegate int D_Push(IntPtr ctx);
    delegate int D_Pop(out IntPtr ctx);
    delegate int D_ModuleLoad(out IntPtr module, IntPtr image, uint n, IntPtr options, IntPtr values);
    delegate int D_ModuleGetFn(out IntPtr fn, IntPtr module, IntPtr name);
    delegate int D_MemAlloc(out ulong ptr, UIntPtr size);
    delegate int D_MemFree(ulong ptr);
    delegate int D_HtoD(ulong dst, IntPtr src, UIntPtr size);
    delegate int D_DtoH(IntPtr dst, ulong src, UIntPtr size);
    delegate int D_Launch(IntPtr fn, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint shared, IntPtr stream,
                          IntPtr parameters, IntPtr extra);
    delegate int D_Sync();
    delegate int D_ErrName(int rc, out IntPtr str);

    // Exact ABI per call (see Cu): the plain cuCtxSynchronize, the _v2 memory calls
    static readonly D_Init _init = Cu.Fn<D_Init>("cuInit");
    static readonly D_DeviceGet _deviceGet = Cu.Fn<D_DeviceGet>("cuDeviceGet");
    static readonly D_Attr _attr = Cu.Fn<D_Attr>("cuDeviceGetAttribute");
    static readonly D_PrimaryRetain _retain = Cu.Fn<D_PrimaryRetain>("cuDevicePrimaryCtxRetain");
    static readonly D_PrimaryRelease _release = Cu.FnAny<D_PrimaryRelease>("cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2");
    static readonly D_PrimaryReset _reset = Cu.Fn<D_PrimaryReset>("cuDevicePrimaryCtxReset");
    static readonly D_Push _push = Cu.Fn<D_Push>("cuCtxPushCurrent");
    static readonly D_Pop _pop = Cu.Fn<D_Pop>("cuCtxPopCurrent");
    static readonly D_ModuleLoad _moduleLoad = Cu.Fn<D_ModuleLoad>("cuModuleLoadDataEx");
    static readonly D_ModuleGetFn _moduleGetFn = Cu.Fn<D_ModuleGetFn>("cuModuleGetFunction");
    static readonly D_MemAlloc _memAlloc = Cu.Fn<D_MemAlloc>("cuMemAlloc_v2");
    static readonly D_MemFree _memFree = Cu.Fn<D_MemFree>("cuMemFree_v2");
    static readonly D_HtoD _htoD = Cu.Fn<D_HtoD>("cuMemcpyHtoD_v2");
    static readonly D_DtoH _dtoH = Cu.Fn<D_DtoH>("cuMemcpyDtoH_v2");
    static readonly D_Launch _launch = Cu.Fn<D_Launch>("cuLaunchKernel");
    static readonly D_Sync _sync = Cu.Fn<D_Sync>("cuCtxSynchronize"); // NOT _v2: this driver's _v2 always returns 709
    static readonly D_ErrName _errName = Cu.Fn<D_ErrName>("cuGetErrorName");

    readonly int _device, _blocks;
    readonly string _cc;
    readonly float[] _seeds, _reference, _hostOut;
    readonly ulong _bytes;
    IntPtr _ctx, _module, _fn;
    ulong _inDev, _outDev;

    public Engine()
    {
        Check(_init(0), "cuInit");
        Check(_deviceGet(out int device, 0), "cuDeviceGet");
        _device = device;
        _attr(out int sm, 16, _device);        // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        _attr(out int maxT, 39, _device);      // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
        _attr(out int major, 75, _device);     // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        _attr(out int minor, 76, _device);     // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        _blocks = Math.Max(1, sm * Math.Max(1, maxT / GpuStress.Threads));
        _cc = $"sm_{major}{minor}";

        int count = _blocks * GpuStress.Threads;
        _bytes = (ulong)count * 4 * 4;
        _seeds = new float[count * 4];
        for (int gid = 0; gid < count; gid++)
            for (int k = 0; k < 4; k++)
                _seeds[gid * 4 + k] = GpuStress.Bases[k] + (gid & 0xFFFF) * GpuStress.SeedScale;
        _reference = new float[_seeds.Length];
        _hostOut = new float[_seeds.Length];
        var seeds = _seeds;
        var reference = _reference;
        Parallel.For(0, seeds.Length, i =>
        {
            float x = seeds[i];
            for (int n = 0; n < GpuStress.Iters; n++)
                x = MathF.FusedMultiplyAdd(-x, x, x) * GpuStress.R;
            reference[i] = x;
        });
        Setup();
    }

    void Setup()
    {
        // The primary context pushed current on this thread (the sequence the CUDA runtime and ffmpeg use)
        Check(_retain(out _ctx, _device), "cuDevicePrimaryCtxRetain");
        Check(_push(_ctx), "cuCtxPushCurrent");
        LoadModule();
        Check(_memAlloc(out _inDev, (UIntPtr)_bytes), "cuMemAlloc(in)");
        Check(_memAlloc(out _outDev, (UIntPtr)_bytes), "cuMemAlloc(out)");
        var h = GCHandle.Alloc(_seeds, GCHandleType.Pinned);
        try
        {
            Check(_htoD(_inDev, h.AddrOfPinnedObject(), (UIntPtr)_bytes), "cuMemcpyHtoD");
        }
        finally
        {
            h.Free();
        }
    }

    void LoadModule()
    {
        Exception? last = null;
        IntPtr nptr = Marshal.StringToHGlobalAnsi("stress");
        try
        {
            foreach (var (version, target) in new[] { ("8.0", _cc), ("6.0", "sm_50") })
            {
                IntPtr img = Marshal.StringToHGlobalAnsi(GpuStress.Ptx(version, target));
                try
                {
                    int rc = LoadDataEx(out IntPtr module, img, out string log);
                    if (rc != 0)
                    {
                        last = new InvalidOperationException($"cuModuleLoadData({target}, PTX {version}) failed ({CuErrorName(rc)}){log}");
                        continue;
                    }
                    if (_moduleGetFn(out IntPtr fn, module, nptr) is var rc2 and not 0)
                    {
                        last = new InvalidOperationException($"cuModuleGetFunction failed ({CuErrorName(rc2)})");
                        continue;
                    }
                    (_module, _fn) = (module, fn);
                    return;
                }
                finally
                {
                    Marshal.FreeHGlobal(img);
                }
            }
        }
        finally
        {
            Marshal.FreeHGlobal(nptr);
        }
        throw last ?? new InvalidOperationException("CUDA kernel not loaded");
    }

    int LoadDataEx(out IntPtr module, IntPtr image, out string log)
    {
        const int ErrLogBuffer = 5, ErrLogSize = 6; // CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        IntPtr buf = Marshal.AllocHGlobal(8192), opts = Marshal.AllocHGlobal(8), vals = Marshal.AllocHGlobal(IntPtr.Size * 2);
        try
        {
            Marshal.WriteInt32(opts, 0, ErrLogBuffer);
            Marshal.WriteInt32(opts, 4, ErrLogSize);
            Marshal.WriteIntPtr(vals, 0, buf);
            Marshal.WriteIntPtr(vals, IntPtr.Size, (IntPtr)8192);
            int rc = _moduleLoad(out module, image, 2, opts, vals);
            log = rc == 0 ? "" : " " + Marshal.PtrToStringAnsi(buf)?.Trim();
            return rc;
        }
        finally
        {
            Marshal.FreeHGlobal(buf);
            Marshal.FreeHGlobal(opts);
            Marshal.FreeHGlobal(vals);
        }
    }

    // Launches the kernel and compares every result with the reference. Bad = mismatches, Fault = CUDA error name.
    public (int Bad, string? Fault) Iterate()
    {
        IntPtr a = Marshal.AllocHGlobal(8), b = Marshal.AllocHGlobal(8), c = Marshal.AllocHGlobal(4),
                argv = Marshal.AllocHGlobal(IntPtr.Size * 3);
        int rc;
        try
        {
            Marshal.WriteInt64(a, (long)_inDev);
            Marshal.WriteInt64(b, (long)_outDev);
            Marshal.WriteInt32(c, GpuStress.Iters);
            Marshal.WriteIntPtr(argv, 0, a);
            Marshal.WriteIntPtr(argv, IntPtr.Size, b);
            Marshal.WriteIntPtr(argv, IntPtr.Size * 2, c);
            rc = _launch(_fn, (uint)_blocks, 1, 1, GpuStress.Threads, 1, 1, 0, IntPtr.Zero, argv, IntPtr.Zero);
        }
        finally
        {
            Marshal.FreeHGlobal(a);
            Marshal.FreeHGlobal(b);
            Marshal.FreeHGlobal(c);
            Marshal.FreeHGlobal(argv);
        }
        if (rc != 0)
            return (0, CuErrorName(rc));
        if ((rc = _sync()) != 0)
            return (0, CuErrorName(rc));
        var h = GCHandle.Alloc(_hostOut, GCHandleType.Pinned);
        try
        {
            rc = _dtoH(h.AddrOfPinnedObject(), _outDev, (UIntPtr)_bytes);
        }
        finally
        {
            h.Free();
        }
        if (rc != 0)
            return (0, CuErrorName(rc));
        int bad = 0;
        for (int i = 0; i < _reference.Length; i++)
            if (BitConverter.SingleToInt32Bits(_reference[i]) != BitConverter.SingleToInt32Bits(_hostOut[i]))
                bad++;
        return (bad, null);
    }

    public bool TryRecover()
    {
        Release(); // best effort: the context may already be corrupted
        for (int attempt = 0; attempt < 3; attempt++)
        {
            try
            {
                _reset(_device); // a faulty kernel leaves the primary context corrupted; reset clears it
                Setup();
                return true;
            }
            catch (InvalidOperationException)
            {
                Thread.Sleep(1000); // the driver needs a moment to come back after a reset
            }
        }
        return false;
    }

    void Release()
    {
        if (_inDev != 0)
        {
            _memFree(_inDev);
            _inDev = 0;
        }
        if (_outDev != 0)
        {
            _memFree(_outDev);
            _outDev = 0;
        }
        _module = _fn = IntPtr.Zero;
        if (_ctx != IntPtr.Zero)
        {
            _pop(out _);        // drop the context we pushed current
            _release(_device);  // release the primary context retain
            _ctx = IntPtr.Zero;
        }
    }

    public void Dispose() => Release();

    static void Check(int rc, string what)
    {
        if (rc != 0)
            throw new InvalidOperationException($"{what} ({CuErrorName(rc)})");
    }

    static string CuErrorName(int rc) =>
        _errName(rc, out var p) == 0 && p != IntPtr.Zero ? Marshal.PtrToStringAnsi(p) ?? $"error {rc}" : $"error {rc}";
}
