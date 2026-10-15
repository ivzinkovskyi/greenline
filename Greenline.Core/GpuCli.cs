namespace Greenline;

// Command line front end shared by the standalone greenline.exe and host apps (e.g. as "myapp gpu ...").
// Run gets the arguments after the command prefix, with --yes/-y already removed. One command per setting: the core
// (global offset or VF curve) and the max voltage are written together as one VF state, so each keeps the other.
public static class GpuCli
{
    // How the commands are spelled in help, prompts and the stress report ("greenline", "myapp gpu")
    public static string Prog { get; set; } = "greenline";

    // Help lines (command, description; an empty command continues the description above), aligned for any Prog
    public static string Usage(string indent = "  ") => string.Join(Environment.NewLine, new (string Cmd, string Text)[]
    {
        ("", "show NVIDIA GPU clocks, offsets and max voltage"),
        ("maxclock <MHz>|off [--yes]", "cap the GPU core clock (lost on reboot)"),
        ("maxvoltage <mV>|off [--yes]", "flatten the core VF curve above <mV> (undervolt); kept by \"core\" and"),
        (null!, "\"curve set\" (lost on reboot)"),
        ("core <MHz> [--yes]", "global core clock offset (whole VF curve shift; replaces a per-point curve)"),
        ("mem <MHz> [--yes]", "memory clock offset, as nvidia-smi shows it (keeps the core curve)"),
        (null!, "offsets are limited to the driver's range, lost on reboot; UXTU/Afterburner overwrite them"),
        ("curve", "show the core VF curve: voltage -> clock and per-point offset (NVAPI)"),
        ("curve set <mV>:<MHz> ... [--yes]", "per-point core offsets, interpolated between anchors and flat"),
        (null!, $"outside them, e.g. \"{Prog} curve set 700:+300 850:+250\" (lost on reboot)"),
        ("curve reset [--yes]", "set all point offsets to 0 (also removes the max voltage)"),
        ("stress [<mV>-<mV>] [<seconds>] [<step mV>]", "pin the core clock (min = max) to each VF curve point"),
        (null!, "in range (default 600..top, 50 mV step, 30 s each) and run a self-checking CUDA"),
        (null!, "kernel there, comparing every result with a CPU reference (catches silent errors);"),
        (null!, "the clock cap and VF curve are restored afterwards; if CUDA is unavailable it"),
        (null!, "aborts before changing anything"),
    }.Select(l => indent + (l.Cmd is null ? "" : $"{Prog} {l.Cmd}".TrimEnd()) switch
    {
        var c when c.Length < 30 => c.PadRight(30) + l.Text,
        var c => c + "  " + l.Text,
    }));

    public static int Run(string[] a, bool yes)
    {
        if (a is ["help" or "-h" or "--help"])
        {
            Console.WriteLine(Usage());
            return 0;
        }
        if (Gpu.Read(out var err) is not { } info)
        {
            Console.Error.WriteLine($"GPU unavailable: {err}.");
            return 3;
        }
        Console.WriteLine(Gpu.Describe(info));
        if (a.Length == 0)
            return 0;
        if (a is ["stress", ..])
            return GpuStress.Run(a[1..], yes);
        if (a is ["curve"])
            return ShowCurve(info.Clock);
        string? bad = null, prompt = null;
        Func<string?>? write = null;
        var core = Gpu.ReadCore(out var coreErr);
        switch (a)
        {
            case ["maxclock", var s]:
                uint cap = 0;
                if (s != "off" && !(uint.TryParse(s, out cap) && cap >= 300 && cap <= info.MaxClock))
                    bad = $"Use off or 300..{info.MaxClock} MHz.";
                else
                {
                    prompt = cap == 0 ? "Remove the GPU core clock cap?" : $"Cap the GPU core clock at {cap} MHz until reboot?";
                    write = () => Nvml(Gpu.SetMaxClock(cap, info.MinSupported));
                }
                break;
            case ["maxvoltage", var s]:
                int? mv = s == "off" ? null : int.TryParse(s, out int v) && v is >= Gpu.MinCapMv and <= Gpu.MaxCapMv ? v : -1;
                if (mv == -1)
                    bad = $"Use off or {Gpu.MinCapMv}..{Gpu.MaxCapMv} mV.";
                else if (core is null)
                    bad = $"Cannot keep the current core offsets ({coreErr}): set them first with \"{Prog} core\" or \"{Prog} curve set\".";
                else
                    (prompt, write) = Core(info, core with { CapMv = mv });
                break;
            case ["core", var s]:
                if (!int.TryParse(s, out int k))
                    bad = "Give the core offset in MHz.";
                else
                    (prompt, write) = Core(info, Gpu.CurveSpec.Global(k, core?.CapMv));
                break;
            case ["mem", var s]:
                if (!int.TryParse(s, out int m))
                    bad = "Give the memory offset in MHz.";
                else if ((bad = Gpu.CheckOffsets(info, 0, m)) is null)
                {
                    prompt = $"Set the GPU memory offset to {m:+0;-0;0} MHz until reboot?";
                    write = () => Nvml(Gpu.SetOffsets(null, m));
                }
                break;
            case ["curve", "set", .. var t]:
                if (Gpu.CurveSpec.Parse(t, out var perr) is not { } spec)
                    bad = perr;
                else if (spec.CapMv is not null)
                    bad = $"The max voltage is its own command: \"{Prog} maxvoltage <mV>\".";
                else
                    (prompt, write) = Core(info, spec with { CapMv = core?.CapMv });
                break;
            case ["curve", "reset"]:
                (prompt, write) = Core(info, Gpu.CurveSpec.Global(0), "Reset all VF point offsets to 0 (and remove the max voltage)?");
                break;
            default:
                Console.WriteLine(Usage());
                return 1;
        }
        if (bad is not null)
        {
            Console.Error.WriteLine(bad);
            return 1;
        }
        if (write is null || prompt is null)
            return 1;
        if (!yes && !Confirm(prompt))
            return 1;
        if (write() is { } werr)
        {
            Console.Error.WriteLine($"Not written: {werr}.");
            return 3;
        }
        Console.WriteLine("OK.");
        if (Gpu.Read(out _) is { } after)
            Console.WriteLine(Gpu.Describe(after));
        return 0;
    }

    // Prompt and write for a core state, checked against the driver's offset range (an invalid one returns a null write)
    static (string?, Func<string?>?) Core(Gpu.Info info, Gpu.CurveSpec spec, string? prompt = null)
    {
        if ((spec.GlobalOffset is { } k ? Gpu.CheckOffsets(info, k, 0) : Gpu.CheckCurve(info, spec)) is { } bad)
        {
            Console.Error.WriteLine(bad);
            return (null, null);
        }
        // A max voltage leak is shown by Describe after the write
        return (prompt ?? $"Set the GPU {spec.Text} until reboot?", () => Gpu.WriteCore(info, spec) is Gpu.Result.Failure f ? f.Error : null);
    }

    static string? Nvml(int rc) => rc == 0 ? null : $"NVML status {rc}{(rc == 4 ? ": run as Administrator" : "")}";

    static int ShowCurve(uint clock)
    {
        if (Gpu.ReadCurve(out var err) is not var (pts, mv))
        {
            Console.Error.WriteLine($"VF curve unavailable: {err}.");
            return 3;
        }
        Console.WriteLine(Gpu.DescribeCurve(pts, mv, clock));
        return 0;
    }

    static bool Confirm(string prompt)
    {
        Console.Write(prompt + " [y/N] ");
        if (string.Equals(Console.ReadLine()?.Trim(), "y", StringComparison.OrdinalIgnoreCase))
            return true;
        Console.WriteLine("Aborted.");
        return false;
    }
}
