namespace Greenline;

// Command line front end shared by the standalone greenline.exe and host apps (e.g. as "myapp gpu ...").
// Run gets the arguments after the command prefix, with --yes/-y already removed.
public static class GpuCli
{
    // How the commands are spelled in help, prompts and the stress report ("greenline", "myapp gpu")
    public static string Prog { get; set; } = "greenline";

    // Help lines (command, description; an empty command continues the description above), aligned for any Prog
    public static string Usage(string indent = "  ") => string.Join(Environment.NewLine, new (string Cmd, string Text)[]
    {
        ("", "show NVIDIA GPU clocks and core/memory offsets (NVML)"),
        ("curve", "show the core VF curve: voltage -> clock and per-point offset (NVAPI)"),
        ("curve set <mV>:<MHz> ... [cap <mV>] [--yes]", "per-point core offsets, interpolated between anchors"),
        (null!, "and flat outside them; cap flattens the curve above <mV> (undervolt), e.g."),
        (null!, $"\"{Prog} curve set 700:+300 850:+250 cap 900\" (lost on reboot; \"{Prog} offset\" overwrites it)"),
        ("curve reset [--yes]", "set all point offsets to 0"),
        ("stress [<mV>-<mV>] [<seconds>] [<step mV>]", "pin the core clock (min = max) to each VF curve point"),
        (null!, "in range (default 600..top, 50 mV step, 30 s each) and run a self-checking CUDA"),
        (null!, "kernel there, comparing every result with a CPU reference (catches silent errors);"),
        (null!, "the clock cap and VF curve are restored afterwards; if CUDA is unavailable it"),
        (null!, "aborts before changing anything"),
        ("clock <MHz> [--yes]", "cap the GPU core clock (0 = remove the cap; lost on reboot)"),
        ("offset <core> <mem> [--yes]", "set core/memory clock offsets in MHz within the driver's range"),
        (null!, "(memory as nvidia-smi shows it; lost on reboot; UXTU/Afterburner overwrite them)"),
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
        if (a is ["curve", "set" or "reset", ..])
        {
            Gpu.CurveSpec? spec = null;
            if (a[1] == "set" && ((spec = Gpu.CurveSpec.Parse(a[2..], out var perr)) is null ? perr : Gpu.CheckCurve(info, spec)) is { } bad)
            {
                Console.Error.WriteLine(bad);
                return 1;
            }
            if (info.CoreOffset is not var (_, omin, omax))
            {
                Console.Error.WriteLine("Driver does not report the core offset range.");
                return 3;
            }
            if (!yes && !Confirm((spec is null ? "Reset all VF point offsets to 0" : $"Set the GPU VF curve: {spec}") +
                                 $" until reboot (\"{Prog} offset\" overwrites it)?"))
                return 1;
            if (Gpu.WriteCurve(spec, omin, omax) is { } werr)
            {
                Console.Error.WriteLine($"VF curve not written: {werr}.");
                return 3;
            }
            Console.WriteLine("OK.");
            return ShowCurve(info.Clock);
        }
        string prompt;
        Func<int> write;
        if (a is ["clock", var cs] && uint.TryParse(cs, out uint mhz))
        {
            if (mhz is > 0 and < 300 || mhz > info.MaxClock)
            {
                Console.Error.WriteLine($"Use 0 (no cap) or 300..{info.MaxClock} MHz.");
                return 1;
            }
            prompt = mhz == 0 ? "Remove the GPU core clock cap?" : $"Cap the GPU core clock at {mhz} MHz until reboot?";
            write = () => Gpu.SetMaxClock(mhz, info.MinSupported);
        }
        else if (a is ["offset", var c, var m] && int.TryParse(c, out int core) && int.TryParse(m, out int mem))
        {
            if (Gpu.CheckOffsets(info, core, mem) is { } bad)
            {
                Console.Error.WriteLine(bad);
                return 1;
            }
            prompt = $"Set GPU offsets core {core:+0;-0;0} MHz, memory {mem:+0;-0;0} MHz until reboot?";
            write = () => Gpu.SetOffsets(core, mem);
        }
        else
        {
            Console.WriteLine(Usage());
            return 1;
        }
        if (!yes && !Confirm(prompt))
            return 1;
        if (write() is var rc and not 0)
        {
            Console.Error.WriteLine($"NVML rejected the change (status {rc}{(rc == 4 ? ": run as Administrator" : "")}).");
            return 3;
        }
        Console.WriteLine("OK.");
        if (Gpu.Read(out _) is { } after)
            Console.WriteLine(Gpu.Describe(after));
        return 0;
    }

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
