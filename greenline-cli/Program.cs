using Greenline;

// greenline — NVIDIA GPU clock cap, core/memory offsets, per-point VF curve and stress test (NVML/NVAPI/CUDA, GPU 0).
// Reads work as any user; writes need an elevated terminal.

if (args is ["help" or "-h" or "--help"])
{
    Console.WriteLine("greenline — NVIDIA GPU tuning (NVML/NVAPI); writes need Administrator and are lost on reboot\n");
    Console.WriteLine(GpuCli.Usage());
    return 0;
}
return GpuCli.Run(args.Where(x => x is not ("--yes" or "-y")).ToArray(), args.Contains("--yes") || args.Contains("-y"));
