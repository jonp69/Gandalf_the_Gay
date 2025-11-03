<#
collect_system_info.ps1
Run from an elevated PowerShell (run as Administrator). Produces a timestamped .txt file with extensive system and Python/pip/torch info for upload.
#>

$now = Get-Date -Format "yyyy-MM-dd_HH-mm"
$out = "system_report_$now.txt"

Write-Host "Generating $out ..."
"=================================================" | Out-File $out
"System report generated: $(Get-Date)" | Out-File $out -Append
"=================================================" | Out-File $out -Append
"" | Out-File $out -Append

"`n--- Windows Version & Systeminfo ---" | Out-File $out -Append
systeminfo 2>&1 | Out-File $out -Append

"`n--- dxdiag (short) ---" | Out-File $out -Append
& dxdiag /t "$env:TEMP\dxdiag_output.txt" 2>$null
if (Test-Path "$env:TEMP\dxdiag_output.txt") {
    Get-Content "$env:TEMP\dxdiag_output.txt" | Out-File $out -Append
    Remove-Item "$env:TEMP\dxdiag_output.txt" -ErrorAction SilentlyContinue
} else {
    "dxdiag not available or failed" | Out-File $out -Append
}

"`n--- NVIDIA SMI (if present) ---" | Out-File $out -Append
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total --format=csv,noheader,nounits 2>&1 | Out-File $out -Append
    "`nFull nvidia-smi output:" | Out-File $out -Append
    nvidia-smi 2>&1 | Out-File $out -Append
} else {
    "nvidia-smi not found on PATH" | Out-File $out -Append
}

"`n--- GPU via WMI / Device Manager info ---" | Out-File $out -Append
Get-WmiObject Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID,AdapterRAM | Format-List | Out-String | Out-File $out -Append

"`n--- PCI device list (Display class) ---" | Out-File $out -Append
Get-PnpDevice -Class Display | Select-Object FriendlyName,InstanceId,Status | Format-List | Out-String | Out-File $out -Append

"`n--- CUDA toolkits (common folders) ---" | Out-File $out -Append
$cudaDir = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
if (Test-Path $cudaDir) {
    Get-ChildItem $cudaDir -Name | Out-File $out -Append
} else {
    "No CUDA installation folders found under Program Files" | Out-File $out -Append
}

"`n--- nvcc (if present) ---" | Out-File $out -Append
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    nvcc --version 2>&1 | Out-File $out -Append
} else {
    "nvcc not found on PATH" | Out-File $out -Append
}

"`n--- Python executables on PATH ---" | Out-File $out -Append
Get-Command python,python3 -ErrorAction SilentlyContinue | Out-String | Out-File $out -Append

"`n--- Python: interpreter and environment probe ---" | Out-File $out -Append
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = "python" }
$pythonProbe = @"
import sys,subprocess
print('sys.executable:', sys.executable)
print('python version:', sys.version.replace('\\n',' '))
try:
    out = subprocess.check_output([sys.executable,'-m','pip','freeze'], stderr=subprocess.STDOUT).decode(errors='ignore')
    print('--- pip freeze (first 300 lines) ---')
    for i,line in enumerate(out.splitlines()):
        if i<300:
            print(line)
        else:
            break
except Exception as e:
    print('pip freeze failed:', e)
try:
    import importlib
    if importlib.util.find_spec('torch') is not None:
        import torch
        print('torch.__version__:', torch.__version__)
        print('torch.version.cuda:', getattr(torch.version,'cuda',None))
        print('cuda available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                print('GPU count:', torch.cuda.device_count())
                print('GPU name:', torch.cuda.get_device_name(0))
            except Exception as e:
                print('torch.cuda device probe error:', e)
    else:
        print('torch not installed')
except Exception as e:
    print('torch import error:', e)
"@
$pythonProbe | Out-File "$env:TEMP\python_probe.py" -Encoding ascii
& $py "$env:TEMP\python_probe.py" 2>&1 | Out-File $out -Append
Remove-Item "$env:TEMP\python_probe.py" -ErrorAction SilentlyContinue

"`n--- pip show important packages ---" | Out-File $out -Append
$pkgs = @("torch","torchvision","xformers","opencv-python","numpy","pip","setuptools","wheel")
foreach ($p in $pkgs) {
    "---- pip show $p ----" | Out-File $out -Append
    & $py -m pip show $p 2>&1 | Out-File $out -Append
}

"`n--- Repository files (if present) ---" | Out-File $out -Append
if (Test-Path ".\requirements.txt") {
    "`n--- requirements.txt ---" | Out-File $out -Append
    Get-Content .\requirements.txt | Out-File $out -Append
} else {
    "requirements.txt not found" | Out-File $out -Append
}
if (Test-Path ".\setup.bat") {
    "`n--- setup.bat ---" | Out-File $out -Append
    Get-Content .\setup.bat | Out-File $out -Append
} else {
    "setup.bat not found" | Out-File $out -Append
}

"`n--- Disk free / drive info ---" | Out-File $out -Append
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free,Root | Out-String | Out-File $out -Append

"`n--- Running processes relevant to GPU/ML ---" | Out-File $out -Append
Get-Process python -ErrorAction SilentlyContinue | Out-String | Out-File $out -Append
Get-Process nv* -ErrorAction SilentlyContinue | Out-String | Out-File $out -Append

"`n--- Recent System Event log (error/warning) ---" | Out-File $out -Append
wevtutil qe System /q:"*[System[(Level=2 or Level=3)]]" /f:text /c:100 2>&1 | Out-File $out -Append

"`nReport complete." | Out-File $out -Append
Write-Host "Report written to $out"
Write-Host "You can now upload $out (it will be in the current directory)."