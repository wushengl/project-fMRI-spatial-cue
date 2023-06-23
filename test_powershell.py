import subprocess

log_file_path = r'logs\ild-fmri-2_2023-05-06.log'
command = f'start powershell.exe -Command "Get-Content -Path "{log_file_path}" -Wait'
subprocess.Popen(command, shell=True)

#subprocess.Popen(['start','powershell.exe','-Command',f'Get-Content -Path "{log_file_path}" -Wait'])

#Get-Content -Path 'logs\ild-fmri-2_2023-05-02.log' -wait
