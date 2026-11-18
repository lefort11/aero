@echo off
setlocal
for %%D in (libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll libgomp-1.dll) do (
    if not "%%~$PATH:D"=="" (
        xcopy /D /Y /Q "%%~$PATH:D" "%~1\" >nul
        if errorlevel 1 exit /b 1
    )
)
exit /b 0
