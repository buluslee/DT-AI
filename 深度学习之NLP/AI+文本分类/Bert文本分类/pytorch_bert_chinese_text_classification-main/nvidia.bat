@ECHO OFF

:: 執行的指令 ( 請注意若要使用 pipe 符號必須在之前加上一個 ^ 符號 )
SET ExecuteCommand=nvidia-smi

:: 單位: 秒
SET ExecutePeriod=1


SETLOCAL EnableDelayedExpansion

:loop

  cls

  echo !date! !time!
  echo 每 !ExecutePeriod! 秒執行一次，指令^: !ExecuteCommand!

  echo.

  %ExecuteCommand%
  
  timeout /t %ExecutePeriod% > nul

goto loop