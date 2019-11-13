@echo off
(powercfg -q | find "Power Scheme GUID" & powercfg -q | find "(Display)" & powercfg -q | find "(Display brightness)") > power_info.txt