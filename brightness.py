import subprocess
import re
import os
import platform

def win_brightness(brit=10,verbose=False):
    p = subprocess.Popen(['datapull.cmd'])
    power_info=open('power_info.txt','r')
    powers=re.findall('(?<=:\s)(.*?)(?=\s\s\()', power_info.read())
    p = subprocess.Popen(["powercfg", "-SetDcValueIndex",powers[0],powers[1],powers[2],str(brit)])
    p = subprocess.Popen(["powercfg", "-S", powers[0]])
    if verbose==True:
        print("Brightness changed to",brit,"percent.")
    

def ubu_brightness(brit=10,verbose=False):
    p = subprocess.Popen(["light", "-S", str(brit)])
    if verbose==True:
        print("Brightness changed to",brit,"percent.")
    
    
def brightness(brit=10,verbose=False):
    if platform.system()=='Windows':
        win_brightness(brit,verbose)
    else:
        ubu_brightness(brit,verbose)

if __name__ == "__main__":
    brightness()