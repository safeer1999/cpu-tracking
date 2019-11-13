import psutil as ps
import time

class Hardware:
    def __init__(self):
        self.file=open('data.csv','a')
        self.file.write('timestamp  ,   cpu ,   ram ,   disk    ,   network\n')
        self.dis=0
    def tracker(self):
        t=time.ctime(time.time())
        cpu=ps.cpu_percent()
        m=ps.virtual_memory()
        ram=((m[0]-m[1])/m[0])*100
        d=ps.disk_io_counters()
        disk=(d[2]/d[4])/100
        #network=ps.sensors_fans()
        network=ps.net_io_counters()
        network=network[1]//(1024*1024)
        self.file.write(str(t)+"    ,   "+str(cpu)+"   ,    "+str(ram)+"   ,    "+str(disk)+"   ,   "+str(network)+"\n")
    """def writer(self):
        try:
            thread.start_new_thread(self.tracker())
        except:
            print("error exception")
        while 1:
            pass"""
def main():
    start=time.time()
    obj=Hardware()
    print("**")
    count=0
    while(True):
        if(time.time()-start>=1):
            count+=1
            print(count)
            obj.tracker()
            print("*")
            start=time.time()
        if(count>=20):
            break
    obj.file.close()
if __name__ == '__main__':
    main()
