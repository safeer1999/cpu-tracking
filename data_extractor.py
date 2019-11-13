import psutil as ps
import time
import pandas as pd
class Hardware:
    def __init__(self):
        self.file=None
        #self.net=ps.net_io_counters()[1]-ps.net_io_counters()[0]
        self.frame=[]
        #self.file.write('timestamp  ,   cpu ,   ram ,   disk    ,   network\n')
    def tracker(self,save):
        t=time.ctime(time.time())
        cpu=ps.cpu_percent()
        m=ps.virtual_memory()
        ram=m[2]*10
        ram=str(ram)
        d=ps.disk_io_counters()
        disk=d[1]-d[0]
        #network=ps.sensors_fans()
        networks=ps.net_io_counters()
        network=networks[1]-networks[0]
        network=str(network)
        network=network[len(network)-2:]
        self.net=networks[1]-networks[0]
        disk=str(disk)
        if(save==1):
            self.file.write(str(t)+"    ,   "+str(cpu)+"   ,    "+ram[1:]+"   ,    "+disk[5:]+"   ,   "+network+"\n")
            self.frame.append([str(t),str(cpu),ram[1:],disk[5:],network])
        else:
            self.frame.append([str(t),str(cpu),ram[1:],disk[5:],network])
    def frameReturn(self):
        #print("**")
        print(pd.DataFrame(self.frame))
        return pd.DataFrame(self.frame,columns=['timestamp','cpu','ram','disk','network'])

    def helper(self,save=1):
        start=time.time()
        #save=int(input("want a CSV or not?\n"))
        if(save==1):
            self.file=open('data.csv','a')
            self.file.write('timestamp  ,   cpu ,   ram ,   disk    ,   network\n')
        count=0
        while(True):
            if(time.time()-start>=1):
                count+=1
                print(count)
                self.tracker(save)
                print("*")
                start=time.time()
            if(count>=20):
                break
        if(save==1):
            self.file.close()
        #print(pd.DataFrame(self.frame))
        return pd.DataFrame(self.frame,columns=['timestamp','cpu','ram','disk','network'])



def main():

    obj=Hardware()
    obj.helper()
    #call dataFrameGenerator() function to get pandas dataframe object
if __name__ == '__main__':
    main()
