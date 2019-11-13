import psutil as ps
import time
import pandas as pd
import os

class Hardware:
    def __init__(self):
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
        
        self.frame.append([str(t),str(cpu),ram[1:],disk[5:],network])
        ddf=pd.DataFrame(self.frame)
        ddf.to_json('dynamic-data.json',orient='records')

    def helper(self,iters,save=1):
        start=time.time()
        #save=int(input("want a CSV or not?\n"))
        count=0
        while(True):
            if(time.time()-start>=1):
                count+=1
                print(count)
                self.tracker(save)
                print("*")
                start=time.time()
            if(count>=iters):
                break
        #print(pd.DataFrame(self.frame))
        if(save==1):
            df=pd.DataFrame(self.frame)
            df.columns=['timestamp','cpu','ram','disk','network']
            df['timestamp']=pd.to_datetime(df['timestamp'])
            df.to_csv('data.csv')
            df.to_json('data.json',orient='records')

        return pd.DataFrame(self.frame)



def main():

    obj=Hardware()
    obj.helper(20)
    #call dataFrameGenerator() function to get pandas dataframe object
if __name__ == '__main__':
    main()
