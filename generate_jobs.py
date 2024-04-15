import pandas as pd
import numpy as np

dfc = pd.DataFrame()

nv = 16
ms = [32,64,128,256,512]
for m in ms:
    df = pd.DataFrame()
    df['Bz'] = np.linspace(-0.6,-0.9,nv)
    df['chi'] = [m for i in range(0,nv)]
    dfc = pd.concat([dfc,df], axis=0)
# print(dfc)

dfc.to_csv('jobs.csv')