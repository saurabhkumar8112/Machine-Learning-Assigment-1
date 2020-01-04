import pandas as pd
import sys
import numpy as np
df = pd.read_csv("fmnist-pca.csv",header=None)  
# arr = np.array(df)[:100,:]
df = df[:int(sys.argv[1])]
print(df[1:5])
df.to_csv("trim2.csv")