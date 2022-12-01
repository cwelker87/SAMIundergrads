import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('/home/daniel/Research_Projects/NYCCT_CosmicFilaments/GAMA_OdeP6g.csv')    #imports CSV file as dataframe 

df_new = df.iloc[:,[0,1,2,4,5]]       #df_new = df[['CATAID','Z','RA','DEC']]  -- can use column designation name also

Z = np.array(df_new.iloc[:,[1]])
RA = np.array(df_new.iloc[:,[2]])
DEC = np.array(df_new.iloc[:,[3]])

Ho = 67.8
sl = 2.99792458e+5

numer = ((1+Z)**2)-1
denom = ((1+Z)**2)+1
H = sl/Ho
 
radius = (numer/denom)*H    # gives radius

for i in Z:
    d = radius

RArad = RA * (np.pi/180)     # converting RA and DEC into radians 
DECrad = DEC * (np.pi/180)

cos = np.cos      # easier this way rather than typing "np.sin" everytime
sin = np.sin

x = d*cos(RArad)*cos(DECrad)
y = d*sin(RArad)*cos(DECrad)
z = d*sin(DECrad)

# Plotting figure of galaxies in 3D Cartesian coordinates
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Galaxies from GAMA DS3 Survey (Z < 0.4)')
ax.scatter(x,y,z)
plt.show()