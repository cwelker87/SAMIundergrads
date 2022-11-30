#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import pearsonr
from scipy.io import readsav
from glob import glob
from tqdm import tqdm
from scipy.io import readsav
from sklearn.preprocessing import normalize
from matplotlib.patches import Circle
import array as arr


# In[2]:


#Test Matrix
a= np.matrix('1 2 3; 3 4 8')
print (a)


# In[3]:


#TEST Plotting in 3D
x=np.random.rand(100)
y=np.random.rand(100)
z=np.random.rand(100)
fig = plt.figure(figsize=(8,8))#scale of graph
ax = fig.add_subplot(111,
                     projection='3d')#makes the graph 3d
ax.scatter(x, y, z,#Adds axis to plot points
           linewidths=1, alpha=.7,
           edgecolor='blue',#changes the color of borders of points
           s = 100,#changes size of points
           c=x)
plt.show()


# # First Attempt
# * I used list instead of arrays
# * If statments are harder to make with list 

# # Second Attempt
# * This time we will use arrays instead of list

# ###### here are just some test to see what i can and cant do with defining an array 
# test=np.array([1,2,3])
# print(test)
# test2= np.array([input(), 2,3])
# print (test2)

# In[55]:


#User Input Points
pntA= np.array([int(input("Input Ax:")),int(input("Input Ay:")),int(input("Input Az:"))])  
pntB= np.array([int(input("Input Bx:")),int(input("Input By:")),int(input("Input Bz:"))])
pntG= np.array([int(input("Input Gx:")),int(input("Input Gy:")),int(input("Input Gz:"))]) 
print(pntA,pntB,pntG)


# In[56]:


#calculating vectors
AB=pntB-pntA
AG=pntG-pntA
BG=pntG-pntB
print("AB:",AB,"AG:",AG,"BG:",BG)

#Creating Unit vectors using (np.linalg.norm) to normalize them
normAB =np.linalg.norm(AB)
U1=AB/normAB #Normalized vector
print("||AB||:",normAB)
print("U1:",U1)
#Calculating the other Unit Vectors 
U3=np.cross(AB,AG) / np.linalg.norm(np.cross(AB,AG))
U2= np.cross(U1,U3)/ np.linalg.norm(np.cross(U1,U3))
print("U2:",U2)
print("U3:",U3)


# In[57]:


#Creating if statment to calculate d 
if all(AG*U1)>0 and all(AG*U1)<normAB:#Case1
    d=abs(np.cross(pntA-pntB,pntA-pntG))/abs(pntG-pntB)
if all(AG*U1)<0:#Case2
    d=np.linalg.norm(AG)
if all(AG*U1)>(normAB):#Case3
    d=np.linalg.norm(BG)
print(d)


# In[58]:


#Plotting ther point and segments
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111,
                     projection='3d')
plt.plot(AB,d,color='palevioletred')
plt.show()


# # Creating an 100 galaxy mock catalog 
# * Now I will create a much larger catalog to see how the code runs over a loop

# In[4]:


#100 point catalog
pntG=np.random.rand(100,3)
print(pntG.shape)
print(pntG)


# In[5]:


#Calling the columns as variables 
Gx=pntG[...,0]
Gy=pntG[...,1]
Gz=pntG[...,2]


# In[6]:


#Polotting Test 
ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
for i in range (0,100):
    ax.scatter(Gx,Gy,Gz,c='lightblue',alpha=.8,edgecolor='blue')
plt.show()


# In[7]:


#Segments 
s1=np.array([0,0,.2])
s2=np.array([.3,.5,.5])
s3=np.array([.80,.90,.80])
s4= np.array([1,.7,.9])

print(s1,s2,s3,s4)

#Making Vectors
s12=s2-s1
s23=s3-s2
s34=s4-s3
print(s12,s23,s34)

#Calculating vectors
norm12 =np.linalg.norm(s12)
norm23 =np.linalg.norm(s23)
norm34 =np.linalg.norm(s34)
u1=s12/norm12
u2=s23/norm23
u3=s34/norm34


# In[36]:


#vector from galaxy to segment edge 

for i in range (100): #Picking each row as a galaxy 
    Sg1=pntG[i]-s1
    Sg2=pntG[i]-s2
    Sg3=pntG[i]-s3
    Sg4=pntG[i]-s4
    print(Sg1)


# In[17]:


#Plotting Galaxies and Fillaments
ax = plt.figure(figsize=(7,7)).add_subplot(projection='3d',facecolor='w')
for i in range (0,100):
    ax.scatter(Gx[i],Gy[i],Gz[i],s=80,cmap='viridis',alpha=.8,edgecolor='black')
    ax.plot(s1,s2,s12,color='k')#So im not sure why they are bent like that but i'm very close
    ax.plot(s2,s3,s23,color='red')
    ax.plot(s3,s4,s34,color='blue')
ax.set_title("Map of Test Galaxies and Fillaments",size=20)
plt.show()


# In[38]:


#Looping and calculating distances

dint=1000

    #First segment is s12 where s 1 is our A and s2 is our B
for i in Sg1:
    if all(Sg1[i]*u1)>0 and all(Sg1[i]*u1)<(norm12):
        d=abs(np.cross(s1-s2,s1-G[i])/abs(G[i]-s2))
    if d < dint:
            dint=d 

    if all(Sg1[i]*u1)<0:
        d=np.linalg.norm(Sg1[i])
    if d < dint:
            dint=d 

    if all(Sg1[i]*u1)>(norm12):
        d=np.linalg.norm(Sg2[i])
    if d < dint:
            dint=d 
#Second Segment is s23 where s2 is our new A and s3 is our B
for i in Sg2:
    if all(Sg2[i]*u2)>0 and all(Sg2[i]*u2)<(norm23):
        d=abs(np.cross(s2-s3,s2-G[i])/abs(G[i]-s3))
    if d < dint:
            dint=d 

    if all(Sg2[i]*u2)<0:
        d=np.linalg.norm(Sg2[i])
    if d < dint:
            dint=d 

    if all(Sg2[i]*u2)>(norm23):
        d=np.linalg.norm(Sg3[i])
    if d < dint:
            dint=d 

#Third Segment is s34 where s3 is our new A and s4 is our B
for i in Sg3:
    if all (Sg3[i]*u3)>0 and all(Sg3*u3)<(norm34):
        d=abs(np.cross(s3-s4,s3-G[i])/abs(G[i]-s4))
    if d < dint:
            dint=d 

    if all(Sg3[i]*u3)<0:
         d=np.linalg.norm(SG3[i])
    if d < dint:
            dint=d 
    
    if all(Sg3[i]*u3)>(norm34):
        d=np.linalg.norm(Sg4[i])
    if d < dint:
            dint=d 

