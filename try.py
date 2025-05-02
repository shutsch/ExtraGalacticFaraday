import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt


phi=np.random.uniform(0.0*np.pi,2.0*np.pi,14500)
ra_deg=phi*180.0/np.pi

p_low= (np.sin(0.0*180.0/np.pi)+1.0)/2.0
p_high= (np.sin(-90.0*180.0/np.pi)+1.0)/2.0
p=np.random.uniform(p_low,p_high,14500)


sin_delta=2*p-1
delta=np.arcsin(sin_delta)

fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111, projection="mollweide")
       
ax.scatter(phi-np.pi, delta, s=1)

# plt.show()     
plt.savefig('survey.png')
