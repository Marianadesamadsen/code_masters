import numpy as np 


Lmax = 40

omax = np.sqrt(Lmax*(Lmax+1))

Tmin = 2*np.pi/omax

dt = Tmin/20

print(Tmin)

Lmax = 16

omax = np.sqrt(Lmax*(Lmax+1))

Tmin = 2*np.pi/omax

print(Tmin)


print(dt*40)

