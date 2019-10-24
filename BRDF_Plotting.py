import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from test4 import model_brdf2
from test4 import theta_i_uniq
from test4 import theta_r_uniq
from test4 import phi_r_uniq
y=theta_r_uniq

def polar2cart(rho,theta,phi):
    x = rho* np.sin(theta)*np.cos(phi)
    y = rho *np.sin(theta)*np.sin(phi)
   
    return(x,y,)

x,y =polar2cart(1,theta_r_uniq,phi_r_uniq)
# X,Y = np.meshgrid(x,y)
fig = plt.figure()
ax = plt.axes(projection="3d")
z = model_brdf2(-45,0,phi_r_uniq,phi_r_uniq)
ax.plot3D(x, y, z, )
plt.show()
