import numpy as np
from EddyDiffusivity import EddyDiffusivity2D

kappa = 1.0

L = 100000.0
xpts = 500
ypts = xpts

x_vec = np.linspace(0, L, num=xpts)
y_vec = x_vec.copy()

qlevels = np.linspace(0, 100, num=100)
gs = x_vec[1] - x_vec[0]

q0 = 100.0
sigma = 30000.0
centx = L / 2.0
centy = centx

xx, yy = np.meshgrid(x_vec, y_vec, indexing='ij')
r2r2 = (xx - centx)**2.0 + (yy - centy)**2.0

# make tracer field
qfield = q0 * np.exp( - r2r2 / sigma**2.0)

# instantiate tool 
ED2D = EddyDiffusivity2D(kappa=kappa, gs=gs, qlevels=qlevels, xpts=xpts, ypts=ypts)

diff_map, diff = ED2D.calDiffusivity(qfield)
analytic = kappa * 4.0 * np.pi**2.0 * r2r2
import matplotlib.pyplot as plt

plt.subplot(131)
CS = plt.contour(x_vec, y_vec, qfield)
plt.clabel(CS, fontsize=9, inline=1)
plt.title('Tracer field')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.subplot(132)
CS = plt.contour(x_vec, y_vec, diff_map)
plt.clabel(CS, fontsize=9, inline=1)
plt.title('Diffusivity - Numerical Solution')
plt.xlabel('x [m]')

plt.subplot(133)
#CS = plt.contourf(x_vec, y_vec, (diff_map - analytic) / analytic * 100.0, np.linspace(0, 100, 20))
plt.plot(x_vec, diff_map[:,int(xpts/2)], color='r')
plt.plot(x_vec, analytic[:,int(xpts/2)], color='k', dashes=(10,3))
plt.title('Diffusivity - Crossection\nAnalytic (dashed black) and Numerical (solid red) Solution')
plt.xlabel('x [m]')
plt.ylabel(r'Eddy diffusivity [$\mathrm{m}^4 \, \mathrm{s}^{-1}$]')

plt.show()
