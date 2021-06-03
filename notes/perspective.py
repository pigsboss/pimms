#!/usr/bin/env python

from sympy import *
import numpy as np
from pymath.common import xyz2ptr,ptr2xyz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

## Check if this is the first run of this code.
if 'S_func' not in globals():
    print('Generating the objective function...')
    phi,theta,psi=symbols('phi theta psi')
    a_x,a_y,a_z=symbols('a_x a_y a_z')
    R_a_phi=Matrix([
        [cos(phi)+a_x**2*(1-cos(phi)), a_x*a_y*(1-cos(phi))-a_z*sin(phi), a_x*a_z*(1-cos(phi))+a_y*sin(phi)],
        [a_x*a_y*(1-cos(phi))+a_z*sin(phi), cos(phi)+a_y**2*(1-cos(phi)), a_y*a_z*(1-cos(phi))-a_x*sin(phi)],
        [a_x*a_z*(1-cos(phi))-a_y*sin(phi), a_y*a_z*(1-cos(phi))+a_x*sin(phi), cos(phi)+a_z**2*(1-cos(phi))]
    ])
    R_z_phi=R_a_phi.subs(a_x,0).subs(a_y,0).subs(a_z,1)
    Y=simplify(R_z_phi*Matrix([[0],[1],[0]]))
    R_Y_theta=simplify(R_a_phi.subs(phi,theta).subs(a_x,Y[0,0]).subs(a_y,Y[1,0]).subs(a_z,Y[2,0]))
    X=simplify(R_Y_theta*R_z_phi*Matrix([[1],[0],[0]]))
    R_X_psi=R_a_phi.subs(phi,psi).subs(a_x,X[0,0]).subs(a_y,X[1,0]).subs(a_z,X[2,0])
    R=simplify(R_X_psi*R_Y_theta*R_z_phi)
    p_x,p_y,p_z=symbols('p_x p_y p_z')
    p=Matrix([[p_x],[p_y],[p_z]])
    c_x,c_y,c_z=symbols('c_x c_y c_z')
    c=Matrix([[c_x],[c_y],[c_z]])
    P=R.subs(psi,-psi)*(p-c)
    f,u,v=symbols('f u v')
    S = ((simplify(R[1,:].subs(psi,-psi)*(p-c))[0,0]/simplify(R[0,:].subs(psi,-psi)*(p-c))[0,0])*f-u)**2 + \
        ((simplify(R[2,:].subs(psi,-psi)*(p-c))[0,0]/simplify(R[0,:].subs(psi,-psi)*(p-c))[0,0])*f-v)**2
    print('Lambdifying the objective function...')
    S_func = lambdify(((p_x,p_y,p_z),(u,v),f,(phi,theta,psi),(c_x,c_y,c_z)),S,"numpy")

## {p_i} are the 4 corners of an A4 paper.
p_vals=[
    (0.0,  0.5*210.0,  0.5*297.0),
    (0.0, -0.5*210.0,  0.5*297.0),
    (0.0,  0.5*210.0, -0.5*297.0),
    (0.0, -0.5*210.0, -0.5*297.0)
]
## {q_i} are the 4 corners of the image of the A4 paper above.
## the image is 100 times smaller than the object itself.
q_vals=[
    ( 0.5*210.0/100.0,  0.5*297.0/100.0),
    (-0.5*210.0/100.0,  0.5*297.0/100.0),
    ( 0.5*210.0/100.0, -0.5*297.0/100.0),
    (-0.5*210.0/100.0, -0.5*297.0/100.0)
]
f_val=27.0 ## HUAWEI's Mate 10 Pro
## the unknown true parameter.
## the camera (a HUAWEI's Mate 10 Pro smartphone) is placed at (-2700mm, 0mm, 0mm) of the WCS.
## its optical axis is pointed at the origin of the WCS, which is also the centre of the A4 paper.
## according to the focal length its lens the zoom factor is 1/100.
omega=(0., 0., 0., -100.*f_val, 0., 0.)
## define a neighborhood of omega
c_x_vals   = -100.0*f_val*np.arange( 0.900, 1.100, 0.001)
c_y_vals   =  100.0*f_val*np.arange(-0.075, 0.075, 0.001)
c_z_vals   =  100.0*f_val*np.arange(-0.075, 0.075, 0.001)
phi_vals   =  0.01*np.arange(-np.pi, np.pi, 0.05)
theta_vals =  0.01*np.arange(-np.pi, np.pi, 0.05)
psi_vals   =  0.05*np.arange(-np.pi, np.pi, 0.05)

if not os.path.exists('figures/gradient-xyz.png'):
    c_x_grid, c_y_grid, c_z_grid = np.meshgrid(c_x_vals, c_y_vals, c_z_vals, indexing='ij')
    phi_grid, theta_grid, _ = xyz2ptr(-c_x_grid, -c_y_grid, c_z_grid)
    m = np.zeros_like(c_y_grid)
    for i in range(len(p_vals)):
        m += S_func(p_vals[i], q_vals[i], f_val, (phi_grid, theta_grid, 0), (c_x_grid, c_y_grid, c_z_grid))
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(131,projection='3d')
    ax.plot_surface(c_x_grid[:,:,75]/100/f_val, c_y_grid[:,:,75]/10/f_val, m[:,:,75]**0.2, cmap='coolwarm')
    ax.set_xlabel(r'$c_x$')
    ax.set_ylabel(r'$c_y$')
    ax = fig.add_subplot(132,projection='3d')
    ax.plot_surface(c_x_grid[:,75,:]/100/f_val, c_z_grid[:,75,:]/10/f_val, m[:,75,:]**0.2, cmap='coolwarm')
    ax.set_xlabel(r'$c_x$')
    ax.set_ylabel(r'$c_z$')
    ax = fig.add_subplot(133,projection='3d')
    ax.plot_surface(c_y_grid[50,:,:]/10/f_val,  c_z_grid[50,:,:]/10/f_val, m[50,:,:]**0.2, cmap='coolwarm')
    ax.set_xlabel(r'$c_y$')
    ax.set_ylabel(r'$c_z$')
    plt.savefig('figures/gradient-xyz.png',dpi='figure')
    plt.close()

if not os.path.exists('figures/gradient-angles.png'):
    phi_grid, theta_grid, psi_grid = np.meshgrid(phi_vals, theta_vals, psi_vals, indexing='ij')
    m = np.zeros_like(phi_grid)
    for i in range(len(p_vals)):
        m += S_func(p_vals[i], q_vals[i], f_val, (phi_grid, theta_grid, psi_grid), (-100.0*f_val, 0, 0))
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(131,projection='3d')
    ax.plot_surface(phi_grid[:,:,63], theta_grid[:,:,63], m[:,:,63]**0.2, cmap='coolwarm')
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')
    ax = fig.add_subplot(132,projection='3d')
    ax.plot_surface(phi_grid[:,63,:], psi_grid[:,63,:], m[:,63,:]**0.2, cmap='coolwarm')
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\psi$')
    ax = fig.add_subplot(133,projection='3d')
    ax.plot_surface(theta_grid[63,:,:], psi_grid[63,:,:], m[63,:,:]**0.2, cmap='coolwarm')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\psi$')
    plt.savefig('figures/gradient-angles.png',dpi='figure')
    plt.close()

if 'HS_func' not in globals():
    print('Generating gradient vector of the objective function...')
    GS = Matrix([
        diff(S,   phi),
        diff(S, theta),
        diff(S,   psi),
        diff(S,   c_x),
        diff(S,   c_y),
        diff(S,   c_z)
    ]) # Gradient of objective function S.
    print('Lambdifying gradient vector of the objective function...')
    GS_func = lambdify(((p_x,p_y,p_z),(u,v),f,(phi,theta,psi),(c_x,c_y,c_z)),GS,"numpy")
    print('Generating Hessian matrix of the objective function...')
    HS = Matrix([
        [diff(diff(S,   phi), phi), diff(diff(S,    phi), theta), diff(diff(S,    phi), psi), diff(diff(S,    phi), c_x), diff(diff(S,    phi), c_y), diff(diff(S,    phi), c_z)],
        [diff(diff(S, theta), phi), diff(diff(S,  theta), theta), diff(diff(S,  theta), psi), diff(diff(S,  theta), c_x), diff(diff(S,  theta), c_y), diff(diff(S,  theta), c_z)],
        [diff(diff(S,   psi), phi), diff(diff(S,    psi), theta), diff(diff(S,    psi), psi), diff(diff(S,    psi), c_x), diff(diff(S,    psi), c_y), diff(diff(S,    psi), c_z)],
        [diff(diff(S,   c_x), phi), diff(diff(S,    c_x), theta), diff(diff(S,    c_x), psi), diff(diff(S,    c_x), c_x), diff(diff(S,    c_x), c_y), diff(diff(S,    c_x), c_z)],
        [diff(diff(S,   c_y), phi), diff(diff(S,    c_y), theta), diff(diff(S,    c_y), psi), diff(diff(S,    c_y), c_x), diff(diff(S,    c_y), c_y), diff(diff(S,    c_y), c_z)],
        [diff(diff(S,   c_z), phi), diff(diff(S,    c_z), theta), diff(diff(S,    c_z), psi), diff(diff(S,    c_z), c_x), diff(diff(S,    c_z), c_y), diff(diff(S,    c_z), c_z)]
    ]) # Hessian of objective function S.
    print('Lambdifying Hessian matrix of the objective function...')
    HS_func = lambdify(((p_x,p_y,p_z),(u,v),f,(phi,theta,psi),(c_x,c_y,c_z)),HS,"numpy")

K=4
omega_init = np.double([
      phi_vals[np.int64(np.size(  phi_vals)*np.random.rand(K))],
    theta_vals[np.int64(np.size(theta_vals)*np.random.rand(K))],
      psi_vals[np.int64(np.size(  psi_vals)*np.random.rand(K))],
      c_x_vals[np.int64(np.size(  c_x_vals)*np.random.rand(K))],
      c_y_vals[np.int64(np.size(  c_y_vals)*np.random.rand(K))],
      c_z_vals[np.int64(np.size(  c_z_vals)*np.random.rand(K))]
])
omega_prev = np.copy(omega_init)
omega_next = np.zeros_like(omega_init)
d = 1
s = np.ones((2,K))
a = np.zeros(K)
g = np.zeros((6,K))
h = np.zeros((6,6,K))
t = np.zeros(K)
r = []
while len(r) < 1000 and np.min(s) > 1e-5:
    l = 0
    while l < 100:
        s[0,:] = s[1,:]
        s[1,:] = 0.
        a[:] = 0.
        g[:] = 0.
        h[:] = 0.
        for k in range(K):
            for i in range(len(p_vals)):
                g[:,k]   += np.squeeze(GS_func(p_vals[i], q_vals[i], f_val, omega_prev[:3, k], omega_prev[3:, k]))
                h[:,:,k] += np.squeeze(HS_func(p_vals[i], q_vals[i], f_val, omega_prev[:3, k], omega_prev[3:, k]))
                s[1,k]   +=             S_func(p_vals[i], q_vals[i], f_val, omega_prev[:3, k], omega_prev[3:, k])
            a[k] = np.sum(g[:,k]**2.0)/np.sum(np.matmul(g[:,k].reshape((1,6)), np.matmul(h[:,:,k], g[:,k]).T))
            if np.abs((s[0,k]-s[1,k]) / s[1,k]) < 1e-2:
                if t[k]<5:
                    t[k] += 1
                else:
                    print('JUMP')
                    a[k] = a[k] * np.random.rand(1)*2.*t[k]
                    g[:,k] = g[:,k] + 2.*(np.random.rand(6)-0.5)*np.sum(g[:,k]**2.0)**0.5
                    t[k] = 0.
            else:
                t[k] = 0.
            omega_next[:, k] = omega_prev[:, k] - a[k] * g[:,k]
            d = np.sum((omega_next - omega_prev)**2.0)
        for k in range(K):
            omega_prev[:, k] = omega_next[:, k]
        l += 1
        print('Outer loop: %d, Inner loop: %d, Residual: %e, Objective Function: %e, Best: %d'%(len(r), l, d, np.min(s), np.argmin(s[1,:])))
        r += (np.min(s),)
    idx = np.argwhere(s[1,:]>np.median(s[1,:])).squeeze()
    omega_prev[:, idx] = omega_prev[:, idx] + a[idx] * np.sum(g[:, idx]**2.0, axis=0)**0.5*np.random.rand(6,len(idx))
