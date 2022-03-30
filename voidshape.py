#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:45:55 2022

@author: georg

todo:
    - separate plotting and calculation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from skimage.measure import EllipseModel

import gudhi

from tqdm import tqdm

def main():
    
    # some settings for plotting
    color_dict = {0: 'r', 1: 'g',2: 'b', 3: 'xkcd:goldenrod'}
    
    density_slice = np.load('./data/density_files/densityslice_LCDM_15_007.npy')
    density_slice = np.log10(density_slice)
    
    N = 256**2
    # work in pixel units
    L = np.array([256,256])
    box_L = np.array([300,300])
    periodic_dimensions = (True,True)
    
    # check whether field is non-degenerate
    # otherwise critical points might not be unique
    # and can not be identified
    if not len(set(density_slice.reshape(N))) == len(density_slice.reshape(N)):
        # if not unique, pertube
        # old version, improve to just perturb where needed
        sort_indices = np.argsort(density_slice.reshape(N))
        differences = np.diff(density_slice.reshape(N)[sort_indices])
        min_diff = np.min(differences[differences>0])
        density_slice.reshape(N)[sort_indices[np.where(differences == 0)]] += min_diff/2
    
    # add visualisation
    # plt.matshow(density_slice)
    
    # calculate persitence:
    # cubical complex
    p_cubical_field = gudhi.PeriodicCubicalComplex(dimensions=density_slice.shape,
                                                   top_dimensional_cells=-density_slice.flatten(),
                                                   periodic_dimensions=periodic_dimensions)
    # sub-level persistence
    persistence_cubical = p_cubical_field.persistence()
    # invert back to super-level persitence
    pers_inverted = [ (pi[0], (-pi[1][1],-pi[1][0])) for pi in persistence_cubical]
    # get the occuring dimensions
    dimensions = set([p[0] for p in pers_inverted])
    # store peristence points as array instead of list of tuples
    pers_array = np.array([ [int(p[0]),p[1][0],p[1][1]] for p in pers_inverted for d in dimensions if p[0]==d ])
    pers_list = [np.array(pers_array[pers_array[:,0]==d,:]) for d in dimensions]
    # sort for descending persistences
    pers_list = [ pl[np.argsort(pl[:,2]-pl[:,1])[::-1],:] for pl in pers_list]
    
    fig, ax = plt.subplots(1,1)
    for dim in [0,1,2]:
        ax.scatter(pers_list[dim][:,1],pers_list[dim][:,2],
                   c=color_dict[dim],label=r'Dimension %i'%dim)
    ax.legend()
    ax.set_xlabel(r'Death density $\log(\delta+1)$')
    ax.set_xlabel(r'Birth density $\log(\delta+1)$')
    

    # get void locations:
    # death of 1-cycle in decreasing superlevel , ie column 1
    
    # max number of voids to use
    N = 500
    # calculate, and plot lines between void centre -> void saddle
    plt.matshow(density_slice)
    void_list = []
    for birth,death in zip(pers_list[1][:N,1],pers_list[1][:N,2]):
        if birth > -np.inf:
            loc_void = np.where(density_slice == birth)
            loc_saddle = np.where(density_slice == death)
            plt.scatter(loc_void[1],loc_void[0],marker='o',s=10,c='r')
            plt.scatter(loc_saddle[1],loc_saddle[0],marker='x',s=10,c='g')
            plt.plot([loc_void[1],loc_saddle[1]],[loc_void[0],loc_saddle[0]])
            distance_temp = np.array([loc_saddle[0][0],loc_saddle[1][0]]) - np.array([loc_void[0][0],loc_void[1][0]])
            distance_temp = np.sum(np.min([np.abs(distance_temp),np.abs(distance_temp-L)],0)**2)**0.5
            void_list.append([loc_void[1][0],loc_void[0][0],death-birth,distance_temp])
    
    # void_array = np.array(void_list)
    
    # fill voids
    
    L1,L2 = density_slice.shape
    # select voids from the sorted list of persistence values
    # void 0 and 1 do not work
    N_select_list = np.mgrid[2:200]#np.mgrid[30]
    
    # fraction of (density_saddle-density_centre) upto which to fill
    # this determines the interior cells of the void
    threshold_fraction = 0.85
    
    void_ellipse_list = []
    ellipse_params_list = []
    void_boundary_point_list = []
    
    # separate plotting and calculation
    fig, ax = plt.subplots(1,3,figsize=(10,5), sharex=True,sharey=True,
                       gridspec_kw = {'wspace':0.05, 'hspace':0.0})
    mask_all = np.zeros(shape=density_slice.shape).astype(int)
    ax[0].matshow(density_slice)
    ax[1].matshow(density_slice*mask_all)
    # ax[2].contour(mask)
    for N_select in tqdm(N_select_list):
        mask = np.zeros(shape=density_slice.shape).astype(int)
        birth,death=pers_list[1][N_select,1],pers_list[1][N_select,2]
        if birth > -np.inf:
            loc_void = np.where(density_slice == birth)
            loc_saddle = np.where(density_slice == death)
            ax[0].scatter(loc_void[1],loc_void[0],marker='o',s=10,c='r')
            ax[0].scatter(loc_saddle[1],loc_saddle[0],marker='x',s=10,c='g')
            ax[0].plot([loc_void[1],loc_saddle[1]],[loc_void[0],loc_saddle[0]])
            distance_temp = np.array([loc_saddle[0][0],loc_saddle[1][0]]) - np.array([loc_void[0][0],loc_void[1][0]])
            distance_temp = np.sum(np.min([np.abs(distance_temp),np.abs(distance_temp-L)],0)**2)**0.5
            void_ellipse_list.append([loc_void[1][0],loc_void[0][0],loc_saddle[1][0],loc_saddle[0][0],death-birth,distance_temp])
        mask[loc_void[0],loc_void[1]]=1
        ax[1].matshow(density_slice*mask)
        frame = max(np.abs(loc_saddle[0]-loc_void[0]),np.abs(loc_saddle[1]-loc_void[1]))*2
        ax[0].set_xlim(loc_void[1]-frame, loc_void[1]+frame)
        ax[0].set_ylim(loc_void[0]-frame, loc_void[0]+frame)
        ax[1].set_xlim(loc_void[1]-frame, loc_void[1]+frame)
        ax[1].set_ylim(loc_void[0]-frame, loc_void[0]+frame)
        ax[2].set_xlim(loc_void[1]-frame, loc_void[1]+frame)
        ax[2].set_ylim(loc_void[0]-frame, loc_void[0]+frame)
        change = True
        threshold = density_slice[loc_saddle]-(1-threshold_fraction)*np.abs(density_slice[loc_saddle] - density_slice[loc_void])
        ax[1].set_title(threshold[0])
        positions = [loc_void]
        neighbours = [(0,1),(0,-1),(1,0),(-1,0)]#(1,1),(1,-1),(-1,-1),(-1,1)]
        while change:
            new_pos = positions.copy()
            for pos in new_pos:
                for n in neighbours:
                    # if density_slice[pos[0]+n[0],pos[1]+n[1]] < threshold and mask[pos[0]+n[0],pos[1]+n[1]] == 0:
                    if (np.roll(np.roll(density_slice,-n[0],axis=0),-n[1],axis=1)[pos] <= threshold and 
                        np.roll(np.roll(mask,-n[0],axis=0),-n[1],axis=1)[pos] == 0):
                        
                        new_pos.append(( (pos[0]+n[0])%L1,(pos[1]+n[1])%L2 ))
                        mask[ (pos[0]+n[0])%L1,(pos[1]+n[1])%L2 ] += 1
                        # print("add")
            positions += new_pos
            new_pos = []
            if len(new_pos) == 0:
                change = False
            ax[1].matshow(mask)
            # ax[2].contour(mask*density_slice)
        mask_all |= mask
        pos_array = np.array([[p[0][0]-loc_void[0][0], p[1][0]-loc_void[1][0] ] for p in positions])
        # center the mask and field, so that contours don't cut the boundary:
        # shift = pos_array.min(0)*(np.abs(pos_array.min(0))>np.abs(pos_array.max(0)))//2
        shift = [L1//2 - loc_void[0][0],L2//2 - loc_void[1][0]]
        # contour_field_1 = mask*density_slice
        contour_field = np.roll(np.roll(mask*density_slice,shift[0],axis=0),shift[1],axis=1)
        contours = measure.find_contours(contour_field, threshold)
        lower_threshold = 0
        while len(contours) == 0:
            contours = measure.find_contours(contour_field, threshold-abs(threshold)*lower_threshold)
            lower_threshold += 0.01
        contours = [ c-shift for c in contours]
        # select longest contour (assuming this is the one surrounging everything ??)
        contour = [ c  for c in contours  if len(c) == np.max([ len(c) for c in contours ])][0]
        ax[2].plot(contour[:,1],contour[:,0])
        # xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, 25),
        #                                 params=(10, 15, 4, 8, np.deg2rad(30)))
        ellipse = EllipseModel()
        if ellipse.estimate(contour):
            # abs(ellipse.residuals(contour))
            ellipse_points = ellipse.predict_xy(np.linspace(0,2*np.pi))
            ellipse_params_list.append(ellipse.params+[np.sum(np.abs(ellipse.residuals(contour)))])
            void_ellipse_list[-1] = void_ellipse_list[-1] + ellipse.params+[np.sum(np.abs(ellipse.residuals(contour)))]
            # look at volume/surface difference
            ax[2].plot(ellipse_points[:,1],ellipse_points[:,0])
    ax[1].matshow(density_slice*mask_all)
    ax[0].set_xlim(0,256)
    ax[0].set_ylim(0,256)
    # ax[1].set_xlim(loc_void[1]-frame, loc_void[1]+frame)
    # ax[1].set_ylim(loc_void[0]-frame, loc_void[0]+frame)
    ax[2].set_xlim(0,256)
    ax[2].set_ylim(0,256)
    ax[2].set_aspect(1)
    
    ellipse_params_list = np.array(ellipse_params_list)
    void_ellipse_list = np.array(void_ellipse_list)
    # 0... void location x
    # 1... void location y
    # 2... saddle location x
    # 3... saddle location y
    # 2... void persistence (log)
    # 3... distance void - saddle
    # 4... ellipse: xc
    # 5... ellipse: yc
    # 6... ellipse: a
    # 7... ellipse: b
    # 8... ellipse: theta
    # 9... ellipse: residual
    
    # add parameter exploration
    
if __name__ == '__main__':
    main()