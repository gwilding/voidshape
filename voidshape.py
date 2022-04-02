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
    ax.set_xlabel(r'Birth density $\log(\delta+1)$')
    ax.set_xlabel(r'Death density $\log(\delta+1)$')
    

    # get void locations:
    # birth of 1-cycle in decreasing superlevel , ie column 1
    
    # let's explore the voids and critical points
    # select how many voids
    N = 500
    # calculate, and plot lines between void centre -> void saddle
    fig, ax = plt.subplots(1,1)
    ax.matshow(density_slice)
    #start with empty list
    void_list = []
    # loop through all one-dimensional persistence pairs
    # this signify the formation of a closed loop (=death) surrounding a void (at a saddle point)
    # and the filling of the loop (=birth) at the centre of the void (at a minimum)
    for death,birth in zip(pers_list[1][:N,1],pers_list[1][:N,2]):
        # we want to exclude the two loops at infinite of the ambient space
        if death > -np.inf:
            # get locations (=indices) where the loop is born and dies
            loc_void = np.where(density_slice == death)
            loc_saddle = np.where(density_slice == birth)
            
            # also calculate the distance between the saddle point and the minimum
            distance_raw = np.array([loc_saddle[0][0],loc_saddle[1][0]]) - np.array([loc_void[0][0],loc_void[1][0]])
            # correct for periodicity (recheck this)
            distance_temp = np.sum(np.min([np.abs(distance_raw),np.abs(distance_raw-L)],0)**2)**0.5
            # store void centre in list
            void_list.append([loc_void[1][0],loc_void[0][0],
                              loc_saddle[1][0],loc_saddle[0][0],
                              birth,
                              death,
                              distance_temp])
            # add saddle point and minimum to plot
            ax.scatter(loc_void[1],loc_void[0],marker='o',s=10,c='r')
            ax.scatter(loc_saddle[1],loc_saddle[0],marker='x',s=10,c='xkcd:goldenrod')
            # add connecting vector, but only of saddle point and central void minimum
            # are NOT separated by an boundary of the box
            if max(np.abs(distance_raw)) < min(L)/2:
                ax.plot([loc_void[1],loc_saddle[1]],[loc_void[0],loc_saddle[0]])
    
    void_list = np.array(void_list)
    
    # fill voids
    # we essentially use a watershed algorithm to flood the voids
    # starting from the centre, and upto a certain threshold
    
    L1,L2 = density_slice.shape
    # select voids from the sorted list of persistence values
    # void 0 and 1 do not work
    N = 200
    
    # fraction of (density_saddle-density_centre) upto which to fill
    # this determines the interior cells of the void
    threshold_factor = 0
    # the neighbouring cells, no diagonal connection
    neighbours = [(0,1),(0,-1),(1,0),(-1,0)]#(1,1),(1,-1),(-1,-1),(-1,1)]
    
    void_coordinates = void_list[:,:2].astype(int)
    wall_coordinates = void_list[:,2:4].astype(int)
    void_birth_death = void_list[:,4:6]
    
    # transpose so that indices work nicely
    # density_slice = density_slice.T
    
    void_boundaries = []
    void_interiors = []
    mask_already_test = np.zeros(shape=density_slice.shape, dtype=bool)
    for void_centre, wall, (birth, death) in zip(tqdm(void_coordinates),wall_coordinates,void_birth_death):
        # fill from the centre (void) to th wall (wall)
        neighbours = np.array([(0,1),(0,-1),(1,0),(-1,0)])
        threshold = birth - (birth - death)*threshold_factor
        
        interior_list = []
        boundary_list = []
        test_positions = [void_centre]

        while len(test_positions):
            # always test and remove first position in array
            pos = test_positions.pop(0)
            mask_already_test[pos[1],pos[0]] = True
            neighbour_cells = (pos + neighbours)%(L1,L2)
            # neighbour_cells = pos + neighbours
            neighbour_cells = [n for n in neighbour_cells if not mask_already_test[n[1],n[0]]]
            if all([density_slice[n[1],n[0]] < threshold for n in neighbour_cells]):
                # all surrounding cells below threshold: add to interior
                interior_list.append(pos)
                # extend test list with cells that have not been tested
                for new_pos in neighbour_cells:
                    mask_already_test[new_pos[1],new_pos[0]] = 1
                    test_positions.append(new_pos)
            elif density_slice[pos[1],pos[0]] < threshold:   
                # not all surrounding cells below threshold: add to boundary
                boundary_list.append(pos)
        mask_already_test *= False
        void_interiors.append(np.array(interior_list))
        # sort by angle towards for now
        ika = np.argsort(np.arctan2(*(np.array(boundary_list) - void_centre).T))
        boundary_list = np.array(boundary_list)[ika]
        # differences = np.sum(np.diff(boundary_list,axis=0)**2,axis=1)
        # while any(differences > 2):
        #     changes = np.where(differences > 2)[0]
        #     pick = changes[np.random.randint(len(changes))]
        #     exchange = boundary_list[pick].copy()
        #     boundary_list[pick] = boundary_list[pick-1].copy()
        #     boundary_list[pick-1] = exchange
        #     differences = np.sum(np.diff(boundary_list,axis=0)**2,axis=1)
        void_boundaries.append(boundary_list)
    
    # transpose back
    # density_slice = density_slice.T
    
    # ellipse fitting
    void_ellipse_list = []
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.matshow(density_slice)
    # for contour, void_centre in zip(tqdm(void_boundaries),void_coordinates):
    for contour, void in zip(tqdm(void_boundaries),void_list):
        xv, yv = void[0], void[1] # void centre
        vw, yw = void[2], void[3] # void wall
        birth, death = void[4], void[5] # birth and death
        # contour = [ c  for c in contours  if len(c) == np.max([ len(c) for c in contours ])][0]
        # shift to centre, then shift back
        # xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, 25),
        #                                 params=(10, 15, 4, 8, np.deg2rad(30)))
        # shift contour to centre to avoid boundary effects
        # if np.min(contour) == 0 and np.max(contour) ==255:
        #     break
        shift = [L1//2 - xv,L2//2 - yv]
        contour = (contour + shift)%(L1,L2)
        
        if len(contour) > 5:
            ellipse = EllipseModel()
            if ellipse.estimate(contour):
                # abs(ellipse.residuals(contour))
                # get points of the fit ellipse
                ellipse_points = ellipse.predict_xy(np.linspace(0,2*np.pi)) - shift
                res = np.sum(np.abs(ellipse.residuals(contour)))
                # store ellipse parameters, including the sum of residuals (goodness of fit)
                xc, yc, a, b, theta =  ellipse.params
                if a < b:
                    a, b = b, a
                xc, yc = np.array([xc, yc]) - shift
                void_ellipse_list.append([xv, yv, vw, yw, birth, death, xc, yc, a, b, theta, res])
                # ellipse_params_list.append(ellipse.params+[np.sum(np.abs(ellipse.residuals(contour)))])
                
                # void_ellipse_list.append(ellipse_params_list)
                
                # void_ellipse_list[-1] = void_ellipse_list[-1] + ellipse.params+[np.sum(np.abs(ellipse.residuals(contour)))]
                # look at volume/surface difference
                ax.scatter(*((contour-shift)%(L1,L2)).T,s=2)
                ax.plot(*ellipse_points.T)
    ax.set_xlim(0,255)
    ax.set_ylim(0,255)
    ax.xlabel(r'x')
    ax.ylabel(r'y')
    
    void_ellipse_list = np.array(void_ellipse_list)
    # 0... void location x
    # 1... void location y
    # 2... saddle location x
    # 3... saddle location y
    # 4... birth
    # 5... death
    # 6... ellipse: xc
    # 7... ellipse: yc
    # 8... ellipse: a
    # 9... ellipse: b
    # 10...ellipse: theta
    # 11...ellipse: residual
    
    # add parameter exploration

    eccentricity = np.sqrt(1-(void_ellipse_list[:,9]/void_ellipse_list[:,8])**2)
    persistence = void_ellipse_list[:,4]-void_ellipse_list[:,5]
    
    
    
    
if __name__ == '__main__':
    main()