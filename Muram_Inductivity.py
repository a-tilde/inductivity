 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:04:10 2023

@author: tilde
"""

# Import statements
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import muram as muram
import MuramMath2 as mm
import sys
import scipy.stats as sps
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy import fftpack
from astropy.io import fits
#import itertools
#import multiprocessing
import argparse

# Define color map
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-2, '#440053'),
    (5e-2, '#424388'),
    (1e-1, '#404388'),
    (2e-1, '#2a788e'),
    (4e-1, '#21a784'),
    (7e-1, '#78d151'),
    (1, '#fde624'),
], N=2560)

#ds = 48e5 # in cm

def ara_to_rom(a):              # Convert arabic numerals to roman 
    if a==1: r='I'
    elif a==2: r='II' 
    elif a==3: r='III'
    elif a==4: r='IV'
    else: print ("Unrecognized numeral"); sys.exit()
    return r

def weighted_Pr (x,y,w):        # Weighted Pearson r
    mx = np.sum(x*w) / np.sum(w)
    my = np.sum(y*w) / np.sum(w)
    sx = np.sum(w*(x-mx)**2) / np.sum(w)
    sy = np.sum(w*(y-my)**2) / np.sum(w)
    sxy = np.sum(w*(y-my)*(x-mx)) / np.sum(w)
    rhoxy = sxy/np.sqrt(sx*sy)
    return rhoxy

def weighted_S (x,y,w):         # Weighted Spearman r
    Rx = sps.rankdata(x)
    Ry = sps.rankdata(y)
    S = weighted_Pr(Rx, Ry, w)
    return S


def build_density_subplot(dBdt,
                          Ecurl,
                          title_string,
                          i,
                          dt,
                          fig,
                          ax,
                          Spearman_r,
                          Pearson_r,
                          linfit,
                          res_sum):
    """
    Plot scatter density between Faraday's induction law's LHS and RHS
    
    """
    
    # Define limits for plots and color bars
    bot_left, top_right = -150,150
    xlim = np.max([-np.min(dBdt),np.max(dBdt)])*0.6
    ylim = np.max([-np.min(Ecurl),np.max(Ecurl)])*0.6
    density = ax.hist2d(dBdt.T.flatten(), 
                        -Ecurl.T.flatten(),
                        range=[[-xlim, xlim],[-ylim, ylim]], 
                        bins=100,
                        vmax=np.max([xlim,ylim])*40, 
                        cmap=white_viridis)
    # Reference line
    ax.plot([bot_left,top_right],
            [bot_left,top_right], 
            color='blue', 
            linewidth=2, 
            label=r'Reference (1:1)')
    # Plot liinear fit
    ax.plot([bot_left,top_right],
            [bot_left*linfit[0][0]+linfit[0][1],top_right*linfit[0][0]+linfit[0][1]],
            color='red',
            linewidth=1.5,
            label=f'y=%.2g'%linfit[0][0]+'x+%.2g'%linfit[0][1]+' [G s$^{-1}$]')
    # Construct legend; change the color parameter to the background color
    ax.plot([-100],[-100], 
            color='white', 
            linewidth=0., 
            label=f'$E$: %.2g' %(linfit[1][0]/len(dBdt)/len(dBdt.T)))
    ax.plot([-100],[-100], 
            color='white', 
            linewidth=0., 
            label=r'$\Sigma${$\frac{\partial{B}}{\partial{t}}+$'+r'$(\nabla\times\overline{E})$}$_z$'+f': %.2g' %(res_sum))
    ax.plot([-100],[-100], 
            color='white', 
            linewidth=0., 
            label=f'Pearson $r$: %.2g' %Pearson_r)
    ax.plot([-100],[-100], 
            color='white', 
            linewidth=0., 
            label=f'Spearman $r$: %.2g' %Spearman_r)
    ax.set_xlabel(r'$\frac{\partial{B_z}}{\partial{t}}$ [G s$^{-1}$]', 
                  fontsize=15)
    
    ax.set_title(title_string, fontsize=15)
    
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.legend(fontsize=9)
    fig.colorbar(density[3], ax=ax, pad=0)

    
def build_residual_subplot(dBdt,
                           Ecurl_frame1,
                           Bmag,
                           fig,
                           ax):
    """
    Plot the difference map between Faraday's induction law's LHS and RHS
    
    """
    # Limit for color bar
    vs=np.max([np.max(dBdt.T+Ecurl_frame1.T),-np.min(dBdt.T+Ecurl_frame1.T)])/4
    
    im = ax.imshow(dBdt.T+Ecurl_frame1.T, 
                   origin='lower', 
                   cmap='PuOr', 
                   vmin=-vs, 
                   vmax=vs)
    cbar = fig.colorbar(im, pad=0, shrink=1)
    ax.contour(Bmag.T,
               levels=[1.2*np.mean(Bmag)],
               colors=['forestgreen'],
               linewidths=0.9)
    ax.set_xlabel(r'pixel', fontsize=15)
    ax.set_title(r'$\frac{\partial{B_z}}{\partial{t}} +$'+r'$ (\nabla\times\overline{E})_z$ [G s$^{-1}$]',
                 fontsize=15)
    
def build_Br_subplot(dBdt,
                     Br,
                     fig,
                     ax):
    """
    Plot context image with the radial component of B
    
    """
    # Limit for color bar
    vs = np.max([np.max(Br.T), -np.min(Br.T)])
    
    im = ax.imshow(Br.T, 
                   origin='lower', 
                   cmap='magma', 
                   vmin=-vs, 
                   vmax=vs)
    cbar = fig.colorbar(im, pad=0, shrink=1)
    ax.contour(dBdt.T,
               levels=[2*np.mean(np.abs(dBdt))],
               colors=['springgreen'],
               linewidths=1.0)
    ax.set_xlabel(r'pixel', fontsize=15)
    ax.set_title(r'$B_z$ [G]',fontsize=15)


def build_powerSpectrasubplot(image_dBdt,
                              image_curlE,
                              ax):
    """
    Compute power spectra of dB/dt and curl E and compare them
    Based on Jessica Lu's script
    
    """

    # Take the fourier transform of the image.
    F1_dBdt = fftpack.fft2(image_dBdt)
    F1_curlE = fftpack.fft2(image_curlE)

    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2_dBdt = fftpack.fftshift(F1_dBdt)
    F2_curlE = fftpack.fftshift(F1_curlE)

    # Calculate a 2D power spectrum
    psd2D_dBdt = np.abs(F2_dBdt)**2
    psd2D_curlE = np.abs(F2_curlE)**2

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D_dBdt = get_azimuthalAverage(psd2D_dBdt)
    psd1D_curlE = get_azimuthalAverage(psd2D_curlE)
    
    # Calculate the differences between peaks and spreads of 1D spectra
    psd1D_dBdt_peak = np.argmax(psd1D_dBdt)
    psd1D_curlE_peak = np.argmax(psd1D_curlE)
    delta_peak = psd1D_dBdt_peak - psd1D_curlE_peak
    delta_spread = find_frac_area_range(psd1D_dBdt, psd1D_dBdt_peak, 0.68
                   )- find_frac_area_range(psd1D_curlE, psd1D_curlE_peak, 0.68)
    
    ax.plot(range(len(psd1D_dBdt)),psd1D_dBdt, 
            color='blue', label=r'$\frac{\partial{B_z}}{\partial{t}}$')
    ax.plot(range(len(psd1D_curlE)),psd1D_curlE, 
            color='orange', label=r'$(-\nabla\times\overline{E})_z$')
    ax.set_xlabel('Spatial frequency',fontsize=18)
    ax.set_yscale('log')
    ax.legend(fontsize=18)
    
    return delta_peak, delta_spread

def find_frac_area_range(data, i_max, area_fraction):
    """
    Auxiliary function for build_powerSpectrasubplot()
    Find the range around the peak where the area under the curve is a 
    fraction (between 0 and 1) of the total area.
    
    """
    total_area_left = sum(data[:i_max])
    total_area_right = sum(data[i_max:])
    
    fraction_area_left = total_area_left * area_fraction
    fraction_area_right = total_area_right * area_fraction
    
    s_left = 0
    s_right = 0
    left, right = i_max, i_max
    
    while sum(data[i_max-s_left:i_max]) < fraction_area_left:
        s_left += 1
        left = i_max - s_left
    
    while sum(data[i_max:i_max+s_right]) < fraction_area_right:
        s_right += 1
        right = i_max + s_right
    
    spread = right-left
    
    return spread

def get_azimuthalAverage(image, center=None):
    """
    Auxiliary function for build_powerSpectrasubplot()
    
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    Based on Jessica Lu's script
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    # Normalize by area under the curve
    radial_prof_norm = radial_prof / np.sum(radial_prof)

    return radial_prof_norm



def get_S(E,B,filename=''):
    """
    Compute and save the vertical component of Poynting flux
    Note that Poynting flux output are stored in ./averagings_output/Sz_fits/
    
    """
    S = mm.cross(E, B).x
    output_filename = 'averagings_output/Sz_fits/'+filename+'.fits'
    return S, output_filename

def get_S_emerg(vz,Bx,By,filename=''):
    """
    Compute and save the emergence term of Poynting flux
    Note that Poynting flux output are stored in ./averagings_output/Sz_fits/
    
    """
    S_emerg = vz * np.sqrt(Bx**2+By**2) # vz * Bh^2
    output_filename = 'averagings_output/Sz_fits/'+filename+'_emerg.fits'
    return S_emerg, output_filename

def degrade_spatially(image, method, degrade_factor):
    """
    Downsample spatial resolution
    Degrade factor is binsize or Gaussian sigma, depending on the method
    
    """
    if method=='Gaussian':
        degraded_image = gaussian_filter(image, sigma=degrade_factor)
    elif method=='Binning':
        degraded_image = image.reshape(int(len(image)/degrade_factor),
                                       degrade_factor,
                                       int(len(image.T)/degrade_factor),
                                       degrade_factor).mean(-1).mean(1)
    else: print ("Unrecognized spatial degrading method"); sys.exit()
    
    return degraded_image

"""
NOTE: the staggered grid functionality is deprecated in this version
    
"""
def COE_to_CE (image):
    """
    Auxiliary function for convert_grid staggered=True option
    CE, used for B_r (r=z=x_muram)
    """
    x, y = np.arange(len(image))*2, np.arange(len(image))*2
    interp = RegularGridInterpolator((x, y), image, 
                                     bounds_error=False, fill_value=None)
    xx, yy = np.arange(len(image))*2+1, np.arange(len(image))*2+1
    X, Y = np.meshgrid(xx, yy, indexing='ij')
    staggered_image = interp((X, Y))
    return staggered_image

def COE_to_PE (image): 
    """
    Auxiliary function for convert_grid staggered=True option
    PE, used for B_phi & v_phi (phi=x=z_muram)
    """
    x, y = np.arange(len(image))*2, np.arange(len(image))*2
    interp = RegularGridInterpolator((x, y), image, 
                                     bounds_error=False, fill_value=None)
    yy = np.arange(len(image))*2+1
    X, Y = np.meshgrid(x, yy, indexing='ij')
    staggered_image = interp((X, Y))
    return staggered_image

def COE_to_TE (image):
    """
    Auxiliary function for convert_grid staggered=True option
    TE, used for B_theta & v_theta (theta=y=y_muram)
    
    """
    x, y = np.arange(len(image))*2, np.arange(len(image))*2
    interp = RegularGridInterpolator((x, y), image, 
                                     bounds_error=False, fill_value=None)
    xx = np.arange(len(image))*2+1
    X, Y = np.meshgrid(xx, y, indexing='ij')
    staggered_image = interp((X, Y))
    return staggered_image

def convert_grid (muram_slice,x1=None,x2=None,y1=None,y2=None, staggered=False,
                  degraded=False, binGau=0, sigma=0):
    """
    Convert spatial grid to desired resolution and/or to staggered grid
    NOTE: the staggered grid functionality is deprecated in this version
    
    """
    
    if staggered and not degraded:
        v_converted = mm.vector(muram_slice.vx[x1:x2,y1:y2],
                                COE_to_TE(muram_slice.vy[x1:x2,y1:y2]),
                                COE_to_PE(muram_slice.vz[x1:x2,y1:y2]))
        B_converted = mm.vector(COE_to_CE(muram_slice.Bx[x1:x2,y1:y2]/(np.sqrt(4*np.pi))),
                                COE_to_TE(muram_slice.By[x1:x2,y1:y2]/(np.sqrt(4*np.pi))),
                                COE_to_PE(muram_slice.Bz[x1:x2,y1:y2]/(np.sqrt(4*np.pi))))
    if degraded and not staggered:
        v_converted = mm.vector(degrade_spatially(muram_slice.vx[x1:x2,y1:y2],
                                                  binGau,
                                                  sigma),
                                degrade_spatially(muram_slice.vy[x1:x2,y1:y2],
                                                  binGau,
                                                  sigma),
                                degrade_spatially(muram_slice.vz[x1:x2,y1:y2],
                                                  binGau,
                                                  sigma))
        B_converted = mm.vector(degrade_spatially(muram_slice.Bx[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                                  binGau,
                                                  sigma),
                                degrade_spatially(muram_slice.By[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                                  binGau,
                                                  sigma),
                                degrade_spatially(muram_slice.Bz[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                                  binGau,
                                                  sigma))
    if degraded and staggered:
        v_converted = mm.vector(degrade_spatially(muram_slice.vx[x1:x2,y1:y2],
                                                  binGau,sigma),
                                COE_to_TE(degrade_spatially(muram_slice.vy[
                                          x1:x2,y1:y2],binGau,sigma)),
                                COE_to_PE(degrade_spatially(muram_slice.vz[
                                          x1:x2,y1:y2],binGau,sigma)))
        B_converted = mm.vector(COE_to_CE(degrade_spatially(muram_slice.Bx[
                                          x1:x2,y1:y2]/(np.sqrt(4*np.pi)),binGau,sigma)),
                                COE_to_TE(degrade_spatially(muram_slice.By[
                                          x1:x2,y1:y2]/(np.sqrt(4*np.pi)),binGau,sigma)),
                                COE_to_PE(degrade_spatially(muram_slice.Bz[
                                          x1:x2,y1:y2]/(np.sqrt(4*np.pi)),binGau,sigma)))
    return v_converted, B_converted






def average_temporally (temp_avg_method, 
                        surface,
                        binGau='',
                        sp_avg_method=0,
                        sigma=0,
                        FOV_coords='Full', 
                        staggered_grid=False, 
                        save_bv=False):
    
    """
    Main function; temporal averaging with other operations called from within
    
    TO-DO HERE: 
        1) Modify the path variable to reflect where MURaM slices are located
        2) Modify the delta_ter variable 
    Everything else is handled by the script file
    
    """
    
    path='./data/'              # path is relative
    iter0 = 138200              # Initial time step
    delta_iter = 200            # For delta_iter = n, select every n'th MURaM frame (i.e. set cadence). Values used for the manuscript: 200, 24, 4
    surfacetype = surface[0]
    height = surface[1]
    ds_native = 48e5            # in cm
    
    output_stats = np.zeros((6,7))
    
    # Create a string for file naming convention
    fig_spadeg_suffix = ''
    grid_suffix=''
    if binGau!='':
        fig_spadeg_suffix = binGau+str(sp_avg_method)+'sigma'+str(sigma)+'_'
    if staggered_grid:
        grid_suffix='_SG'
    
    # Create plot spaceholder variables
    fig = plt.figure(figsize=[32,19])
    ax1 = fig.add_subplot(4, 6, 1)
    ax2 = fig.add_subplot(4, 6, 2)
    ax3 = fig.add_subplot(4, 6, 3)
    ax4 = fig.add_subplot(4, 6, 4)
    ax5 = fig.add_subplot(4, 6, 5)
    ax6 = fig.add_subplot(4, 6, 6)
    ax7 = fig.add_subplot(4, 6, 7)
    ax8 = fig.add_subplot(4, 6, 8)
    ax9 = fig.add_subplot(4, 6, 9)
    ax10 = fig.add_subplot(4, 6, 10)
    ax11 = fig.add_subplot(4, 6, 11)
    ax12 = fig.add_subplot(4, 6, 12)
    ax13 = fig.add_subplot(4, 6, 13)
    ax14 = fig.add_subplot(4, 6, 14)
    ax15 = fig.add_subplot(4, 6, 15)
    ax16 = fig.add_subplot(4, 6, 16)
    ax17 = fig.add_subplot(4, 6, 17)
    ax18 = fig.add_subplot(4, 6, 18)
    ax19 = fig.add_subplot(4, 6, 19)
    ax20 = fig.add_subplot(4, 6, 20)
    ax21 = fig.add_subplot(4, 6, 21)
    ax22 = fig.add_subplot(4, 6, 22)
    ax23 = fig.add_subplot(4, 6, 23)
    ax24 = fig.add_subplot(4, 6, 24)
    
    ax1.set_ylabel(r'$(-\nabla\times\overline{E}^{'+ara_to_rom(temp_avg_method)
                   +'})_z$ [G s$^{-1}$] ', fontsize=15)
    ax7.set_ylabel(r'pixel', 
                   fontsize=15)
    ax13.set_ylabel(r'pixel', 
                   fontsize=15)
    ax19.set_ylabel(r'Power spectrum', fontsize=15)
    axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20,ax21,ax22,ax23,ax24]
    
    # Define field of view
    if FOV_coords[-1]!='Full':
        x1,x2,y1,y2 = FOV_coords[0],FOV_coords[1],FOV_coords[2],FOV_coords[3]
    else: x1,x2,y1,y2 = None, None, None, None
    FOV_coords = FOV_coords[-1]
    
    for i in range(6): # Loop over six cadences
        
        # Check for surface type
        if surfacetype=='z':
            kind = 'yz'
            frame0=muram.MuramSlice(path,iter0,kind,height)
        elif surfacetype=='tau':
            kind = 'tau'
            frame0=muram.MuramTauSlice(path,iter0,float(height))
        else: print ("Unrecognized surface type"); sys.exit()
        
        # Stagger the grid as needed. NOTE: staggered grid functionality is deprecated, may not work correctly
        if staggered_grid or sp_avg_method==2:
            v_frame0,B_frame0 = convert_grid(frame0,
                                             x1,x2,y1,y2,
                                             staggered_grid,
                                             sp_avg_method==2, 
                                             binGau, 
                                             sigma)
        else:
            v_frame0=mm.vector(frame0.vx[x1:x2,y1:y2],
                               frame0.vy[x1:x2,y1:y2],
                               frame0.vz[x1:x2,y1:y2])
            B_frame0=mm.vector(frame0.Bx[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                               frame0.By[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                               frame0.Bz[x1:x2,y1:y2]/(np.sqrt(4*np.pi)))
        
        # Save B & v without doing anything else (e.g. inputs for PDFI); frame 0
        if save_bv and i==0:
            if temp_avg_method==1 or temp_avg_method==3: sys.exit()
            bv_filename_prefix='data/fits_bv/'+surfacetype+height+'_ta'+str(temp_avg_method)+'_'+fig_spadeg_suffix+'_'
            vx0file = fits.PrimaryHDU(v_frame0.y)
            vx0file.writeto(bv_filename_prefix+'vx'+str(iter0), overwrite=True)
            vy0file = fits.PrimaryHDU(v_frame0.z)
            vy0file.writeto(bv_filename_prefix+'vy'+str(iter0), overwrite=True)
            vz0file = fits.PrimaryHDU(v_frame0.x)
            vz0file.writeto(bv_filename_prefix+'vz'+str(iter0), overwrite=True)
            Bx0file = fits.PrimaryHDU(B_frame0.y)
            Bx0file.writeto(bv_filename_prefix+'Bx'+str(iter0), overwrite=True)
            By0file = fits.PrimaryHDU(B_frame0.z)
            By0file.writeto(bv_filename_prefix+'By'+str(iter0), overwrite=True)
            Bz0file = fits.PrimaryHDU(B_frame0.x)
            Bz0file.writeto(bv_filename_prefix+'Bz'+str(iter0), overwrite=True)
        
        if temp_avg_method==1:
            """
            Temporal averaging method I
            
            """
            if save_bv: sys.exit() # averaging for PDFI should be done on its (PDFIs) outputs instead
            
            # Electric fields, Poynting flux, and emerging term of Poynting flux
            E_frame1=mm.cross(B_frame0,v_frame0)
            S_frame1, output_filename = get_S(E_frame1,
                                              B_frame0,
                                              str(temp_avg_method)+'_'+surfacetype+height+'_'+fig_spadeg_suffix+FOV_coords+grid_suffix)
            S_frame1_emerg, output_filename_emerg = get_S_emerg(v_frame0.x, 
                                                                B_frame0.y, 
                                                                B_frame0.z,
                                                                str(temp_avg_method)+'_'+surfacetype+height+'_'+fig_spadeg_suffix+FOV_coords+grid_suffix)
            
            for j in range(i+1): # Loop over the preceding frames
                iter1 = iter0 + delta_iter*(j+1)
                
                # Open an appropriate MURaM slice (tau or z)
                if surfacetype=='z':
                    frame1=muram.MuramSlice(path,iter1,kind,height)
                elif surfacetype=='tau':
                    frame1=muram.MuramTauSlice(path,iter1,float(height))
                
                """
                Spatial averaging methods II (whether binning or PSF) are implemented within each temporal averaging case
                
                """
                if staggered_grid or sp_avg_method==2:  # Staggered grid functionality is deprecated
                    v_frame1, B_frame1 = convert_grid(frame1,
                                                      x1,x2,y1,y2,
                                                      staggered_grid,
                                                      sp_avg_method==2,
                                                      binGau, 
                                                      sigma)
                else:
                    v_frame1=mm.vector(frame1.vx[x1:x2,y1:y2],
                                       frame1.vy[x1:x2,y1:y2],
                                       frame1.vz[x1:x2,y1:y2])
                    B_frame1=mm.vector(frame1.Bx[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                       frame1.By[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                       frame1.Bz[x1:x2,y1:y2]/(np.sqrt(4*np.pi)))
                
                
                # Perform temporal averaging of E-fields and Poynting flux
                E_frame_toadd=mm.cross(B_frame1,
                                       v_frame1)
                E_frame1 = mm.vector_sum(E_frame1,
                                         E_frame_toadd)
                
                S_frame_toadd=get_S(E_frame_toadd, 
                                    B_frame1)[0]
                S_frame1 = S_frame1 + S_frame_toadd
                
                S_frame1_toadd_emerg = get_S_emerg(v_frame1.x, 
                                                   B_frame1.y, 
                                                   B_frame1.z)[0]
                S_frame1_emerg = S_frame1_emerg + S_frame1_toadd_emerg
                
            
            # Pull info from MURaM slices
            tau0,nv0,shape0,time0=muram.read_slice(path,
                                                   iter0,
                                                   kind,
                                                   height)
            tau1,nv1,shape1,time1=muram.read_slice(path,
                                                   iter1,
                                                   kind,
                                                   height)
            
            # time step, pixel size, and temporal averaging factor for Poynting flux calculations (assuming (v x B) x B)
            dt = time1-time0
            ds = ds_native*(i+2)
            S_TAfactor = i+2
        
        elif temp_avg_method==2:
            """
            Temporal averaging method II
            
            """
            if staggered_grid or sp_avg_method==2: # Staggered grid functionality is deprecated
                v_frame1, B_frame1 = convert_grid(frame0,
                                                  x1,x2,y1,y2,
                                                  staggered_grid,
                                                  sp_avg_method==2,
                                                  binGau, 
                                                  sigma)
                B_frame0 = convert_grid(frame0, 
                                        x1,x2,y1,y2, 
                                        staggered_grid,
                                        sp_avg_method==2, 
                                        binGau, 
                                        sigma)[1]
            else:
                v_frame1=mm.vector(frame0.vx[x1:x2,y1:y2],
                                   frame0.vy[x1:x2,y1:y2],
                                   frame0.vz[x1:x2,y1:y2])
                B_frame1=mm.vector(frame0.Bx[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                   frame0.By[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                   frame0.Bz[x1:x2,y1:y2]/(np.sqrt(4*np.pi)))
                B_frame0=mm.vector(frame0.Bx[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                   frame0.By[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                   frame0.Bz[x1:x2,y1:y2]/(np.sqrt(4*np.pi)))
            
            for j in range(i+1):
                iter1 = iter0 + delta_iter*(j+1)
                if surfacetype=='z':
                    frame1=muram.MuramSlice(path,
                                            iter1,
                                            kind,
                                            height)
                elif surfacetype=='tau':
                    frame1=muram.MuramTauSlice(path,
                                               iter1,
                                               float(height))
                
                if staggered_grid or sp_avg_method==2:
                    v_frame_toadd,B_frame_toadd=convert_grid(frame1,
                                                             x1,x2,y1,y2, 
                                                             staggered_grid,
                                                             sp_avg_method==2,
                                                             binGau,
                                                             sigma)
                else:
                    v_frame_toadd=mm.vector(frame1.vx[x1:x2,y1:y2],
                                            frame1.vy[x1:x2,y1:y2],
                                            frame1.vz[x1:x2,y1:y2])
                    B_frame_toadd=mm.vector(frame1.Bx[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                            frame1.By[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                            frame1.Bz[x1:x2,y1:y2]/(np.sqrt(4*np.pi)))
                
                v_frame1=mm.vector_sum(v_frame1,v_frame_toadd)
                B_frame1=mm.vector_sum(B_frame1,B_frame_toadd)
            
            # The quantities are divided by the square root of ds's multiplying factor
            if save_bv:
                vx1file = fits.PrimaryHDU(v_frame1.y/(i+2))
                vx1file.writeto(bv_filename_prefix+'vx'+str(iter1),
                                overwrite=True)
                vy1file = fits.PrimaryHDU(v_frame1.z/(i+2))
                vy1file.writeto(bv_filename_prefix+'vy'+str(iter1), 
                                overwrite=True)
                vz1file = fits.PrimaryHDU(v_frame1.x/(i+2))
                vz1file.writeto(bv_filename_prefix+'vz'+str(iter1), 
                                overwrite=True)
                Bx1file = fits.PrimaryHDU(B_frame1.y/(i+2))
                Bx1file.writeto(bv_filename_prefix+'Bx'+str(iter1), 
                                overwrite=True)
                By1file = fits.PrimaryHDU(B_frame1.z/(i+2))
                By1file.writeto(bv_filename_prefix+'By'+str(iter1), 
                                overwrite=True)
                Bz1file = fits.PrimaryHDU(B_frame1.x/(i+2))
                Bz1file.writeto(bv_filename_prefix+'Bz'+str(iter1), 
                                overwrite=True)
                if i<5: continue
                else: sys.exit()
            
            E_frame1=mm.cross(B_frame1,v_frame1)
            S_frame1, output_filename = get_S(E_frame1,
                                              B_frame1,
                                              str(temp_avg_method)+'_'+surfacetype+height+'_'+fig_spadeg_suffix+FOV_coords+grid_suffix)
            S_frame1_emerg, output_filename_emerg = get_S_emerg(v_frame1.x, 
                                                                B_frame1.y, 
                                                                B_frame1.z,
                                                                str(temp_avg_method)+'_'+surfacetype+height+'_'+fig_spadeg_suffix+FOV_coords+grid_suffix)
            
            tau0,nv0,shape0,time0=muram.read_slice(path,
                                                   iter0,
                                                   kind,
                                                   height)
            tau1,nv1,shape1,time1=muram.read_slice(path,
                                                   iter1,
                                                   kind,
                                                   height)
            dt = time1-time0
            ds = ds_native*(i+2)**2
            S_TAfactor = (i+2)**3
            
            
            dBdt = mm.vector((B_frame_toadd.x-B_frame0.x)/dt,
                             (B_frame_toadd.y-B_frame0.y)/dt,
                             (B_frame_toadd.z-B_frame0.z)/dt)
            B_mag = np.sqrt(B_frame_toadd.x**2+B_frame_toadd.y**2+B_frame_toadd.z**2)
            Br = B_frame_toadd.x
            
        elif temp_avg_method==3:
            """
            Temporal averaging method III
            
            """
            if save_bv: sys.exit() # averaging for PDFI should be done on its (PDFIs) outputs instead
            
            iter1 = iter0 + delta_iter*(i+1)
            if surfacetype=='z':
                frame1=muram.MuramSlice(path,iter1,kind,height)
            elif surfacetype=='tau':
                frame1=muram.MuramTauSlice(path,iter1,float(height))
            
            tau0,nv0,shape0,time0=muram.read_slice(path,
                                                   iter0,
                                                   kind,
                                                   height)
            tau1,nv1,shape1,time1=muram.read_slice(path,
                                                   iter1,
                                                   kind,
                                                   height)
            dt = time1-time0
            ds = ds_native*2
            S_TAfactor = 2
            
            if staggered_grid or sp_avg_method==2: # Staggered grid functionality is deprecated
                v_frame0, B_frame0 = convert_grid(frame0,
                                                  x1,x2,y1,y2,
                                                  staggered_grid,
                                                  sp_avg_method==2,
                                                  binGau, 
                                                  sigma)
                v_frame1, B_frame1 = convert_grid(frame1,
                                                  x1,x2,y1,y2,
                                                  staggered_grid,
                                                  sp_avg_method==2,
                                                  binGau, 
                                                  sigma)
            else:
                v_frame1=mm.vector(frame1.vx[x1:x2,y1:y2],
                                   frame1.vy[x1:x2,y1:y2],
                                   frame1.vz[x1:x2,y1:y2])
                B_frame1=mm.vector(frame1.Bx[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                   frame1.By[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                   frame1.Bz[x1:x2,y1:y2]/(np.sqrt(4*np.pi)))
            
            E_frame1=mm.vector_sum(mm.cross(B_frame1,v_frame1),
                                   mm.cross(B_frame0,v_frame0))
            S_frame1 = get_S(mm.cross(B_frame1,v_frame1),B_frame1)[0
                   ] + get_S(mm.cross(B_frame0,v_frame0),B_frame0)[0]
            output_filename = 'averagings_output/Sz_fits/'+ str(temp_avg_method)+'_'+surfacetype+height+'_'+fig_spadeg_suffix+str(FOV_coords)+grid_suffix +'.fits'
            S_frame1_emerg = get_S_emerg(v_frame1.x,B_frame1.y,B_frame1.z)[0
                         ] + get_S_emerg(v_frame0.x,B_frame0.y,B_frame0.z)[0]
            output_filename_emerg = 'averagings_output/Sz_fits/'+ str(temp_avg_method)+'_'+surfacetype+height+'_'+fig_spadeg_suffix+str(FOV_coords)+grid_suffix +'_emerg.fits'
            
        elif temp_avg_method==4:
            """
            Temporal averaging method IV
            
            """
            iter1 = iter0 + delta_iter*(i+1)
            if surfacetype=='z':
                frame1=muram.MuramSlice(path,iter1,kind,height)
            elif surfacetype=='tau':
                frame1=muram.MuramTauSlice(path,iter1,float(height))
            
            tau0,nv0,shape0,time0=muram.read_slice(path,
                                                   iter0,
                                                   kind,
                                                   height)
            tau1,nv1,shape1,time1=muram.read_slice(path,
                                                   iter1,
                                                   kind,
                                                   height)
            dt = time1-time0
            ds = ds_native*4
            S_TAfactor = 8
            
            if staggered_grid or sp_avg_method==2: # Staggered grid functionality is deprecated
                v_frame0, B_frame0 = convert_grid(frame0,
                                                  x1,x2,y1,y2,
                                                  staggered_grid,
                                                  sp_avg_method==2,
                                                  binGau, 
                                                  sigma)
                v_frame1, B_frame1 = convert_grid(frame1,
                                                  x1,x2,y1,y2,
                                                  staggered_grid,
                                                  sp_avg_method==2,
                                                  binGau, 
                                                  sigma)
            else:
                v_frame1=mm.vector(frame1.vx[x1:x2,y1:y2],
                                   frame1.vy[x1:x2,y1:y2],
                                   frame1.vz[x1:x2,y1:y2])
                B_frame1=mm.vector(frame1.Bx[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                   frame1.By[x1:x2,y1:y2]/(np.sqrt(4*np.pi)),
                                   frame1.Bz[x1:x2,y1:y2]/(np.sqrt(4*np.pi)))
            
            # The quantities are divided by the square root of ds's multiplying factor
            if save_bv:
                vx1file = fits.PrimaryHDU(mm.vector_sum(v_frame1,v_frame0).y/2)
                vx1file.writeto(bv_filename_prefix+'vx'+str(iter1),
                                overwrite=True)
                vy1file = fits.PrimaryHDU(mm.vector_sum(v_frame1,v_frame0).z/2)
                vy1file.writeto(bv_filename_prefix+'vy'+str(iter1), 
                                overwrite=True)
                vz1file = fits.PrimaryHDU(mm.vector_sum(v_frame1,v_frame0).x/2)
                vz1file.writeto(bv_filename_prefix+'vz'+str(iter1), 
                                overwrite=True)
                Bx1file = fits.PrimaryHDU(mm.vector_sum(B_frame1,B_frame0).y/2)
                Bx1file.writeto(bv_filename_prefix+'Bx'+str(iter1), 
                                overwrite=True)
                By1file = fits.PrimaryHDU(mm.vector_sum(B_frame1,B_frame0).z/2)
                By1file.writeto(bv_filename_prefix+'By'+str(iter1), 
                                overwrite=True)
                Bz1file = fits.PrimaryHDU(mm.vector_sum(B_frame1,B_frame0).x/2)
                Bz1file.writeto(bv_filename_prefix+'Bz'+str(iter1), 
                                overwrite=True)
                if i<5: continue
                else: sys.exit()
            
            E_frame1=mm.cross(mm.vector_sum(B_frame1,B_frame0),
                              mm.vector_sum(v_frame1,v_frame0))
            S_frame1, output_filename = get_S(E_frame1,
                                              mm.vector_sum(B_frame1,B_frame0),
                                              str(temp_avg_method)+'_'+surfacetype+height+'_'+fig_spadeg_suffix+FOV_coords+grid_suffix)
            S_frame1_emerg, output_filename_emerg = get_S_emerg(
                            mm.vector_sum(v_frame0,v_frame1).x,
                            mm.vector_sum(B_frame0,B_frame1).y,
                            mm.vector_sum(B_frame0,B_frame1).z,
                            str(temp_avg_method)+'_'+surfacetype+height+'_'+fig_spadeg_suffix+FOV_coords+grid_suffix)
        
        else: print ("Unrecognized temporal averaging method"); sys.exit()
        
        if temp_avg_method!=2:
            B_mag = np.sqrt(B_frame1.x**2+B_frame1.y**2+B_frame1.z**2)
            Br = B_frame1.x
            dBdt = mm.vector((B_frame1.x-B_frame0.x)/dt,
                             (B_frame1.y-B_frame0.y)/dt,
                             (B_frame1.z-B_frame0.z)/dt)
        
        
        
        # Divide S_z by the appropriate temporal averaging factor
        Sz = S_frame1 / S_TAfactor
        Sz_emerg = S_frame1_emerg / S_TAfactor
        
        """
        Spatial averaging methods I (whether binning or PSF) are implemented here, 
        outside of temporal averaging cases, i.e. when TA has already been applied
        
        """
        if sp_avg_method==1 and binGau!='':
            # Select the vertical component
            dBzdt = degrade_spatially(dBdt.x, 
                                      binGau, 
                                      sigma)
            Ecurlz = degrade_spatially(E_frame1.curl2D_x(), 
                                       binGau, 
                                       sigma)
            B_mag = degrade_spatially(B_mag, 
                                      binGau, 
                                      sigma)
            Br = degrade_spatially(Br, 
                                   binGau, 
                                   sigma)
            Sz = degrade_spatially(Sz, 
                                   binGau, 
                                   sigma)
            Sz_emerg = degrade_spatially(Sz_emerg, 
                                         binGau, 
                                         sigma)
            szfile = fits.PrimaryHDU(degrade_spatially(Sz, 
                                                       binGau, 
                                                       sigma))
            szfile.writeto(output_filename, overwrite=True)
        else:
            # Select the vertical component
            dBzdt = dBdt.x
            Ecurlz = E_frame1.curl2D_x()
            szfile = fits.PrimaryHDU(Sz)
            szfile.writeto(output_filename, overwrite=True)
            
        # Remove padding -- this is done because FOV borders have artifacts after 
        # spatial averaging, partularly when PSF is applied
        if binGau == 'Binning':
            dBzdt = dBzdt[1:-1,1:-1]
            Ecurlz = Ecurlz[1:-1,1:-1]
            B_mag = B_mag[1:-1,1:-1]
            Br = Br[1:-1,1:-1]
            Sz = Sz[1:-1,1:-1]
            Sz_emerg = Sz_emerg[1:-1,1:-1]
        elif binGau == 'Gaussian':
            dBzdt = dBzdt[sigma*2:-sigma*2,sigma*2:-sigma*2]
            Ecurlz = Ecurlz[sigma*2:-sigma*2,sigma*2:-sigma*2]
            B_mag = B_mag[sigma*2:-sigma*2,sigma*2:-sigma*2]
            Br = Br[sigma*2:-sigma*2,sigma*2:-sigma*2]
            Sz = Sz[sigma*2:-sigma*2,sigma*2:-sigma*2]
            Sz_emerg = Sz_emerg[sigma*2:-sigma*2,sigma*2:-sigma*2]
            
        szfile = fits.PrimaryHDU(Sz)
        szfile.writeto(output_filename, overwrite=True)
        
        szfile_emerg = fits.PrimaryHDU(Sz_emerg)
        szfile_emerg.writeto(output_filename_emerg, overwrite=True)
        
        # Obtain stats, weighted version
        # w = np.sqrt(dBzdt.T.flatten()**2 + Ecurlz.T.flatten()**2/ds) #weights
        # Spearman_r = weighted_S(dBzdt.T.flatten(), -Ecurlz.T.flatten()/ds, w)
        # Pearson_r = weighted_Pr(dBzdt.T.flatten(), -Ecurlz.T.flatten()/ds, w)
        # linfit = np.polyfit(dBzdt.T.flatten(),-Ecurlz.T.flatten()/ds,deg=1,
        #                     w=w,full=True)
        
        # Obtain stats
        res_sum = np.sum(np.abs(dBzdt+Ecurlz/ds))/len(dBzdt)/len(dBzdt.T)
        Spearman_r = sps.spearmanr(dBzdt.T.flatten(),
                                   -Ecurlz.T.flatten()/ds)[0]
        Pearson_r = sps.pearsonr(dBzdt.T.flatten(),
                                 -Ecurlz.T.flatten()/ds)[0]
        linfit = np.polyfit(dBzdt.T.flatten(), 
                            -Ecurlz.T.flatten()/ds, 
                            deg=1,
                            full=True)
        
        
        # Density plot
        if kind=='tau':
            title_string1 = r'$\tau=$'+height+', $\Delta t$ = %1.3g'%dt+'s'
        else: title_string1 = r'$z=$'+height+', $\Delta t$ = %1.3g'%dt+'s'
        if binGau=='Binning':
            title_string2 =r'; bin $\sigma=$' + str(sigma)
        elif binGau=='Gaussian': title_string2 =r'; bin $\sigma=$' + str(sigma)
        else: title_string2=''
        title_string = title_string1 + title_string2 + grid_suffix
        build_density_subplot(dBzdt,
                              Ecurlz/ds,
                              title_string,
                              i, (i+1)*dt,
                              fig, axs[i],
                              Spearman_r, Pearson_r, linfit, res_sum)
        
        # Residual plot
        build_residual_subplot(dBzdt,
                               Ecurlz/ds,
                               B_mag,
                               fig,axs[i+6])
        # B-field plot
        build_Br_subplot(dBzdt,
                         Br,
                         fig,axs[i+12])
        # Power spectra subplot and stats
        delta_ps_peak, delta_ps_spread = build_powerSpectrasubplot(dBzdt,
                                                                   Ecurlz/ds,
                                                                   axs[i+18])
        
        # Store stats in output array
        output_stats[i] = [linfit[0][0], 
                           linfit[1][0]/len(dBzdt)/len(dBzdt.T),
                           Spearman_r,
                           Pearson_r,
                           res_sum,
                           delta_ps_peak,
                           delta_ps_spread]
        
    # Save figure. Note that figures are stored in ./figures subdirectory
    plt.tight_layout()
    plt.savefig('./figures/tempavg'+str(temp_avg_method) + '_' + surfacetype + height +'_'+fig_spadeg_suffix + FOV_coords + grid_suffix + '.pdf')
    plt.close()
    
    # Columns in output file that encode inputs
    binGau_dict = {'':0, 'Binning':1, 'Gaussian':2}
    FOV_dict = {'QS':0, 'AR&QS':1, 'spot':2}
    surfacekind_dict = {'tau':0, 'z':1}
    input_encoders = [temp_avg_method,
                      binGau_dict[binGau],
                      sp_avg_method,
                      sigma,
                      surfacekind_dict[surface[0]],
                      float(surface[1]),
                      FOV_dict[FOV_coords],
                      float(staggered_grid)]
    
    output = [np.concatenate((row, input_encoders)) for row in output_stats]
    
    # Last column in output text file
    descriptor = str(temp_avg_method)+'_'+surface[0]+surface[1]+'_'+binGau+str(sp_avg_method)+str(sigma)+'_'+FOV_coords+str(staggered_grid)
    
    return output, descriptor




# To be read by the script file

input_matrix = [
    [1,2,3,4],                      # TA methods
    [('tau', '1.000'),              # Surfaces, heights
     ('tau', '0.100'),
     ('tau', '0.010'),
     ('z', '0556'),
     ('z', '0557'),
     ('z', '0560'),
     ('z', '0561'),
     ('z', '0564'),
     ('z', '0565')
     ],
    ['','Binning','Gaussian'],      # Spatial averaging method families
    [1,2],                          # Spatial averaging methods within each family
    [2,4,8],                        # Spatial averaging factors
    [[350,750,850,1250, 'AR&QS'],   # FOVs
     [350,750,1350,1750, 'QS'],
     [1171,1475,925,1229, 'spot'],
     ['Full']
     ],
    [False,True],                   # Staggered grid or not
    [False,True]
    ]

parser = argparse.ArgumentParser(
                    description='Refer to input_matrix in Muram_Inductivity.py')
parser.add_argument('tavg', 
                    type=int,
                    help='Temporal averaging method (I-IV)')
parser.add_argument('slice', 
                    type=int,
                    help='Muram slice with surface kind and height')
parser.add_argument('BinGau', 
                    type=int,
                    help='Binning or Gaussian degradation (or none)')
parser.add_argument('method', 
                    type=int,
                    help='Spatial degradation method (I or II)')
parser.add_argument('sigma', 
                    type=int,
                    help='Spatial degradation factor')
parser.add_argument('FOV', 
                    type=int,
                    help='Field of view')
parser.add_argument('grid', 
                    type=int,
                    help='Whether it is staggered or not')
parser.add_argument('save_bv', 
                    type=int,
                    help='Whether to only save B and v vectors')
args = parser.parse_args()




out, descriptor = average_temporally(input_matrix[0][args.tavg], 
                                       input_matrix[1][args.slice],
                                       input_matrix[2][args.BinGau], 
                                       input_matrix[3][args.method],
                                       input_matrix[4][args.sigma], 
                                       input_matrix[5][args.FOV],
                                       input_matrix[6][args.grid],
                                       input_matrix[7][args.save_bv])

file = open("output.txt", "a")
np.savetxt(file, out, fmt='%.4g', newline=' '+descriptor+'\n')
file.close()
sys.exit()