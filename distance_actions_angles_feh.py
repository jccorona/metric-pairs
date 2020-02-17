import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time
import scipy
from  galpy.util import bovy_coords

## Reading the data from a fits file ##
hdulist = fits.open('file.fits')
tbdata = hdulist[1]
tbdata.data.shape
tbdata.header
tbdata = hdulist[1].data # assume the first extension is a table
feh = tbdata.field('FEH')
mg_feh = tbdata.field('MG_FE')
JR = tbdata.field("JR_kpckms")
Jz = tbdata.field("Jz_kpckms")
Lz = tbdata.field("Lz_kpckms")
TR = tbdata.field("TR_rad")
TP = tbdata.field("TP_rad")
TZ = tbdata.field("TZ_rad")

## index to select stars with proper action,angle and feh values ## 
idx_in = np.where((feh!=-9) &(alpha!=-9)&(mg_feh!=-999.)&(JR<9000*8*220)&(Jz<9000*8*220)&(Lz<9000*8*220)&(np.isfinite(TZ))&(np.isfinite(TR))&(TP<9999.99) &(TR<9999.99) &(TZ<9999.99))

JR = tbdata.field("JR_kpckms")[idx_in]
Jz = tbdata.field("Jz_kpckms")[idx_in]
Lz = tbdata.field("Lz_kpckms")[idx_in]
TR = tbdata.field("TR_rad")[idx_in]
TP = tbdata.field("TP_rad")[idx_in]
TZ = tbdata.field("TZ_rad")[idx_in]
pmra = tbdata.field("PMRA")[idx_in]
pmdec = tbdata.field("PMDEC")[idx_in]
ra  = tbdata.field("RA")[idx_in]
dec = tbdata.field("DEC")[idx_in]
feh =  tbdata.field('FEH')[idx_in]
RV = tbdata.field('RV_kms')[idx_in]
d_kpc = tbdata.field('d_mcmc_pc')[idx_in]/1000


### coordinate transformation to X,Y,Z, U,V,W ###
lb = bovy_coords.radec_to_lb(
                ra,dec,
                degree=True,epoch=2000
                )
l = lb[:,0]
b = lb[:,1]

xyz = bovy_coords.lbd_to_XYZ(l,b,d_kpc,degree=True)

x_HC = xyz[:,0]
y_HC = xyz[:,1]
z_HC = xyz[:,2]

pmlpmb = bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec,degree=True, epoch=2000.0)
pml = pmlpmb[:,0]
pmb = pmlpmb[:,1]
vxvyvz = bovy_coords.vrpmllpmbb_to_vxvyvz(RV, pml,pmb,l,b,d_kpc,XYZ=False,degree=True)
vx_HC = vxvyvz[:,0]
vy_HC = vxvyvz[:,1]
vz_HC = vxvyvz[:,2]

## Creating arrays for the dataset so the pairwise distance computation is done efficiently ##
data_arrange = np.zeros((len(feh),17))
data_arrange[:,0] = Lz
data_arrange[:,1] = JR
data_arrange[:,2] = Jz
data_arrange[:,3] = TP
data_arrange[:,4] = TR
data_arrange[:,5] = TZ
data_arrange[:,6] = feh
data_arrange[:,7] = vx_HC
data_arrange[:,8] = vy_HC
data_arrange[:,9] = vz_HC
data_arrange[:,10] = x_HC
data_arrange[:,11] = y_HC
data_arrange[:,12] = z_HC
data_arrange[:,13] = ra
data_arrange[:,14] = dec
data_arrange[:,15] = d_kpc
data_arrange[:,16] = RV

var_JR = np.var(JR)
var_Jz = np.var(Jz)
var_Lz = np.var(Lz)
var_thetaR = 2.94
var_thetaP = 0.03
var_thetaz = 2.94
var_array = np.array([var_Lz,var_JR,var_Jz, var_thetaP, var_thetaR, var_thetaz])


## Defining the bins for the histogram when considering V,X and d_proj##
#log_bin_V = np.arange(-1,3.05,0.05)
#log_bin_x = np.arange(-2.5,1.05,0.05)
#log_bin_dproj = np.arange(-4,1.05,0.05)
#bin_V = 10**log_bin_V
#bin_x = 10**log_bin_x
#bin_dproj = 10**log_bin_dproj
#count_all = np.zeros((bin_metric.size-1, bin_V.size-1, bin_x.size-1))
#count_all = np.zeros((bin_metric.size-1, bin_V.size-1, bin_dproj.size-1))
############################################

## Defining the bins for the histogram in actions-angles and metallicity, feh ##
log_bin_metric = np.arange(-2.5,2.01, 0.01)
bin_feh = np.arange(0,2.01,0.01)
bin_metric = 10**log_bin_metric
count_all = np.zeros((bin_metric.size-1, bin_feh.size-1))
time_tot = 0.

#n = np.array([])
#w = np.array([])
#metric = np.array([])

Niter = len(data_arrange[:,0])
npor = 0
for ii in range(Niter-1):
    progress = ii/np.float(Niter)*100
    if progress > npor:
       print("{:d}% completed".format(npor))
       npor += 1

    start = time.time()
    star = data_arrange[ii,:3]
    other_stars = data_arrange[ii+1:,:3]
    star2 = data_arrange[ii,3]
    other_stars2 = data_arrange[ii+1:,3]
    star3 = data_arrange[ii,4]
    other_stars3 = data_arrange[ii+1:,4]
    star4 = data_arrange[ii,5]
    other_stars4 = data_arrange[ii+1:,5]
    star_feh = data_arrange[ii,6]
    other_starsfeh = data_arrange[ii+1:,6]
    ## distance between three actions ##    
    dist = (star-other_stars)**2
    dist /= var_array[:3]
    ## distance for the angles ##
    diff2 = np.fabs(star2 - other_stars2)
    diff2[diff2>np.pi] = 2*np.pi - diff2[diff2>np.pi]
    dist2 = diff2**2
    dist2 /= var_array[3]
    diff3 = np.fabs(star3 - other_stars3) 
    diff3[diff3>np.pi] = 2*np.pi - diff3[diff3>np.pi]
    dist3 = diff3**2
    dist3 /= var_array[4]
    diff4 = np.fabs(star4 - other_stars4) 
    diff4[diff4>np.pi] = 2*np.pi - diff4[diff4>np.pi]
    dist4 = diff4**2
    dist4 /= var_array[5]
    ## distance in metallicity ##
    dist_feh  = np.fabs(star_feh - other_starsfeh)
    metric_Jtheta= (np.sum(dist,axis=1) + dist2 + dist3 + dist4)
    metric_Jtheta = np.sqrt(metric_Jtheta)/np.sqrt(6)
    ## this saves the histogram with the defined bins ##
    counts,edges = np.histogramdd([metric_Jtheta, dist_feh],bins=[bin_metric, bin_feh])
    count_all += counts
    time_tot += (time.time()-start)/60.
    
    ## This calculates distances in X,V, and d_proj ##
    #starU = data_arrange[ii,7]
    #other_starsU = data_arrange[ii+1:,7]
    #starV = data_arrange[ii,8]
    #other_starsV = data_arrange[ii+1:,8]
    #starW = data_arrange[ii,9]
    #other_starsW = data_arrange[ii+1:,9]
    #starx = data_arrange[ii,10]
    #other_starsx = data_arrange[ii+1:,10]
    #stary = data_arrange[ii,11]
    #other_starsy = data_arrange[ii+1:,11]
    #starz = data_arrange[ii,12]
    #other_starsz = data_arrange[ii+1:,12]
    #star_ra = data_arrange[ii,13]
    #other_starsra = data_arrange[ii+1:,13]
    #star_dec = data_arrange[ii,14]
    #other_starsdec = data_arrange[ii+1:,14]
    #distU = (starU-other_starsU)**2
    #distV = (starV -other_starsV)**2
    #distW = (starW - other_starsW)**2
    #distx = (starx - other_starsx)**2
    #disty = (stary - other_starsy)**2
    #distz = (starz - other_starsz)**2
    #dist5 = np.sqrt(distU + distV + distW)
    #dist6 = np.sqrt(distx + disty + distz)
    #theta = np.arccos(np.sin(np.radians(star_dec))*np.sin(np.radians(other_starsdec)) + np.cos(np.radians(star_dec))*np.cos(np.radians(other_starsdec))*np.cos(np.radians(star_ra) -np.radians(other_starsra)))
    #d_proj = theta*((data_arrange[ii,15]+ data_arrange[ii+1:,15])/2)
    #counts,edges = np.histogramdd([metric_Jtheta, dist5, d_proj],bins=[bin_metric, bin_V, bin_dproj])
    
    #### Defining an index to save the actual pairs up to metric_Jtheta <=10**(-1.3) ####
    #idx = np.where((metric_Jtheta >10**(-2.5))&(metric_Jtheta <=10**(-1.3)))[0]
    #w = np.append(w,ii+1+idx)
    #n = np.append(n,np.full_like(idx,ii))
    #metric  = np.append(metric, metric_Jtheta[idx])
    
    ##this is Kamdar+2019 selection##
    #idx = np.where((dist5<1.5)&(dist6>0.002)&(dist6<0.02))[0]
    #w = np.append(w,ii+1+idx)
    #n = np.append(n,np.full_like(idx,ii))
    #metric  = np.append(metric, metric_Jtheta[idx])

#np.savetxt('stars_actions_first_bin.txt', np.column_stack((n,w,(metric))), fmt='%d %d %.6e')
np.save('hist_6d_X.npy',count_all)

print (time_tot)
