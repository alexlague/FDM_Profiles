import numpy as np
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM

###################### CDM #############################

def nfw(r,r_s,rho_s):
    return rho_s / ((r/r_s)*(1+(r/r_s))**2)

def find_halo_radius_cdm(radius, dm_density, mean_rho_box):
    valid_pts  = dm_density>0
    fitted_nfw = curve_fit(nfw, radius[valid_pts], dm_density[valid_pts]*1e25)[0] # need to scale for fit
    fit_r_s    = fitted_nfw[0]
    fit_rho_s  = fitted_nfw[1]
    
    integrand  = lambda r: nfw(r, fit_r_s, fit_rho_s) * r**2
    integral   = lambda R: quad(integrand,0,R)[0] / quad(lambda x: x**2,0,R)[0] # int_0^R r^2 rho(r) dr / int_0^R r^2 dr
    sol_for_R  = lambda R_halo: integral(R_halo) - 1e25*200*mean_rho_box.value
    R_halo     = fsolve(sol_for_R, 10)[0]
    
    return R_halo, fit_r_s, fit_rho_s


###################### FDM #############################

def soliton(r,r_c,rho_0):
    return rho_0 / (1+9.1e-2*(r/r_c)**2)**8

def find_halo_radius_fdm(radius, dm_density, mean_rho_box):
    if dm_density.mean() > 0.1:
        dm_density *= 5.38937467957e-28
    
    if mean_rho_box > 0.1:
        mean_rho_box *= 5.38937467957e-28
    
    zero_pts   = dm_density.value == 0.
    last_zero  = radius[zero_pts][-1] # radius of transition between soliton and nfw
    
    # nfw tail
    valid_pts  = (dm_density>0) & (radius > last_zero)
    fitted_nfw = curve_fit(nfw, radius[valid_pts].value, dm_density[valid_pts].value*1e25, bounds=([5,1e-1],[40,1e2]))[0]
    fit_r_s    = fitted_nfw[0]
    fit_rho_s  = fitted_nfw[1]
    print(fit_r_s, fit_rho_s)

    # soliton core
    valid_pts  = (dm_density>0) & (radius <= last_zero)
    fitted_sol = curve_fit(soliton, radius[valid_pts], dm_density[valid_pts]*1e25, bounds=([0.1,1e-1],[3,1e2]))[0]
    fit_r_c    = fitted_sol[0]
    fit_rho_0  = fitted_sol[1]
    print(fit_r_c, fit_rho_0)
    
    # solve for halo radius with nfw tail
    integrand  = lambda r: nfw(r, fit_r_s, fit_rho_s) * r**2
    integral   = lambda R: quad(integrand,0,R)[0] / quad(lambda x: x**2,0,R)[0]
    sol_for_R  = lambda R_halo: integral(R_halo) - 200*mean_rho_box.value*1e25
    R_halo     = fsolve(sol_for_R, 10)

    return R_halo, fit_r_s, fit_rho_s, fit_r_c, fit_rho_0


def find_core_radius(M_halo, redshift, hubble, omega_matter):
    '''
    From Schive et al.
    '''
    m22      = 1.
    cosmo    = FlatLambdaCDM(H0=hubble*100, Om0=omega_matter, Tcmb0=2.725)
    xi       = (18*np.pi**2 + 82*(cosmo.Om(redshift)-1) - 39*(cosmo.Om(redshift)-1)**2) / cosmo.Om(redshift)
    xi0      = (18*np.pi**2 + 82*(cosmo.Om(0)-1) - 39*(cosmo.Om(0)-1)**2) / cosmo.Om(0)
    r_core   = 1.6 * m22**(-1./2) * 1./(1+redshift)**(1./2) * (xi/xi0)**(-1./6) * (M_halo/1e9)**(-1./3) #kpc 
    
    return r_core
