import numpy as np
import yt
import astropy.units as u
from astropy.constants import c, hbar
'''
def velocity_magnitude(field, data):
        code_to_velocity_units = 1./127439397.362 # 1 / code_velocity    
        
        return np.sqrt((data['deposit','all_cic_velocity_x']-halo.mean('all_cic_velocity_x'))**2+
                       (data['deposit','all_cic_velocity_y']-halo.mean('all_cic_velocity_y'))**2+
                       (data['deposit','all_cic_velocity_z']-halo.mean('all_cic_velocity_z'))**2) * code_to_velocity_units
'''
def add_cdm_fields(ds, velocity_magnitude):
    '''
    '''
    
    #halo = ds.sphere(halo_center, (halo_radius, "kpc"))
    #code_to_velocity_units = 1./127439397.362 # 1 / code_velocity 
    
    ds.add_field(("deposit", "velocity_magnitude"), function=velocity_magnitude, units="cm/s", force_override=True)

    return


######################################

def fdens(field, data):
        return data.ds.arr(data['FDMDensity'], 'code_mass/code_length**3')

def FDMMomentum_x(field, data):
    _M           = yt.units.g
    _L           = yt.units.cm
    _t           = yt.units.s
    code_density = 5.38937467957e-28
    psi_units    = (_M/_L**3)**(1./2)
    psi2_to_phys = psi_units**2 * code_density
    #z            = ds.current_redshift
    
    fdm_velocity_x = (data["enzo","Re_Psi"] * data["enzo","Im_Psi_gradient_x"] -
                      data["enzo","Im_Psi"] * data["enzo","Re_Psi_gradient_x"]) / 2 * psi2_to_phys * (1+z)
    return fdm_velocity_x
    
def FDMMomentum_y(field, data):
    _M           = yt.units.g
    _L           = yt.units.cm
    _t           = yt.units.s
    code_density = 5.38937467957e-28
    psi_units    = (_M/_L**3)**(1./2)
    psi2_to_phys = psi_units**2 * code_density
    #z            = ds.current_redshift
    
    fdm_velocity_y = (data["enzo","Re_Psi"] * data["enzo","Im_Psi_gradient_y"] -
                      data["enzo","Im_Psi"] * data["enzo","Re_Psi_gradient_y"]) / 2 * psi2_to_phys * (1+z)
    return fdm_velocity_y

def FDMMomentum_z(field, data):
    _M           = yt.units.g
    _L           = yt.units.cm
    _t           = yt.units.s
    code_density = 5.38937467957e-28
    psi_units    = (_M/_L**3)**(1./2)
    psi2_to_phys = psi_units**2 * code_density
    #z            = ds.current_redshift
    
    fdm_velocity_z = (data["enzo","Re_Psi"] * data["enzo","Im_Psi_gradient_z"] -
                      data["enzo","Im_Psi"] * data["enzo","Re_Psi_gradient_z"]) / 2 * psi2_to_phys * (1+z)
    return fdm_velocity_z
'''
def Re_Psi_Shifted(field, data):
    phi          = (data["enzo","x"] * mean_velocity_x + 
                    data["enzo","y"] * mean_velocity_y + 
                    data["enzo","z"] * mean_velocity_z)
    psi_shifted  = data["enzo","Re_Psi"]*np.cos(phi) - data["enzo","Im_Psi"]*np.sin(phi)
    psi_shifted *= np.sqrt(psi2_to_phys)
    return psi_shifted

def Im_Psi_Shifted(field, data):
    phi          = (data["enzo","x"] * mean_velocity_x + 
                    data["enzo","y"] * mean_velocity_y + 
                    data["enzo","z"] * mean_velocity_z)
    psi_shifted  = data["enzo","Im_Psi"]*np.cos(phi) + data["enzo","Re_Psi"]*np.sin(phi)
    psi_shifted *= np.sqrt(psi2_to_phys)
    return psi_shifted
'''

def Root_FDMDensity(field, data):
    _M           = yt.units.g
    _L           = yt.units.cm
    _t           = yt.units.s
    code_density = 5.38937467957e-28
    psi_units    = (_M/_L**3)**(1./2)
    psi2_to_phys = psi_units**2 * code_density
    
    return np.sqrt(data["enzo","FDMDensity"] * psi2_to_phys)

def FDMVelocity_dispersion(field, data):
    m            = 1e-22 * u.eV / c**2
    m            = m.cgs
    #z            = ds.current_redshift
    coef         = (hbar/m/2/(z+1)).cgs
    c_units      = coef.unit
    coef         = coef.value
    
    psi_magnitude  = np.sqrt(data["enzo","Re_Psi_Shifted"]**2 + 
                             data["enzo","Im_Psi_Shifted"]**2)
    
    E_internal     = (data["enzo","Root_FDMDensity_gradient_x"]**2 +
                      data["enzo","Root_FDMDensity_gradient_y"]**2 +
                      data["enzo","Root_FDMDensity_gradient_z"]**2)
    
    
    mag_grad_psi   = np.sqrt(data["enzo","Re_Psi_Shifted_gradient_x"]**2 + 
                             data["enzo","Im_Psi_Shifted_gradient_x"]**2 +
                             data["enzo","Re_Psi_Shifted_gradient_y"]**2 + 
                             data["enzo","Im_Psi_Shifted_gradient_y"]**2 +
                             data["enzo","Re_Psi_Shifted_gradient_z"]**2 + 
                             data["enzo","Im_Psi_Shifted_gradient_z"]**2)
    
    velocity_disp  = (mag_grad_psi**2) / psi_magnitude**2# - E_internal / psi_magnitude**2
    #velocity_disp  = E_internal/ psi_magnitude**2
    velocity_disp *= coef * yt.units.cm**4 / yt.units.s**2
    
    return np.sqrt(velocity_disp)


def add_fdm_fields(ds, part=1):
        
        '''
        _M           = yt.units.g
        _L           = yt.units.cm
        _t           = yt.units.s
        
        m            = 1e-22 * u.eV / c**2
        m            = m.cgs
        z            = ds.current_redshift
        
        coef         = (hbar/m/2/(z+1)).cgs
        c_units      = coef.unit
        coef         = coef.value
        
        code_density = 5.38937467957e-28
        psi_units    = (_M/_L**3)**(1./2)
        psi2_to_phys = psi_units**2 * code_density
        '''
        global z
        z = ds.current_redshift
        
        if part == 1:
            
            ds.add_field(('fdens'), function=fdens, units='g/cm**3', sampling_type='cell',force_override=True)
            ds.add_gradient_fields(("enzo","Re_Psi"))
            ds.add_gradient_fields(("enzo","Im_Psi"))
            ds.add_field(("enzo", "FDMMomentum_x"), function=FDMMomentum_x, units="g/cm**4", sampling_type='cell', force_override=True)
            ds.add_field(("enzo", "FDMMomentum_y"), function=FDMMomentum_y, units="g/cm**4", sampling_type='cell', force_override=True)
            ds.add_field(("enzo", "FDMMomentum_z"), function=FDMMomentum_z, units="g/cm**4", sampling_type='cell', force_override=True)
        
        '''
        halo      = ds.sphere(halo_center, (halo_radius, "kpc"))
        mean_rho  = halo.mean("FDMDensity") # currently dimensionless (assume code units)
        mean_rho *= 5.38937467957e-28 * yt.units.g / yt.units.cm**3
        
        # j/rho term
        mean_velocity_x = halo_sphere.mean("FDMMomentum_x") / mean_rho
        mean_velocity_y = halo_sphere.mean("FDMMomentum_y") / mean_rho
        mean_velocity_z = halo_sphere.mean("FDMMomentum_z") / mean_rho
        
        ds.add_field(("enzo", "Re_Psi_Shifted"), function=Re_Psi_Shifted, units="g**(1./2)/cm**(3./2)", sampling_type='cell', force_override=True)
        ds.add_field(("enzo", "Im_Psi_Shifted"), function=Im_Psi_Shifted, units="g**(1./2)/cm**(3./2)", sampling_type='cell', force_override=True)
        '''
        if part==2:
            
            ds.add_gradient_fields(("enzo","Re_Psi_Shifted"))
            ds.add_gradient_fields(("enzo","Im_Psi_Shifted"))
            
            # for internal energy, might neglect
            ds.add_field(("enzo", "Root_FDMDensity"), function=Root_FDMDensity, units="g**(1./2)/cm**(3./2)", sampling_type='cell', force_override=True)
            ds.add_gradient_fields(("enzo","Root_FDMDensity"))
            
            # finally, the velocity dispersion
            ds.add_field(("enzo", "FDMVelocity_dispersion"), function=FDMVelocity_dispersion, units="cm/s", sampling_type='cell', force_override=True)
        
