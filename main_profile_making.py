import numpy as np
from astropy.cosmology import Planck15
import yt
import argparse
from radial_averages import cdm_density_profile, fdm_density_profile, cdm_velocity_profile, fdm_velocity_profile
from add_fields import add_cdm_fields, add_fdm_fields
from halo_radius import find_core_radius #find_halo_radius_cdm, find_halo_radius_fdm
from parallel_fft_halo_finder import halo_finding_iterations

#yt.enable_parallelism()

# Take argument
parser = argparse.ArgumentParser()
parser.add_argument("seed", type=str, help="Name seed to analyze")
parser.add_argument("DM", type=str, help="Type of DM: CDM, FDM, BOTH")
parser.add_argument("snap", type=str, help="Snapshot number: DD0055, RD0006")
args = parser.parse_args()

#seed_numb = '13589'
seed_numb = args.seed
#directory = '/scratch/r/rbond/alague/Nbody/xinyu_li_shared_files/Profile_Runs/' + seed_numb + '/'
directory = '/scratch/r/rbond/alague/Nbody/xinyu_li_shared_files/test_halo_run/'
snapshot  = args.snap

plotting_radius = 50

if str(args.DM) == 'CDM' or str(args.DM) == 'BOTH':
        file_name    = directory + 'cdm/' + snapshot + '/' + snapshot 
        ds           = yt.load(file_name)
        ytfield      = 'all_cic'
        halo_center  = ds.find_max(ytfield)[1]
        all_data     = ds.all_data()
        mean_rho_box = all_data['deposit', ytfield].mean()
        #rho_crit = Planck15.critical_density(ds.current_redshift)
        #mean_rho_box = rho_crit * ds.omega_matter
        
        radius, dm_density = cdm_density_profile(ds, halo_center, plotting_radius)
        # MOST RECENT CHANGE WITH PARALLEL FFT HALO FINDER
        #R_halo, r_s, rho_s = find_halo_radius_cdm(radius, dm_density, mean_rho_box)
        R_halo, M_halo = halo_finding_iterations(ds, mean_rho_box, 'cdm')
        
        halo         = ds.sphere(halo_center, (R_halo, "kpc")) 

        def velocity_magnitude(field, data):
                code_to_velocity_units = 1./127439397.362 # 1 / code_velocity
                return np.sqrt((data['deposit','all_cic_velocity_x']-halo.mean('all_cic_velocity_x'))**2+
                               (data['deposit','all_cic_velocity_y']-halo.mean('all_cic_velocity_y'))**2+
                               (data['deposit','all_cic_velocity_z']-halo.mean('all_cic_velocity_z'))**2) * code_to_velocity_units

        added_fields = add_cdm_fields(ds, velocity_magnitude)

        vel_radius, vel_disp   = cdm_velocity_profile(ds, halo_center, plotting_radius, R_halo)
        
        data = np.array([radius.value, dm_density.value, vel_radius.value, vel_disp.value])
        np.savetxt(seed_numb + '/cdm/' + snapshot  + '_cdm_profiles.dat', data)
        np.savetxt(seed_numb + '/cdm/' + snapshot  + '_cdm_halo_properties.dat', [R_halo, M_halo, ds.current_redshift])
                
                
############################################### 
###############################################

if str(args.DM) == 'FDM' or str(args.DM) == 'BOTH':
        file_name    = directory + 'fdm/' + snapshot + '/' + snapshot
        ds           = yt.load(file_name)
        ytfield      = 'FDMDensity'
        
        _M           = yt.units.g
        _L           = yt.units.cm
        _t           = yt.units.s
        code_density = 5.38937467957e-28
        psi_units    = (_M/_L**3)**(1./2)
        global psi2_to_phys    
        psi2_to_phys = psi_units**2 * code_density
        
        halo_center  = ds.find_max(ytfield)[1]
        all_data     = ds.all_data()
        mean_rho_box = all_data[ytfield].mean()
        #rho_crit = Planck15.critical_density(ds.current_redshift)
        #mean_rho_box = rho_crit * ds.omega_matter
        
        radius, dm_density = fdm_density_profile(ds, halo_center, plotting_radius)
        #R_halo, r_s, rho_s, r_c, rho_c = find_halo_radius_fdm(radius, dm_density, mean_rho_box)
        R_halo, M_halo = halo_finding_iterations(ds, mean_rho_box, 'fdm')
       

        print("### FOUND HALO RADIUS ###")
        
        halo      = ds.sphere(halo_center, (R_halo, "kpc"))
        mean_rho  = halo.mean("FDMDensity") # currently dimensionless (assume code units)
        mean_rho *= 5.38937467957e-28 * yt.units.g / yt.units.cm**3 #mea density for the halo
        
        
        added_fields = add_fdm_fields(ds, part=1)
        
        # j/rho term
        mean_velocity_x = halo.mean("FDMMomentum_x") / mean_rho
        mean_velocity_y = halo.mean("FDMMomentum_y") / mean_rho
        mean_velocity_z = halo.mean("FDMMomentum_z") / mean_rho
        
        
        print("### FOUND MEAN VELOCITY ###")

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

        ds.add_field(("enzo", "Re_Psi_Shifted"), function=Re_Psi_Shifted, units="g**(1./2)/cm**(3./2)", sampling_type='cell', force_override=True)
        ds.add_field(("enzo", "Im_Psi_Shifted"), function=Re_Psi_Shifted, units="g**(1./2)/cm**(3./2)", sampling_type='cell', force_override=True)
        added_fields = add_fdm_fields(ds, part=2)
        
        print("### FIELDS ADDED ###")

        vel_radius, vel_disp   = fdm_velocity_profile(ds, halo_center, plotting_radius, R_halo)

        print("### FOUND VELOCITY DISPERSION ###")

        r_c = find_core_radius(M_halo, ds.current_redshift, ds.hubble_constant, ds.omega_matter)
        
        print("### FOUND CORE RADIUS ###")

        data = np.array([radius.value, dm_density.value, vel_radius.value, vel_disp.value])
        #np.savetxt(seed_numb + '/fdm/' + snapshot  + '_fdm_profiles.dat', data)
        #np.savetxt(seed_numb + '/fdm/' + snapshot  + '_fdm_halo_properties.dat', [R_halo, M_halo, ds.current_redshift, r_c])
        np.savetxt('FDM_Small_Box/' + snapshot  + '_fdm_profiles.dat', data)
        np.savetxt('FDM_Small_Box/' + snapshot  + '_fdm_halo_properties.dat', [R_halo, M_halo, ds.current_redshift, r_c])
#else:
 #       print('Incorrect DM type: CDM, FDM or BOTH')
