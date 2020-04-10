import yt
import numpy as np

def cdm_density_profile(ds, halo_center, plotting_radius):
    plotting_sphere = ds.sphere(halo_center, (plotting_radius, "kpc"))

    density_profile = yt.create_profile(plotting_sphere, 'radius',('deposit','all_density'),
                                        units = {'radius': 'kpc'},
                                        extrema = {'radius': ((0.1, 'kpc'), (plotting_radius, 'kpc'))},
                                        weight_field='all_count',
                                        n_bins=32)
    radius   = density_profile.x
    density  = density_profile['deposit','all_density']
    
    return radius, density


def cdm_velocity_profile(ds, halo_center, plotting_radius, halo_radius):
    plotting_sphere = ds.sphere(halo_center, (plotting_radius, "kpc"))
    
    velocity_profile = yt.create_profile(plotting_sphere, 'radius',('deposit','velocity_magnitude'),
                                         units = {'radius': 'kpc'},
                                         extrema = {'radius': ((.05*halo_radius, 'kpc'), (2*halo_radius, 'kpc'))},
                                         weight_field='all_count',
                                         n_bins=32, logs={'radius':False})
    
    radius   = velocity_profile.x
    vel_disp = velocity_profile.standard_deviation['deposit','velocity_magnitude']

    return radius, vel_disp


#########################################################

def fdm_density_profile(ds, halo_center, plotting_radius):
    plotting_sphere = ds.sphere(halo_center, (plotting_radius, "kpc"))

    density_profile = yt.create_profile(plotting_sphere, 'radius', 'FDMDensity',
                                        units = {'radius': 'kpc'},
                                        extrema = {'radius': ((0.1, 'kpc'), (plotting_radius, 'kpc'))},
                                        n_bins=32)
    radius   = density_profile.x
    density  = density_profile[('enzo', 'FDMDensity')]

    return radius, density



def fdm_velocity_profile(ds, halo_center, plotting_radius, halo_radius):
    plotting_sphere = ds.sphere(halo_center, (plotting_radius, "kpc"))

    velocity_profile = yt.create_profile(plotting_sphere, 'radius', 'FDMVelocity_dispersion',
                                         units = {'radius': 'kpc'},
                                         extrema = {'radius': ((0.05*halo_radius, 'kpc'), (2*halo_radius, 'kpc'))},
                                         n_bins=32, logs={'radius':False})

    radius   = velocity_profile.x
    vel_disp = velocity_profile[('enzo', 'FDMVelocity_dispersion')]

    return radius, vel_disp

