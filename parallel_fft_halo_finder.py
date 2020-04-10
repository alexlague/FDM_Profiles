################################
# PARALLEL HALO FINDING
# 
# 
################################


import numpy as np
import pyfftw
import yt
import h5py
import multiprocessing
import os.path
import time

#yt.enable_parallelism()
#### INITIALIZATION ####

# Halo finding parameters
error_margin = 0.1
max_steps    = 50
threshold    = 200.

#### FUNCTION DEFINITIONS ####

def top_hat_filter(radius, distances, ngrid):
    '''
    Cubic array of 1 in ball of radius R 
    Normalized to add up to 1
    '''
    
    top_hat_indexes  = distances <= radius
    top_hat_region   = np.zeros((ngrid, ngrid, ngrid))
    top_hat_region[top_hat_indexes] = 1.0
    top_hat_region  /= np.sum(top_hat_region) # normalization

    return top_hat_region


def is_halo(radius, grid_density, distances, ngrid):
    '''
    Checks if there is any point
    where the smoothed density exceeds
    the threshold to be considered a halo
    The smoothing kernel varies and is
    specified by the smoothing radius (input)
    '''
    # Parallel part
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    pyfftw.interfaces.cache.enable()
    # Convolution with top-hat in Fourier space
    #smoothed_density = np.fft.ifftn(np.fft.fftn(grid_density) * np.fft.fftn(top_hat_filter(radius, distances, ngrid)))
    smoothed_density = pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.fftn(grid_density) 
                                                         * pyfftw.interfaces.numpy_fft.fftn(top_hat_filter(radius, distances, ngrid)))
    smoothed_density = np.real(smoothed_density)
    thresh_density   = threshold * np.mean(smoothed_density)
    
    if np.max(smoothed_density) > thresh_density:    
        #print("Max greater than thresh")
        print(np.unravel_index(np.argmax(smoothed_density, axis=None), smoothed_density.shape))
        return False
    else:
        #print("Max NOT greater than thresh")
        return True


def fdens(field, data):
    '''
    Creates the FDM density field with correct
    density units
    '''
    return data.ds.arr(data['FDMDensity'], 'code_mass/code_length**3')

#### MAIN FUNCTION ####

def halo_finding_iterations(ds, mean_rho_box, dm):
    # Initial values of radius and step size
    current_radius = 30.
    delta          = 100.
    
    # Load data and snapshot information
    #ds         = yt.load(snapshot + '/' + snapshot)
    boxsize    = 2e3 # in kpc
    ngrid      = ds.domain_dimensions[0]
    redshift   = ds.current_redshift
    #ytfield    = 'fdens'
    
    # Add FDM density with proper units
    if dm == 'fdm':
        ds.add_field(('fdens'), function=fdens, units='g/cm**3',
                     sampling_type='cell',force_override=True)
    
    #reg        = ds.r[:,:,:]
    #mean_rho   = reg.mean(ytfield).in_units('Msun/kpc**3').value
    
    # Switch to CMS Units 
    if mean_rho_box > 0.1:
        mean_rho_box *= 5.38937467957e-28
    
    # Top-hat filter
    box_at_z  = boxsize / (1 + redshift)
    axis      = np.linspace(-box_at_z/2, box_at_z/2, ngrid)
    mesh      = np.meshgrid(axis, axis, axis)
    distances = np.sqrt(mesh[0]**2 + mesh[1]**2 + mesh[2]**2) # also in kpc
    
    # Create density field
    all_data_level_0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    if dm == 'cdm':
        ytfield = ('deposit', 'all_cic')
    elif dm =='fdm':
        ytfield = 'fdens'
    else:
        print('Wrong DM type')
    
    grid_density     = all_data_level_0[ytfield]
    
    for step in range(max_steps):
        if step == 0:
            old_result = True

        # Prevent R-delta<0
        if current_radius <= 0:
            current_radius += delta
            while current_radius <= delta:
                delta /= 2
        
        # Evaluate if there is a halo at that R
        current_result = is_halo(current_radius, grid_density, distances, ngrid)
        
        # Find closer value to TRUE halo radius
        if current_result and ~old_result:
            current_radius -= delta
        elif current_result and old_result:
            delta       /= 2
            current_radius -= delta
        elif ~current_result and old_result:
            delta       /= 2
            current_radius += delta
        else:
            current_radius += delta
        
        print("Radius = " + str(current_radius))
        print("Delta = " + str(delta))
        
        # If close enough to TRUE radius, stop
        if delta <= error_margin:
            break
    '''
    # Save halo mass and redshift to file
    with open('halo_masses.dat', 'a') as file_object:
        
        # Calculate halo radius and mass
        print('Found Halo at Redshift ' + str(redshift))
    '''
    cms_to_kpc      = 1.47751334e+31
    radius_estimate = current_radius + delta/2
    mass_estimate   = 4*np.pi/3 *threshold * mean_rho_box * cms_to_kpc  * radius_estimate**3 
    
    #file_object.write('\n' + str(redshift) + '\t' + str(mass_estimate)  + '\t' + str(radius_estimate))
    print("Radius= ", radius_estimate)
    return radius_estimate, mass_estimate
    



#snapshot = 'DD0055'
#x = halo_finding_iterations(snapshot)


'''
#### PARALLEL LOOP OVER SNAPSHOTS ####

# Creating file list to loop through
file_list   = []
file_exists = True
file_number = 0
file_name   = 'DD0000'

while file_exists == True:
    file_list.append(file_name)
    file_number += 1
    file_name   = 'DD' + str(file_number).rjust(4,'0')
    file_exists = os.path.exists(file_name)

# Creating file to store data
f = open('halo_masses.dat','w+')
f.write('Redshift' + '\t' + 'Halo Mass (Msun)' + '\t' + 'Halo Radius (kpc)')
f.close()

# Parallel loop
#n_cores = multiprocessing.cpu_count()
#p = multiprocessing.Pool(n_cores)
#p.map(halo_finding_iterations, file_list[::4])

for snapshot in file_list[20:][::6]:
    #time1 = time.time()
    x = halo_finding_iterations(snapshot)
    #time2 = time.time()
    #print(time2-time1)

'''
