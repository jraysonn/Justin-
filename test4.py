import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
import random
import matplotlib.pyplot as plt
from pyswarms.utils.search import RandomSearch
 
def compute_half_angle_vector(theta_i, phi_i, theta_r, phi_r):
    '''
    Half angle vecotr is define as the halfway direction between the viewing and illumination direction
    '''

    incident_normal_vector = np.array([np.cos(phi_i) * np.sin(theta_i),
                                            np.sin(phi_i) * np.sin(theta_i),
                                            np.cos(theta_i)]
                                           )
    exitant_normal_vector = np.array([np.cos(phi_r) * np.sin(theta_r),
                                           np.sin(phi_r) * np.sin(theta_r),
                                           np.cos(theta_r)]
                                          )
    half_angle_vector = (incident_normal_vector + exitant_normal_vector) / 2.

    return half_angle_vector


def compute_bistatic_angle(theta_i, phi_i, theta_r, phi_r):
    """
    Compute the bistatic angle, which is defined as the angle between the incident normal vector and the half angle
    vector
    """

    bistatic_angle = 0.5 * np.arccos(np.cos(theta_i) * np.cos(theta_r)
                                          + np.sin(theta_i) * np.sin(theta_r) * np.cos(phi_r - phi_i))
    return bistatic_angle


def compute_theta_n(theta_i, theta_r, bistatic_angle):
    """
    Compute the normal angle given incident and exitant thetas
    """

    cos_theta_n = (np.cos(theta_i) + np.cos(theta_r)) / (2 * np.cos(bistatic_angle))
    theta_n = np.arccos(cos_theta_n)

    return theta_n


def Fresnel_R( n, k, angle):
    """
    Compute the effective reflectivity over S and P polarizations using the Fresnel equations
    :param n: the index of refraction of the material
    :return: the effective Fresnel reflectivity
    """

    # R_s = (
    #               (np.cos(angle) - n * np.sqrt(1 - ((1 / n) * np.sin(angle)) ** 2.0)) /
    #               (np.cos(angle) + n * np.sqrt(1 - ((1 / n) * np.sin(angle)) ** 2.0))
    #       ) ** 2.0
    #
    # R_p = (
    #               (np.sqrt(1 - ((1 / n) * np.sin(angle)) ** 2.0) - n * np.cos(angle)) /
    #               (np.sqrt(1 - ((1 / n) * np.sin(angle)) ** 2.0) + n * np.cos(angle))
    #       ) ** 2.0
    #
    # R_eff = 0.5 * (R_s + R_p)

    n_complex = n + 1j*k
    rs = ( (np.cos(angle) -  np.sqrt(n_complex**2 - ( np.sin(angle)) ** 2.0)) /
           (np.cos(angle) + np.sqrt(n_complex ** 2 - (np.sin(angle)) ** 2.0))
          )
    rp = ( (np.cos(angle) * n_complex**2 -  np.sqrt(n_complex**2 - ( np.sin(angle)) ** 2.0)) /
           (np.cos(angle) * n_complex ** 2 + np.sqrt(n_complex ** 2 - (np.sin(angle)) ** 2.0))
          )
    R_eff = 0.5 * (np.abs(rs)**2.0 + np.abs(rp)**2.0)

    return R_eff


def SO( theta_n, angle, tau, omega):
    """
    Compute the shadowing obscuration function

    :param angle: the angle under consideration, normally the bistatic angle
    :param tau: the fall-off of shadowing in the backward scatter direction
    :param omega: the fall-off of shadowing in the forward scatter direction
    :param theta_n: the normal angle of the surface direction
    :return: shadowing funrction for the
    """

    so = (1 + (theta_n / omega) * np.exp(-2 * angle / tau)) / (1 + (theta_n / omega))
    return so


def BRDF_fs(theta_n, B, sigma, n, k):
    """
    Compute the facet normal distribution function

    :param B: scaling parameter
    :param theta_n: the surface normal angle
    :param sigma: the mean square value of the facet slope
    :param n: the index of refraction
    :return:
    """

    R_0 = Fresnel_R( n, k, 0)

    # numer = (R_0 * B)
    # denom = (4 * (np.cos(theta_n) ** 3.0) * (sigma ** 2.0 + np.tan(theta_n) ** 2.0))
    rho_fs = (R_0 * B) / (4 * (np.cos(theta_n) ** 3.0) * (sigma ** 2.0 + np.tan(theta_n) ** 2.0))

    return rho_fs

def model_brdf2(theta_i,phi_i,theta_r,phi_r):
    # Unpack the parameters that we are trying to optimize
    # n = random.uniform(3.4,3.7)
    n=3.9980423984083644
    # k = random.uniform(2.2,2.4)
    k =3.6193271231926114
    # tau =random.uniform(np.deg2rad(0),np.deg2rad(3))
    tau =0.02474270739371105
    # omega =random.uniform(np.deg2rad(0),np.deg2rad(5))
    omega=0.0008442950435641663
    # sigma = random.uniform(np.deg2rad(0),np.deg2rad(6))
    sigma= 0.17453292519935268
    # B = random.uniform(np.deg2rad(.001),np.deg2rad(2))
    B=0.02127289125747736
    # rho_d =  random.uniform(np.deg2rad(0.00),np.deg2rad(0.05))
    rho_d =0.09999999998597375
    # rho_v = random.uniform(np.deg2rad(0.00),np.deg2rad(0.0006))
    rho_v = 0.09999999999617754
    # print("values")
    # print(n)
    # print(k)
    # print(tau)
    # print(omega)
    # print(sigma)
    # print(B)
    # print(rho_d)
    # print(rho_v)

    theta_i = theta_i
    theta_r = theta_r
    phi_i = phi_i

    # phi_i = np.deg2rad(phi_i_uniq)
    phi_r = phi_r

    # Compute the necessary angles
    beta = compute_bistatic_angle(theta_i, phi_i, theta_r, phi_r)
    theta_n = compute_theta_n(theta_i, theta_r, beta)

    # Compute the necessary variables
    BRDF_fs_theta_n = BRDF_fs(theta_n, B, sigma,  n, k) # 1/sr
    SO_theta_n = SO(theta_n, beta, tau, omega)
    R0 =  Fresnel_R(n, k, 0)
    R_beta = Fresnel_R(n, k, beta)
    cos_theta_n = np.cos(theta_n)
    cos_theta_i = np.cos(theta_i)
    cos_theta_r = np.cos(theta_r)

    # Compute the modeled BRDF
    term1 = R_beta * ((BRDF_fs_theta_n * cos_theta_n**2.0)/(R0 * cos_theta_i * cos_theta_r)) * SO_theta_n
    model_brdf2 = term1 + rho_d + (2*rho_v)/(cos_theta_i + cos_theta_r)
    # print(csv_brdf.shape)
    # print(model_brdf2.shape)
    residual = csv_brdf - model_brdf2
    # print(residual)
    avg= np.sum(residual)/len(residual)
    print('avg')
    print(avg)
    
    # if -20 < avg >20:
    print(residual)
    # plt.scatter(csv_brdf, residual)
    # plt.show()
    # if   -20 < avg < 20:
         
    # return print(residual)
    # else:
    # return np.sum(residual)/len(residual)
    return model_brdf2
    

# define parameters
# Load in the dataset for the BRDF of solar panels
x_measurements = np.genfromtxt('./White_paintt.csv', delimiter = ",")[1:, :]
csv_wavelength = x_measurements[:,0]
csv_theta_i = x_measurements[:,1]
csv_phi_i = x_measurements[:,2]
csv_theta_r = x_measurements[:,3]
csv_phi_r = x_measurements[:,4]
csv_brdf = x_measurements[:, -1]
    # For each wavelength, read in the corresponding data
unique_wavelengths = np.unique(csv_wavelength)
for uniq_wave in unique_wavelengths:
    where_uniq = np.where(csv_wavelength == uniq_wave)
    theta_i_uniq = csv_theta_i[where_uniq]
    phi_i_uniq = csv_phi_i[where_uniq]
    theta_r_uniq = csv_theta_r[where_uniq]
    phi_r_uniq = csv_phi_r[where_uniq]
    brdf_uniq = csv_brdf[where_uniq]
    ndims = 8
    x_max = 10 * np.ones(ndims)
    x_min = -1 * x_max
#dictionary with x,y values
    values = {
       "theta_i": theta_i_uniq,
       "theta_r":theta_r_uniq
    }

    bounds = (x_min, x_max)
    # c2 being 11 is wrong, we need it be be a fraction
    # options = {'c1':2.639402428815788,'c2':8.487522659302403 , 'w' : 3.895007209666742}
    # g = RandomSearch(GlobalBestPSO, n_particles=40, dimensions=20,
               # options=options, objective_func= model_brdf2, iters=10, n_selection_iters=10)
    # best_score, best_options = g.search()
   
    # optimizer = GlobalBestPSO(n_particles=8, dimensions=8, options=options ) #, bounds=bounds)

    # now run the optimization, pass a=1 and b=100 as a tuple assigned to args
    # cost, pos = optimizer.optimize(model_brdf2, 1000, x_measurements = x_measurements) 