import numpy as np
import pandas as pd

def load_data():
    path_1 = 'data/DMTA_rh/'
    path_2 = 'data/axel_DMTA_data/'

    data = [
        pd.read_table('{}Rh_DMTA_000Storacell15Benecel_1Hz.txt'.format(path_1), decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}Rh_DMTA_025Storacell15Benecel_1Hz.txt'.format(path_1), decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}Rh_DMTA_050Storacell15Benecel_1Hz.txt'.format(path_1), decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}Rh_DMTA_100Storacell15Benecel_1Hz.txt'.format(path_1), decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}Rh_DMTA_150Storacell15Benecel_1Hz.txt'.format(path_1), decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}Rh_DMTA_200Storacell15Benecel_1Hz.txt'.format(path_1), decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}DMTA_Stora05_BC075_1Hz.txt'.format(path_2), decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}DMTA_Stora05_BC10_1Hz.txt'.format(path_2),  decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}DMTA_Stora05_BC125_1Hz.txt'.format(path_2), decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}DMTA_Stora05_BC175_1Hz.txt'.format(path_2), decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}DMTA_Stora05_BC20_1Hz.txt'.format(path_2),  decimal=',', comment='#', encoding='utf-16'),
        pd.read_table('{}DMTA_Stora05_BC25_1Hz.txt'.format(path_2),  decimal=',', comment='#', encoding='utf-16')
    ]

    # Load data:
    # *Minimum of the phase shift angle gradient wrt. temperature
    # *Storage modulus at the minimum of the phase shift angle gradient wrt. temperature
    # *Storage modulus at room temperature (25C)
    # *Loss modulus at room temperature (25C)
    # *Gelification temperature
    # *Yield stress
    # *Yield strain
    # *Young's modulus
    # *Storacell pulp concentration
    # *Benecel pulp concentration
    # *Density of the foam
    min_grad_angle = [ np.min(np.gradient(data[i]['Phase Shift Angle(rad)'], data[i]['Temperature(C)'])) for i in range(len(data)) ]
    storage_modulus_at_min_grad_angle = [ data[i]['Storage Modulus(Pa)'][np.argmin(np.gradient(data[i]['Phase Shift Angle(rad)'], data[i]['Temperature(C)']))] for i in range(len(data)) ]
    storage_modulus = [ data[i]['Storage Modulus(Pa)'][np.argmin((data[i]['Temperature(C)'] - 25)**2)] for i in range(len(data)) ]
    loss_modulus = [ data[i]['Loss Modulus(Pa)'][np.argmin((data[i]['Temperature(C)'] - 25)**2)] for i in range(len(data)) ]

    def t_gel_fun(temp, angle):
        grad = np.gradient(angle, temp)

        grad_max = max(2, np.argmax(grad))
        grad_min = min(len(temp - 3), np.argmin(grad))

        low_t_fit = temp[grad_max - 2:grad_max + 3]
        low_d_fit = angle[grad_max - 2:grad_max + 3]

        hi_t_fit = temp[grad_min - 2:grad_min + 3]
        hi_d_fit = angle[grad_min - 2:grad_min + 3]

        low_p = np.polyfit(low_t_fit, low_d_fit, 1)
        hi_p = np.polyfit(hi_t_fit, hi_d_fit, 1)
        t_gel = (low_p[1] - hi_p[1]) / (hi_p[0] - low_p[0])
        return t_gel

    t_gel = [ t_gel_fun(data[i]['Temperature(C)'], data[i]['Phase Shift Angle(rad)']) for i in range(len(data)) ]

    yield_stress = [
        0.03973, 0.05190, 0.06381, 0.1207, 0.1094, 0.1971,
        0.04862, 0.04193, 0.04339, 0.1422, 0.1351, 0.2408
    ]
    yield_strain = [
        6.540, 6.274, 4.846, 4.760, 8.040, 5.085,
        5.052, 4.715, 8.315, 5.659, 5.293, 5.203
    ]
    youngs_modulus = [
        0.607, 1.098, 1.211, 2.645, 1.336, 4.210,
        0.999, 0.756, 0.616, 2.742, 2.166, 5.816
    ]

    conc_storacell = [
        0.00, 0.25, 0.50, 1.00, 1.50, 2.00,
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50
    ]
    conc_benecel = [
        1.50, 1.50, 1.50, 1.50, 1.50, 1.50,
        0.75, 1.00, 1.25, 1.75, 2.00, 2.50
    ]
    lab_density = [
        20.11, 24.70, 29.23, 48.95, 50.50, 56.20,
        33.10, 32.40, 19.60, 42.90, 41.20, 62.80
    ]

    # Define configuration parameters
    columns = [
        'Min grad angle(rad/C)',
        'Storage modulus at min grad angle(Pa)',
        'Storage modulus at room temperature(Pa)',
        'Loss modulus(Pa)',
        'Gelification temperature(C)',
        'Yield stress(Pa*10^6)',
        'Yield strain(%)',
        "Young's modulus(Pa*10^6)",
        'Storacell(%)',
        'Benecel(%)',
        'Density(kg/m^3)'
    ]

    return [min_grad_angle, storage_modulus_at_min_grad_angle, storage_modulus, loss_modulus, t_gel, yield_stress, yield_strain, youngs_modulus, conc_storacell, conc_benecel, lab_density], columns