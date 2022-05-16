###make lite version of catalog - most needed columns and sig figs only
###v1 only deals with Table1 (biggest), future versions should handle other tables

import numpy as np
from astropy.table import Table


#################################################################################
#################################################################################
###params - make configurable where appropriate

file_format = 'fits'
file_dir = '../../../../survey_data/VLASS/catalogues/'
components_file_name = 'CIRADA_VLASS1QL_table1_components_v1.fits'

cfile = file_dir + components_file_name


need_cols = ['Component_name', 'RA', 'DEC', 'E_RA', 'E_DEC', 'Total_flux',
             'E_Total_flux', 'Peak_flux', 'E_Peak_flux', 'Isl_rms', 'S_Code',
             'DC_Maj', 'E_DC_Maj', 'DC_Min', 'E_DC_Min', 'DC_PA', 'E_DC_PA',
             'Subtile', 'Source_name', 'Peak_to_ring', 'Duplicate_flag',
             'Quality_flag', 'NN_dist']

###columns for which precision isnt determined automatically
pdict = {'Peak_to_ring': 5, 'NN_dist': 2}

#################################################################################
#################################################################################
###functionality


def lighten_file(datafile, outcols, folder='', file_format='fits', errorcolid='E_',
                 presdict={}):
    'make data table lighter and write new file'
    
    data = Table.read(folder+datafile, format=file_format)
    data = data[need_cols]
    
    ###define which columns need reduced SF (only numeric quanitities) using errors
    ecols = [col for col in outcols if errorcolid in col]
    qcols = [col.replace(errorcolid, '') for col in ecols]
    
    for i in range(len(ecols)):
        qc = qcols[i]
        ec = ecols[i]
        error_array = np.array(data[ec])
        emin = np.min(error_array[(error_array>0)]) ##find min non-0 error for column
        ##use 2 sig figs minumum
        if emin > 0.01:
            sigfigs = 2
        else:
            sigfigs = int(np.abs(np.floor(np.log10(emin)))) + 1
        
        ###round value and error to sigfigs
        data[qc] = np.round(data[qc], sigfigs)
        data[ec] = np.round(data[ec], sigfigs)

    ##additional columns that aren't determined automatically
    xcols = list(presdict.keys())
    for col in xcols:
        sigfigs = presdict[col]
        data[col] = np.round(data[col], sigfigs)
    
    ###write file
    namesplit = datafile.split('.')
    outname = folder + namesplit[0] + '_lite.' + namesplit[1]

    data.write(outname, format=file_format)

    return
