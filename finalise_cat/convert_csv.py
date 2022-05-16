####script to take csv file outputs from CIRADA VLASS pipeline and convert format
####adds in metadata column metadata too (i.e. units)
####incorporated this into finalise_cat prior to Manitoba transition

import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy.time import Time

############################################################################
############################################################################
##define parameters - pop thes into a config file at some point

tab1file = 'CIRADA_VLASS1QL_table1_components_v1.csv'
tab2file = 'CIRADA_VLASS1QL_table2_hosts_v1.csv'
tab3file = 'CIRADA_VLASS1QL_table3_subtile_info_v1.csv'

outdir = '../../CIRADA_output_files/v1/fits_format/'
indir = '../../CIRADA_output_files/v1/'

outformat = 'fits'
informat = 'csv'

############################################################################
############################################################################
###def functions

def add_units_to_columns(table, unitdict):
    'takes astropy table and dictionary of column names and units and adds units to table'
    qcols = list(unitdict.keys())
    cols = table.colnames
    
    for col in cols:
        if col in qcols:
            table[col].unit = unitdict[col]
    
    return(table)


def convert_col_dtype(table, dtdict):
    'convert dtype of column, e.g. float -> int'
    
    tcols = list(dtdict.keys())
    cols = table.colnames
    
    for col in cols:
        if col in tcols:
            if dtdict[col] == Time:
                table[col] = dtdict[col](np.array(table[col]))
            else:
                table[col] = np.array(table[col]).astype(dtdict[col])
    
    return(table)


def update_metadata(table, unitdict, dtdict):
    'update table metadata - convert dtypes and add units as needed'
    
    table = convert_col_dtype(table=table, dtdict=dtdict)
    table = add_units_to_columns(table=table, unitdict=unitdict)
    
    return(table)


def output_catalogue(compfile, hostfile, stfile, compu, compt, hostu, hostt,
                     stu, stt, informat='csv', outformat='fits', indir='', outdir='',
                     overwrite=False):
    'read in catalogue files, add metadata, output updated files'
    
    filelist = [compfile, hostfile, stfile]
    udicts = [compu, hostu, stu]
    tdicts = [compt, hostt, stt]

    if outformat == 'votable':
        outext = 'xml'
    else:
        outext = outformat
    
    for i in range(len(filelist)):
        filename = filelist[i]
        ud = udicts[i]
        td = tdicts[i]
        newfile = filename.split('.')[0] + '.' + outext
        data = Table.read(indir+filename, format=informat)
        data = update_metadata(table=data, unitdict=ud, dtdict=td)
        data.write(outdir+newfile, format=outformat, overwrite=overwrite)
    
    return


############################################################################
############################################################################
###column units and dtype - make this configurable (e.g. file with all columns, dtype, unit)

udict1 = {'RA': u.deg, 'DEC': u.deg, 'E_RA': u.deg, 'E_DEC': u.deg,
          'Total_flux': u.mJy, 'E_Total_flux': u.mJy, 'Peak_flux': u.mJy/u.beam,
          'E_Peak_flux': u.mJy/u.beam, 'Maj': u.arcsec, 'E_Maj': u.arcsec,
          'Min': u.arcsec, 'E_Min': u.arcsec, 'PA': u.deg, 'E_PA': u.deg,
          'Isl_Total_flux': u.mJy, 'E_Isl_Total_flux': u.mJy, 'Isl_rms': u.mJy/u.beam,
          'Isl_mean': u.mJy/u.beam, 'Resid_Isl_rms': u.mJy/u.beam,
          'Resid_Isl_mean': u.mJy/u.beam, 'RA_max': u.deg, 'DEC_max': u.deg,
          'E_RA_max': u.deg, 'E_DEC_max': u.deg, 'Xposn': u.pixel, 'E_Xposn': u.pixel,
          'Yposn': u.pixel, 'E_Yposn': u.pixel, 'Xposn_max': u.pixel,
          'E_Xposn_max': u.pixel, 'Yposn_max': u.pixel, 'E_Yposn_max': u.pixel,
          'Maj_img_plane': u.arcsec, 'E_Maj_img_plane': u.arcsec, 'Min_img_plane': u.arcsec,
          'E_Min_img_plane': u.arcsec, 'PA_img_plane': u.deg, 'E_PA_img_plane': u.deg,
          'DC_Maj': u.arcsec, 'E_DC_Maj': u.arcsec, 'DC_Min': u.arcsec,
          'E_DC_Min': u.arcsec, 'DC_PA': u.deg, 'E_DC_PA': u.deg,
          'DC_Maj_img_plane': u.arcsec, 'E_DC_Maj_img_plane': u.arcsec,
          'DC_Min_img_plane': u.arcsec, 'E_DC_Min_img_plane': u.arcsec,
          'DC_PA_img_plane': u.deg, 'E_DC_PA_img_plane': u.deg, 'QL_image_RA': u.deg,
          'QL_image_DEC': u.deg, 'NVSS_distance': u.arcsec, 'FIRST_distance': u.arcsec,
          'NN_dist': u.arcsec, 'BMAJ': u.arcsec, 'BMIN': u.arcsec, 'BPA': u.deg}

typedict1 = {'Component_id': np.int, 'Isl_id': np.int, 'Duplicate_flag': np.int,
             'Quality_flag': np.int}


udict2 = {'RA_Source': u.deg, 'DEC_Source': u.deg, 'Total_flux_source': u.mJy,
          'E_Total_flux_source': u.mJy, 'Peak_flux_source': u.mJy/u.beam,
          'E_Peak_flux_source': u.mJy/u.beam, 'Median_rms': u.mJy/u.beam,
          'Angular_size': u.arcsec, 'RA_Host': u.deg, 'DEC_Host': u.deg,
          'Host_sep': u.arcsec, 'W1_mag': u.mag, 'E_W1_mag': u.mag, 'Abs_dpa': u.deg}

typedict2 = {'N_components': np.int, 'Source_reliability_flag': np.int}


udict3 = {'BMAJ': u.deg, 'BMIN': u.deg, 'BPA': u.deg, 'CRVAL1': u.deg,
          'CRVAL2': u.deg, 'OBSRA': u.deg, 'OBSDEC': u.deg,
          'Mean_isl_rms': u.mJy/u.beam, 'SD_isl_rms': u.mJy/u.beam,
          'Peak_flux_p25': u.mJy/u.beam, 'Peak_flux_p50': u.mJy/u.beam,
          'Peak_flux_p75': u.mJy/u.beam, 'Peak_flux_max': u.mJy/u.beam}

typedict3 = {'Image_version': np.int, 'DATE-OBS': Time, 'DATE': Time,
             'N_components': np.int, 'N_empty_islands': np.int}

############################################################################
############################################################################
###main

output_catalogue(compfile=tab1file, hostfile=tab2file, stfile=tab3file,
                 compu=udict1, compt=typedict1, hostu=udict2, hostt=typedict2,
                 stu=udict3, stt=typedict3, informat=informat, outformat=outformat,
                 indir=indir, outdir=outdir)

