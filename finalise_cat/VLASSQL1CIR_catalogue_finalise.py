###finalise CIRADA catalogues
##add reliability flag to Host table
##add Legacy survey cutout urls to Host and component tables
##trim subtile info to required columns only

import numpy as np, pandas as pd, argparse, os
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u


########################################
#time of code check - not important for code to run
import time
start_time = time.time()
########################################


######################################################################
######################################################################
###set params

###file io
#input_directory = '../CIRADA_VLASSQL1_catalogue_v1_vetting/'
#output_directory = '../v1/'
input_directory = ''
output_directory = ''

###define these automatically based on image names
#comptab_out = output_directory + 'CIRADA_VLASS_components.csv'
#hosttab_out = output_directory + 'CIRADA_VLASS_hosts.csv'
#sitab_out = output_directory + 'CIRADA_VLASS_subtile_info.csv'

output_files_in = 'catalogue_output_files/' ###include '/' in name!



######################################################################
######################################################################
###def functions

def col_rename(df, old_cols, new_cols):
    ###renames selected columns in dataframe to new col names
    ###list columns in df
    collist = list(df.columns)
    
    ##change column names in list
    for i in range(len(collist)):
        col = collist[i]
        if col in old_cols:
            collist[i] = new_cols[old_cols.index(col)]

    ##rename columns in DF
    df.columns = collist
    return(df)


def find_centroid(df, n_components):
    ###determine central position without flux weighting
    ###i.e. simple mean position
    ###account for len(df) == 0
    
    if len(df)>0:
        ###make for generic number of components
        racols = ['RA_1', 'RA_2', 'RA_3'][:n_components]
        decols = ['DEC_1', 'DEC_2', 'DEC_3'][:n_components]
    
        ###set up numpy arrays to allow weighted averages
        ra = np.array(df[racols])
        dec = np.array(df[decols])
    
        ###set up central positions
        ra_cen = np.average(ra, axis=1)
        dec_cen = np.average(dec, axis=1)
    
        ##Dec good but need to deal with ra wrpping issues (e.g. RA_1>359 RA_2<1)
        amin = np.min(ra, axis=1)
        amax = np.max(ra, axis=1)
        dra = amax - amin
        ###subtract 180 from unwrapped ra_cen, add 360 if <0
        afilt = (dra>350)
    
        ra_cen[afilt] = ra_cen[afilt] - 180
        ra_cen[(ra_cen<0)] = ra_cen[(ra_cen<0)] + 360
    
    else:
        ###return None if no data
        ra_cen, dec_cen = None, None
    
    return(ra_cen, dec_cen)


def make_url(ra, dec, survey='dr8', s_arcmin=3, s_px=512, format='fits'):
    ###convert coords to string
    ra, dec = str(np.round(ra,5)), str(np.round(dec, 5))
    
    ###set pixscale
    s_arcsec = 60*s_arcmin
    pxscale = s_arcsec/s_px
    
    ###convert image scales to string
    s_px, pxscale = str(s_px), str(np.round(pxscale, 4))
    
    url0 = 'http://legacysurvey.org/viewer/cutout.'
    url1 = '?ra='
    url2 = '&dec='
    url3 = '&layer='
    url4 = '&pixscale='
    url5 = '&size='
    
    url = url0+format+url1+ra+url2+dec+url3+survey+url4+pxscale+url5+s_px
    
    return(url)


def finalise_components(infile, outfile):
    ###finalise the component table
    ###add in urls
    ###set BMAJ/BMIN to arcsec?
    
    ##columns already exist so replacing with df.assign takes care of ordering
    
    comptab = pd.read_csv(infile)
    
    ##create urls
    cutout = []
    for i in range(len(comptab)):
        RA = comptab.iloc[i]['RA']
        DEC = comptab.iloc[i]['DEC']
        cutout.append(make_url(ra=RA, dec=DEC, survey='vlass1.2'))
    
    comptab = comptab.assign(QL_cutout = cutout)

    ##convert beam size to arcsec
    comptab = comptab.assign(BMAJ = 3600*comptab.BMAJ)
    comptab = comptab.assign(BMIN = 3600*comptab.BMIN)
    
    ###write to file
    comptab.to_csv(outfile, index=False)

    return


def finalise_hosts(host_in, comp_in, outfile):
    ###finalise the host table
    ###find DPA (comp/host) for doubles and flag high vals
    ###find triple where central component isn't nearest host and flag
    ###rename n_components to N_components
    
    ##columns to add to host table
    add_cols = ['Abs_dpa', 'Central_component', 'Source_reliability_flag',
                'VLASS_cutout_url', 'Host_cutout_url']
    
    finalcols = ['Source_name', 'Source_type', 'N_components', 'RA_Source',
                 'DEC_Source', 'Total_flux_source', 'E_Total_flux_source',
                 'Peak_flux_source', 'E_Peak_flux_source', 'Median_rms',
                 'Angular_size', 'Component_name_1', 'Component_name_2',
                 'Component_name_3', 'Peak_flux_ratio_12', 'Peak_flux_ratio_13',
                 'Peak_flux_ratio_23', 'Host_objID', 'Host_name', 'RA_Host',
                 'DEC_Host', 'P_Host', 'Host_sep', 'W1_mag', 'E_W1_mag',
                 'Abs_dpa', 'Central_component', 'Source_reliability_flag',
                 'VLASS_cutout_url', 'Host_cutout_url']
    
    dpa_limit=30 ###min diffence between host and componet PAs to flag doubles at
    
    comptab = pd.read_csv(comp_in)
    hosttab = pd.read_csv(host_in)
    
    ###subset components to Duplicate_flag<2 & Quality_flag==0
    comptab = comptab[(comptab['Duplicate_flag']<2)
                      & (comptab['Quality_flag']==0)].reset_index(drop=True)
    
    ###rename N_components
    hosttab = col_rename(df=hosttab, old_cols=['n_components', 'median_rms'],
                         new_cols=['N_components', 'Median_rms'])
    
    ###subset doubles and triples
    doubles = hosttab[(hosttab['N_components']==2)].reset_index(drop=True)
    triples = hosttab[(hosttab['N_components']==3)].reset_index(drop=True)
    
    ###column lists
#    comppos_cols = ['Component_name', 'RA', 'DEC']
#    double_cols = list(doubles.columns) + ['RA_1', 'DEC_1', 'RA_2', 'DEC_2']
#    triple_cols = double_cols + ['RA_3', 'DEC_3']

    comppos_cols = ['Component_name', 'RA', 'DEC', 'Peak_flux']
    double_cols = list(doubles.columns) + ['RA_1', 'DEC_1', 'RA_2', 'DEC_2',
                                           'Peak_flux_1', 'Peak_flux_2']
    triple_cols = double_cols + ['RA_3', 'DEC_3', 'Peak_flux_3']
    
    
    ###add in componen position info to doubles and triples and rename columns
    doubles = pd.merge(doubles, comptab[comppos_cols], left_on='Component_name_1',
                       right_on='Component_name', how='inner')
    doubles = pd.merge(doubles, comptab[comppos_cols], left_on='Component_name_2',
                       right_on='Component_name', how='inner')
                       
#    doubles = col_rename(df=doubles, old_cols=['RA_x', 'DEC_x', 'RA_y', 'DEC_y'],
#                         new_cols=['RA_1', 'DEC_1', 'RA_2', 'DEC_2'])
    doubles = col_rename(df=doubles, old_cols=['RA_x', 'DEC_x', 'RA_y', 'DEC_y',
                                               'Peak_flux_x', 'Peak_flux_y'],
                         new_cols=['RA_1', 'DEC_1', 'RA_2', 'DEC_2',
                                   'Peak_flux_1', 'Peak_flux_2'])

    doubles = doubles[double_cols]
    
    
    triples = pd.merge(triples, comptab[comppos_cols], left_on='Component_name_1',
                       right_on='Component_name', how='inner')
    triples = pd.merge(triples, comptab[comppos_cols], left_on='Component_name_2',
                       right_on='Component_name', how='inner')
    triples = pd.merge(triples, comptab[comppos_cols], left_on='Component_name_3',
                       right_on='Component_name', how='inner')
    
#    triples = col_rename(df=triples, old_cols=['RA_x', 'DEC_x', 'RA_y', 'DEC_y', 'RA', 'DEC'],
#                         new_cols=['RA_1', 'DEC_1', 'RA_2', 'DEC_2', 'RA_3', 'DEC_3'])
    triples = col_rename(df=triples, old_cols=['RA_x', 'DEC_x', 'RA_y', 'DEC_y',
                                               'RA', 'DEC', 'Peak_flux_x', 'Peak_flux_y',
                                               'Peak_flux'],
                         new_cols=['RA_1', 'DEC_1', 'RA_2', 'DEC_2', 'RA_3', 'DEC_3',
                                   'Peak_flux_1', 'Peak_flux_2', 'Peak_flux_3'])

    triples = triples[triple_cols]
    
    ##NEW
    ################################################################################
    ###find peak flux ratios for double/triples
    doubles = doubles.assign(Peak_flux_ratio_12=doubles['Peak_flux_1']/doubles['Peak_flux_2'])
    
    triples = triples.assign(Peak_flux_ratio_12=triples['Peak_flux_1']/triples['Peak_flux_2'])
    triples = triples.assign(Peak_flux_ratio_13=triples['Peak_flux_1']/triples['Peak_flux_3'])
    triples = triples.assign(Peak_flux_ratio_23=triples['Peak_flux_2']/triples['Peak_flux_3'])
    
    frats = pd.concat([doubles[['Source_name', 'Peak_flux_ratio_12']],
                       triples[['Source_name', 'Peak_flux_ratio_12', 'Peak_flux_ratio_13',
                                'Peak_flux_ratio_23']]]).reset_index(drop=True)
    
#    print('CHECK1: ', len(hosttab))
    hosttab  = pd.merge(hosttab, frats, on='Source_name', how='left')
#    print('CHECK2: ', len(hosttab))
    ################################################################################
    
    
    ###for doubles find PA between components, Comp_1 and host, and the difference of these
    c1pos = SkyCoord(ra=np.array(doubles['RA_1']), dec=np.array(doubles['DEC_1']),
                     unit='deg')
    c2pos = SkyCoord(ra=np.array(doubles['RA_2']), dec=np.array(doubles['DEC_2']),
                     unit='deg')
    hostpos = SkyCoord(ra=np.array(doubles['RA_Host']), dec=np.array(doubles['DEC_Host']),
                       unit='deg')
    
    pa12 = c1pos.position_angle(c2pos).deg
    pa1h = c1pos.position_angle(hostpos).deg
    dpa = pa12 - pa1h
    
    ##convert dpa to -180 < dpa < 180
    dpa[(dpa<-180)] = dpa[(dpa<-180)] + 360
    dpa[(dpa>180)] = dpa[(dpa>180)] - 360
    
    doubles = doubles.assign(Abs_dpa = abs(dpa))
    
    ###for triples find central source position -> central component, and if closest to host
    ###can only do if len(triples>0)
#    if len(triples)>0:
    cen_a, cen_d = find_centroid(df=triples, n_components=3)
    
    if len(triples)>0:
        cenpos = SkyCoord(ra=cen_a, dec=cen_d, unit='deg')
        tcpos1 = SkyCoord(ra=np.array(triples['RA_1']), dec=np.array(triples['DEC_1']),
                          unit='deg')
        tcpos2 = SkyCoord(ra=np.array(triples['RA_2']), dec=np.array(triples['DEC_2']),
                          unit='deg')
        tcpos3 = SkyCoord(ra=np.array(triples['RA_3']), dec=np.array(triples['DEC_3']),
                          unit='deg')
        tchostpos = SkyCoord(ra=np.array(triples['RA_Host']),
                             dec=np.array(triples['DEC_Host']),
                             unit='deg')
    
        ##--find separation of every component from cenpos and hostpos
        sep1cen = tcpos1.separation(cenpos).arcsec
        sep1host = tcpos1.separation(tchostpos).arcsec
    
        sep2cen = tcpos2.separation(cenpos).arcsec
        sep2host = tcpos2.separation(tchostpos).arcsec

        sep3cen = tcpos3.separation(cenpos).arcsec
        sep3host = tcpos3.separation(tchostpos).arcsec
    
        ###central component and closest to host
        cencomp = np.argmin([sep1cen, sep2cen, sep3cen], axis=0) + 1 ##add one to match comp #
        closest_to_host = np.argmin([sep1host, sep2host, sep3host], axis=0) + 1
    
        ###make cencomp and closest dtype==int
        cencomp = cencomp.astype(int)
        closest_to_host = closest_to_host.astype(int)

    else:
        ###if no triples make None for columns
        cencomp = None
        closest_to_host = None

    triples = triples.assign(Central_component = cencomp)
    triples = triples.assign(closest_to_host = closest_to_host)

    
    ###merge doubles and triples QA and flux ratio columns with hosttab
    hosttab = pd.merge(hosttab, doubles[['Source_name', 'Abs_dpa']], on='Source_name',
                       how='left')
    hosttab = pd.merge(hosttab, triples[['Source_name', 'Central_component',
                                         'closest_to_host']], on='Source_name', how='left')
#    hosttab = pd.merge(hosttab, doubles[['Source_name', 'Abs_dpa', 'Peak_flux_ratio_12']],
#                       on='Source_name', how='left')
#    hosttab = pd.merge(hosttab, triples[['Source_name', 'Central_component',
#                                         'closest_to_host', 'Peak_flux_ratio_13',
#                                         'Peak_flux_ratio_12']], on='Source_name', how='left')
#    print(hosttab.columns)

    ###create flag column
    rflag = np.zeros(len(hosttab))
    
    ncomp = np.array(hosttab['N_components'])
    adpa = np.array(hosttab['Abs_dpa'])
    ccomp = np.array(hosttab['Central_component'])
    hcomp = np.array(hosttab['closest_to_host'])
    
    dflag = (adpa > 30) ##filter for doubles to flag
    tflag = (ncomp == 3) & (ccomp!=hcomp) ##filter for triples to flag
    
    rflag[dflag] = 1
    rflag[tflag] = 2
    
    ###set to dtype == int and assign to dataframe
    rflag = rflag.astype(int)
    
    hosttab = hosttab.assign(Source_reliability_flag = rflag)
    
    
    ###add in legacy survey urls - both centred on host
    unwise_url, vlass_url = [], []
    for i in range(len(hosttab)):
        RA = hosttab.iloc[i]['RA_Host']
        DEC = hosttab.iloc[i]['DEC_Host']
        unwise_url.append(make_url(ra=RA, dec=DEC, survey='unwise-neo4'))
        vlass_url.append(make_url(ra=RA, dec=DEC, survey='vlass1.2'))
    
    hosttab = hosttab.assign(VLASS_cutout_url = vlass_url)
    hosttab = hosttab.assign(Host_cutout_url = unwise_url)

    ###SET SOURCE_TYPE == BT -> AD!!!!
    source_type = np.array(hosttab['Source_type'])

    source_type[(source_type=='BT')] = 'AD'

    hosttab = hosttab.assign(Source_type = source_type)
    
    ###ENSURE COLUMNS MATCH ORDER IN DOCUMENT - use finalcols list
    hosttab = hosttab[finalcols]

    ###write file
    hosttab.to_csv(outfile, index=False)
    
    return


def finalise_subtiles(infile, outfile):
    ###finalise subtile info table
    finalcols = ['Subtile', 'Image_version', 'Tile', 'Epoch', 'BMAJ', 'BMIN',
                 'BPA', 'LATPOLE', 'LONPOLE', 'CRVAL1', 'CRVAL2', 'DATEOBS',
                 'OBSRA', 'OBSDEC', 'FIELD', 'DATE', 'ORIGIN', 'Mean_isl_rms',
                 'SD_isl_rms', 'Peak_flux_p25', 'Peak_flux_p50', 'Peak_flux_p75',
                 'Peak_flux_max', 'N_components', 'N_empty_islands', 'Subtile_url']
    
    ###read in and subset to required columns
    subtile_info = pd.read_csv(infile)
    subtile_reduced = subtile_info[finalcols]
    
    ###output file
    subtile_reduced.to_csv(outfile, index=False)
    return


def write_catalogue(comps_in, comps_out, hosts_in, hosts_out, subtiles_in, subtiles_out,
                    output_to=output_files_in):
    ##output finalised 3-table catalogue
    
    finalise_components(infile=comps_in, outfile=comps_out)
    print('Component Table finalised')
    
    finalise_hosts(host_in=hosts_in, comp_in=comps_in, outfile=hosts_out)
    print('Host Table finalised')
    
    finalise_subtiles(infile=subtiles_in, outfile=subtiles_out)
    print('Subtile Table finalised')
    
    ###move output to it's own directory'
    dlist = os.listdir()
    
    ###make sure output directory exists!
    if output_to not in dlist:
        os.mkdir(output_to)
    
    ###move files
    for file in [comps_out, hosts_out, subtiles_out]:
        os.rename(src=file, dst=output_to+file)
    
    return


def parse_args():
    'parse command line arguments'
    parser = argparse.ArgumentParser(description="finalise Manitoba VLASS catalogue")
    parser.add_argument("components", help="radio components table",
                        action='store')
    parser.add_argument("hosts", help="unWISE hosts table",
                        action='store')
    parser.add_argument("subtiles", help="VLASS subtiles table",
                        action='store')

    args = parser.parse_args()
    return(args)

######################################################################
######################################################################
###run code


if __name__ == '__main__':
    args = parse_args()
    
    ###use component name prefix as file prefix for output
    pfx = pd.read_csv(args.components, nrows=1)['Component_name'].iloc[0].split(' ')[0]
    print(pfx)
    
    comptab_out = pfx + '_components.csv'
    hosttab_out = pfx + '_hosts.csv'
    sitab_out = pfx + '_subtile_info.csv'
    
    write_catalogue(comps_in=args.components, comps_out=comptab_out,
                    hosts_in=args.hosts, hosts_out=hosttab_out,
                    subtiles_in=args.subtiles, subtiles_out=sitab_out)



#finalise_hosts(host_in=hosttab_in, comp_in=comptab_in, outfile=hosttab_out)



##############################################################################
##############################################################################
#########################################
#time taken for code to run
print('END: ', np.round(time.time() - start_time, 2), 's')
