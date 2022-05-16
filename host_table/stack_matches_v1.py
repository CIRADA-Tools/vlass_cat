###stack matched catalogues into a single table
### look for duplicate unWISE sourced and remove keeping highest w1_Rel

import numpy as np, pandas as pd, os, argparse
from astropy.coordinates import SkyCoord


########################################
#time of code check - not important for code to run
import time
start_time = time.time()
########################################

################################################################################
###def params

target_directory = 'LR_output/'

#sfile = 'input_data/CIRADA_VLASS1QL_source_candidates_v1b.csv'
#cfile = '../../../../CIRADA_output_files/CIRADA_VLASS1QL_table1_components_v01.csv'

#tab2_outname = '../../../../CIRADA_output_files/CIRADA_VLASS1QL_table2_hosts_v01a.csv'
#tab1_outname = '../../../../CIRADA_output_files/CIRADA_VLASS1QL_table1_components_v01a.csv'

tab2_outname = 'VLASS_table2_hosts.csv'
tab1_outname = 'VLASS_table1_components.csv'


################################################################################
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


def source_name(ra, dec, aprec=2, dp=5, prefix='VLASS1QLCIR'):
    ###create source name to nearest arcsec based on ra/dec
    ###truncated to dp so compatible with IAU standards
    ra, dec = np.array(ra), np.array(dec)
    
    cat = SkyCoord(ra=ra, dec=dec, unit='deg')
    
    astring = cat.ra.to_string(sep='', unit='hour', precision=dp, pad=True)
    dstring = cat.dec.to_string(sep='', precision=dp, pad=True, alwayssign=True)
    
    ###truncation index
    tind = aprec+7
    
    sname = [prefix + ' J' + astring[i][:tind] + dstring[i][:tind] for i in range(len(astring))]
    
    return(sname)


def stack_catalogues(target_folder):
    fstart_time = time.time() ###set timer for testing
    
    ###stack up individual catalogues output from LR matching
    catlist = os.listdir(target_folder)
    catlist = [target_folder + file for file in catlist if 'LR_matches' in file]
    
    ###set up empty data frame
    columns = ['radio_ID', 'w1_ID', 'w1_LR', 'w1_Rel', 'w1_n_cont', 'w1_separation',
               'unwise_detid', 'ra', 'dec', 'fluxlbs', 'dfluxlbs', 'w1_mag', 'w1_mag_err']
    
    stacked = pd.DataFrame(columns=columns)
    
    fcount = np.arange(0, len(catlist), 500)
    
    
    ###open each file
    for i in range(len(catlist)):
        if i in fcount: ###for testing purposes only - provide whitespace buffer for terminal
            print(' ')
            print('------------------------------')
            print(i, '/', len(catlist), ' --- ' , np.round(time.time() - fstart_time, 2), 's')
            print('------------------------------')
            print(' ')
        
        cat = catlist[i]
        df = pd.read_csv(cat)
        stacked = pd.concat([stacked, df], axis=0).reset_index(drop=True)
    
    ###for some reason some rows are duplicated - not sure why but ra/dec info isn't
    ###can be fixed by requiring ra to be finite
    ##testing shows this doesn't remove any unique info
    
    ###clean cat - ra finite and 0 < w1_rel < 1
    gfilt = (stacked.ra>=0) & (stacked.ra<=360) & (stacked.w1_Rel>=0) & (stacked.w1_Rel<=1)
    stacked = stacked[gfilt].reset_index(drop=True)
    
    ###sort by w1_Rel and drop_duplicate radio_ID
    stacked = stacked.sort_values(by='w1_Rel', ascending=False).reset_index(drop=True)
    stacked = stacked.drop_duplicates(subset='radio_ID', keep='first').reset_index(drop=True)

    #add in unwise name
    stacked = stacked.assign(Host_name = source_name(ra=stacked['ra'], dec=stacked['dec'],
                                                       prefix='WISEU'))

    ###remove duplicate unWISE objects and sort by RA
    stacked = stacked.drop_duplicates(subset='Host_name')
    stacked = stacked.sort_values(by='ra', ascending=True).reset_index(drop=True)

    ###write to file

    return(stacked)


def make_table2(stacked_matches, source_file, component_file,
                outfilename_sources='source_table.csv',
                outfilename_components='component_table.csv'):
    
    ###columns to output
    tab2_cols = ['Source_name', 'Source_type', 'n_components', 'RA_Source', 'DEC_Source',
                 'Total_flux_source', 'E_Total_flux_source', 'Peak_flux_source',
                 'E_Peak_flux_source', 'median_rms', 'Angular_size', 'Component_name_1',
                 'Component_name_2', 'Component_name_3', 'Host_objID', 'Host_name',
                 'RA_Host', 'DEC_Host', 'P_Host', 'Host_sep', 'W1_mag', 'E_W1_mag',
                 'VLASS_cutout_url', 'Host_cutout_url']
    
    mdata = stacked_matches
    sdata = pd.read_csv(source_file)
    
    ####convert columns in source table
    oldcolnames = ['RA', 'DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux',
                   'mean_rms']
    newcolnames = ['RA_Source', 'DEC_Source', 'Total_flux_source', 'E_Total_flux_source',
                   'Peak_flux_source', 'E_Peak_flux_source','Source_local_rms']
    sdata = col_rename(sdata, old_cols=oldcolnames, new_cols=newcolnames)

    ###set up missing columns
#    sdata = sdata.assign(Source_type = np.array(sdata.n_components))

    ###rename mdat columns
    ocolnames = ['radio_ID', 'w1_ID', 'w1_Rel', 'w1_separation', 'w1_mag', 'w1_mag_err',
                 'ra', 'dec']
    ncolnames = ['Source_name', 'Host_objID', 'P_Host', 'Host_sep', 'W1_mag', 'E_W1_mag',
                 'RA_Host', 'DEC_Host']

    mdata = col_rename(mdata, old_cols=ocolnames, new_cols=ncolnames)

    ####set up missing columns
    mdata = mdata.assign(VLASS_cutout_url = np.zeros(len(mdata)))
    mdata = mdata.assign(Host_cutout_url = np.zeros(len(mdata)))
    
    
    ###set up table 2
    tab2 = pd.merge(mdata, sdata, on='Source_name', how='inner')
    tab2 = tab2[tab2_cols]
    
    ##set up DF with component_name, source_name, and p
    ##i.e. each component listed once in a single column, to enable update of component table

    f1 = tab2[['Component_name_1', 'Source_name', 'Source_type', 'P_Host']]
    f2 = tab2[['Component_name_2', 'Source_name', 'Source_type', 'P_Host']]
    f3 = tab2[['Component_name_3', 'Source_name', 'Source_type', 'P_Host']]

    colnames = ['Component_name', 'Source_name', 'Source_type', 'P_Host']

    f1.columns = colnames
    f2.columns = colnames
    f3.columns = colnames

    f2 = f2[(f2.Source_type==2)].reset_index(drop=True)
    f3 = f3[(f3.Source_type==3)].reset_index(drop=True)

    ##combine into a single dataframe
    finalise = pd.concat([f1, f2, f3], axis=0).reset_index(drop=True)
    
    ###ensuring that each component is only assigned to a single source
    finalise = finalise.sort_values(by='P_Host', ascending=False).reset_index(drop=True)
    finalise = finalise.drop_duplicates(subset='Component_name',
                                        keep='first').reset_index(drop=True)
    
    fsources = finalise.drop_duplicates(subset='Source_name',
                                        keep='first')[['Source_name']].reset_index(drop=True)


    ###merge with tab2 to remove sources with duplicate component assignments
    ###more specifically ensuring that each component is only assigned to a single source
    tab2 = pd.merge(tab2, fsources, on='Source_name', how='inner').reset_index(drop=True)
    
    ###reset tab2 column_x (x==2,3) to nan if not a compnent
    c2 = np.array(tab2.Component_name_2)
    c3 = np.array(tab2.Component_name_3)
    
    ##filter out no component assigned
    c2filt = (c2=='0') | (c2=='0.0')
    c3filt = (c3=='0') | (c3=='0.0')
    
    c2[c2filt] = np.nan
    c3[c3filt] = np.nan
    
    ###assign to tab2
    tab2 = tab2.assign(Component_name_2 = c2)
    tab2 = tab2.assign(Component_name_3 = c3)
    
    ####update component table with source_name/type
    comptab = pd.read_csv(component_file)
    compcols = list(comptab.columns)

    comptab = pd.merge(comptab, finalise[['Component_name', 'Source_name', 'Source_type']],
                   on='Component_name', how='left')

    comptab = comptab.assign(Source_name = np.array(comptab.Source_name_y))
    comptab = comptab.assign(Source_type = np.array(comptab.Source_type_y))

    comptab = comptab[compcols]
    
    ###write tables to file
    tab2.to_csv(outfilename_sources, index=False)
    comptab.to_csv(outfilename_components, index=False)
    
    return


def finalise_host_ids(tdir, sfile, cfile, t1_out='Table1_components.csv',
                      t2_out='Table2_components.csv'):
    ###combine stack_catalogues and make_tab2 to run as a single function
    ###takes directory of individual catalogues, stacks them and outputs:
    ###updated component table (Table1) with source information
    ###Source+Host table (Table2)
    
    stacked_matches = stack_catalogues(target_folder=tdir)
    
    make_table2(stacked_matches=stacked_matches, source_file=sfile, component_file=cfile,
                outfilename_sources=t2_out, outfilename_components=t1_out)
    
    return


def parse_args():
    'parse command line arguments'
    parser = argparse.ArgumentParser(description="find hosts")
    parser.add_argument("source_candidates", help="table of candidate sources",
                        action='store')
    parser.add_argument("components", help="component table",
                        action='store')
                        
    args = parser.parse_args()
    return(args)


################################################################################
###run code

if __name__ == '__main__':
    args = parse_args()
    finalise_host_ids(tdir=target_directory, sfile=args.source_candidates,
                      cfile=args.components, t1_out=tab1_outname, t2_out=tab2_outname)


###WORKS

################################################################################
###testing


################################################################################
#########################################
#time taken for code to run
print('END: ', np.round(time.time() - start_time, 2), 's')
