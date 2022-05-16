####finding isolated and close double candidate sources from CIRADA VLASS QL comp cat
####provides input for LR host finding
####only run on recommended components

import numpy as np, pandas as pd, argparse
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky

from collections import Counter
import wget, os


########################################
#time of code check - not important for code to run
import time
start_time = time.time()
########################################

##############################################################################
##############################################################################
###def input parameters

test_subset = True


outfile = 'VLASS_source_candidates.csv'



minflux = 1 ###minimum peak flux to use for components (1mJy looks good)
isolim = 40 ###min number of arcsecs to NN for isolated sources (40" where dNN converges)
dtlim = 10 ###max number of arcsecs to nearest neighbour for double/triple finding
fratlim = 5 ###maximum flux ratio to allow for doubles (5 accounts for 90% of flux ratio distribution)
bp_limit = 20 ###minimum flux for isolated bright sources
brat_min = 10 ###minimum component flux ratio associated with bright point sources

source_prefix = 'RAW'


imsize = 512
angsize = 2 ##angular size of cutout in arcmin




##############################################################################
##############################################################################
###def functions

def source_name(ra, dec, aprec=2, dp=5, prefix='VLASS1QLCIR'):
    ###create source name to nearest arcsec based on ra/dec
    ###truncated to dp so compatible with IAU standards
    ra, dec = np.array(ra), np.array(dec)
    
    cat = SkyCoord(ra=ra, dec=dec, unit='deg')
    
    astring = cat.ra.to_string(sep='', unit='hour', precision=dp, pad=True)
    dstring = cat.dec.to_string(sep='', precision=dp, pad=True, alwayssign=True)
    
    ###truncation index
    tind = aprec+7
    
    if prefix == 'RAW':
        sname = ['J' + astring[i][:tind] + dstring[i][:tind] for i in range(len(astring))]
    else:
        sname = [prefix + ' J' + astring[i][:tind] + dstring[i][:tind] for i in
                 range(len(astring))]
    
    return(sname)


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


def add_neighbour_info(df, poscat, add_cols, n_neighbour=1):
    ####finds nth nearest neighbour and brings in important columns
    n_neighbour = n_neighbour+1 ##accounts for self matching of catalogue
    neighbours = match_coordinates_sky(poscat, poscat, nthneighbor=n_neighbour)
    
    for col in add_cols:
        newcol = col + '_' + str(n_neighbour)
        df[newcol] = np.array(df[col].iloc[neighbours[0]])
    
    df['sep_'+str(n_neighbour)] = neighbours[1].arcsec
    
    return(df)


def add_source_properties(df, n_components):
    ##combine component properties in to source properties
    
    ###parent columns susbet to only n_components used
    tfcols = ['Total_flux_1', 'Total_flux_2', 'Total_flux_3'][:n_components]
    etfcols = ['E_Total_flux_1', 'E_Total_flux_2', 'E_Total_flux_3'][:n_components]
    pfcols = ['Peak_flux_1', 'Peak_flux_2', 'Peak_flux_3'][:n_components]
    epfcols = ['E_Peak_flux_1', 'E_Peak_flux_2', 'E_Peak_flux_3'][:n_components]
    rmscols = ['Isl_rms_1', 'Isl_rms_2', 'Isl_rms_3'][:n_components]
    
    ###source fluxes
    df = df.assign(Total_flux_source = df[tfcols].sum(axis=1))
    df = df.assign(E_Total_flux_source = np.sqrt((df[etfcols]**2).sum(axis=1)))
    df = df.assign(Peak_flux_source = df[pfcols].max(axis=1))
    df = df.assign(E_Peak_flux_source = df[epfcols].max(axis=1))
    
    ###component flux weights
    df = df.assign(sfrac_1 = df['Total_flux_1']/df['Total_flux_source'])
    if n_components>1:
        df = df.assign(sfrac_2 = df['Total_flux_2']/df['Total_flux_source'])
    if n_components>2:
        df = df.assign(sfrac_3 = df['Total_flux_3']/df['Total_flux_source'])
    
    ###source median rms
    df = df.assign(median_rms = df[rmscols].median(axis=1))

    ###add n_components
    df = df.assign(n_components = n_components*np.ones(len(df)))
    
    return(df)


def flux_weighted_centroid(df, n_components):
    ###determine central position based on flux weighting
    
    ###make for generic number of components
    racols = ['RA_1', 'RA_2', 'RA_3'][:n_components]
    decols = ['DEC_1', 'DEC_2', 'DEC_3'][:n_components]
    wcols = ['sfrac_1', 'sfrac_2', 'sfrac_3'][:n_components]
    
    ###set up numpy arrays to allow weighted averages
    ra = np.array(df[racols])
    dec = np.array(df[decols])
    sweights = np.array(df[wcols])
    
    ###set up central positions
    ra_cen = np.average(ra, axis=1, weights=sweights)
    dec_cen = np.average(dec, axis=1, weights=sweights)
    
    ##Dec good but need to deal with ra wrpping issues (e.g. RA_1>359 RA_2<1)
    amin = np.min(ra, axis=1)
    amax = np.max(ra, axis=1)
    dra = amax - amin
    ###subtract 180 from unwrapped ra_cen, add 360 if <0
    afilt = (dra>350)
    
    ra_cen[afilt] = ra_cen[afilt] - 180
    ra_cen[(ra_cen<0)] = ra_cen[(ra_cen<0)] + 360
    
    ###append source position to data frame
    df = df.assign(RA_source = ra_cen)
    df = df.assign(DEC_source = dec_cen)
    
    return(df)


def make_source(df, ncomp):
    ##combine component properties in to source properties,
    ##find centroid and name
    ###combine properties - flux weighted centroid last as it requires total flux
    df = add_source_properties(df=df, n_components=ncomp)
    df = flux_weighted_centroid(df=df, n_components=ncomp)

    ###add in source_name
    df = df.assign(Source_name = source_name(ra=df['RA_source'],
                                             dec=df['DEC_source'],
                                             prefix=source_prefix))
    
    return(df)


def select_sources(component_file, output, minflux=1, isolim=40, dtlim=10, fratlim=5,
                   bp_limit=20, brat_min=10):

    ###function selecting all sources and outputting single data frame
    need_cols = ['Component_name', 'RA', 'DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux',
                 'E_Peak_flux', 'Isl_rms', 'Maj', 'NN_dist', 'S_Code', 'Peak_to_ring', 'Subtile']
    
    outcols = ['Source_name', 'Source_type', 'RA', 'DEC', 'Total_flux', 'E_Total_flux',
               'Peak_flux', 'E_Peak_flux', 'median_rms', 'Angular_size', 'n_components',
               'Component_name_1', 'Component_name_2', 'Component_name_3',
               'Peak_flux_ratio_12', 'Peak_flux_ratio_13', 'Peak_flux_ratio_23']

    compcat = pd.read_csv(component_file)

    ###subset on d/q_flags and columns - move minflux to source selection!
    data = compcat[(compcat.Duplicate_flag<2) & (compcat.Quality_flag==0)].reset_index(drop=True)
    data = data[need_cols]

    ###find isolated and possible double/triples
    
    ######################################################################
    ###isolated - these are easy!
    iso = data[(data.NN_dist>isolim) & (data.Peak_flux>minflux)].reset_index(drop=True)
    iso = iso.assign(n_components = np.ones(len(iso))) ###add in number of components
    iso = iso.assign(Source_name = source_name(ra=iso['RA'], dec=iso['DEC'], prefix=source_prefix))
    iso = iso.assign(Source_type = ['SC' for i in range(len(iso))])
    
    ###need to add component_name_1, component_name_2, component_name_3 to iso
    #iso = iso.assign(Component_name_1 = iso.Component_name)
    cnan = np.zeros(len(iso))
    cnan[(cnan==0)] = np.nan
    iso = iso.assign(Component_name_2 = cnan)
    iso = iso.assign(Component_name_3 = cnan)
    
    ###add in peak flux ratios
    iso = iso.assign(Peak_flux_ratio_12 = cnan)
    iso = iso.assign(Peak_flux_ratio_13 = cnan)
    iso = iso.assign(Peak_flux_ratio_23 = cnan)

    ###rename relavent columns
    iso_old = ['Component_name', 'Isl_rms', 'Maj']
    iso_new = ['Component_name_1', 'median_rms', 'Angular_size']

    iso = col_rename(df=iso, old_cols=iso_old, new_cols=iso_new)
    iso = iso[outcols]

    
    ######################################################################
    ###set up double/triple candidates
    dtcand = data[(data.NN_dist<dtlim)].reset_index(drop=True)

    ###set up coordinate catalogue
    dtcat = SkyCoord(ra=np.array(dtcand['RA']), dec=np.array(dtcand['DEC']), unit='deg')

    ###define columns to include from neighbours
    meascols = ['Component_name', 'RA', 'DEC', 'Peak_flux', 'E_Peak_flux',
                'Total_flux', 'E_Total_flux', 'Isl_rms', 'S_Code']

    dtcand = add_neighbour_info(df=dtcand, poscat=dtcat, add_cols=meascols,
                                n_neighbour=1)
    dtcand = add_neighbour_info(df=dtcand, poscat=dtcat, add_cols=meascols,
                                n_neighbour=2)

    ###add in distance to 3rd nearest neighbour to ensure isolated
    dtnn3 = match_coordinates_sky(dtcat, dtcat, nthneighbor=4) ###finds 3rd NN - confirm iso trips
    dtcand = dtcand.assign(sep_4 = dtnn3[1].arcsec)

    ##rename original component columns to '_1'
    ccols_old = need_cols.copy()
    ccols_new = [i+'_1' for i in ccols_old]
    col_rename(df=dtcand, old_cols=ccols_old, new_cols=ccols_new)

    ###put in flux ratios
    dtcand = dtcand.assign(Peak_flux_ratio_12 = dtcand['Peak_flux_1']/dtcand['Peak_flux_2'])
    dtcand = dtcand.assign(Peak_flux_ratio_13 = dtcand['Peak_flux_1']/dtcand['Peak_flux_3'])
    dtcand = dtcand.assign(Peak_flux_ratio_23 = dtcand['Peak_flux_2']/dtcand['Peak_flux_3'])
    
    ##info for column renaming when sources selected
    mc_old = ['RA_source', 'DEC_source', 'Total_flux_source', 'E_Total_flux_source',
              'Peak_flux_source', 'E_Peak_flux_source']
    mc_new = ['RA', 'DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux']
    
    #######################################################################
    ##doubles - nn(above flux lim) < 40" 2nn >10"
    ###only use those with flux ratios < fratlim (default==5)
    ###setting min S_ratio == 1 takes care of pair duplication

    doubles = dtcand[(dtcand['sep_3']>isolim) & (dtcand['sep_2']<dtlim)
                     & (dtcand['Peak_flux_1']>minflux) & (dtcand['Peak_flux_ratio_12']>=1)
                     & (dtcand['Peak_flux_ratio_12']<=fratlim)].reset_index(drop=True)

    ###create source list with combined properties from components
    doubles = make_source(df=doubles, ncomp=2)
    
    ###add source type
    doubles = doubles.assign(Source_type = ['CD' for i in range(len(doubles))])
    
    ###add angular size to multi-component_sources
    doubles = doubles.assign(Angular_size = doubles['sep_2'])
    
    ###limit to only relavent component info and columns
    dnan = np.zeros(len(doubles))
    dnan[(dnan==0)] = np.nan
    doubles = doubles.assign(Component_name_3 = dnan)
    doubles = doubles.assign(Peak_flux_ratio_13 = dnan)
    doubles = doubles.assign(Peak_flux_ratio_23 = dnan)
    
    doubles = col_rename(df=doubles, old_cols=mc_old, new_cols=mc_new)[outcols]

    #######################################################################
    ###blended triples
    ###set Peak_flux_ratio_12 > fratlim ensures not in doubles and brightest component first
    hidden_triples = dtcand[(dtcand['sep_3']>isolim) & (dtcand['sep_2']<dtlim)
                            & (dtcand['Peak_flux_1']>minflux)
                            & (dtcand['Peak_flux_ratio_12']<brat_min)
                            & (dtcand['Peak_flux_ratio_12']>fratlim)
                            & (dtcand['S_Code_1']=='M')].reset_index(drop=True)

    ###create source list with combined properties from components
    hidden_triples = make_source(df=hidden_triples, ncomp=2)
    hidden_triples = hidden_triples.assign(Source_type = ['BT' for i in range(len(hidden_triples))])
    
    ###add angular size to multi-component_sources
    hidden_triples = hidden_triples.assign(Angular_size = hidden_triples['sep_2'])
    
    ###limit to only relavent component info and columns
    htnan = np.zeros(len(hidden_triples))
    htnan[(htnan==0)] = np.nan
    hidden_triples = hidden_triples.assign(Component_name_3 = htnan)
    hidden_triples = hidden_triples.assign(Peak_flux_ratio_13 = htnan)
    hidden_triples = hidden_triples.assign(Peak_flux_ratio_23 = htnan)
    
    hidden_triples = col_rename(df=hidden_triples, old_cols=mc_old, new_cols=mc_new)[outcols]
    
    #######################################################################
    ####triples - dNN2 < 10", need to ensure max extant not greater than 10"
    ### no flux ratio restrictions - core will skew and false positive rate should be low

    triples = dtcand[(dtcand['sep_4']>isolim) & (dtcand['sep_3']<dtlim) & (dtcand['sep_2']<dtlim)
                     & (dtcand['Peak_flux_ratio_12']>=1) & (dtcand['Peak_flux_ratio_13']>=1)
                     & (dtcand['Peak_flux_ratio_12']<brat_min)
                     & (dtcand['Peak_flux_ratio_13']<brat_min)
                     & (dtcand['Peak_flux_1']>minflux)].reset_index(drop=True)

    ###are all 3 components within 10"?!
    tc2cat = SkyCoord(ra=np.array(triples['RA_2']), dec=np.array(triples['DEC_2']),
                      unit='deg')
    tc3cat = SkyCoord(ra=np.array(triples['RA_3']), dec=np.array(triples['DEC_3']),
                      unit='deg')

    triples = triples.assign(sep_23 = tc2cat.separation(tc3cat).arcsec)
    triples = triples[(triples['sep_23']<dtlim)].reset_index(drop=True)

    ###make triple sources
    triples = make_source(df=triples, ncomp=3)

    ###ensure no duplicates
    triples = triples.drop_duplicates(subset='Source_name').reset_index(drop=True)
    triples = triples.assign(Source_type = ['CT' for i in range(len(triples))])
    
    ###add angular size to multi-component_sources
    triples = triples.assign(Angular_size = triples[['sep_2', 'sep_3', 'sep_23']].max(axis=1))
    
    ###limit to only relavent component info and columns
    triples = col_rename(df=triples, old_cols=mc_old, new_cols=mc_new)[outcols]
    
    ######################################################################
    ###re-add bright point sources
    ## flux ratios > frat for 1-2 and 1-3, minimum flux 20mJy
    ## separation from 4th nn use full component catalogue
    ## ensure >40" to make comparable to iso selection
    bright_point = dtcand[(dtcand['Peak_flux_ratio_12']>brat_min)
                          & (dtcand['Peak_flux_ratio_13']>brat_min)
                          & (dtcand['Peak_flux_1']>bp_limit) & (dtcand['sep_4']>isolim)
                          ].reset_index(drop=True)

    bright_point = make_source(df=bright_point, ncomp=1)
    bright_point = bright_point.assign(Source_type = ['BS' for i in range(len(bright_point))])
    
    ###add angular size to multi-component_sources
    bright_point = bright_point.assign(Angular_size = bright_point['Maj_1'])
    
    ###limit to only relavent component info and columns
    bnan = np.zeros(len(bright_point))
    bnan[(bnan==0)] = np.nan
    bright_point = bright_point.assign(Component_name_2 = bnan)
    bright_point = bright_point.assign(Component_name_3 = bnan)
    bright_point = bright_point.assign(Peak_flux_ratio_12 = bnan)
    bright_point = bright_point.assign(Peak_flux_ratio_13 = bnan)
    bright_point = bright_point.assign(Peak_flux_ratio_23 = bnan)
    
    bright_point = col_rename(df=bright_point, old_cols=mc_old, new_cols=mc_new)[outcols]
    
    ######################################################################
    ###need to ensure no repeats of componenets between source class
    ###set na for comp3 in doubles and 2 and 3 in singles (iso and bright point)
    ###create single table
    
    source_list = pd.concat([iso, doubles, hidden_triples, triples, bright_point])
    
    ###sort by RA
    source_list = source_list.sort_values(by='RA', ascending=True).reset_index(drop=True)
    

    ###write to file
    source_list.to_csv(output, index=False)

    return


def target_images(target_file, imdir_file, output, tacol='RA', tdcol='DEC',
                  iacol='cenRA', idcol='cenDec'):
    ###create list of target images for Likelihood ratio
    ###output table of source_names, nearest image centre, distance, and image_name
    targetlist = pd.read_csv(target_file)
    imlist = pd.read_csv(imdir_file)
    
    ###cross match catalogues
    tcat = SkyCoord(ra=np.array(targetlist[tacol]), dec=np.array(targetlist[tdcol]),
                    unit='deg')
    icat = SkyCoord(ra=np.array(imlist[iacol]), dec=np.array(imlist[idcol]), unit='deg')
    
    txi = tcat.match_to_catalog_sky(icat)
    
    outdf = pd.DataFrame({'Source_name': np.array(targetlist['Source_name']),
                          'Coad_ID': np.array(imlist['Coad_ID'].iloc[txi[0]]),
                          'd_cen_Coad': np.array(txi[1].arcsec)})

    ###write to file
    outdf.to_csv(output, index=False)
    
    return


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


def make_filename(objname, survey='DECaLS-DR8', format='fits'):
    ###just take Julian coords of name to eliminate white space - eliminate prefix
    name = objname.split(' ')[1]
    
    filename = name + '_' + survey + '.' + format
    
    return(filename)



def grab_cutouts(target_file, survey, namecol='Component_name', acol='RA', dcol='DEC',
                 odir='', udir='', cutout_s=angsize, pxsize=imsize, format='fits'):
    ###assumes target_file is csv
    targets = pd.read_csv(target_file)
    
    for i in range(len(targets)):
        target = targets.iloc[i]
        name = target[namecol]
        a = target[acol]
        d = target[dcol]
        ###download urls
        url = make_url(ra=a, dec=d, survey=survey, s_arcmin=cutout_s, s_px=pxsize,
                       format=format)
        ###filenames
        ofile = odir + make_filename(objname=name, survey=survey, format=format)
        ###get files
        wget.download(url=url, out=ofile)
    
    return


def parse_args():
    'parse command line arguments'
    parser = argparse.ArgumentParser(description="find candidate sources for LR")
    parser.add_argument("components", help="radio components table",
                        action='store')
                        
    args = parser.parse_args()
    return(args)


##############################################################################
###run code

if __name__ == '__main__':
    args = parse_args()
    select_sources(component_file=args.components, output=outfile)





###############################################################################
##############################################################################
##############################################################################
###testing


##############################################################################
##############################################################################
#########################################
#time taken for code to run
print('END: ', np.round(time.time() - start_time, 2), 's')
