####code to process VLASS component table that is the result of PyBDSF and vlad phase 1
#### written by Yjan Gordon (yjan.gordon@umanitoba.ca) for the CIRADA project
#### 17th April 2020
###run as vlass_compcat_vlad_stage2.py pybdsf_output_file subtile_file nvss_file first_file

import sys
import numpy as np, pandas as pd, argparse
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import coordinates as coords, units as u
from astropy.wcs import WCS


################################################################################
###define parameters

###these could also be added to the config file!
#components_file = '../M_work/M_vlad/VLASS1_UOFM_QL_Catalogue_20200416_204640.csv'
#subtile_file = '../../CIRADA_output_files/CIRADA_VLASS1QL_table3_subtile_info_v01.csv'
#first_file = '../../../../survey_data/FIRST/catalogue/first_14dec17.fits'
#nvss_file = '../../../../survey_data/NVSS/CATALOG41.FIT'

#outfile = 'CIRADA_VLASS1QL_table1_components_v01.csv'
outfile = 'VLASS_components.csv'


####config file parameters (and default values):
#name_prefix = 'VLASS1QLCIR' ##prefix in component naming - now determined from image names
ra_col = 'RA'
dec_col = 'DEC'
flux_units_in = 'mJy' ##flux units used in input pybdsf table - astropy unit strings
flux_units_out = 'mJy' ##desired output flux units - astropy unit strings
size_units_in = 'arcsec' ##size columns (Maj/Min etc) units used in input - astropy unit strings
size_units_out = 'arcsec' ##desired output size units - astropy unit strings
duplicate_search_arcsec = '2' ##search radius (arcsec) to use for duplicate flagging
sig_noise_threshold = '5' ##Peak_flux to Isl_rms to use in quality flagging
peak_to_ring_threshold = '2' ##peak_to_ring value to use in quality flagging
peak_ring_dist = '20' ##NN distance at which to start using peak_to_ring in qual flag

##convert strings of config params to useable where needed
flux_units_in = u.Unit(flux_units_in)
flux_units_out = u.Unit(flux_units_out)
size_units_in = u.Unit(size_units_in)
size_units_out = u.Unit(size_units_out)
duplicate_search_arcsec = float(duplicate_search_arcsec)*u.arcsec
sig_noise_threshold = float(sig_noise_threshold)
peak_to_ring_threshold = float(peak_to_ring_threshold)
peak_ring_dist = float(peak_ring_dist)



####if dropping columns from input to redo here, specify
####THIS IS A HACK to ensure the data is accurate, these columns should really be fixed in code that produces the input data
revlad = False
redocolumns = []
#redocolumns = ['Xposn', 'E_Xposn', 'Yposn', 'E_Yposn', 'Xposn_max',
#               'E_Xposn_max', 'Yposn_max', 'E_Yposn_max',
#               'Duplicate_flag', 'Quality_flag', 'Source_name',
#               'Source_type', 'NN_dist']



################################################################################
###define functions

def obtain_eg_image_name(subtile_data, filename_col='Subtile_url'):
    ###obtain an example image name to use in definening component name prefix
    
    name_list = list(subtile_data[filename_col])
    ###make sure all rows have a filename
    names = [name for name in name_list if type(name) == str]
    if len(names) != len(name_list):
        print('WARNING: not all subtile rows have filename - was catenator.py run on multiple data without flushing?')

    egname = names[0]
    
    return(egname)


def determine_source_name_prefix_from_image_names(subtile_url):
    ##use the VLASS image file names determine the appropriate component name prefix
    ##accounts for epoch and image type, e.g. VLASS1SE versuse VLASS2QL etc.
    ###dependent on filename format remaining consistent
    
    ###remove directory structure in filename url (if present)
    if '/' in subtile_url:
        image_filename = subtile_url.rsplit('/', 1)[1]
    else:
        image_filename = subtile_url

    ###extract epoch and image type from filename
    namesplit = image_filename.split('.')

    epoch = namesplit[0].upper()
    imtype = namesplit[2].upper() ##ensures upper case for IAU approved component name

    prefix = epoch + imtype + 'CIR'
    
    return(prefix)


def source_name(ra, dec, aprec=2, dp=5, prefix=''):
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


def colscale(df, collist, scalefactor):
    ###scale multiple columns in pandas df by same factor
    ###useful for conversions between Jy->mJy and deg->arcsec for example
    for col in collist:
        df[col] = scalefactor*df[col]
    return(df)


def drop_input_cols(df, dropcols):
    'columns to drop from input data'
    ###useful for hacking around bugs in columns produced in input that can be reproduced correctly here - shouldn't be used as long term solution
    dfcols = list(df.columns)
    
    ###remove columns
    for col in dropcols:
        if col in dfcols:
            dfcols.remove(col)

    return(df[dfcols])


def find_duplicates(df, acol='RA', dcol='DEC', pos_err=2*u.arcsec):
    ###find duplicates and flag

    ###create SN column to sort by - may replace with q_flag later
    df['SN'] = df['Peak_flux']/df['Isl_rms']
    
    #2) sort by SN/qflag, subset dist<2"
    df = df.sort_values(by='SN', ascending=False).reset_index(drop=True)
    
    #####DONT subset duplicates!
    ###tun search around on entire catalogue (sorted) and use index you dumbass!
    dfpos = SkyCoord(ra=np.array(df[acol]), dec=np.array(df[dcol]), unit='deg')
    dsearch = dfpos.search_around_sky(dfpos, seplimit=pos_err)
    
    ###create dataframe for easy manipulation - not actually neccesary just cleaner
    dsdf = pd.DataFrame({'ix1': dsearch[0], 'ix2': dsearch[1], 'ix3': dsearch[2].arcsec})
    
    ###subset to ix1 != ix2 - reduces 4M to 500k
    dsdf = dsdf[(dsdf['ix1']!=dsdf['ix2'])].reset_index(drop=True)
    
    ###is index of preferred components where fist instance in ix1 occurs before ix2? - I think so
    ix1, ix2 = list(dsdf['ix1']), list(dsdf['ix2'])
    prefcomp = [i for i in ix1 if ix1.index(i) < ix2.index(i)] ##this takes a while
    
    ###use pref comp to filter dup array and reflag
    dupflag = np.zeros(len(df)) ##all set to zero
    dupflag[np.unique(ix1)] = 2 ##flags all duplicates
    dupflag[prefcomp] = 1 ##reflags preferred duplicates
    
    df['Duplicate_flag'] = dupflag
    
    return(df)


def q_flag(df, snmin=5, prmax=2, prdist=20):
    ###Q_flag
    ##3 fold:
    ## 1) Tot < Peak (GFIT)
    ## 2) Peak < 5*Isl_rms (SN)
    ## 3) dNN >= 20" && Peak_to_ring < 2 (PR)
    
    ###combine in to single flag value via binary bit addition
    ## PR = 1; SN = 2; GFIT = 4
    ## weights GFIT highest, then SN, then P2R
    gfit, sn, pr = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    
    ###necessary column arrays
    stot = np.array(df['Total_flux'])
    speak = np.array(df['Peak_flux'])
    rms = np.array(df['Isl_rms'])
    dnn = np.array(df['NN_dist'])
    ptr = np.array(df['Peak_to_ring'])
    
    ###flag individual critera
    gfit[(speak > stot)] = 4
    sn[(speak < snmin*rms)] = 2
    pr[(dnn>=prdist) & (ptr<prmax)] = 1
    
    qflag = gfit + sn + pr
    
    df = df.assign(Quality_flag = qflag)

    return(df)


##x/y positions (requires subtile info)
###need to create WCS info for each image - WCS(dict of series)
def recover_xy(cdata, imdat, acol='RA', dcol='DEC', apcol='RA_max', dpcol='DEC_max',
               eacol='E_RA', edcol='E_DEC', eapcol='E_RA_max', edpcol='E_DEC_max'):
    ##make colnames generic within function
    ###recover X/Yposn from image header info
    #fstart_time = time.time()
    
    #cunit, ctype, cdelt, crval, crpix, naxis
    ###values that are constant for ALL QL images
    nax = 3722
    cta = 'RA---SIN'
    ctd = 'DEC--SIN'
    cu = 'deg'
    crp = 1861.0
    cda = -0.0002777777777778
    cdd = 0.0002777777777778
    
    ###create a list of WCS transforms
    ###create a base 2D WCS for all images - need to move inside of loop
    wcslist = []
    for i in range(len(imdat)):
        aref, dref = imdat.CRVAL1.iloc[i], imdat.CRVAL2.iloc[i]
        ql_wcs = WCS({'NAXIS':2, 'NAXIS1':nax, 'NAXIS2':nax})
        ql_wcs.wcs.ctype = [cta, ctd]
        ql_wcs.wcs.cunit = [cu, cu]
        ql_wcs.wcs.crpix = [crp, crp]
        ql_wcs.wcs.crval = [aref, dref]
        ql_wcs.wcs.cdelt = [cda, cdd]
        wcslist.append(ql_wcs)
    
    ###obtain index of subtile in imdat to determine wcs to use for row in catalogue
    stlist = list(imdat['Subtile'])

    ##need to loop through cdata and append x/y coords to list
    cpos_cat = SkyCoord(ra=np.array(cdata[acol]), dec=np.array(cdata[dcol]), unit='deg')
    mpos_cat = SkyCoord(ra=np.array(cdata[apcol]), dec=np.array(cdata[dpcol]), unit='deg')
    xpos, ypos, xpmax, ypmax = [], [], [], []
    for i in range(len(cdata)):
        skypos = cpos_cat[i]
        maxpos = mpos_cat[i]
        stile = cdata.iloc[i]['Subtile']
        sti = stlist.index(stile)
        pxcoords = wcslist[sti].world_to_pixel(skypos)
        pxcoords_max = wcslist[sti].world_to_pixel(maxpos)
        
        xpos.append(float(pxcoords[0]))
        ypos.append(float(pxcoords[1]))

        xpmax.append(float(pxcoords_max[0]))
        ypmax.append(float(pxcoords_max[1]))

    ##add x/y columns to cdata
    ###errors in px coords estimate via error in pos/|cdelt|
    ###e.g. E_Xposn = E_RA/|CDELT1|
    cdata = cdata.assign(Xposn = xpos)
    cdata = cdata.assign(E_Xposn = np.array(cdata[eacol])/cdd)
    cdata = cdata.assign(Yposn = ypos)
    cdata = cdata.assign(E_Yposn = np.array(cdata[edcol])/cdd)

    cdata = cdata.assign(Xposn_max = xpmax)
    cdata = cdata.assign(E_Xposn_max = np.array(cdata[eapcol])/cdd)
    cdata = cdata.assign(Yposn_max = ypmax)
    cdata = cdata.assign(E_Yposn_max = np.array(cdata[edpcol])/cdd)
    
    return(cdata)


####nn dist needs to be limited to unique components that aren't empty islands
def add_nn_dist(df, acol='RA', dcol='DEC'):
    
    ###create column for row number - allows easy remerge
    df['dfix'] = df.index
    
    ###subset those to be used in the NN search (D_flag<2 & S_Code!='E')
    ucomps = df[(df['Duplicate_flag']<2) & (df['S_Code']!='E')].reset_index(drop=True)
    
    ##create sky position catalogue and self match to nearest OTHER component
    poscat = SkyCoord(ra=np.array(ucomps[acol]), dec=np.array(ucomps[dcol]), unit='deg')
    self_x = coords.match_coordinates_sky(poscat, poscat, nthneighbor=2)
    
    ###create new column in ucomps
    ucomps = ucomps.assign(NN_dist = self_x[1].arcsec)
    
    ###merge with df, fill na with -99
    df = pd.merge(df, ucomps[['dfix', 'NN_dist']], on='dfix', how='left')
    df['NN_dist'] = df['NN_dist'].fillna(-99)
    
    return(df)


def catalogue_process(df, subtiles, outputcols, acol='RA', dcol='DEC', old_cols=[],
                      new_cols=[], nanflag=-99, xfirst=True, fdata='first_14dec17.fits',
                      ndata='CATALOG41.FIT', xnvss=True, name_prefix='VLASS'):
    ###process catalogue
    
    ##0) scale units flux[Jy -> mJy], size[deg -> arcsec]
    catcols = list(df.columns)
    ###define columns to scale
    sizecols = [col for col in catcols if 'Maj' in col]
    sizecols = sizecols + [col for col in catcols if 'Min' in col]
    fluxcols = [col for col in catcols if 'flux' in col]
    fluxcols = fluxcols + ['Isl_rms', 'Isl_mean', 'Resid_Isl_rms', 'Resid_Isl_mean']
    ##scale columns
    size_scale = size_units_in.to(size_units_out)
    flux_scale = flux_units_in.to(flux_units_out)
    df = colscale(df=df, collist=sizecols, scalefactor=size_scale)
    df = colscale(df=df, collist=fluxcols, scalefactor=flux_scale)
    
    ##1) rename PyBDSF_Source_id to Component_name
    df = col_rename(df=df, old_cols=old_cols, new_cols=new_cols)
    
    ##2) create truncated component name in IAU format
    ##
    #still needs to be done here till I can work out where Michelle's code assigns the name prefix. if it's possible to make that image file dependent there it would make sense to move this to her code.
    df = df.assign(Component_name=source_name(ra=df[acol], dec=df[dcol], prefix=name_prefix))

    ##3) reset S_Code for empty islands to 'E'
    ##now performed in earlier code
#    scode, sid = np.array(df['S_Code']), np.array(df['Component_id'])
#    scode[(sid<0)] = 'E'
#    df = df.assign(S_Code = scode)

    ##3) add flag for duplicates (and if duplicated is it the preferred duplicate to use
    ###now performed in earlier code
#    df = find_duplicates(df=df, acol=acol, dcol=dcol, pos_err=duplicate_search_arcsec)

    ##4) add NN_distance for all unique, only recommended - has to be post duplicates
    ##now performed in earlier code
#    df = add_nn_dist(df=df, acol=acol, dcol=dcol)

    ##5) add Quality flag - has to be done post NN info
    ###now performed in earlier code
#    df = q_flag(df, snmin=sig_noise_threshold, prmax=peak_to_ring_threshold,
#                prdist=peak_ring_dist)
    
    ##6) create x/y_postitions plus errors from RA/DEC and fits header info
    ##now performed in earlier code
#    df = recover_xy(cdata=df, imdat=subtiles)

    ##7) redo first distance to first/nvss (all components)
    ##now dependent on revlad argument (to be removed entirely once completely happy with input data quality
    if xfirst == True:
        first = Table(fits.open(fdata)[1].data).to_pandas()
        fcat = SkyCoord(ra=np.array(first[acol]), dec=np.array(first[dcol]), unit='deg')
        dcat = SkyCoord(ra=np.array(df[acol]), dec=np.array(df[dcol]), unit='deg')
    
        dxf = dcat.match_to_catalog_sky(fcat)
    
        df = df.assign(FIRST_distance = dxf[1].arcsec)
    
    if xnvss == True:
        nvss = Table(fits.open(ndata)[1].data).to_pandas()
        ncat = SkyCoord(ra=np.array(nvss['RA(2000)']), dec=np.array(nvss['DEC(2000)']), unit='deg')
        dcat = SkyCoord(ra=np.array(df[acol]), dec=np.array(df[dcol]), unit='deg')
        
        dxn = dcat.match_to_catalog_sky(ncat)
        
        df = df.assign(NVSS_distance = dxn[1].arcsec)
    
    ####possibly, if combined in to sources - unlikely for v1, add as dummy columns
    ####can be updated when Table 2: Source table is added (has to be done after then)
    ##now performed in earlier code
#    dummystring = np.zeros(len(df), dtype=bool).astype(str)
#    dummystring[(True)] = 'N/A'
#    ##7) add Source_name
#    df = df.assign(Source_name = dummystring)
#
#    ##8) add Source_type
#    df = df.assign(Source_type = dummystring)

    ###tidy up
    #mop up any NaNs from e.g. PyBDSF failures
    ##now performed in earlier code
#    df = df.fillna(nanflag)
#    #sort by RA ascending and reset index
#    df = df.sort_values(by='RA', ascending=True).reset_index(drop=True)
#    #set columns == pcat1
    df = df[outputcols]

    return(df)


###wrapper for cat_process that loads data
def vprocess_io(vlass_in, subtiles_in, first_in, nvss_in, vlass_out, pubcols_only=True,
                oldcolnames=[], newcolnames=[], redovlad=revlad):
    ###loads data and outputs csv file
    vlass = pd.read_csv(vlass_in)
    imdata = pd.read_csv(subtiles_in)
    
    name_prefix = determine_source_name_prefix_from_image_names(obtain_eg_image_name(imdata))
    
    ###filter out unwanted columns from input if redoing value-add
    vlass = drop_input_cols(df=vlass, dropcols=redocolumns)
    
    ##define columns to output
    if pubcols_only==True:
        colnames = ['Component_name', 'Component_id','Isl_id', 'RA', 'DEC', 'E_RA',
                    'E_DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux',
                    'Maj', 'E_Maj', 'Min', 'E_Min', 'PA', 'E_PA', 'Isl_Total_flux',
                    'E_Isl_Total_flux', 'Isl_rms', 'Isl_mean', 'Resid_Isl_rms',
                    'Resid_Isl_mean', 'RA_max', 'DEC_max', 'E_RA_max', 'E_DEC_max',
                    'S_Code', 'Xposn', 'E_Xposn', 'Yposn', 'E_Yposn', 'Xposn_max',
                    'E_Xposn_max', 'Yposn_max', 'E_Yposn_max', 'DC_Maj', 'E_DC_Maj',
                    'DC_Min', 'E_DC_min', 'DC_PA', 'E_DC_PA', 'Tile', 'Subtile',
                    'NVSS_distance', 'FIRST_distance', 'Peak_to_ring', 'Duplicate_flag',
                    'Quality_flag', 'Source_name', 'Source_type', 'QL_cutout', 'NN_dist']
    else:
        colnames = ['Component_name', 'Component_id', 'Isl_id', 'RA', 'DEC', 'E_RA',
                    'E_DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux',
                    'Maj', 'E_Maj', 'Min', 'E_Min', 'PA', 'E_PA', 'Isl_Total_flux',
                    'E_Isl_Total_flux', 'Isl_rms', 'Isl_mean', 'Resid_Isl_rms',
                    'Resid_Isl_mean', 'RA_max', 'DEC_max', 'E_RA_max', 'E_DEC_max',
                    'S_Code', 'Xposn', 'E_Xposn', 'Yposn', 'E_Yposn', 'Xposn_max',
                    'E_Xposn_max', 'Yposn_max', 'E_Yposn_max', 'Maj_img_plane',
                    'E_Maj_img_plane', 'Min_img_plane', 'E_Min_img_plane',
                    'PA_img_plane', 'E_PA_img_plane','DC_Maj', 'E_DC_Maj', 'DC_Min',
                    'E_DC_Min', 'DC_PA', 'E_DC_PA', 'DC_Maj_img_plane', 'E_DC_Maj_img_plane',
                    'DC_Min_img_plane', 'E_DC_Min_img_plane', 'DC_PA_img_plane',
                    'E_DC_PA_img_plane','Tile', 'Subtile', 'QL_image_RA', 'QL_image_DEC',
                    'NVSS_distance', 'FIRST_distance', 'Peak_to_ring', 'Duplicate_flag',
                    'Quality_flag', 'Source_name', 'Source_type', 'QL_cutout',
                    'NN_dist', 'BMAJ', 'BMIN', 'BPA']

    ##process catalogue
    vlass = catalogue_process(df=vlass, subtiles=imdata, acol=ra_col, dcol=dec_col,
                              fdata=first_in, ndata=nvss_in, outputcols=colnames,
                              old_cols=oldcolnames, new_cols=newcolnames,
                              name_prefix=name_prefix,
                              xfirst=redovlad,
                              xnvss=redovlad)

    ###write file
    vlass.to_csv(vlass_out, index=False)

    return


def parse_args():
    'parse command line arguments'
    parser = argparse.ArgumentParser(description="tidy up and add flags to component table")
    parser.add_argument("pybdsf_sources", help="pybdsf source list",
                        action='store')
    parser.add_argument("subtiles", help="subtile data",
                        action='store')
    parser.add_argument("nvss", help="NVSS positions file",
                        action='store')
    parser.add_argument("first", help="FIRST positions file",
                        action='store')
                    
    args = parser.parse_args()
    return(args)


################################################################################
###main code

###columns to rename
oldcols = ['PyBDSF_Source_id']
newcols = ['Component_id']


if __name__ == '__main__':
    args = parse_args()
    vprocess_io(vlass_in=args.pybdsf_sources, subtiles_in=args.subtiles,
                first_in=args.first, nvss_in=args.nvss, vlass_out=outfile,
                pubcols_only=False, oldcolnames=oldcols, newcolnames=newcols)


####end
