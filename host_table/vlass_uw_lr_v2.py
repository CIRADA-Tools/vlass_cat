####runs v1 of likelihood-ratio (LR) code wirtten by Leah Morabito for LoTSS
####wrapper written by Yjan Gordon (May 2020) for CIRADA and applies LR code to VLASS and unWISE
####wrapper takes vlass source catalogue and directory of unWISE images as inputs
####unWISE images and associated source catalogues are downloaded on-the-fly
####files are processed to useble format and fed in to LR code
####outputs matches and associated radio and host data
####images and catalogues downloaded are removed once used along with intermediate step files

import subprocess, os, shutil, pandas as pd, numpy as np, wget, argparse
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from urllib.error import HTTPError
from subset_unwise import *

########################################
#time of code check - not important for code to run
import time
start_time = time.time()
########################################

#######################################################################################
###def params

##output folder
outfolder = 'LR_output/'

##input files
#vf_in = 'input_data/CIRADA_VLASS1QL_source_candidates_v1b.csv'
#uw_in = 'input_data/CIRADA_VLASS1QL_input_unWISE_image_list_v1b.csv'

##likelihood ratio code to call
lr_file = 'likelihood_ratio_matching.py'

##data parameters
w1maglim = 18 ###vega magnitude limit to consider (default to 18)
psf_uw = 2.641100 ###fwhm of unwise
badflag = -99

###columns required from unWISE catalogue
uwcols = ['unwise_detid', 'ra', 'dec', 'fluxlbs', 'dfluxlbs']


#######################################################################################
###def functions

def download_url(url, target, max_attempts=5):
    # Often encounter the following error:
    # urllib.error.HTTPError: HTTP Error 504: Gateway Time-out
    # Repeat the download attempt for up to `max_attempts` tries
    # Return True if the download was successful
    for attempt in range(max_attempts):
        try:
            wget.download(url=url)
            return True
        except HTTPError:
            continue

    print(f"Failed to download image {target}")
    return False



def vega_to_nmgy(mag):
    ###convert vega magnitude to flux value in nMgy
    mag = np.array(mag)
    
    flux = 10**((22.5 - mag)/2.5)
    
    return(flux)


def nmgy_to_vega(flux, flagval=badflag):
    ##converts flux in nMgy to vega magnitude
    flux = np.array(flux)
    
    ###account for negative values
    flux_no_neg = flux.copy()
    
    ##set array filter for bad flux values
    badflux = (flux<=0)
    
    flux_no_neg[badflux] = 1
    
    ##convert to vega mags
    vega = 22.5 - 2.5*np.log10(flux_no_neg)
    
    ###flag bad magnitudes
    vega[badflux] = flagval
    return(vega)


def mag_err(dflux, flagval=badflag):
    ##estimate delta magnitude from flux error
    dflux = np.array(dflux) ##assumed this is relative error in flux (noise/signal)
    
    ###flag bad errors
    bad_filter = (dflux<=0)
    
    e_flux = dflux.copy()
    e_flux[bad_filter] = 1
    
    e_flux = 2.5*np.log10(1+e_flux)
    
    e_flux[bad_filter] = flagval
    return(e_flux)


def unwise_cat_process(file, outfname='test.fits', use_cols=uwcols, fcol='fluxlbs',
                       band='w1', ow=False):
    ###process unWISE cataloge info from fits file and return Table
    data = Table(fits.open(file)[1].data).to_pandas()
    
    ##subset to rquired columns
    data = data[use_cols]
    
    dfcol = 'd'+fcol
    ###create magnitude info
    mag = nmgy_to_vega(flux=data[fcol])
    flux_rel_noise = np.array(data[dfcol])/np.array(data[fcol])
    emag = mag_err(dflux=flux_rel_noise)
    
    data[band + '_mag'] = mag
    data[band + '_mag_err'] = emag
    
    ##rename flux cols to include band
    cols = list(data.columns)
    cols[cols.index(fcol)] = band + '_' + fcol
    cols[cols.index(dfcol)] = band + '_' + fcol + '_err'
    
    ###add in mask and stargal columns - these allow everything at present
    data['Mask'] = np.zeros(len(data))
    data['stargal'] = np.ones(len(data))

    Table.from_pandas(data).write(outfname, format='fits', overwrite=ow)
    return


def new_header(header, band='w1', survey='unWISE-neo4'):
    ##create new header for legacy survey cutout cube
    newhead = header.copy()
    
    ###remove/replace keywords for new header
    keylist = list(header.keys())
    remkeys = ['NAXIS3', 'BANDS', 'BAND0', 'BAND1']
    
    for key in remkeys:
        if key in keylist:
            del(newhead[key])

    ###change keyword NAXIS and add in BAND == 'w1' and comment on processing
    ###add in CDELT1, CDELT2 keywords (replaces CD1_1, CD2_2)
    newhead['NAXIS'] = 2
    newhead['BAND'] = band
    newhead['SURVEY'] = survey
    newhead['CDELT1'] = newhead['CD1_1']
    newhead['CDELT2'] = newhead['CD2_2']
    newhead['Comment'] = '  processed for LR VLASS xIDs as part of the CIRADA project.'
    return(newhead)


def mask_image(image_data, fmin):
    immask = image_data.copy()
    
    ###mask out nan/inf
    immask[np.isnan(immask)] = 1
    immask[np.isinf(immask)] = 1
    
    immask[image_data>fmin] = 0
    immask[image_data<=fmin] = 1
    return(immask)


def write_image_file(image_data, header_info, filename):
    ##write a fits file of image data
    hdu = fits.PrimaryHDU(image_data)
    hdu.header = header_info
    hdu.writeto(filename, overwrite=False)
    return


def mask_image_file(fname_in, maskfile='testmask.fits', imarray_index=0,
                    maglim=w1maglim, psf=psf_uw, band='w1', survey='unWISE-neo4'):
    ###process the legacy survey obtained files into image (and mask if option selected)
    hdu = fits.open(fname_in)
    
    ###extract header and image
    oldhead = hdu[0].header
    imdim = oldhead['NAXIS']
    if imdim == 2:
        image = hdu[0].data
    else:
        image = hdu[0].data[imarray_index]
    
    ##create new header
    nhead = new_header(oldhead, band=band, survey=survey)

    ##mask image
    ##estimate flux per pixel to use as mask based on source mag limit
    pixel_scale = nhead['CDELT2']*3600
    flim_int = vega_to_nmgy([maglim])[0]
    ps_area = np.pi*psf**2
    flim_px = flim_int/ps_area
                                 
    image_mask = mask_image(image_data=image, fmin=flim_px)
                             
    ##mask header
    nhead_mask = nhead.copy()
    nhead_mask['Comment'] = '  MASKED IMAGE'
    nhead_mask['Comment'] = '  minimum px value coresponds to point source with w1<18mag'
    nhead_mask['Comment'] = '  assumes unWISE PSF FWHM == ' + str(np.round(psf, 2)) +'"'
                                 
    ##write mask to file
    write_image_file(image_data=image_mask, header_info=nhead_mask, filename=maskfile)
    return


def process_vfile(file_in, file_out, single_component=False, multi_component=False):
    ###processes CIRADA VLASS source csv into fits format for Morabito LR code
    vcat = pd.read_csv(file_in)
    #make Source_id col as needed by morabito code
    vcat = vcat.assign(Source_id = np.array(vcat['Source_name']))
    if single_component==True:
        vcat = vcat[(vcat['n_components']==1)].reset_index(drop=True)
    elif multi_component==True:
        vcat = vcat[(vcat['n_components']>1)].reset_index(drop=True)
    
    vtab = Table.from_pandas(vcat)
    vtab.write(file_out, format='fits', overwrite=True)
    return


def get_unwise_urls(directory_file):
    ###get the url list for the catalogues and images needed
    ### - a subsetted directory should be provided
    df = pd.read_csv(directory_file)
    
    coad_list = list(df['Coad_ID'])
    
    cat0 = 'https://faun.rc.fas.harvard.edu/unwise/release/cat/'
    cat1 = '.1.cat.fits'
    im0 = 'http://unwise.me/data/neo4/unwise-coadds/fulldepth/'
    im1 = '/unwise-'
    im3 = '-w1-img-u.fits'
    
    catlist = [cat0 + coad_list[i] + cat1 for i in range(len(coad_list))]
    imlist = [im0 + coad_list[i][:3] + '/' + coad_list[i] + im1 + coad_list[i] + im3
              for i in range(len(coad_list))]
    
    return(catlist, imlist)


def uwimurl(coadID):
    ###build the url string for one unwise target
    im0 = 'http://unwise.me/data/neo4/unwise-coadds/fulldepth/'
    im1 = '/unwise-'
    im3 = '-w1-img-u.fits'
    
    imurl = im0 + coadID[:3] + '/' + coadID + im1 + coadID + im3
    
    return(imurl)


def uwcaturl(coadID):
    ###build the url string for one unwise target
    cat0 = 'https://faun.rc.fas.harvard.edu/unwise/release/cat/'
    cat1 = '.1.cat.fits'
    
    caturl = cat0 + coadID + cat1
    return(caturl)


def uwcatname(coadID):
    ##determines file name of raw unwise cat from coadID
    filename = coadID + '.1.cat.fits'
    return(filename)


def uwimname(coadID):
    ###determinesfile name of raw unwise image from coadID
    filename = 'unwise-' + coadID + '-w1-img-u.fits'
    return(filename)


####need to incorporate downloading urls and processing them in to batch_run_lr
###need to load radio cat and write directory of uW images that correspond to this footprint

def target_uw_ims(vfile_in, ifile_in):
    ###take radio cat and uw image directory and create list of images to use
    vdata = pd.read_csv(vfile_in)
    idata = pd.read_csv(ifile_in)
    
    vcat = SkyCoord(ra=np.array(vdata['RA']), dec=np.array(vdata['DEC']), unit='deg')
    icat = SkyCoord(ra=np.array(idata['cenRA']), dec=np.array(idata['cenDec']), unit='deg')
    
    vxi = vcat.match_to_catalog_sky(icat)
    
    vdata = vdata.assign(Coad_ID = np.array(idata['Coad_ID'].iloc[vxi[0]]))
    vdata = vdata.assign(uw_sep = np.array(vxi[1].arcsec))
    
    targets = pd.merge(idata, vdata[['Coad_ID']].drop_duplicates(subset='Coad_ID'),
                       on='Coad_ID', how='right').reset_index(drop=True)
    
    return(targets, vdata)


def morabito_lr(command_args, fieldname, file_ext, to_folder):
    ###run Morabito LR code and polish up output files
    ###alter output file name to w1_LR_matches_field.dat where field == e.g. 1493p166
    match_file_x = 'w1_LR_matches_' + fieldname + file_ext #(e.g. _sc.csv)
    wcatfile = command_args[2] ###WISE cat file
    no_match = [] ###list of fields without a match
    
    ###run LR code
    output = subprocess.Popen(command_args, stdout=subprocess.PIPE, universal_newlines=True)
    output.wait() ##wait till process has finishes before continuing loop
    
    ###need to merge matches with processed cat file before deleting
    matches = Table.read('w1_LR_matches.dat', format='ascii').to_pandas()
    info = Table(fits.open(wcatfile)[1].data).to_pandas()
    infocols = ['unwise_detid', 'ra', 'dec', 'fluxlbs', 'dfluxlbs',
                'w1_mag', 'w1_mag_err']
        
    matchinfo = pd.merge(matches, info[infocols], left_on='w1_ID', right_on='unwise_detid',
                         how='left')
                
    if len(matchinfo) < 1:
        print('WARNING: matches x info is empty - ', fieldname)
        no_match.append(fieldname)

    ###save merged file to output_dir
    matchinfo.to_csv(outfolder+match_file_x, index=False)

    return(no_match)


def batch_run_lr(ifile_in, vfile_in, outfolder='LR_output/', countstep=200):
    ##batch run Morabito LR code moving files as produced
    ##create individual image outputs for hosts (stack as second phase)
    ##merge with uw catalogue info
    ##separate runs with isolated and close-double (different beam size in config)
    ###need to account for files already downloaded and timeout errors on download
    
    fstart_time = time.time() ###set timer for testing
    
    ###lists to append fields with no matches (separate for simple, complex)
    nm_simple, nm_complex = [], []
    
    
    ##process vlass catalogue into fits_format
    process_vfile(file_in=vfile_in, file_out='vlass_cat_sc.fits', single_component=True)
    process_vfile(file_in=vfile_in, file_out='vlass_cat_dt.fits', multi_component=True)

#    ###################################################################
#    ###replace with function to obtain needed images from full manifest
#    ifile_cat = pd.read_csv(ifile_in)
#    tfields = list(ifile_cat['Coad_ID'].drop_duplicates())
#    ###################################################################
    ifile_cat = subset_needed_unWISE_images(srcfile=vfile_in, imfile=ifile_in)
    tfields = list(ifile_cat['Coad_ID'])


    ###check if target fields already been processed
    od_filelist = os.listdir(outfolder)
    
    gotfields = [i for i in tfields if 'w1_LR_matches_'+i+'_sc.csv' in od_filelist
                 and 'w1_LR_matches_'+i+'_dt.csv' in od_filelist]
    
    tfields = [i for i in tfields if i not in gotfields]
    

    fcount = np.arange(0, len(tfields), countstep)
    
    for i in range(len(tfields)):
        if i in fcount: ###for testing purposes only - provide whitespace buffer for terminal vis
            print(' ')
            print(' ')
            print('------------------------------')
            print(i, '/', len(tfields), ' --- ' , np.round(time.time() - fstart_time, 2), 's')
            print('------------------------------')
            print(' ')
            print(' ')
    
        field = tfields[i]

        ###get url for unWISE image and cat
        imurl = uwimurl(field)
        caturl = uwcaturl(field)
        
        imfile_u = uwimname(field)
        catfile_u = uwcatname(field)
        
        ###download files
#        wget.download(imurl)
#        wget.download(caturl)
        download_url(url=imurl, target=field+'_im')
        download_url(url=caturl, target=field+'_cat')

        ###set file names for cat and mask image - as being deleted keep in wd
        maskfile = 'uw_' + field + '_mask.fits'
        wcatfile = field + '_pw1cat.fits'
        
        ###need to process cat and image and output files for LR code
        ###mask image
        mask_image_file(fname_in=imfile_u, maskfile=maskfile)
        ###process cat data
        unwise_cat_process(file=catfile_u, outfname=wcatfile)
        
        ###set command arguments for running LR code
        command_args_sc = ['python3', lr_file, wcatfile, 'vlass_cat_sc.fits', maskfile, '--overwrite',
                        '--config_file=lr_config_sc.txt']
        command_args_dt = ['python3', lr_file, wcatfile, 'vlass_cat_dt.fits', maskfile, '--overwrite',
                           '--config_file=lr_config_dt.txt']
            
        ###files to remove after LR run
        remove_files = [maskfile, wcatfile, imfile_u, catfile_u, field+'_pw1cat_master.fits',
                        field+'_pw1cat_master_masked.fits']


        ###Run LR separately for doubles/triples
        nm_sc = morabito_lr(command_args=command_args_sc, fieldname=field, file_ext='_sc.csv',
                            to_folder=outfolder)
        nm_dt = morabito_lr(command_args=command_args_dt, fieldname=field, file_ext='_dt.csv',
                            to_folder=outfolder)

        ##add fields to list
        nm_simple = nm_simple + nm_sc
        nm_complex = nm_complex + nm_dt
        
        
        ###remove field specific files
        remove_files = [maskfile, wcatfile, imfile_u, catfile_u, field+'_pw1cat_master.fits',
                        field+'_pw1cat_master_masked.fits']
        for file in remove_files:
            os.remove(file)



        ###tidy up folder at end
        if i == len(tfields)-1:
            magmatch_files = os.listdir()
            magmatch_files = [file for file in magmatch_files if 'w1_matched_mags' in file]
            also_remove = ['w1_Q0_estimates.dat', 'w1_q0_estimate.png',
                           'w1_magnitude_distributions.png', 'w1_LR_values.png',
                           'w1_LR_matches.dat', 'w1_Fleuren_random_catalogue_masked.fits',
                           'w1_Fleuren_no_counterparts.dat', 'unmasked_area.dat',
                           'Sky_coverage.png', 'w1_Fleuren_random_catalogue.fits',
                           'vlass_cat_sc_masked.fits', 'vlass_cat_sc.fits',
                           'vlass_cat_dt_masked.fits', 'vlass_cat_dt.fits']

            to_remove = also_remove + magmatch_files
            directory_list = os.listdir() ###allows process to complete if a file not in dir
            for file in to_remove:
                if file in directory_list:
                    os.remove(file)

    ##output lists of fields with no matches
    nomatch_sc = pd.DataFrame({'Coad_ID': nm_sc})
    nomatch_dt = pd.DataFrame({'Coad_ID': nm_dt})
    
    nomatch_sc.to_csv(outfolder+'unWISE_fields_no_matches_sc.csv', index=False)
    nomatch_dt.to_csv(outfolder+'unWISE_fields_no_matches_dt.csv', index=False)
    
    return


def parse_args():
    'parse command line arguments'
    parser = argparse.ArgumentParser(description="find hosts")
    parser.add_argument("source_candidates", help="table of candidate sources",
                        action='store')
    parser.add_argument("unwise_images", help="list of unWISE images to parse through",
                        action='store')
        
    args = parser.parse_args()
    return(args)


#######################################################################################

if __name__ == '__main__':
    args = parse_args()
    batch_run_lr(ifile_in=args.unwise_images, vfile_in=args.source_candidates,
                 outfolder=outfolder, countstep=200)


################################################################################
################################################################################
#########################################
#time taken for code to run
print('END: ', np.round(time.time() - start_time, 2), 's')




