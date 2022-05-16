###function to subset the unwise images required for data set from manifest of all unwise images


import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u



def subset_needed_unWISE_images(srcfile, imfile, acol_src='RA', dcol_src='DEC',
                                acol_im='cenRA', dcol_im='cenDec'):
    'load source and unwise image files and subset those images needed for LR'
    
    source_data = pd.read_csv(srcfile)
    imdata = pd.read_csv(imfile)

    ###need to create skycat for sources and images to find the nearest image to each source
    srccat = SkyCoord(ra=np.array(source_data[acol_src]),
                      dec=np.array(source_data[dcol_src]),
                      unit='deg')
    imcat = SkyCoord(ra=np.array(imdata[acol_im]),
                     dec=np.array(imdata[dcol_im]),
                     unit='deg')

    ####match source to nearest image
    sxim = srccat.match_to_catalog_sky(imcat)

    ###select only images that are the closest to sources in list - only need each image once
    imidx = np.unique(sxim[0])

    imdir = imdata.iloc[imidx]
    
    return(imdir)



