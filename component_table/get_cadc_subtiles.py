import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.cadc import Cadc
import warnings

###mute bloody astropy warnings -- may want to switch this off later?
warnings.filterwarnings('ignore')


########################################
#time of code check - not important for code to run
import time
start_time = time.time()
########################################

####code for getting CADC urls of VLASS subtiles based on meta -- will return all epochs for position



####set up list of central image positions to find data for (n~35k)
subtile_file = '/Users/yjangordon/Documents/science/survey_data/VLASS/catalogues/CIRADA_VLASS1QL_table3_subtile_info_v1.fits' ###change as appropriate


qlinfo = Table.read(subtile_file, format='fits')

qlposcat = SkyCoord(ra=qlinfo['OBSRA'], dec=qlinfo['OBSDEC'])



####set up cadc query
cadc = Cadc()



###can also use any input table with publisherID column in with cadc.get_data_urls -- this should be faster!

def write_data_urls_to_file(data):
    ##data must include publiserID column
    urlist = cadc.get_data_urls(data)
    tab = Table()
    Table['cadc_url'] = urllist
    Table.write('vlass_ql_cadc_urllist.fits', format='fits')
    return





def getcadc_imurl(coord, collection='VLASS'):
    result = cadc.query_region(coord, collection=collection)
    urls = cadc.get_data_urls(result)
    if type(urls) != list:
        urls = list(urls)
    return urls



test = qlposcat[:10]


def getallimurls(cat, collection='VLASS'):
    t0 = time.time()
    n = len(cat)
    nstep = np.arange(0, len(cat), 1)
    urllist = []
    for i in range(len(cat)):
        if i in nstep:
            telapsed = np.round(time.time()-t0, 2)
            print(i, '/', n, '  time='+str(telapsed)+' s')
        urllist = urllist +  getcadc_imurl(cat[i])
    return urllist





#########################################################################
print('END: ', np.round(time.time()-start_time, 2), 's')
