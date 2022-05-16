###code to produce urls for cutouts based on VLASS QL images at CADC

import numpy as np, urllib
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
from astropy import units as u


####test coordinates
#qltest = 'https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VLASS/VLASS1.2.ql.T09t22.J141001-043000.10.2048.v1.I.iter1.image.pbcor.tt0.subim.fits'
#sttest = 'J141001-043000'
#comptest = 'VLASS1QLCIR J140820.90-041007.9'
#postest = SkyCoord(ra=212.08712166890496, dec=-4.16888789803483, unit='deg')
#rtest = 1.5*u.arcmin
#
#cadc_urls = '../other_survey_data/CADC_subtile_urls_epoch1.fits'
#
#subtile_urls = Table.read(cadc_urls, format='fits')
#
#tpos = SkyCoord(ra=24.601498156281927, dec=+33.2552769199096, unit='deg')
#ttile = 'J013823+333000'
#turl = subtile_urls[(subtile_urls['Subtile']==ttile)]['cadc_url'][0]





def get_ql_cadc_urls(component_data,
                     urllist='../other_survey_data/CADC_subtile_urls_epoch1.fits',
                     joinkey='Subtile'):
    'grab subtile CADC urls and append to component list -- assumes join_key is a column in both data sets'
    
    urlinfo = Table.read(urllist)
    outdata = join(component_data, urlinfo, keys=joinkey, join_type='left')
    
    return outdata



def get_cutout_url(ql_url, coords, radius=1.5*u.arcmin):
    standard_front = 'https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/sync?ID=ad%3AVLASS%2F'
    #coords assumes astropy sky coord, radius now an astropy angle
    #ql_url is the url of the CADC hosted 1sq deg QL image
    encoded_ql = urllib.parse.quote(ql_url.split("/")[-1])
    encoded_ql = encoded_ql.replace('%3F','&').replace('?','&')
    cutout_end = f"&CIRCLE={coords.ra.value}+{coords.dec.value}+{radius.to(u.deg).value}"
    return standard_front+encoded_ql+cutout_end


def add_cutout_url_to_data(data, subtile_file=cadcurlfile, radius=1.5*u.arcmin,
                           subtile_key='Subtile',
                           acol='RA', dcol='DEC', punits='deg',
                           outcolname='cadc_cutout_url'):
    'use component positions and subtile_url to add cutout urls to data'
    
    ###note that this can reorder the table, add idx column to join on and remove after
    data['idx'] = np.arange(len(data))+1
    
    add_sturl = get_ql_cadc_urls(component_data=data, urllist=subtile_file,
                                 joinkey=subtile_key)

    curls = []
                                 
    ###makes sky coords once then extract
    poscat = SkyCoord(ra=add_sturl[acol], dec=add_sturl[dcol], unit=punits)
                                 
    for i in range(len(add_sturl)):
        pos = poscat[i]
        sturl = add_sturl['cadc_url'][i]
        curls.append(get_cutout_url(ql_url=sturl, coords=pos, radius=radius))

    add_sturl[outcolname] = curls

    ##rejoin with data
    data = join(data, add_sturl[['idx', outcolname]], keys='idx', join_type='left')
    
    ###sort by idx so in same order as input and remove idx
    data.sort('idx')
    data.remove_column('idx')
    
    return data




####need to list available CADC images and select from here (version numbers may differ)


