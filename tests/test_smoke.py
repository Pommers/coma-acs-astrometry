from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from src.acs_astrometry import Settings, reconcile_acs_catalog, validate_with_jwst

rng=np.random.default_rng(123)
n=160
w=WCS(naxis=2)
w.wcs.crpix=[500,500]
w.wcs.cd=np.array([[-0.05/3600,0],[0,0.05/3600]])
w.wcs.crval=[195.24,28.02]
w.wcs.ctype=['RA---TAN','DEC--TAN']
p=Path('data/coma_astrometry/test_synthetic.fits')
fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=np.zeros((1000,1000),dtype=np.float32),header=w.to_header(),name='SCI')]).writeto(p,overwrite=True)
x=rng.uniform(100,900,n); y=rng.uniform(100,900,n)
ra,dec=w.pixel_to_world_values(x+4.3+rng.normal(0,0.1,n),y-9.2+rng.normal(0,0.1,n))
ra[0],dec[0]=w.pixel_to_world_values(x[0]+25,y[0]-30)
df=pd.DataFrame({'source_id':range(n),'pointing':['2-0']*n,'x':x,'y':y,'x_wcs':ra,'y_wcs':dec,'type':['gc']*n})
res,summ=reconcile_acs_catalog(df,{'2-0':p})
assert summ.iloc[0]['n_outlier']==1, summ
assert res.iloc[0].astrom_status=='outlier'
assert np.isnan(res.iloc[0].ra_reg)
assert (res.type==df.type).all()
assert np.allclose(res.x,df.x)
assert abs(summ.iloc[0].dx_bulk_pix-4.3)<0.05
assert abs(summ.iloc[0].dy_bulk_pix+9.2)<0.05
ids=np.arange(1,81)
anc=df.iloc[ids][['source_id','x_wcs','y_wcs']].copy()
anc.rename(columns={'x_wcs':'jwst_ra','y_wcs':'jwst_dec'},inplace=True)

# synthetic independent measurement 0.01 arcsec east/north
from astropy.coordinates import SkyCoord
import astropy.units as u
coords=SkyCoord(anc.jwst_ra.to_numpy()*u.deg,anc.jwst_dec.to_numpy()*u.deg)
shifted=coords.spherical_offsets_by(0.01*u.arcsec,-0.015*u.arcsec)
anc['jwst_ra']=shifted.ra.deg;anc['jwst_dec']=shifted.dec.deg
val,sm=validate_with_jwst(res,summ,anc)
assert sm.iloc[0].jwst_status=='validated'
assert val.iloc[1].astrom_status=='validated'
assert val.iloc[0].astrom_status=='outlier'
assert np.allclose(val.ra_reg.iloc[1:],df.x_wcs.iloc[1:])
print('PASS: bulk fit, outlier protection, original values, JWST validation')
p.unlink()
