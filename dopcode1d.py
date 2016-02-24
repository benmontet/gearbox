import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
import numpy.random as rand
import emcee
from scipy.io.idl import readsav
from pysynphot import observation
from pysynphot import spectrum
from timeit import default_timer as timer

def setfig(fig,**kwargs):
    """Tim's handy plot tool
    """
    if fig:
        pl.figure(fig,**kwargs)
        pl.clf()
    elif fig==0:
        pass
    else:
        pl.figure(**kwargs)
        
def plothist(data, bins=100, over=False, **kwargs):
    if not over: setfig(2)
    foo = pl.hist(data,bins=bins,histtype='step',**kwargs)

def get_iodine(wmin, wmax):
    file = '/Users/johnjohn/Dropbox (Caltech Exolab)/research/dopcode_new/ftskeck50.sav'
    sav = readsav(file)
    wav = sav.w
    iod = sav.s
    use = np.where((wav >= wmin) & (wav <= wmax))
    return wav[use], iod[use]

def extend(arr, nel, undo=False, wavelength=False):
    new = arr.copy()
    if undo:
        return new[nel:len(new)-nel]
    else:
        if wavelength:
            dw = arr[1]-arr[0]
            lower = np.arange(arr.min()-nel*dw, arr.min(), dw)
            upper = np.arange(arr.max(), arr.max()+nel*dw, dw)
            new = np.append(np.append(lower, arr), upper)
        else:
            new = np.append(np.append(np.zeros(nel)+arr[0], new), np.zeros(nel)+arr[-1])
        return new
        
def numconv(y, kern):
    ynew = y.copy()
    lenex = 10
    ynew = extend(ynew, lenex)
    new = fftconvolve(ynew, kern, mode='full')
    new /= kern.sum()
    nel = len(kern)+lenex*2
    new = new[nel/2:len(new)-nel/2+1]
    return new

def rebin_spec_new(wold, sold, wnew):
    """ Define the left and right limits of the first Wnew pixel. Keep in mind
    that pixels are labled at their centers. """    
    dw_new = wnew[1]-wnew[0]
    w_left = wnew.min() - 0.5 * dw_new
    w_right = w_left + dw_new
    Nsub = 10. # use 10 sub pixels for the template across new pixel 
    """ Create a finely sampled 'sub-grid' across a pixel. We'll later integrate
    the stellar spectrum on this fine grid for each pixel  """
    wfine_sub = np.linspace(w_left, w_right, Nsub, endpoint=False)
    Npixels = len(wnew) #Number of pixels in the new wavelength scale
    """ Copy an individual pixel subgrid Npixels times, aka 'tiling' """
    wsub_tile = np.tile(wfine_sub, Npixels)
    """ Create a separate array that ensures that each pixel subgrid is 
    properly incremented, pixel-by-pixel following the Wnew wavelength scale"""
    step = np.repeat(np.arange(Npixels), Nsub) * dw_new
    wfine = wsub_tile + step #Finely sampled new wavelength scale
    dsub = wfine[1] - wfine[0]
    wfine += dsub/2. # Center each subgrid on the pixel
    ifunction = interp1d(wold, sold) #Create spline-interpolation function
    sfine = ifunction(wfine) #Calculate interpolated spectrum 
    sfine_blocks = sfine.reshape([Npixels,Nsub]) #Prepare for integration
    snew = np.sum(sfine_blocks, axis=1)/Nsub #Efficient, vector-based integration! 
    return snew


def rebin_spec(wave, specin, wavnew):
    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
    return obs.binflux

   
def jjgauss(x, *a):
    """ For free parameters create a Gaussian defined at x
    """
    return a[0] * np.exp(-0.5*((x - a[1])/a[2])**2) 

class Chunk(object):
    def __init__(self, num, bstar=False):
        info = np.load('/Users/johnjohn/Dropbox (Caltech Exolab)/research/dopcode_new/info9407/info9407_'+str(num)+'.npy')
        self.info = info
        pad = 10
        self.wiod, self.siod  = get_iodine(info.wiod[0].mean()-pad, info.wiod[0].mean()+pad)
        self.iodinterp = interp1d(self.wiod, self.siod)
        if not bstar:
            d = readsav('/Users/johnjohn/Dropbox (Caltech Exolab)/research/dopcode_new/dsst9407ad_rj85.dat')
            use = np.where(((d.sdstwav > self.wiod.mean()-pad) & (d.sdstwav < self.wiod.mean()+pad)))
            self.wstar = d.sdstwav[use]
            self.sstar = d.sdst[use]
            self.bstar = False
        else: 
            self.bstar = True
        initpar = info.par[0]
        self.initpar = np.append(initpar[2:14], initpar[15:19])
        self.initpar[9] += info.w0[0]
        self.initpar[0:9] = 0.0
        self.initpar[12:15] = 0.0
        self.obchunk = info.obchunk[0]
        self.initpar[15] = 1. # Continuum offset
        self.initpar = np.append(self.initpar, 0.) # Continuum slope
#        print self.initpar[15:]
        if len(self.obchunk) == 79:
            self.obchunk = np.append(self.obchunk, self.obchunk[-1])
        if len(self.obchunk) == 81:
            self.obchunk = self.obchunk[0:len(self.obchunk)-1]
        self.dstep = np.zeros(len(self.initpar))
        self.dstep[0:9] = 0.01
        self.dstep[12:15] = 0.01
        self.dstep[15] = 0.05
        self.dstep[9] = 1e-6
        self.dstep[10] = 5e-6
        self.dstep[11] = 1e-3
        pp = info.psfpix[0]
        self.ippix = pp[2:14]
        ps = info.psfsig[0]
        self.ipsig = ps[2:14]
        self.xchunk = np.arange(80)
        self.xover = np.arange(-10,90,0.25)
        self.xip = np.arange(-15,15,0.25)
        self.c = info.c[0]
        mod = self.model(self.initpar)
        if len(self.obchunk) != 80:
            print len(self.obchunk), len(mod)
        self.initpar[15] = np.median(self.obchunk/mod)
       
    def __call__(self, par):
        model = self.model(par)
        lnprob = (self.obchunk-model)**2
        return -lnprob[3:len(lnprob)-1].sum()
    
    def model(self, par):
        wobs = par[9] + par[11]*self.xchunk
        wover = par[9] + par[11]*self.xover
        z = par[10]
        try:
            if self.bstar:
                starover = wover*0 + 1. # No stellar spectrum if B star
            else:
                starinterp = interp1d(self.wstar*(1+z), self.sstar)
                starover = starinterp(wover)
            iodover = self.iodinterp(wover)
        except ValueError:
            print "star:   ", self.wstar.min(), self.wstar.max()
            print "iodine: ", self.wiod.min(), self.wiod.max()
            print "wover:  ", wover.min(), wover.max()
        product = starover * iodover
        ipar = np.append(par[0:9],par[12:15])
        ip = self.gpfunc(ipar)
#        print wover.min(), wobs.min(), wobs.max(), wover.max()
        cont = par[15] + par[16]*self.xchunk
#        print par[15], par[16]
        model = rebin_spec_new(wover, numconv(product, ip), wobs)*cont
        return model
        
    def lm_model(self, x, *par):
        model = self.model(par)
        return model
        
    def gpfunc(self, pars, censig=0.80):
        ip = jjgauss(self.xip, 1.0, 0.0, censig)
        for i, sig in enumerate(self.ipsig):
            gau = jjgauss(self.xip, pars[i], self.ippix[i], self.ipsig[i])
            ip += gau
        return ip / ip.sum() 

    def emcee_fitter(self, nwalkers, niter, nburn):
        p0 = self.initpar
        #s, p0 = self.lm_fitter() # initialize parameters using least-sq fit
        ndims = len(p0)
        p0arr = np.array([])
        for i, p in enumerate(p0):
            if p != 0:
                amp = self.dstep[i] * p
            else:
                amp = self.dstep[i]
            randp = p + amp*(rand.random(nwalkers) - 0.5) 
            p0arr = np.append(p0arr, randp)
        p0arr.shape = (ndims, nwalkers)
        p0arr = p0arr.transpose()
        
        sampler = emcee.EnsembleSampler(nwalkers,ndims,self)
        print "Starting burn-in with "+str(nburn)+" links"  
        pos, prob, state = sampler.run_mcmc(p0arr, nburn)
        sampler.reset()
        print "Starting main MCMC run with "+str(nwalkers)+" walkers and "+str(niter)+ " links."
        foo = sampler.run_mcmc(pos, niter, rstate0=state)
        m = np.argmax(sampler.flatlnprobability)
        bpar = sampler.flatchain[m,:]
        return sampler, bpar
        
    def lm_fitter(self):
        bestpar, cov = curve_fit(self.lm_model, self.xchunk, self.obchunk, p0=self.initpar)
        return self.model(bestpar), bestpar

def dop_driver(nwalkers, niter, nburn):
    ind = np.arange(10, dtype=int)+400
    setfig(2)
    vmed = np.zeros(len(ind))
    vorig = np.zeros(len(ind))
    size = 100
    rarray = rand.random(size=size)*nwalkers*niter
    rind = rarray.argsort()
    vchain = np.array([])
    for j, i in enumerate(ind):
        print "On chunk "+str(i)
        print
        ch = Chunk(i)
        s, bp = ch.emcee_fitter(nwalkers, niter, nburn)
        vorig[j] = ch.initpar[10]*3e8
        vmed[j] = np.median(s.flatchain[:,10]*3e8)
        vchain = np.append(vchain, s.flatchain[rind,10]*3e8)
#        v = s.flatchain[:,10]*3e8
#        if i == ind[0]: range = (v.min(), v.max())
#        plothist(v, over=True, range=range)
    return vmed, vorig, vchain
  
def dop_driver_lm():
    ind = np.arange(660)
    zlm = np.zeros(len(ind))
    dz = np.zeros(len(ind))
    for i in ind:
        ch = Chunk(i)
        s, bp = ch.lm_fitter()
        zlm[i] = bp[10]
        print i
        zorig = ch.info.par[0][12]
        dz[i] = (zlm[i] - zorig)/zlm[i]
    return zlm, dz
  
def get_info():
    d = readsav('infoha9407_rj13.2142') 
    pre = '/Users/johnjohn/Dropbox (Caltech Exolab)/research/dopcode_new/info9407/'
    for i in range(700):
        print i,
        info = d.infoarr[i]
        filename = pre+'info9407_'+str(i)
        np.save(filename, info)
    
#vmed, vorig, vchain = dop_driver(10, 500, 50)

start = timer()
zlm, dz = dop_driver_lm()
end = timer()
print end-start
ch = Chunk(20)
s, bp = ch.lm_fitter()
#s, bp = ch.emcee_fitter(40, 2500, 250)
bm = ch.model(bp)
pl.plot(ch.obchunk)
pl.plot(bm)
#pl.plot((bm-ch.obchunk+1)/ch.obchunk, 'o')
#plothist(s.flatchain[:,10]*ch.c)
pl.show()