import numpy as np
import pylab as plt
import math
import warnings
warnings.filterwarnings("ignore")
import kplr
client = kplr.API()
from scipy.stats import chi2

if __name__ == '__main__':
	list1=np.linspace(100,3000,10)
	for l in list1:
		#koi = client.planet('Kepler-14b')
		# generate some non-uniformly spaced times.
		ts_orig = np.load('Kepler-14_b_time.npy')#np.linspace(0,1000.,N) + 0.1*np.random.uniform(-1,1,N)
		dt=ts_orig[3]-ts_orig[2]
		print("dt=",dt)
		
		# Now create a periodic light curve, period P, chosen
		# to be a divisor of the 1000 day length of the data
		
		P = 6.79012361
		T0 = 138.088
		print("period= ", P)
		print("T0= ", T0)
		
		ts1=(T0+(P*0))
		ts2=(T0+(P*10))
		print(ts1,ts2)

		condition= np.logical_and(ts_orig>=ts1,ts_orig<=ts2) #making the window Johan suggests
		condition1= np.logical_and(ts_orig>=(T0+(P*2)),ts_orig<=(T0+(P*3)))


		#################################
		def find_nearest(array,value):
		    idx = (np.abs(array-value)).argmin()
		    return array[idx]
		    
		
		
		ts_new=ts_orig[condition]

		ts1_ind=np.where(ts_orig==ts_new[0])
		ts2_ind=np.where(ts_orig==ts_new[-1])
		ts1_ind=ts1_ind[0][0]#-1
		ts2_ind=ts2_ind[0][0]#+2
		print(ts1_ind)
		print(ts2_ind)
		#ts=ts_orig[condition]
		
		N0 = ((np.nanmax(ts_new)-np.nanmin(ts_new))/P)
		N0_int = int(N0) + 1
		print("number of transits: ", N0_int)

		#I = np.zeros_like(ts)
		#ok = np.abs(phs) < 0.05
		I_orig=np.load('Kepler-14_b_flux.npy')#[ok] = (np.cos(np.pi*phs[ok]/0.05)+1)/2.
		#I=I_orig[2238:6227]#2237:6224
		I=I_orig[condition]#[ts1_ind:ts2_ind]#2237:6224
		
		
		# N = number of data points
		N = len(I)
		
		print(N)
		ts=np.linspace(0,N,N)
		where_are_NaNs = np.isnan(I)
		I[where_are_NaNs] = 1.0
		
		#maskI=np.isfinite(I)
		#I=I[maskI]
		#ts=ts[maskI]
		
		#ts[np.where(np.isnan(I)==True)]=np.nan#1.0
		#I[np.where(np.isnan(I)==True)]=np.nan#1.0
		
		
		
		phs=np.mod(ts-ts[0]+(P/dt)/2, (P/dt))/(P/dt)

		print(phs)

		
		plt.figure(1)
		#plt.plot(ts/dt,I)
		plt.plot(phs,I)
		plt.show()

		#ts=ts[np.where(np.isnan(I)==False)]
		#I=I[np.where(np.isnan(I)==False)]
		#I=I[np.where(np.isnan(ts)==False)]
		#ts=ts[np.where(np.isnan(ts)==False)]
		
		N = len(I)
		
		
		# subtract mean level
		I -= np.nanmean(I)#.mean()
		
			
		# beat into column vector form. Will use these later.
		I_new = np.matrix(I.reshape(N,1))


		# choose the data length which is the data width in the paper 
		Tw = len(I)#P*12#0.254263157895#*P/2
			

		#phis = 2.*np.pi*(ts - ts[0])/Tw #with noise
		phis = 2.*np.pi*np.mod(ts/(P/dt),1) # without noise

		# beat into colummn vector form
		phis = phis.reshape(N,1)

		
		NCPT = len(I_new)

		
		#nharms = np.linspace(1,NCPT/2,NCPT//2) #with noise
		nharms = np.linspace(1,NCPT/N0_int,NCPT//N0_int) #without noise(24 because 2*12 full transits in data)

		
		pmat = nharms*phis
		print(np.size(pmat)/10e6)
		
		# cosine and sine versions

		C = np.cos(pmat)
		S = np.sin(pmat)
			

		Tmat = np.matrix(np.hstack((C,S)))
		
		
		R = (Tmat.T*Tmat).I * (Tmat.T*I_new)
		
		
		fit = np.array(Tmat*R).reshape(N)
		dat = np.array(I_new).reshape(N)

		# Now plot phase-folded
		isort = np.argsort(phs)
		phs = phs[isort]
		fit = fit[isort]
		dat = dat[isort]
	####################################################
		plt.figure(2)
		plt.plot(phs,fit,'r')
		plt.plot(phs,dat,',b')
		#plt.show()

		#print(np.shape(R))
		#R = np.array(R).reshape(NCPT,1)
		plt.figure(3)
		#plt.step(nharms/N0_int,R[:NCPT/2],'.b',where='mid',label="Cosine")
		#plt.step(nharms/N0_int,R[NCPT/2:],'.r',where='mid',label="Sine")
		plt.step(nharms,R[:NCPT/N0_int],'.b',label="Cosine")
		plt.step(nharms,R[NCPT/N0_int:],'.r',label="Sine")
		plt.legend()
		#plt.xlim(0,30)
		plt.ylim(-1.5e-4,0.5e-4)
		#plt.step(nharms,R[:NCPT],'.b',where='mid')
		#plt.step(nharms,R[NCPT:],'.r',where='mid')

		phs_new=np.mod(ts-(P/dt)/2, (P/dt))/(P/dt)
		Inew=np.array(Tmat*R)
		print(len(Inew))
		plt.figure(4)
		plt.plot(phs_new,Inew)
		plt.show()

