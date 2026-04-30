import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import PchipInterpolator

class augmentor:
    def __init__(self,dataset_csv_path):
        self.path = dataset_csv_path
        data = pd.read_csv(self.path)
        self.columns = data.columns
        self.data_array = np.array(data)
        self.detectors = ['A','B','C','D','F']
        self.indices = {}
        self.amplitudes = {}
        for detector in self.detectors:
            self.indices[detector] = [np.where(data.columns == c)[0][0] for c in data.columns if f"P{detector}" in c and "amp" not in c]
            self.amplitudes[detector] = np.where(data.columns == f"P{detector}amp")[0][0]
        self.rows = []
        for row in self.data_array:
            self.rows.append(pulses(row,self.detectors,self.indices,self.amplitudes))
            
        self.noisyData = None
        self.noiseIntensities = None
        self.noiseSNR = None
            
            
    def getPulses(self,row):
        return self.rows[row]
    
    def createNoisyData(self,timesteps,weights,intensities=[1e-9, 1e-8, 5e-8],max_tries=100):
        self.noisyData = None
        out_data = []
        agg_snr = 0.0

        for i in range(len(self.rows)):
            aRiseOffset = 0.0
            out_row = np.zeros(self.data_array.shape[1])
            out_row[0] = self.data_array[i][0]
            out_row[1] = self.data_array[i][1]
            snr = 0.0

            for j in self.detectors:
                signal = self.rows[i].getSignal(j)
                
                kt = self.data_array[i][self.indices[j]].copy()
                amplitude = self.data_array[i][self.amplitudes[j]]
                
                success = False
                
                for _ in range(max_tries):
                    try:
                        signal.add_noise(timesteps, weights, intensities)
                        snr += signal.getSNR()
                        kt, amplitude = signal.getKeyTimes()
                        success =True
                        break
                    except IndexError:
                        continue
                        
                if j == 'A':
                    aRiseOffset = kt[1]
                
                out_row[self.indices[j]] = kt - aRiseOffset
                out_row[self.amplitudes[j]] = np.array(amplitude)
                
            out_data.append(out_row.tolist())
            agg_snr += snr/len(self.detectors)
            self.noiseSNR = snr_db = 10 * np.log10(agg_snr/len(self.rows))

        print(f"Average dB SNR: {self.noiseSNR} dB")
        print(f"Noise Intensities:\n\tShot: {intensities[0]}\n\tPink: {intensities[1]}\n\tBrownian: {intensities[2]}")
        self.noiseIntensities = intensities
        self.noisyData = pd.DataFrame(out_data, columns=self.columns)
    
    def saveNoisyData(self,outdir=None):
        if outdir is None:
            outdir = ""
        self.noisyData.to_csv(os.path.join(outdir,f"augmented_shot{self.noiseIntensities[0]}_pink{self.noiseIntensities[1]}_brown{self.noiseIntensities[2]}_snr{np.round(self.noiseSNR,2)}.csv"),index=False)
        
    def toReduced(self,outdir=None,noisy=False,returnDF=False,save=True):
        if outdir is None:
            outdir = ""
        if noisy:
            data = self.noisyData
        else:
            data = pd.DataFrame(self.data_array,columns=self.columns)
            
        new_rows = []
        for i,row in data.iterrows():
            new = [row['Row']]
            for d in self.detectors:
                new.append(row[f'P{d}r20'])
            for d in self.detectors:
                new.append(row[f'P{d}r50']-row[f'P{d}r20'])
            for d in self.detectors:
                new.append(row[f'P{d}f20']-row[f'P{d}r40'])
            for d in self.detectors:
                new.append(row[f'P{d}f80']-row[f'P{d}r80'])
            new.append(row['y'])
            del new[1]
            new_rows.append(new)
            
        colNames = []
        colNames.append('Row')
        for d in self.detectors:
            colNames.append(f'P{d}start')
        for d in self.detectors:
            colNames.append(f'P{d}rise')
        for d in self.detectors:
            colNames.append(f'P{d}fall')
        for d in self.detectors:
            colNames.append(f'P{d}width')
        colNames.append('y')
        del colNames[1]
        output = pd.DataFrame(new_rows,columns=colNames)
        output['Row'] = output['Row'].astype(int)
        output.sort_values(by=['Row','y'],inplace=True)
        output.reset_index(drop=True,inplace=True)
        if save:
            if noisy:
                output.to_csv(os.path.join(outdir,f"reduced-augmented_shot{self.noiseIntensities[0]}_pink{self.noiseIntensities[1]}_brown{self.noiseIntensities[2]}_snr{np.round(self.noiseSNR,2)}.csv"),index=False)
            else:
                output.to_csv(os.path.join(outdir,f"reduced-original.csv"),index=False)
        if returnDF:
            return output
        
    
    def plotSignals(self,row):
        pulse = self.getPulses(row)
        fig,axs = plt.subplots(1,4,figsize=(16,4))
        fig2,axs2 = plt.subplots(1,5,figsize=(20,4))
        
        data_desc = ["Original Signal","Interpolated Signal","Noisy Interpolated","Noisy Output"]
        for i,d in enumerate(self.detectors):
            signal = pulse.getSignal(d)
            times,voltages = signal.sampleFromSpline(1000)
            kt,a = signal.getKeyTimes()
            vol = a*np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0,0.95,0.9,0.8,0.4,0.2])
            kt = np.append(kt,kt[-1]*3)
            vol = np.append(vol,0.01*a)
            plot_times = [signal.times,times,signal.noisyTimes,kt]
            plot_voltages = [signal.voltages,voltages,signal.noisyVoltages,vol]
            ct = 0
            for ax,t,v in zip(axs,plot_times,plot_voltages):
                ax.plot(t,v,label=f"Pulse {d}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Voltage")
                ax.set_title(data_desc[ct])
                ax.legend()
                axs2[i].plot(t,v,label=data_desc[ct])
                ct += 1
            axs2[i].set_xlabel("Time")
            axs2[i].set_ylabel("Voltage")
            axs2[i].set_title(f"Pulse {d}")
            axs2[i].legend()
        
            
class pulses:
    def __init__(self,data_row,detectors,indices,amplitudes):
        self.pulses  = {}
        for detector in detectors:
            times = data_row[indices[detector]]
            amplitude = data_row[amplitudes[detector]]
            self.pulses[detector] = signal(times,amplitude)
            
    def getSignal(self,pulse):
        return self.pulses[pulse]
            
class signal:
    def __init__(self,times,amplitude):
        voltages = amplitude*np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0,0.95,0.9,0.8,0.4,0.2])
        voltages = np.append(voltages,0.01*amplitude)
        self.voltages = voltages
        times = np.append(times,times[-1]*3)
        self.times = times
        self.spline = PchipInterpolator(self.times,self.voltages)
        self.noisyTimes = None
        self.noisyVoltages = None
        
    def getOriginalSignal(self):
        return self.times,self.voltages
        
    def sampleFromSpline(self,timesteps):
        times = np.linspace(self.times[0],self.times[-1],timesteps)
        voltages = self.spline(times)
        return times,voltages
    
    def add_noise(self,timesteps,weights,intensities=[1e-9,1e-8,5e-8]):
        times,voltages = self.sampleFromSpline(timesteps)
        new_voltages = np.zeros(len(voltages))
        new_voltages += weights[0]*noise.add_shot_noise(voltages,intensities[0])
        new_voltages += weights[1]*noise.add_pink_noise(voltages,intensities[1])
        new_voltages += weights[2]*noise.add_brown_noise(voltages,intensities[2])
        self.noisyTimes = times
        self.noisyVoltages = new_voltages
        
    def getSNR(self):
        t,signal = self.sampleFromSpline(len(self.noisyVoltages))
        noise = self.noisyVoltages - signal
        p_signal = np.sqrt(np.mean(signal**2))
        p_noise = np.sqrt(np.mean(noise**2))
        return (p_signal / p_noise)**2
    
    def getKeyTimes(self):
        amplitude = np.max(self.noisyVoltages)
        max_index = self.noisyVoltages.argmax()
        normalized = self.noisyVoltages/amplitude

        riseV = normalized[0:max_index+1].copy()
        fallV = normalized[max_index:-1].copy()
        riseT = self.noisyTimes[0:max_index+1].copy()
        fallT = self.noisyTimes[max_index:-1].copy()

        riseVals = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
        fallVals = np.array([0.95,0.9,0.8,0.4,0.2])

        riseTimes = []
        
        if riseV[0] < 0.1:
            riseVals = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
        else:
            riseTimes.append(riseT[0])

        for v in riseVals:
            below = np.where(riseV < v)[0][-1]
            above = np.where(riseV > v)[0][0]
            riseTimes.append(riseT[below] + (v-riseV[below])*(riseT[above]-riseT[below])/(riseV[above]-riseV[below]))
            riseV = riseV[below::]
            riseT = riseT[below::]
        riseTimes.append(riseT[-1])

        fallTimes = []

        for v in fallVals:
            below = np.where(fallV < v)[0][0]
            above = np.where(fallV > v)[0][-1]
            fallTimes.append(fallT[above] + (v-fallV[above])*(fallT[below]-fallT[above])/(fallV[below]-fallV[above]))
            fallV = fallV[above::]
            fallT = fallT[above::]

        return  np.array(riseTimes+fallTimes),amplitude
    
class noise:
    @staticmethod
    def add_shot_noise(signal,events=1e-9):
        return np.random.poisson(signal/events)*events
    
    @staticmethod
    def add_pink_noise(signal, intensity=1e-8):
        n = len(signal)
        white_noise = np.random.randn(n)

        white_noise_fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n)

        filter_mag = np.zeros_like(freqs)
        filter_mag[1:] = 1.0 / np.sqrt(freqs[1:])
        pink_noise_fft = white_noise_fft * filter_mag

        pink_noise = np.fft.irfft(pink_noise_fft, n)
        pink_noise /= np.std(pink_noise)

        return signal + (pink_noise * intensity)
    
    @staticmethod
    def add_brown_noise(signal, intensity=5e-8):
        white_noise = np.random.normal(0, 1, len(signal))

        brown_noise = np.cumsum(white_noise)
        brown_noise = brown_noise / np.max(np.abs(brown_noise))

        return signal + (intensity * brown_noise)
    
    
if __name__ == "__main__":
    import sys
    np.random.seed(42)
    
    src_csv = "../data/data_rearranged_copy.csv"
    out_dir = "../data/augmented/"
    os.makedirs(out_dir, exist_ok=True)
    
    profiles = [
        ("brown_heavy", [0.2, 0.2, 0.6], [5e-9, 2e-8, 8e-8]),
        ("shot_heavy",  [0.8, 0.1, 0.1], [1e-9, 7e-8, 3e-7]),
    ]
    
    aug = augmentor(src_csv)
    for name, weights, intensities in profiles:
        print(f"\n=== Profile: {name} ===")
        aug.createNoisyData(timesteps=150, weights=weights, intensities=intensities)
        out_path = os.path.join(out_dir, f"augmented_{name}.csv")
        aug.noisyData.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")