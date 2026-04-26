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
            out_row[0] = self.data_array[i][0] # Index 0: Row
            out_row[26] = self.data_array[i][26] # Index 26: energy
            out_row[27] = self.data_array[i][27] # Index 27: y
            
            snr = 0.0

            for j in self.detectors:
                signal = self.rows[i].getSignal(j)
                tries = 0
                while tries < max_tries:
                    try:
                        signal.add_noise(timesteps, weights, intensities)
                        snr += signal.getSNR()
                        
                        kt, amplitude = signal.getKeyTimes()
                        break
                    except IndexError as e:
                        tries += 1
                        
                if j == 'A':
                    aRiseOffset = kt[0]

                keyTimes = kt.copy()
                amplitude_vals = np.array(amplitude)
                
                out_row[self.indices[j]] = keyTimes - aRiseOffset
                out_row[self.amplitudes[j]] = amplitude_vals
                
            out_data.append(out_row.tolist())
            agg_snr += snr/len(self.detectors)
            self.noiseSNR = snr_db = 10 * np.log10(agg_snr/len(self.rows))

        print(f"Average dB SNR: {self.noiseSNR} dB")
        print(f"Noise Intensities:\n\tShot: {intensities[0]}\n\tPink: {intensities[1]}\n\tBrownian: {intensities[2]}")
        self.noiseIntensities = intensities
        self.noisyData = pd.DataFrame(out_data, columns=self.columns)
        
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
    def __init__(self, times, amplitude):
        # The CSV passes features as an array of 4 values: [Start, Rise, Fall, Width]
        t_start = times[0]
        t_rise_dur = times[1]
        t_fall_dur = times[2]
        t_width_dur = times[3]
        
        # We must reconstruct an absolute, strictly increasing timeline for the Interpolator
        
        # Point 1: Pulse Begins (10% height)
        t1 = t_start
        v1 = 0.1
        
        # Point 2: Peak (100% height). Occurs after the Rise duration.
        t2 = t1 + t_rise_dur
        v2 = 1.0
        
        # Point 3: Half-Fall (50% height). Occurs approximately 'Width' time after start.
        t3 = t1 + t_width_dur
        v3 = 0.5
        
        # Point 4: End of Fall (10% height). Occurs after the Fall duration.
        t4 = t3 + t_fall_dur
        v4 = 0.1
        
        # Point 5: Pulse Tail (1% height). Extends out to simulate decay.
        t5 = t4 + abs(t4 * 0.5) + 1e-3 
        v5 = 0.01
        
        # Build the 5-point curve arrays
        self.times = np.array([t1, t2, t3, t4, t5])
        self.voltages = amplitude * np.array([v1, v2, v3, v4, v5])
        
        # FAILSAFE: PchipInterpolator crashes if X-values go backwards or are identical.
        # If a sensor was "dead" (0 rise, 0 fall), we force a tiny microsecond step forward.
        for idx in range(1, len(self.times)):
            if self.times[idx] <= self.times[idx-1]:
                self.times[idx] = self.times[idx-1] + 1e-4
                
        # Now len(times) == 5 and len(voltages) == 5. The interpolator will succeed.
        self.spline = PchipInterpolator(self.times, self.voltages)
        
        self.noisyTimes = None
        self.noisyVoltages = None
        
    def getOriginalSignal(self):
        return self.times, self.voltages
        
    def sampleFromSpline(self, timesteps):
        times = np.linspace(self.times[0], self.times[-1], timesteps)
        voltages = self.spline(times)
        return times, voltages
    
    def add_noise(self, timesteps, weights, intensities=[1e-9, 1e-8, 5e-8]):
        times, voltages = self.sampleFromSpline(timesteps)
        new_voltages = np.zeros(len(voltages))
        new_voltages += weights[0] * noise.add_shot_noise(voltages, intensities[0])
        new_voltages += weights[1] * noise.add_pink_noise(voltages, intensities[1])
        new_voltages += weights[2] * noise.add_brown_noise(voltages, intensities[2])
        self.noisyTimes = times
        self.noisyVoltages = new_voltages
        
    def getSNR(self):
        if self.noisyVoltages is None:
            return 0.0
        t, signal = self.sampleFromSpline(len(self.noisyVoltages))
        noise_diff = self.noisyVoltages - signal
        p_signal = np.sqrt(np.mean(signal**2))
        p_noise = np.sqrt(np.mean(noise_diff**2))
        return (p_signal / p_noise)**2
    
    def getKeyTimes(self):
        if self.noisyVoltages is None or self.noisyTimes is None:
            raise ValueError("Noise must be added before extracting Key Times.")
        
        v = self.noisyVoltages
        t = self.noisyTimes
        amplitude = np.max(v)
        peak_idx = np.argmax(v)
        
        # Start
        start_time = t[0]
        
        # Rise Time
        v_rise = v[:peak_idx+1]
        t_rise = t[:peak_idx+1]
        
        # Failsafe for noisy data that never drops below 10% or 90%
        try:
            t10 = t_rise[np.where(v_rise >= 0.1 * amplitude)[0][0]]
            t90 = t_rise[np.where(v_rise >= 0.9 * amplitude)[0][0]]
            rise_time = t90 - t10
        except IndexError:
            rise_time = t_rise[-1] - t_rise[0]
            
        # Fall Time
        v_fall = v[peak_idx:]
        t_fall = t[peak_idx:]
        try:
            t90_f = t_fall[np.where(v_fall <= 0.9 * amplitude)[0][0]]
            t10_f = t_fall[np.where(v_fall <= 0.1 * amplitude)[0][0]]
            fall_time = t10_f - t90_f
        except IndexError:
            fall_time = t_fall[-1] - t_fall[0]
            
        # Width (FWHM)
        try:
            t50_rise = t_rise[np.where(v_rise >= 0.5 * amplitude)[0][0]]
            t50_fall = t_fall[np.where(v_fall <= 0.5 * amplitude)[0][0]]
            width = t50_fall - t50_rise
        except IndexError:
            width = 0.0
            
        return np.array([start_time, rise_time, fall_time, width]), amplitude
    
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
    # Define the dataset path (we will just use Train set)
    train_filepath = "../data/train_data.csv"
    output_file = "../data/train_augmented_data.csv"
    
    print(f"Loading original training data from {train_filepath}...")
    
    original_df = pd.read_csv(train_filepath)
    all_dataframes = [original_df]
    
    # Initialize the augmentor
    dataset = augmentor(train_filepath)
    
    # define 6 physical noise profiles
    # (Short, Pink, Brownian)
    # Combining these 6 with the 1 original would essentially give 7x expansion
    profiles = [
        ([0.2, 0.2, 0.6], [5e-9, 2e-8, 8e-8]),
        ([0.8, 0.1, 0.1], [1e-9, 7e-8, 3e-7]),
        ([0.1, 0.8, 0.1], [1e-9, 5e-8, 1e-8]),
        ([0.33, 0.33, 0.33], [2e-9, 2e-8, 5e-8]),
        ([0.4, 0.3, 0.3], [5e-10, 5e-9, 1e-8]),
        ([0.5, 0.2, 0.3], [1e-8, 1e-7, 5e-7])
    ]
    
    # Now generate the augmented arrays
    for i, (weights, intensities) in enumerate(profiles):
        print(f"\nGenerating noisy dataset {i+1}/6...")
        
        # Used 150 like Bobby had in his jupyter
        dataset.createNoisyData(150, weights, intensities=intensities)
        
        if dataset.noisyData is None:
            raise ValueError("No Noise created!")
        
        # append to the all_dataframes
        all_dataframes.append(dataset.noisyData)
        
    # Concat the all_dataframes
    augmented_df = pd.concat(all_dataframes, ignore_index = True)
    
    # save this as the new training dataset
    augmented_df.to_csv(output_file, index = False)
    print(f"\nSUCCESS! Saved the expanded dataset to {output_file}")
    print(f"Original rows: {len(original_df)} -> New rows: {len(augmented_df)}")