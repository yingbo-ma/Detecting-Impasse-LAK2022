# Python codes for extracting acoustic-prosodic features
import audiofile
import opensmile

signal, sampling_rate = audiofile.read("absolute_path_of_audio_file") # read an audio file

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02, # select feature set
    feature_level=opensmile.FeatureLevel.Functionals
)

X = smile.process_signal(signal, sampling_rate).values.tolist()

for index in range(len(X)):
    # details for the feature list: https://audeering.github.io/opensmile-python/usage.html

    # 11 loudness-related features
    loudness_sma3_amean = X[index][10]
    loudness_sma3_stddevNorm = X[index][11]
    loudness_sma3_percentile20 = X[index][12]
    loudness_sma3_percentile50 = X[index][13]
    loudness_sma3_percentile80 = X[index][14]
    loudness_sma3_pctlrange0 = X[index][15]
    loudness_sma3_meanRisingSlope = X[index][16]
    loudness_sma3_stddevRisingSlope = X[index][17]
    loudness_sma3_meanFallingSlope = X[index][18]
    loudness_sma3_stddevFallingSlope = X[index][19]
    loudnessPeaksPerSec = X[index][-7]

    # 10 pitch-related features
    F0semitoneFrom275Hz_sma3nz_amean = X[index][0]
    F0semitoneFrom275Hz_sma3nz_stddevNorm = X[index][1]
    F0semitoneFrom275Hz_sma3nz_percentile200 = X[index][2]
    F0semitoneFrom275Hz_sma3nz_percentile500 = X[index][3]
    F0semitoneFrom275Hz_sma3nz_percentile800 = X[index][4]
    F0semitoneFrom275Hz_sma3nz_pctlrange02 = X[index][5]
    F0semitoneFrom275Hz_sma3nz_meanRisingSlope = X[index][6]
    F0semitoneFrom275Hz_sma3nz_stddevRisingSlope = X[index][7]
    F0semitoneFrom275Hz_sma3nz_meanFallingSlope = X[index][8]
    F0semitoneFrom275Hz_sma3nz_stddevFallingSlope = X[index][9]

    # 2 shimmer-related features
    shimmerLocaldB_sma3nz_amean = X[index][32]
    shimmerLocaldB_sma3nz_stddevNorm = X[index][33]

    # 2 jitter-related features
    jitterLocal_sma3nz_amean  = X[index][30]
    jitterLocal_sma3nz_stddevNorm = X[index][31]

    # 16 MFCCs-related features
    mfcc1_sma3_amean = X[index][22]
    mfcc1_sma3_stddevNorm = X[index][23]
    mfcc2_sma3_amean = X[index][24]
    mfcc2_sma3_stddevNorm = X[index][25]
    mfcc3_sma3_amean = X[index][26]
    mfcc3_sma3_stddevNorm = X[index][27]
    mfcc4_sma3_amean = X[index][28]
    mfcc4_sma3_stddevNorm = X[index][29]
    mfcc1V_sma3nz_amean = X[index][-20]
    mfcc1V_sma3nz_stddevNorm = X[index][-19]
    mfcc2V_sma3nz_amean = X[index][-18]
    mfcc2V_sma3nz_stddevNorm = X[index][-17]
    mfcc3V_sma3nz_amean = X[index][-16]
    mfcc3V_sma3nz_stddevNorm = X[index][-15]
    mfcc4V_sma3nz_amean = X[index][-14]
    mfcc4V_sma3nz_stddevNorm = X[index][-13]

    # 5 spectralFlux-related features
    spectralFlux_sma3_amean = X[index][20]
    spectralFlux_sma3_stddevNorm = X[index][21]
    spectralFluxV_sma3nz_amean = X[index][-22]
    spectralFluxV_sma3nz_stddevNorm = X[index][-21]
    spectralFluxUV_sma3nz_amean = X[index][-8]
