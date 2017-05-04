#parameters
inputSize = 300.0
classNum = 80
bBoxLossAlpha = 1.0
sMin = 0.2
sMax = 0.95
boxRatios = [1.0, 1.0, 2.0, 3.0, 0.5, 1.0 / 3.0]
conv4_3Ratios = [1.0, 0.5, 2.0]
conv4_3Scale = 0.07

GpuMemory = 0.8

outShapes = None
defaults = None

confidence = 0.5
nmsIOU = 0.5