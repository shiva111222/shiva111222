'''
Title: Timing Control and Frame Rate Measurement Script

Description:
This Python script is designed to measure the execution time of a set of calculations over a specific number of iterations (counter). 
It implements timing control to  achieve a desired frame rate (FRAMERATE). The code calculates the time between frames (FRAME) based on the desired frame rate.
The script includes a loop that performs the calculations, and within that loop, there's another nested loop to run the calculations repeatedly. 
The calculated frame execution times are used to adjust the sleep time for achieving the desired frame rate.
After all iterations are complete, the script calculates and displays statistics related to the achieved frame rate, 
including the average frame time and total execution time. It also plots the measured frame times using matplotlib for visualization.
'''

#Code:
import rospy
import time
from numpy import mean
import matplotlib.pyplot as plt
import math

FRAMERATE = 30
FRAME = 1.0 / FRAMERATE

myTimer = 0.0
counter = 2000
runtime = counter * FRAME
TIME_CORRECTION = 0.0
dataStore = []

print("START COUNTING: FRAME TIME", FRAME, "RUN TIME:", runtime)

myTime = newTime = time.time()
masterTime = myTime

for ii in range(counter):
    # Perform calculations here
    
    for jj in range(1000):
        x = 100
        y = 23 + ii
        z = math.cos(x)
        z1 = math.sin(y)

    newTime = time.time()
    myTimer = newTime - myTime
    timeError = FRAME - myTimer

    sleepTime = timeError + (TIME_CORRECTION / 1.5)
    sleepTime = max(sleepTime, 0.0)
    time.sleep(sleepTime)

    time2 = time.time()
    measuredFrameTime = time2 - myTime
    TIME_CORRECTION = FRAME - measuredFrameTime
    dataStore.append(measuredFrameTime * 1000)
    myTime = time.time()

endTime = time.time() - masterTime
avgTime = endTime / counter

print("FINISHED COUNTING")
print("REQUESTED FRAME TIME:", FRAME, "AVG FRAME TIME:", avgTime)
print("REQUESTED TOTAL TIME:", runtime, "ACTUAL TOTAL TIME:", endTime)
print("AVERAGE ERROR:", FRAME - avgTime, "TOTAL ERROR:", runtime - endTime)
print("AVERAGE SLEEP TIME:", mean(dataStore), "AVERAGE RUNTIME:", (FRAME * 1000) - mean(dataStore))

plt.plot(dataStore)
plt.show()
