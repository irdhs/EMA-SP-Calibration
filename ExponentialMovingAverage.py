import matplotlib.pyplot as plt
import pandas as pd
from math import atan2, degrees


def pixel2degree(px): #tool to convert the pixel error to visual degree error
    h = 23  # Monitor height in cm
    d = 60  # Distance between monitor and participant in cm
    r = 768  # Vertical resolution of the monitor
    deg_per_px = degrees(atan2(.5 * h, d)) / (.5 * r) #parameters to convert the error
    size_in_deg = px * deg_per_px #size in degree = pixel error * degree error per pixel
    return size_in_deg #convert pixel error to visual degree error


df = pd.read_csv("hasil/DataAsli.csv", sep=",", decimal=".", names=['Time', 'X_Gaze', 'Y_Gaze', 'X_Obj', 'Y_Obj'])
df.head() #read the data rawa on csv file

df['X_Gaze'] = pixel2degree(df['X_Gaze']) #convert X Gaze pixel error to visual degree error
df['Y_Gaze'] = pixel2degree(df['Y_Gaze']) #convert Y Gaze pixel error to visual degree error
df['X_Obj'] = pixel2degree(df['X_Obj']) #convert X Object pixel error to visual degree error
df['Y_Obj'] = pixel2degree(df['Y_Obj']) #convert Y Object pixel error to visual degree error
window = 20 #filtering level of EMA (period)
df['EMA_X_Gaze'] = df['X_Gaze'].ewm(span=window).mean().shift(int(window / -2)) #shifting window on X Gaze
df['EMA_Y_Gaze'] = df['Y_Gaze'].ewm(span=window).mean().shift(int(window / -2)) #shifting window on X Gaze
df_filtered = df[['Time', 'EMA_X_Gaze', 'EMA_Y_Gaze', 'X_Obj', 'Y_Obj']].copy() #processing the EMA
df_filtered.drop(df.tail(int(window/2)).index,inplace=True) #first initial predictor of EMA
df_filtered.drop_duplicates(subset='Time', keep="last",inplace=True) #get the result
df_filtered.to_csv(r'hasil/Filter_EMA_result.csv', index=False, header=False) #save the result to csv file

df.plot(kind='line', x='Time', y=['X_Gaze', 'EMA_X_Gaze'], color=['red', 'blue']) #plotting EMA result on graphic(X Gaze)
df.plot(kind='line', x='Time', y=['Y_Gaze', 'EMA_Y_Gaze'], color=['red', 'blue']) #plotting EMA result on graphic(Y Gaze)
 
plt.show()

exec(open(r'nslrHMMclassification.py').read())
