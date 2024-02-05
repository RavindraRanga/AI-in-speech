#!/usr/bin/env python
# coding: utf-8

# # A RANGA RAVINDRA
# # BL.EN.U4AIE21002

# In[3]:


'''#Q.(A1): Load the recorded speech file into your python workspace. Once loaded, plot the graph for the speech signal.
            You may use the below code from librosa asa reference.'''

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
y, sr = librosa.load('AI in speech processing.wav')
librosa.display.waveshow(y)


# In[7]:


'''#Q.(A2): Observe the length & magnitude range of the signalfrom theplot. Observing the plot, 
            relate tothe spoken words and silence between the words.'''

print('length of signal:',len(y), 'samples')
print('magnitude range:',np.min(y), 'to',np.max(y))
print('sampling rate:', sr,"Hz")


# In[15]:


'''#Q.(A3): Take a small segment of the signal and play it. '''

import IPython.display as ipd

start = int(1.5 * sr)
end = int(3 * sr)
segment = y[start:end]
librosa.display.waveshow(segment, color = 'blue')
ipd.Audio(segment, rate=sr)


# In[18]:


'''# Q.(A4): Play around with your recorded speech signal for various segments. Understand the nature of the signal. 
             Also observe with abruptly segmented speech, how the perception of that speech is affected.'''

segment_1_start = int(0 * sr)
segment_1_end = int(2 * sr)
segment_2_start = int(2.5 * sr)
segment_2_end = int(3.2 * sr)
segment_3_start = int(1.7 * sr)
segment_3_end = int(3.5 * sr)
segment_1 = y[segment_1_start:segment_1_end]
segment_2 = y[segment_2_start:segment_2_end]
segment_3 = y[segment_3_start:segment_3_end]
print("Playing Segment 1:")

plt.figure(figsize=(10, 5))
librosa.display.waveshow(segment_1, color = 'blue')
plt.title('Segment-1')
plt.show()
ipd.Audio(segment_1, rate=sr)


# In[19]:


print("Playing Segment 2:")

plt.figure(figsize=(10, 5))
librosa.display.waveshow(segment_2, color = 'blue')
plt.title('Segment-2')
plt.show()
ipd.Audio(segment_2, rate=sr)


# In[20]:


print("Playing Segment 3:")

plt.figure(figsize=(10, 5))
librosa.display.waveshow(segment_3, color = 'blue')
plt.title('Segment-3')
plt.show()
ipd.Audio(segment_3, rate=sr)


# In[ ]:




