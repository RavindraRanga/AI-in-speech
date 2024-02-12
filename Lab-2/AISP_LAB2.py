#!/usr/bin/env python
# coding: utf-8

# # A RANGA RAVINDRA
# # BL.EN.U4AIE21002

# # (A1) Find the first derivative of your speech signal with finite difference method. Listen to the first derivative signal and the original speech signal. 

# In[10]:


import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd

y, sr = librosa.load('AI in speech processing.wav')
derivative = np.diff(y)
derivative = np.append(derivative, 0)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, color = 'red')
plt.title('Original Signal')

plt.subplot(2, 1, 2)
librosa.display.waveshow(derivative, sr=sr, color = 'red')
plt.title('First Derivative')

plt.tight_layout()
plt.show()

print("Original Signal:")
display(Audio(data=y, rate=sr))


print("First Derivative Signal:")
Audio(data=derivative, rate=sr)


# # (A2) Detect the points of zero crossing in the first derivative signal. Compare the average length between two consecutive zero crossings for speech and silence regions. Observe the pattern.

# In[11]:


zero_crossing = np.where(np.diff(np.sign(derivative_1)))[0]
diff = np.diff(zero_crossing)
threshold = 1000
speech_regions = diff[diff > threshold]
silence_regions = diff[diff <= threshold]

avg_length_speech = np.mean(speech_regions)
avg_length_silence = np.mean(silence_regions)

print("Average length between consecutive zero crossings in speech regions:", avg_length_speech)
print("Average length between consecutive zero crossings in silence regions:", avg_length_silence)

plt.figure(figsize=(10, 5))
plt.plot(diff, label='All regions',color = 'purple')
plt.plot(np.arange(len(speech_regions)), speech_regions, 'ro', label='Speech regions',color = 'red')
plt.plot(np.arange(len(speech_regions), len(speech_regions) + len(silence_regions)), silence_regions, 'bo', label='Silence regions',color = 'blue')
plt.title('Pattern of Zero Crossings')
plt.xlabel('Zero Crossing Difference')
plt.ylabel('Difference between Consecutive Zero Crossings')
plt.legend()
plt.show()

print("Pattern of Zero Crossings:")
print("All regions:", diff)
print("Speech regions:", speech_regions)
print("Silence regions:", silence_regions)


# # (A3) Speak 5 of your favorite words. Observe the length of the speech signals. Compare the lengths of your spoken words with those of your project team-mate.

# In[14]:


word_files_mine = ['computer.wav', 'Science.wav', 'Engineering.wav', 'Artificial.wav', 'Intelligence.wav']
word_files_team_mate = ['computer_m.wav', 'science_m.wav', 'engineering_m.wav', 'artificial_m.wav', 'intelligence_m.wav']
words = ['computer', 'Science', 'Engineering', 'Artificial', 'Intelligence']
word_lengths_mine = []
word_lengths_teammate = []

for word_file in word_files_mine:
    signal, sr = librosa.load(word_file, sr=None)
    length_seconds = len(signal) / sr
    word_lengths_mine.append(length_seconds)

for word_file in word_files_team_mate:
    signal, sr = librosa.load(word_file, sr=None)
    length_seconds = len(signal) / sr
    word_lengths_teammate.append(length_seconds)

print("Lengths of the spoken words MINE:", word_lengths_mine)
print("Lengths of the spoken words TeamMate:", word_lengths_teammate)

bar_width = 0.35
index = np.arange(len(words))
plt.figure(figsize=(12, 6))
plt.bar(index - bar_width/2, word_lengths_mine, bar_width, label='My Words', color='red')
plt.bar(index + bar_width/2, word_lengths_teammate, bar_width, label="Teammate's Words", color='skyblue')
plt.xlabel('Words')
plt.ylabel('Length (seconds)')
plt.title('Comparison of Spoken Words Length')
plt.xticks(index, words)
plt.legend()

plt.show()


# # A4. Select a sentence which can be used for making a statement or asking a question. Ex: “You are going to college on Sunday(./?)”. Record two signals – one with making the statement while other with asking question. Study the two signals and compare them.

# In[12]:


y, rs = librosa.load('statementQ.wav')
question, rs = librosa.load('statementQ.wav')
plt.figure(figsize=(10, 5))
librosa.display.waveshow(question, sr=rs,color='red')
plt.title('Question')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
print('Question Audio')
ipd.Audio(y, rate=rs)


# In[13]:


y, rs = librosa.load('statementA.wav')
statement, rs = librosa.load('statementA.wav')
plt.figure(figsize=(10, 5))
librosa.display.waveshow(statement, sr=rs,color='red')
plt.title('STATEMENT')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
print('Statement Audio')
ipd.Audio(y, rate=rs)


# In[ ]:


import IPython.display as ipd
AI in speech processing.wav

