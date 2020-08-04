# Denoiser
**An AI model to remove the specific noise from the noisy input audio using the essentials of Deep Learning.**

![img](https://steamuserimages-a.akamaihd.net/ugc/155775956858409027/07547CE6405B1FE32EDECEF2D5635C9871E33F04/)

### Idea
A deep learning model is used to take input audio and detect the type of noises present in the audio. Then, a 'noise reducer' is used to remove the similar kind of audio from the input file and creats a noise free clean audio file.

### Dataset
Dataset used is: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)

Dataset Size: 6GB

Contains predefined 10-fold most common noises, stated below:

    air_conditioner
    car_horn
    children_playing
    dog_bark
    drilling
    engine_idling
    gun_shot
    jackhammer
    siren
    street_music

### Folders
    model : Saved model
    noise : contains 10 type of noise samples
    results : contains the resulted clean audio
    sample_dataset : contains 54 audio samples from the dataset
    test_audio : audios used for testing performance
    UrbanSound8K : metadata file for the real audio along with it's labels
    preprocess.py : Contains the preprocessing performed on the data:
                - converting audio file into spectrogram
                - Spectro to mfcc 
                - Feature extraction on melspectrograms
                - reshaping to 2D to CSV form
                - train/test split and save as csv
    train.py : - Retrieve the data from csv 
               - Reshape to One Hot to CNN required form
               - Model formation and compilation
               - Saving model with test score
    test.py : - Loads model
              - Inputs audio file and Preprocess
              - Predict the NOISE present
              - Removes corresponding noises using 'noise reducer'
              - Saves the final output
    
    
### Usage
        1. python3 preprocess.py
        2. pyhton3 train.py
        3. python3 test.py
        
        
### Results

The training and testing of the model was done on the server with : 

60GB RAM 

16GB GPU 

Training duration: 3 min for 40 epochs, 50 batch size.

    Training Accuracy: 97.1%        Training Loss: 0.07
    Validation Accuracy: 79.5%      Validation Loss: 0.4
    
**Input Audio 1:** [noisy-1.wav](https://drive.google.com/file/d/1e5FI30J-grBRXg68a8o2Fdw_V_5DdGTy/view?usp=sharing)

**Result :** [cleaned-1.wav](https://drive.google.com/file/d/1L-ndhO4sWllQOe9Isq-CB6ZC6mevjawU/view?usp=sharing)

input audio: 

![img](https://raw.githubusercontent.com/immohann/Denoiser/master/results/noisy1.png?token=ALRSXZUSLOPZCPRJ7CHSEI27BJLLU)

output :

![img](https://raw.githubusercontent.com/immohann/Denoiser/master/results/clean1.png?token=ALRSXZUSF5UINRDY45EPXI27BJLL6)

    
**Input Audio 2:** [noisy-2.wav](https://drive.google.com/file/d/1_vEW7WtA8-758ZgY4-QZnVKTSyQpFrKw/view?usp=sharing)

**Result :** [cleaned-2.wav](https://drive.google.com/file/d/1o9r5YFMahuN41Ik2HNZkvX55PWwaFjcc/view) 

input audio: 

![img](https://raw.githubusercontent.com/immohann/Denoiser/master/results/noisy2.png?token=ALRSXZXD3RWERXUEI7G2QHC7BJLMG)

output :

![img](https://raw.githubusercontent.com/immohann/Denoiser/master/results/clean2.png?token=ALRSXZRDLWDEU5BDHSWVE5S7BJLMQ)

### Conclusion
Results obtained are remarkable, still a number of things can be done to improve the performance:

- Reduction in real audio loss using HQ filters
- Increase number of noise-removal sample to get more accurate results.
- More generalized method of filtering can be performed.
- Do ping for quality updates and ideas.
    
    
**[MIT License](https://github.com/immohann/Denoiser/blob/master/LICENSE)**
    

              
               
