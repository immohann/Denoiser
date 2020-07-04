# Denoiser
An AI model to remove the specific noise from the noisy input audio using the essentials of Deep Learning.

### Idea
A deep learning model is used to take input audio and detect the type of noises present in the audio. Then, a 'noise reducer' is used to remove the similar kind of audio from the input file and creats a noise free clean audio file.

### Dataset
Dataset used is: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)

Dataset Size: 9GB

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
    
### Results

The training and testing of the model was done on the server with : 

32GB RAM 

16GB GPU 

Training duration: 3 min for 40 epochs, 50 batch size.

    Training Accuracy: 97.1%        Training Loss: 0.07
    Validation Accuracy: 79.5%      Validation Loss: 0.4
    
Input Audio 1: [noisy-1.wav]()

Result : [cleaned-1.wav]()
    
Input Audio 1: [noisy-1.wav]()

Result : [cleaned-1.wav]()

### Conclusion
Results obtained are pretty promising. Although, a number of things can be done to improve the performance:

- Increase number of noise-removal sample to get more accurate results.
- More generalized method of filtering can be performed.
- Do ping for quality updates and ideas.
    
    

              
               
