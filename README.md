To run the CRF model:
Python3 ner.py —model CRF
This command would run the general implementation with the features given in the reference code. 

To run the model with the additional features, containing all the features: 
- uncomment line 265 and comment line 264
- uncomment line 331 and comment line 332
- uncomment line 314 and comment line 315 and 316
- uncomment line 428 and comment line 429


To load a pre-trained weight and run only inference: 
Uncomment line 347 and line 425

File names and the feature: 
Eng.testb.out: Original CRF model
Eng.testb_punc_pos_stop.out: Punctuation + POS cluster + Stop Word
eng.testb_punc_stop.out: Punctuation + Stop Word
Eng.testb_punc.out: Punctuation
eng.testb_ngram.out: n-grams
Eng.testb_CRF_all_features.out: Punctuation + POS cluster + Stop Word + TF-IDF
eng.testb_all_features_ngram.out: Punctuation + POS cluster + Stop Word + TF-IDF + n-grams 


File name for trained model weight: 
Optimizer_weights_10epochs.npy: CRF model trained for 10 epochs with the original feature set
