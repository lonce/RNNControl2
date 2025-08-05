## RNNControl

Jupyter notebooks that run experiments investigating the effect of conditional parameters on audio synthesis.   
The code here is closely integrated with [audioDataloader](https://github.com/muhdhuz/AudioDataloader) and [paramManager](https://github.com/lonce/paramManager).
Clone those repositories and run the notebooks from this repo alongside them as follows: 

.  
+--model  
+--README.md  
+--paramManager  
+--audioDataloader  
+--explore_generation.ipynb  
+--main.ipynb  
+--data  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|+--dataset  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|+--dataparams

**Files**  
* main.ipynb: for training and saving model (an RNN)
* explore_generation.ipynb: to do cool stuff i.e. audio synthesis with the trained model
* model: pre-trained models can be found here 

**Dependencies**  
* pytorch 0.4.0
* [audioDataloader](https://github.com/muhdhuz/AudioDataloader)
* [paramManager](https://github.com/lonce/paramManager)
  

**Authors**  
* Muhammad Huzaifah

- - -
NOTE: this project is incomplete, run at your own risk.  

**To do**  
 * Explore different primer for generation
 * Speed up generation
 * Explore conditional parameters






