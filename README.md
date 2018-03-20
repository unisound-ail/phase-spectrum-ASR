# Phase spectrum for ASR

This repo is for investigating the performance of phase spectrum (specificly
Modified Group Delay Feaure) on Auto Speech Recognition.

### Training

1. **Dataset.**

     [LibriSpeech](http://www.openslr.org/12/) (CC BY 4.0)

2. **Preprocess data**
   ```
   python preprocess.py --dataset libri
   ```

3. **Train a model**
   ```
   python train.py
   ```

## Reference

1. [Significance of the Modified Group Delay Feature in Speech Recognition](http://pdfs.semanticscholar.org/9e7f/f44e4e7f056fae2e03072d9073e6fe68232a.pdf).
