# LIBS-Spectra-Classification

This is the code for part of a project I worked on which was concerned with classifying rocks based on 1D-spectra. The dataset (not included in the repo) is composed of _Measurement Points_ (MP) made up of an 8x8 grid of _shots_ where each shot is one such 1D-spectrum. Each shot appears as in the figure below and two experiments were conducted.
![Spectrum](1D_spectra.png | =500x500)
The first one generated more data using an online dataset provided by the National Institute of Standards and Technology (NIST) and used an autoencoder in an attempt to denoise the real data. In this case each single shot was taken as a single stand-alone sample. The second experiment used the entire MP as input to the model and applied pooling to improve performance.
