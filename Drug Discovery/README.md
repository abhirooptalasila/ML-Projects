# Bioactivity Prediction

<img src="logo.png" width="512"/>

Small molecules can potentially bind to a variety of bimolecular targets and whilst counter-screening against a wide variety of targets is feasible it can be rather expensive and probably only realistic for when a compound has been identified as of particular interest. For this reason there is considerable interest in building computational models to predict potential interactions. With the advent of large data sets of well annotated biological activity such as ChEMBL and BindingDB this has become possible.

Downloaded Bioactivity data from chembl database. Calculated Lepinski descriptors, which are used to find likelihood being a drug like molecule. Used acetylcholinesterase as target protein. Calculated molecular descriptors using PaDEL descriptor software and used regression models to predict bioactivity (pIC50 values).  