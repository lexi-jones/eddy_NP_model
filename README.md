# eddy_NP_model
Initialize a mesoscale eddy flow from the streamfunction, and model the evolution of the nutrient and phytoplankton response during the eddy  spin-up and spin-down.

Set up environment
	
	conda create --name eddy_NP_model
	
	conda install -c anaconda numpy
	
	conda install -c conda-forge matplotlib jupyterlab parcels jupyter cartopy ffmpeg

Directory contents

	eddy_NP_model_functions.py: contains the functions needed to run the eddy NP model and generate animations and figures

	run_eddy_NP_model.py: runs a model simulation

	interactive_run.ipynb: use to run the model and generate figs in a notebook interface
