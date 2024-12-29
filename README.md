# inductivity
Automated inductivity runs

Package used to generate multi-panel inductivity plots (scatter, residual, context, spatial frequency), store Poynting flux maps, and save statistical output files

CONTAINS:
1) Muram_Inductivity.py -- python routine, the backend 
2) run_Muram_Inductivity.sh -- script that iteratively calls on the python routine and closes it after each iteration to prevent memory overload.

BEFORE RUNNING:
1) Make sure the correct MURaM slice are in ./data or change the path variable (located at the beginning of the main function, average_temporally())! Download correct slices (to make sure they are correct, in Muram_Inductivity.py check the iter0 parameter (also at the beginning of average_temporally(), should match the time step of the first MURaM slice) and the delta_iter parameter (same locations, determines cadence). The code currently runs through six time steps excluding the initial one (hence the number of subplots in each row), so for instance, if the initial one is 138200 and delta_iter=200, then the MURaM slices corresponding to time steps 138400, 138600, 138800, 139000, 139200, and 139400 should also be present. Take care to include all the z and tau heights at these time steps, or edit the input matrix at the end of Muram_Inductivity.py to match your input files
2) mkdir ./figures (subdirectory to store figures)
3) mkdir ./averagings_output/Sz_fits (subdirectory to store Poynting flux files)

RUNNING THE SCRIPT:

Run the run_Muram_Inductivity.sh script from terminal to perform temporal and/or spatial averaging on any set of heights, slices, FOVs. The arrays at the beginning of the script correspond to the input matrix in Muram_Inductivity.py, at the end of the code. Refer there for descriptions, or use the help function. The script iteratively runs through every combination of averaging method, surface, and FOV.

AFTER RUNNING:

Make sure the file output.txt, where the statistical parameters are stored, as well as the subdirectories where figures and Poynting flux files are stored, are either renamed or moved elsewhere. If this step is not taken, the next call to run_Muram_Inductivity.sh can overwrite these files, resulting in loss of data.

INTERPRETING output.txt

There are six lines of statistical parameters per each iteration of Muram_Inductivity.py, corresponding to the six cadences mentioned above. These correspond to averaging between the 1st and the 2nd, the 1st and the 3rd, ..., and the 1st and the 7th frames.
The leftmost seven columns encode the output (the similarity index is constructed from linear fit (column 1), Spearman r (column 3), and Pearson r (column 4)); the following eight columns encode the input parameters for Muram_Inductivity.py; the string in the rightmost column is a semi-linguistic representation of these input parameters.

FOR ADDITIONAL INFO:
Muram_Inductivity.py contains more detailed comments on specific procedures.
