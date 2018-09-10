This `ezfit` project is a set of scripts to perform a numu disappearance analysis in icecube. In short, this statistical analysis tool is designed to measure the numerical values of neutrino mass squared differences and the corresponding mixing angles, given that only the three standard neutrino flavors exist (i.e. no sterile neutrinos).

Overall Principle
-----------------

The measurement is done via fitting a 3D template to the data histogram. The 3D template is built from 3D histograms with 6 different data types (i.e. members) based on simulations. The members involved are numu CC, nue CC, nutau CC, nu NC (which includes numu NC, nue NC, and nutau NC), atmospheric muons, and noise-triggered events. The signal for this analysis is numu CC, while the rest are all considered as backgrounds.

This analysis tool takes into account 28 sources of uncertainties, 16 of which are included in the final unblinding. Among the 16 uncertainties included, 6 are related to atmospheric neutrino and muon fluxes, whereas another 5 are associated to the probability of neutrino interactions. Additional 5 parameters are related to detector response, while the last 2 uncertainties takes into account the absolute scalings of the individual histograms. The settings for each nuisance parameter can be changed via `nuisance_textfiles/nuparams_template.txt`.

How to run the tool
-------------------

To perform one fit, two steps are required.

First, one needs to make the MC template and a data histogram using `generate_template.py`:

> elims $ python generate_template.py --outfile outputs/test_template.p --nuparam_textfile nuparam_textfiles/nuparams_template.txt 

Based on the user's settings for all parameters in `nuparam_textfiles/nuparams_template.txt`, a mc template is included. Further non-standard settings, such as the histogram binning and members to be included, can be modified inside `generate_template.py`. Note that the above command line uses pseudo-data as the data histogram; to use real data, add `--fit_data` flag to the command line.

After building the templates and data histogram, the second step is to perform the minimization using `fit_template.py`:

> elims $ python fit_template.py --test_statistics modchi2 --template outputs/test_template.p --outfile outputs/fit_test_template.p

One can change `modchi2` to `chi2`/`poisson`/`barlow` for other definitions of test statistics. Additional settings, such as verbose and demanding fixed oscillation parameters, can be added with extra flags.

To learn how a fitter works in general, please take a look at:
http://icecube.umd.edu/~elims/Fitter/Basics

To learn how to use ezFit, visit:
http://icecube.umd.edu/~elims/Fitter/ezFit

How to perform numu disappearance analysis
------------------------------------------

While the above commands are used to perform one fit, the most important aspect of a measurement analysis is to determine 90% confidence level (C.L.) of the measured value. To determine the 1 sigma errorbars for the two neutrino oscillation parameters, many fits are performed at fixed values of oscillation parameters within certain ranges. These fits are performed with the use of a cluster of machines. Within IceCube, a condor job submission system is used (see https://research.cs.wisc.edu/htcondor/). In condor, `dags` are written to define jobs and the corresponding arguments for submitting jobs.

Therefore, to perform numu disappearance analysis, a dag that define all jobs to determine the 90% C.L. is written via Bash script in `dagman/*.sh`. The dag for performing numu disappearance analysis is located in `dagman/fit_90CL.dag`. Additional dags for systematic effects on histograms, for convergence tests, for statistical biases, for N-1 tests, and for Feldman-Cousin, can also be found in `dagman/` folder.

Additional Notes
----------------
