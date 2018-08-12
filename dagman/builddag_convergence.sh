#!/bin/sh

maindir="/data/condor_builds/users/elims/ezfits/clean_ezfit/"
outdir="/data/i3store0/users/elims/clean_fit/fits/"
nuparamdir="/data/i3store0/users/elims/clean_fit/nuparam_textfiles/convergence_nohplane/"

HISTsub="${maindir}/dagman/temp.submit"
HISTscript="${maindir}/generate_template.py"

FITsub="${maindir}/dagman/fit.submit"
FITscript="${maindir}/fit_template.py"
FITargs="--test_statistics modchi2  --verbose 1"

#########################################################
#### define DAG arguments 
#########################################################
for nufile in `ls ${nuparamdir}`; do

    base=${nufile:0:-4}
    
    ### templates
    toutfile="${outdir}/template_${base}.p"
    targs="--outfile ${toutfile} --nuparam_textfile ${nuparamdir}/${nufile}"
    command="python ${HISTscript} ${targs}"
    tjobname=conv_temp_${base}
    tJOBID=conv_temp_${base}
    echo JOB $tJOBID ${HISTsub}
    echo VARS $tJOBID JOBNAME=\"$tjobname\" command=\"$command\"
    #echo PRIORITY $hJOBID 8

    ### fit
    foutfile="${outdir}/fit_${base}.p"
    fargs="${FITargs} --template ${toutfile} --outfile ${foutfile}"
    command="python ${FITscript} ${fargs}"
	    
    fjobname=conv_fit_${base}
    fJOBID=conv_fit_${base}
    echo JOB $fJOBID ${FITsub}
    echo VARS $fJOBID JOBNAME=\"$fjobname\" command=\"$command\"
    echo PARENT $tJOBID CHILD $fJOBID
done
