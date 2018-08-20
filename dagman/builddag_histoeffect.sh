#!/bin/sh

maindir="/data/condor_builds/users/elims/ezfits/clean_ezfit/"
outdir="/data/i3store0/users/elims/clean_fit/fits/histoeffect/"
nuparamdir="/data/i3store0/users/elims/clean_fit/nuparam_textfiles/histoeffect/"

HISTsub="${maindir}/dagman/temp.submit"
HISTscript="${maindir}/generate_template.py"

#########################################################
#### define DAG arguments 
#########################################################
for nufile in `ls ${nuparamdir}`; do

    base=${nufile:0:-4}
    
    ### templates
    toutfile="${outdir}/histoeffect_${base}.p"
    targs="--outfile ${toutfile} --nuparam_textfile ${nuparamdir}/${nufile}"
    command="python ${HISTscript} ${targs}"
    tjobname=heffect_temp_${base}
    tJOBID=heffect_temp_${base}
    echo JOB $tJOBID ${HISTsub}
    echo VARS $tJOBID JOBNAME=\"$tjobname\" command=\"$command\"
    echo PRIORITY $tJOBID 8

done
