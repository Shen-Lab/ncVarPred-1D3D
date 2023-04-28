#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=san       #Set the job name to "JobExample5"
#SBATCH --time=30:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=200G                  #Request 2560MB (2.5GB) per node
#SBATCH --output=ExampleOut.%j      #Send stdout/err to "ExampleOut.[jobID]"

#OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132821719863             #Set billing account to 123456

source activate ncvarpred_1d3d


