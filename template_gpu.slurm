#!/bin/sh
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=infer       #Set the job name to "JobExample5"
#SBATCH --time=4:40:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=40G                  #Request 2560MB (2.5GB) per node
#SBATCH --output=ExampleOut.%j      #Send stdout/err to "Example5Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132821719374             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=email_address    #Send all emails to email_address 

source activate py38


