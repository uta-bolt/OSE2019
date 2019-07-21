#!/bin/bash
# a sample job submission script to submit a job to the sandyb partition on Midway1

# set the job name to hello
#SBATCH --job-name=dp

#SBATCH --time=00:10:00 ## walltime requested

# send output to outputDP.out
#SBATCH --output=outputDP.out
#SBATCH --error=outputDP.err ## error
# receive an email when job starts, ends, and fails
#SBATCH --mail-type=BEGIN,END,DAIL

# this job requests 1 core. Cores can be selected from various nodes.
#SBATCH --ntasks=1

# there are many partitions on Midway1 and it is important to specify which
# partition you want to run your job on. Not having the following option, the
# broadwl partition on Midway1 will be selected as the default partition
#SBATCH --partition=broadwl

# Run the process
./VFI
