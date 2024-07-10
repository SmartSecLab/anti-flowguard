#!/bin/bash
#SBATCH --job-name=FlowGuard           # Job name
#SBATCH --partition=LocalQ             # Partition to submit to
#SBATCH --ntasks=4                     # Number of tasks (MPI processes)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=4            # Number of tasks per node
#SBATCH --time=05-00:00:00             # Walltime (hh:mm:ss)
#SBATCH --output=output/slurm%j.out    # Output file name (%j expands to jobID)
#SBATCH --error=output/slurm%j.err     # Error file name (%j expands to jobID)

# activate the environment
source venv/bin/activate

# run the file
python3 -m run