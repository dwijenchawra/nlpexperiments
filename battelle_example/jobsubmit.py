import os
import numpy as np

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
scratch = os.environ['SCRATCH']
data_dir = os.path.join(scratch, '/nlp_testing/')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)

for ngmin in np.arange(1, 5, 1):
    for ngmax in np.arange(ngmin, 5, 1):
        for mindf in np.arange(ngmin, 5, 1):
            for maxdf in np.arange(ngmin, 5, 1):
                for c in [0.01, 0.1, 1, 10, 100]:
                    for penalty in ['l1', 'l2']:
                        if count == 3:
                            quit()

                        jobname = ""

                        job_file = os.path.join(
                            job_directory, "%s.job" % lizard)
                        lizard_data = os.path.join(data_dir, lizard)

                        # Create lizard directories
                        mkdir_p(lizard_data)

                        with open(job_file) as fh:
                            fh.writelines("#!/bin/bash\n")
                            fh.writelines(
                                "#SBATCH --job-name=%s.job\n" % lizard)
                            fh.writelines(
                                "#SBATCH --output=.out/%s.out\n" % lizard)
                            fh.writelines(
                                "#SBATCH --error=.out/%s.err\n" % lizard)
                            fh.writelines("#SBATCH --time=00:30:00\n")
                            fh.writelines("#SBATCH --mem=8000\n")
                            fh.writelines("#SBATCH --qos=normal\n")
                            fh.writelines(
                                "Rscript $HOME/project/LizardLips/run.R %s potato shiabato\n" % lizard_data)

                        os.system("sbatch %s" % job_file)