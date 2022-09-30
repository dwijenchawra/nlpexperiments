import os
import numpy as np
import time

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
script_directory = "%s/.job/script" % os.getcwd()
out_directory = "%s/.job/out" % os.getcwd()
err_directory = "%s/.job/err" % os.getcwd()

# Make top level directories
mkdir_p(job_directory)
mkdir_p(script_directory)
mkdir_p(out_directory)
mkdir_p(err_directory)



count = 0
for ngmin in np.arange(1, 5, 1):
    for ngmax in np.arange(ngmin, 5, 1):
        for mindf in np.arange(ngmin, 5, 1):
            for maxdf in np.arange(ngmin, 5, 1):
                for c in [0.01, 0.1, 1, 10, 100]:
                    for penalty in ['l1', 'l2']:
                        if count == 1:
                            quit()

                        jobname = "logreg_testing"
                        params = [ngmin, ngmax, mindf, maxdf, c, penalty]
                        stringified_params = [str(i) for i in params]
                        filename = "_".join(stringified_params)

                        job_file = os.path.join(
                            script_directory, "%s.sh" % filename)
                        
                        out_file = os.path.join(
                            out_directory, "%s.out" % filename)
                        err_file = os.path.join(
                            err_directory, "%s.err" % filename)

                        # Create lizard directories

                        with open(job_file, "w") as fh:
                            fh.writelines("#!/bin/bash\n")
                            fh.writelines(
                                "#SBATCH --job-name=%s\n" % jobname)
                            # fh.writelines(
                                # "#SBATCH --output=../out/%s.out\n" % filename)
                            fh.writelines(
                                "#SBATCH --output=%s\n" %out_file)
                            fh.writelines(
                                "#SBATCH --error=%s\n" % err_file)
                            fh.writelines("#SBATCH --time=00:30:00\n")
                            fh.writelines("#SBATCH --mem=8000\n")
                            fh.writelines("#SBATCH --account=cis220051\n")
                            fh.writelines("#SBATCH --partition=shared\n")
                            fh.writelines(
                                "python ./modeloptimization.py %g %g %d %d %g %s" % (ngmin, ngmax, mindf, maxdf, c, penalty))

                        os.system("sbatch %s" % job_file)
                        time.sleep(1)
                        
                        count += 1
