import subprocess
import os
import numpy as np
import time

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.rfjob" % os.getcwd()
script_directory = "%s/.rfjob/script" % os.getcwd()
out_directory = "%s/.rfjob/out" % os.getcwd()
err_directory = "%s/.rfjob/err" % os.getcwd()

# Make top level directories
mkdir_p(job_directory)
mkdir_p(script_directory)
mkdir_p(out_directory)
mkdir_p(err_directory)

# count = 0
# for ngmin in np.arange(1, 5, 1):
#     for ngmax in np.arange(ngmin, 5, 1):
#         for mindf in np.arange(0.006, 0.001, -0.001):
#             for maxdf in [1.0]:
#                 for c in [0.01, 0.1, 1, 10, 100]:
#                     for penalty in ['l1', 'l2']:
#                         count += 1
# print(count)
# exit()

count = 0

maxdf = 1.0
for ngmin in np.arange(1, 5, 1):
    for ngmax in np.arange(ngmin, 5, 1):
        for mindf in np.arange(0.003, 0.03, 0.001):
            for featuretype in ['bow', 'tfidf']:
                # if count == 1:
                #     quit()

                jobname = "rf_testing"
                params = [ngmin, ngmax, mindf, maxdf, featuretype]
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
                    fh.writelines(
                        "#SBATCH --output=%s\n" % out_file)
                    fh.writelines(
                        "#SBATCH --error=%s\n" % err_file)
                    fh.writelines("#SBATCH --time=10:00:00\n")
                    fh.writelines("#SBATCH --mem=32000\n")
                    fh.writelines("#SBATCH --cores=32\n")
                    fh.writelines("#SBATCH --account=cis220051\n")
                    fh.writelines("#SBATCH --partition=shared\n")
                    fh.writelines(
                        "python ./rfmodeloptimization.py %g %g %g %g %s %s" % (ngmin, ngmax, mindf, maxdf, featuretype, filename))

                subprocess.call(['sbatch %s' % job_file], shell=True)
                time.sleep(0.1)
                
                count += 1
