#!/bin/bash
#MSUB -N kk_nemo_gpu_test
#MSUB -q gpu
#MSUB -l nodes=1:ppn=4:gpus=1
#MSUB -l walltime=0:24:00:00
#MSUB -l pmem=8000mb
#MOAB -d /work/ws/nemo/fr_kk486-kk486-0/video_fusion/
#MOAB -o /work/ws/nemo/fr_kk486-kk486-0/video_fusion/out_std_${MOAB_JOBID}.out
#MOAB -e /work/ws/nemo/fr_kk486-kk486-0/video_fusion/error_std_${MOAB_JOBID}.err

source /home/fr/fr_fr/fr_kk486/.bashrc

conda activate pytorch_video

echo 'env:' $CONDA_DEFAULT_ENV
echo 'env:' $CONDA_PREFIX
echo 'pythonpath:' $PYTHONPATH
echo "path: $PATH"

echo 'which python:' $(which python)

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";

# ================================================== #
# Begin actual Code

/home/fr/fr_fr/fr_kk486/anaconda3/envs/pytorch_video/bin/python   work/ws/nemo/fr_kk486-kk486-0/video_fusion/training_script.py

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";