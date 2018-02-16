source $HOME/.bashrc
source activate matador
. /etc/profile.d/modules.sh
module purge
module load rhel7/default-peta4
module load intel/mkl/2017.4 intel/impi/2017.4/intel
