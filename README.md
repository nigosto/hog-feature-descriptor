# hog-feature-descriptor

A collection of different parallel implementations of the Histograms of Oriented Gradients feature descriptor, optimized for HPC workloads.

## Instructions for running the experiments
If you have installed `pyenv`, then it should automatically detect the python version from `.python-version`. Otherwise, you would need to install it manually.

After setting the correct version of the python interpreter, run:
```bash
python -m venv .venv # create virtual environment

source .venv/bin/activate # activate the virtual environment

python -m pip install -r requirements.txt # install the packages, listed in requirements.txt
```

After that you can run the experiments that are inside the `scripts` directory and profile them in the following way:

```bash
perf stat -r 3 python3 -m scripts.vectorized.run_parallel
```

An example output may look like this:

```bash
 Performance counter stats for 'python3 -m scripts.vectorized.run_parallel' (3 runs):

         626026,22 msec task-clock                       #    4,022 CPUs utilized               ( +-  1,50% )
             37347      context-switches                 #   59,657 /sec                        ( +-  9,99% )
              6272      cpu-migrations                   #   10,019 /sec                        ( +-  0,91% )
           3346526      page-faults                      #    5,346 K/sec                       ( +-  0,88% )
     2190430882258      cycles                           #    3,499 GHz                         ( +-  0,58% )
     4581210619670      instructions                     #    2,09  insn per cycle            
      995029429180      branches                         #    1,589 G/sec                       ( +-  0,02% )
        3084432124      branch-misses                    #    0,31% of all branches             ( +-  0,44% )

            155,66 +- 2,16 seconds time elapsed  ( +-  1,39% )
```

IMPORTANT: Keep in mind that you may need to adjust the scripts - for example change the dataset directory, change the number of workers to match your hardware, etc.

To run the distributed version, run the following command on a SLURM cluster:

```bash
sbatch run.sh
```