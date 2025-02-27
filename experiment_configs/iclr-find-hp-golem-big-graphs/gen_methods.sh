# source experiment_configs/iclr-find-hp-golem-big-graphs/gen_methods.sh > experiment_configs/iclr-find-hp-golem-big-graphs/methods.yaml

LAMBDAS1=(0.02 0.002 0.0002)
LAMBDAS2=(2.0 5.0 8.0)
THRESHOLDS=(0.1 0.2 0.3)
LRS=(0.001)

for LAMBDA1 in "${LAMBDAS1[@]}"
do
    for LAMBDA2 in "${LAMBDAS2[@]}"
    do
        for THR in "${THRESHOLDS[@]}"
        do
            for LR in "${LRS[@]}"
            do
              echo "golem_ev__lambd1=${LAMBDA1}-lambd2=${LAMBDA2}-lr=${LR}-wthreshold=${THR}:
  lambda_1: ${LAMBDA1}
  lambda_2: ${LAMBDA2}
  equal_variances: True
  learning_rate:  ${LR}
  w_threshold: ${THR}

  __run__:
    n_cpus: 1
    n_gpus: 0
    length: short
    grouped: True
"
echo "golem_nv__lambd1=${LAMBDA1}-lambd2=${LAMBDA2}-lr=${LR}-wthreshold=${THR}:
  lambda_1: ${LAMBDA1}
  lambda_2: ${LAMBDA2}
  equal_variances: False
  learning_rate:  ${LR}
  w_threshold: ${THR}

  __run__:
    n_cpus: 1
    n_gpus: 0
    length: short
    grouped: True
"
            done
        done
    done
done



# TODO: Check seeding
