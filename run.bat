@echo off
cls

set "args="

REM NOTE: if the model checkpoint flag is defined, will use that / not create a new model.

REM model checkpoint in
set "args=%args% --in_model_checkpoint_file_path=Z:\neural_network\data\checkpoints\fashion_mnist.model_checkpoint.pb"

REM new model characteristics
set "args=%args% --layer_sizes=784,512,256,10"
set "args=%args% --intermediate_activation=SIGMOID"
set "args=%args% --output_activation=SOFTMAX"

REM input data paths
set "args=%args% --train_data_file_path=Z:\neural_network\data\fashion_mnist\train.csv"
set "args=%args% --test_data_file_path=Z:\neural_network\data\fashion_mnist\test.csv"

REM output model checkpoint path
set "args=%args% --out_model_checkpoint_file_path=Z:\neural_network\data\checkpoints\fashion_mnist.model_checkpoint.pb"

REM train params
set "args=%args% --cost=MEAN_SQUARED"
set "args=%args% --learn_rate=0.0005"
set "args=%args% --momentum=0.9"
set "args=%args% --regularization=0.0"
set "args=%args% --train_batch_size=64"
set "args=%args% --test_batch_size=128"
set "args=%args% --num_epochs=5"

REM misc flags
set "args=%args% --stderrthreshold=0"

bazel run src:main -- %args%
