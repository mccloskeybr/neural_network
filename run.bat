@echo off
cls

set "args="

REM NOTE: if the model checkpoint flag is defined, will use that / not create a new model.

REM model checkpoint in
REM set "args=%args% --in_model_checkpoint_file_path=Z:\neural_network\data\checkpoints\fashion_mnist.model_checkpoint.pb"

REM new model characteristics
set "args=%args% --layer_sizes=784,512,512,10"
set "args=%args% --intermediate_activation=SIGMOID"
set "args=%args% --output_activation=SOFTMAX"

REM input data paths
set "args=%args% --train_data_file_path=Z:\neural_network\data\mnist\train.csv"
set "args=%args% --test_data_file_path=Z:\neural_network\data\mnist\test.csv"

REM output model checkpoint path
set "args=%args% --out_model_checkpoint_file_path=Z:\neural_network\data\checkpoints\mnist.model_checkpoint.pb"

REM train params
set "args=%args% --cost=MEAN_SQUARED"
set "args=%args% --learn_rate=0.01"
set "args=%args% --momentum=0.1"
set "args=%args% --regularization=0.0"
set "args=%args% --train_batch_size=128"
set "args=%args% --test_batch_size=128"
set "args=%args% --num_epochs=10"

REM misc flags
set "args=%args% --stderrthreshold=0"

bazel run src:main -- %args%
