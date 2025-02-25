# panns_transfer_to_ESC50


To run:

1) Edit config file:
Path to CSV file
Path to Audio files

In Anaconda Prompt:
2) Setup variables
set DATASET_DIR=C:\Users\imspr\ESC-50-master\audio #(not actually used here since it defined in config)

set WORKSPACE=C:\Users\imspr\ESC-50-CODE

set PRETRAINED_CHECKPOINT_PATH=C:\Users\imspr\Cnn14_mAP=0.431.pth

set CUDA_VISIBLE_DEVICES=0
3) Calculate features
python ESC-50-CODE/code_2/utils/features.py pack_audio_files_to_hdf5 --workspace=%WORKSPACE% --dataset_dir=%DATASET_DIR%

4) Setup the data directory to the new file created
set DATASET_DIR=C:\Users\imspr\ESC-50-CODE\features\waveform.h5

5) Run the main function
python ESC-50-CODE/code_2/pytorch/main.py train --dataset_dir=%DATASET_DIR% --workspace=%WORKSPACE% --holdout_fold=5 --model_type=Transfer_Cnn14 --pretrained_checkpoint_path=%PRETRAINED_CHECKPOINT_PATH% --loss_type=clip_nll --augmentation mixup --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=4000 --cuda
