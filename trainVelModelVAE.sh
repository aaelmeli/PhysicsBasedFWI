
mkdir ./VaeVelCurvA14
rm -rf /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/checkpoints/VaeVelCurvA14/*.txt
no_proxy=localhost python trainVal4dVel2.py --dataroot /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/deepwave/deepwave/Notebooks/curvelA/allmodels/ --name VaeVelCurvA14 --model Vae --direction AtoB --input_nc 7 --output_nc 1 --display_port 9987 --n_epochs 500 --n_epochs_decay 50  --batch_size 1  --gpu_ids 5 --no_html --display_freq 1 --print_freq 1 --lr 0.0001 --verbose --save_epoch_freq 20
