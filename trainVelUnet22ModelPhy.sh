
mkdir ./unet2
# rm -rf /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/checkpoints/MarmousiOpenUnet22/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/MarmousiN/ --name MarmousiOpenUnet2 --model Unet22 --direction AtoB --input_nc 18 --output_nc 1 --display_port 8881 --n_epochs 5000 --n_epochs_decay 0 --batch_size 8 --gpu_ids 1 --no_html --display_freq 1 --print_freq 1 --lr 0.003 --verbose --save_epoch_freq 50
