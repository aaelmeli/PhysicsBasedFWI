mkdir ./vae38ccLatent8
rm -rf  /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/checkpoints/MarmousiOpenFWIVAE38ccLatent8/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/MarmousiN/  --name MarmousiOpenFWIVAE38ccLatent8 --model Vae3 --direction AtoB --input_nc 18 --output_nc 1 --display_port 9990 --n_epochs 200 --n_epochs_decay 200  --batch_size 1 --gpu_ids 7 --no_html --display_freq 1 --print_freq 1 --lr 0.0005 --verbose --save_epoch_freq 25
