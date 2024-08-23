mkdir ./vaeFull65ccLatent16
rm -rf /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/checkpoints/MarmousiOpenFWIvaeFull65/*.txt
no_proxy=localhost python trainValLatent4dVel2Old.py --dataroot /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/MarmousiBig5/ --name MarmousiOpenFWIvaeFull65 --model Vae3Full --direction AtoB --input_nc 24 --output_nc 1 --n_epochs 500 --n_epochs_decay 500  --batch_size 1 --gpu_ids 6 --no_html --display_freq 1 --print_freq 1 --lr 0.0005 --verbose --save_epoch_freq 25 
