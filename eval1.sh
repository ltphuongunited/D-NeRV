python train.py --model_type HDNeRV3 --model_size S \
        --eval_only --weight checkpoints/HDNeRV3/S.pth -b 2 --method normal -p 500

python train.py --model_type HDNeRV3 --model_size M \
        --eval_only --weight checkpoints/HDNeRV3/M.pth -b 2 --method normal -p 500

python train.py --model_type HDNeRV3 --model_size L \
        --eval_only --weight checkpoints/HDNeRV3/L.pth -b 2 --method normal -p 500

python train.py --model_type HDNeRV3 --model_size S \
        --eval_only --weight checkpoints/HDNeRV3/S.pth -b 2 --method cabac -p 500

python train.py --model_type HDNeRV3 --model_size M \
        --eval_only --weight checkpoints/HDNeRV3/M.pth -b 2 --method cabac -p 500

python train.py --model_type HDNeRV3 --model_size L \
        --eval_only --weight checkpoints/HDNeRV3/L.pth -b 2 --method cabac -p 500

python train.py --model_type HDNeRV2 --model_size M \
        --eval_only --weight checkpoints/HDNeRV2/M.pth -b 2 --method cabac -p 500

python train.py --model_type HDNeRV2 --model_size L \
        --eval_only --weight checkpoints/HDNeRV2/L.pth -b 2 --method cabac -p 500