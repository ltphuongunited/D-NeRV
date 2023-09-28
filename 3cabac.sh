python train.py --model_type HDNeRV3 --model_size L \
        --eval_only --weight checkpoints/HDNeRV3/L.pth -b 4 -p 500 --method cabac --qp -34

python train.py --model_type HDNeRV3 --model_size L \
        --eval_only --weight checkpoints/HDNeRV3/L.pth -b 4 -p 500 --method cabac --qp -38

python train.py --model_type HDNeRV3 --model_size L \
        --eval_only --weight checkpoints/HDNeRV3/L.pth -b 4 -p 500 --method cabac --qp -42

python train.py --model_type HDNeRV3 --model_size M \
        --eval_only --weight checkpoints/HDNeRV3/M.pth -b 4 -p 500 --method cabac --qp -34

python train.py --model_type HDNeRV3 --model_size M \
        --eval_only --weight checkpoints/HDNeRV3/M.pth -b 4 -p 500 --method cabac --qp -38
        
python train.py --model_type HDNeRV3 --model_size M \
        --eval_only --weight checkpoints/HDNeRV3/M.pth -b 4 -p 500 --method cabac --qp -42

python train.py --model_type HDNeRV3 --model_size S \
        --eval_only --weight checkpoints/HDNeRV3/S.pth -b 4 -p 500 --method cabac --qp -34

python train.py --model_type HDNeRV3 --model_size S \
        --eval_only --weight checkpoints/HDNeRV3/S.pth -b 4 -p 500 --method cabac --qp -38

python train.py --model_type HDNeRV3 --model_size S \
        --eval_only --weight checkpoints/HDNeRV3/S.pth -b 4 -p 500 --method cabac --qp -42



