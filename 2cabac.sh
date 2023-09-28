CUDA_VISIBLE_DEVICES=1  python train.py --model_type HDNeRV2 --model_size L \
        --eval_only --weight checkpoints/HDNeRV2/L.pth -b 4 -p 500 --method cabac --qp -34

CUDA_VISIBLE_DEVICES=1  python train.py --model_type HDNeRV2 --model_size L \
        --eval_only --weight checkpoints/HDNeRV2/L.pth -b 4 -p 500 --method cabac --qp -38

CUDA_VISIBLE_DEVICES=1  python train.py --model_type HDNeRV2 --model_size L \
        --eval_only --weight checkpoints/HDNeRV2/L.pth -b 4 -p 500 --method cabac --qp -42

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size M \
        --eval_only --weight checkpoints/HDNeRV2/M.pth -b 4 -p 500 --method cabac --qp -34

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size M \
        --eval_only --weight checkpoints/HDNeRV2/M.pth -b 4 -p 500 --method cabac --qp -38
        
CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size M \
        --eval_only --weight checkpoints/HDNeRV2/M.pth -b 4 -p 500 --method cabac --qp -42

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size S \
        --eval_only --weight checkpoints/HDNeRV2/S.pth -b 4 -p 500 --method cabac --qp -34

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size S \
        --eval_only --weight checkpoints/HDNeRV2/S.pth -b 4 -p 500 --method cabac --qp -38

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size S \
        --eval_only --weight checkpoints/HDNeRV2/S.pth -b 4 -p 500 --method cabac --qp -42



hdnerv2:
  compression_dir: compression/hdnerv2
  embedding_path: "ompressed_embedding.ncc"
  raw_decoder_path: "raw_decoder.pt"
  stream_path: "compressed_decoder.nnc"
  compressed_decoder_path: "compressed_decoder_converted.pt"