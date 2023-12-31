import nnc
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from model import *
from dahuffman import HuffmanCodec
from pathlib import Path
import yaml


def load_yaml(fname):
    fname = Path(fname)
    with fname.open("rt") as file:
        config = yaml.safe_load(file)
    return config

def state(full_model):
    encoder_state, decoder_state = (
        full_model.state_dict().copy(),
        full_model.state_dict().copy(),
    )
    
    if type(full_model) is HDNeRV2:
        ckpt_dict = load_yaml("compression/hdnerv2.yaml")
    else:
        ckpt_dict = load_yaml("compression/hdnerv3.yaml")
    
    decoder_list, encoder_list = ckpt_dict["decoder"], ckpt_dict["encoder"]

    for key in decoder_list:
        del encoder_state[key]  # Encoder

    for key in encoder_list:
        del decoder_state[key]  # Decoder


    if type(full_model) is HDNeRV2:
        encoder_model = HDNeRV2Encoder(full_model)
        decoder_model = HDNeRV2Decoder(full_model)
    else:
        encoder_model = HDNeRV3Encoder(full_model)
        decoder_model = HDNeRV3Decoder(full_model)
    
    encoder_model.load_state_dict(encoder_state)
    encoder_model.eval()

    decoder_model.load_state_dict(decoder_state)
    decoder_model.eval()

    return encoder_model, decoder_model


# DeepCABAC
def embedding_compress(dataloader, encoder_model, compressdir, embedding_path, eqp, model_size):
    embedding = defaultdict()

    compressdir_size = os.path.join(compressdir, model_size)
    if not os.path.exists(compressdir_size):
        os.mkdir(compressdir_size)

    embedding_path_size = os.path.join(compressdir_size, embedding_path) 

    for batch_idx, (video, norm_idx, keyframe, backward_distance, frame_mask) in tqdm(enumerate(dataloader)):
        video = video.cuda()
        if type(encoder_model) is HDNeRV2Encoder:
            feature = encoder_model(video)
        else:
            feature = encoder_model(video[:,:,1:-1,:,:])
        embedding[str(batch_idx)] = feature.cpu().detach().numpy()

    bit_stream = nnc.compress(
        parameter_dict=embedding,
        bitstream_path=embedding_path_size,
        qp=eqp,
        return_bitstream=True,
    )

    embedding = nnc.decompress(embedding_path_size, return_model_information=True)
    return embedding[0], len(bit_stream)


def dcabac_compress(
    full_model, decoder_model_org, stream_path, mqp, compressdir, compressed_decoder_path, model_size):

    compressdir_size = os.path.join(compressdir, model_size)
    if not os.path.exists(compressdir_size):
        os.mkdir(compressdir_size)
    
    compressed_decoder_path_size = os.path.join(compressdir_size, compressed_decoder_path)
    stream_path_size = os.path.join(compressdir_size, stream_path)

    bit_stream = nnc.compress_model(
        decoder_model_org,
        bitstream_path=stream_path_size,
        qp=int(mqp),
        return_bitstream=True,
    )
    nnc.decompress_model(stream_path_size, model_path=compressed_decoder_path_size)

    if type(full_model) is HDNeRV2:
        decoder_model = HDNeRV2Decoder(full_model)
    else:
        decoder_model = HDNeRV3Decoder(full_model)

    decoder_model.load_state_dict(torch.load(compressed_decoder_path_size))
    decoder_model.eval()

    return decoder_model, len(bit_stream)


# def dcabac_decoding(embedding_path, stream_path, compressed_decoder_path, decoder_dim):
#     embedding = nnc.decompress(embedding_path, return_model_information=True)

#     nnc.decompress_model(stream_path, model_path=compressed_decoder_path)
#     decoder_model = NeRV3DDecoder(NeRV3D(decode_dim=decoder_dim))
#     decoder_model.load_state_dict(torch.load(compressed_decoder_path))
#     decoder_model.eval()

#     return embedding[0], decoder_model.cuda()


# Traditional Compression
def compute_quantization_params(tensor, num_bits=8):
    def calculate_scale(min_value, max_value):
        return (max_value - min_value) / (2**num_bits - 1)

    global_min, global_max = tensor.min(), tensor.max()
    quantization_params = [[global_min, calculate_scale(global_min, global_max)]]

    for axis in range(tensor.dim()):
        axis_min, axis_max = (
            tensor.min(axis, keepdim=True)[0],
            tensor.max(axis, keepdim=True)[0],
        )

        if axis_min.nelement() / tensor.nelement() < 0.02:
            axis_scale = calculate_scale(axis_min, axis_max).to(torch.float16)
            quantization_params.append([axis_min.to(torch.float16), axis_scale])

    return quantization_params


def quantize_tensor(tensor, min_value, scale_factor, num_bits=8):
    expanded_min_value = min_value.expand_as(tensor)
    expanded_scale_factor = scale_factor.expand_as(tensor)

    quantized_tensor = (
        ((tensor - expanded_min_value) / expanded_scale_factor)
        .round()
        .clamp(0, 2**num_bits - 1)
    )
    approximated_tensor = expanded_min_value + expanded_scale_factor * quantized_tensor
    quantization_error = (tensor - approximated_tensor).abs().mean()

    return quantized_tensor, approximated_tensor, quantization_error


def quantize_tensor_full(tensor, num_bits=8):
    quantization_params = compute_quantization_params(tensor, num_bits)
    quantized_tensors, approximated_tensors, errors = [], [], []

    for min_value, scale_factor in quantization_params:
        quantized_tensor, approximated_tensor, error = quantize_tensor(
            tensor, min_value, scale_factor, num_bits
        )
        quantized_tensors.append(quantized_tensor)
        approximated_tensors.append(approximated_tensor)
        errors.append(error)

    best_error = min(errors)
    best_index = errors.index(best_error)
    best_quantized_tensor = quantized_tensors[best_index].to(torch.uint8)

    quantization_result = {
        "quantized": best_quantized_tensor,
        "min": quantization_params[best_index][0],
        "scale": quantization_params[best_index][1],
    }

    return quantization_result, approximated_tensors[best_index]


def quantize_model(model, num_bits=8):
    # Create a copy of the model for quantization
    quantized_model = deepcopy(model)

    # Retrieve the state dictionary for both quantized and original weights
    quantized_state_dict, original_state_dict = [
        quantized_model.state_dict() for _ in range(2)
    ]

    # Iterate through the original weights, quantizing all except encoders
    for key, weight in original_state_dict.items():
        # if "encoder" not in key:
        quantization_data, approximated_weight = quantize_tensor_full(
            weight, num_bits
        )

        quantized_state_dict[key] = quantization_data
        original_state_dict[key] = approximated_weight

    # Load the approximated weights back into the quantized model
    quantized_model.load_state_dict(original_state_dict)

    return quantized_model, quantized_state_dict


def huffman_encoding(quantized_embedding, quantized_weights):
    # Embedding
    quantized_values_list1 = quantized_embedding["quantized"].flatten().tolist()

    min_scale_length1 = (
        quantized_embedding["min"].nelement() + quantized_embedding["scale"].nelement()
    )
    # Compute the frequency of each unique value
    unique_values1, counts1 = np.unique(quantized_values_list1, return_counts=True)
    value_frequency1 = dict(zip(unique_values1, counts1))

    # Generate Huffman coding table
    codec1 = HuffmanCodec.from_data(quantized_values_list1)
    symbol_bit_dictionary1 = {
        key: value[0] for key, value in codec1.get_code_table().items()
    }

    # Compute total bits for quantized data
    total_bits1 = sum(
        freq * symbol_bit_dictionary1[num] for num, freq in value_frequency1.items()
    )

    # Include the overhead for min and scale storage
    total_bits1 += min_scale_length1 * 16  # (16 bits for float16)
    
    quantized_values_list = []
    min_scale_length = 0
    # Decoder weights
    for key, layer_weights in quantized_weights.items():
        quantized_values_list.extend(layer_weights["quantized"].flatten().tolist())

        min_scale_length += (
            layer_weights["min"].nelement() + layer_weights["scale"].nelement()
        )

    # Compute the frequency of each unique value
    unique_values, counts = np.unique(quantized_values_list, return_counts=True)
    value_frequency = dict(zip(unique_values, counts))

    # Generate Huffman coding table
    codec = HuffmanCodec.from_data(quantized_values_list)
    symbol_bit_dictionary = {
        key: value[0] for key, value in codec.get_code_table().items()
    }

    # Compute total bits for quantized data
    total_bits = sum(
        freq * symbol_bit_dictionary[num] for num, freq in value_frequency.items()
    )

    # Include the overhead for min and scale storage
    total_bits += min_scale_length * 16  # (16 bits for float16)
    # total_bits += total_bits1 / 24
    full_bits_per_param = total_bits / len(quantized_values_list)

    return full_bits_per_param, total_bits, total_bits1

# def huffman_encoding(quantized_embedding, quantized_weights):
#     # Embedding
#     quantized_values_list = quantized_embedding["quantized"].flatten().tolist()

#     min_scale_length = (
#         quantized_embedding["min"].nelement() + quantized_embedding["scale"].nelement()
#     )
    
#     # Decoder weights
#     for key, layer_weights in quantized_weights.items():
#         quantized_values_list.extend(layer_weights["quantized"].flatten().tolist())

#         min_scale_length += (
#             layer_weights["min"].nelement() + layer_weights["scale"].nelement()
#         )

#     # Compute the frequency of each unique value
#     unique_values, counts = np.unique(quantized_values_list, return_counts=True)
#     value_frequency = dict(zip(unique_values, counts))

#     # Generate Huffman coding table
#     codec = HuffmanCodec.from_data(quantized_values_list)
#     symbol_bit_dictionary = {
#         key: value[0] for key, value in codec.get_code_table().items()
#     }

#     # Compute total bits for quantized data
#     total_bits = sum(
#         freq * symbol_bit_dictionary[num] for num, freq in value_frequency.items()
#     )

#     # Include the overhead for min and scale storage
#     total_bits += min_scale_length * 16  # (16 bits for float16)
#     full_bits_per_param = total_bits / len(quantized_values_list)

#     return full_bits_per_param, total_bits
def dequantize_tensor(quantization_data):
    quantized_tensor = quantization_data["quantized"]
    min_value = quantization_data["min"]
    scale_factor = quantization_data["scale"]

    expanded_min_value = min_value.expand_as(quantized_tensor)
    expanded_scale_factor = scale_factor.expand_as(quantized_tensor)

    reconstructed_tensor = expanded_min_value + expanded_scale_factor * quantized_tensor
    return reconstructed_tensor


# def normal_compression(
#     full_model,
#     dataloader,
#     encoder_model,
#     decoder_model_org
# ):
#     # Embedding
#     embedding_list = []


#     for batch_idx, (video, norm_idx, keyframe, backward_distance, frame_mask) in tqdm(enumerate(dataloader)):
#         video = video.cuda()
#         if type(full_model) is HDNeRV2:
#             feature = encoder_model(video)
#         else:
#             feature = encoder_model(video[:,:,1:-1,:,:])
#         embedding_list.append(feature.cpu().detach().numpy())

#     # Quantization
#     print('Quantization')
#     quantize_embedding, _ = quantize_tensor_full(
#         torch.Tensor(embedding_list), num_bits=8
#     )

#     quantized_model, quantized_model_state = quantize_model(decoder_model_org, num_bits=8)
#     # Huffman Encoding
#     print('Huffman Encoding')
#     bits_per_param, total_bits = huffman_encoding(
#         quantize_embedding, quantized_model_state
#     )

#     # Dequantization
#     print('Dequantization')
#     embedding = dequantize_tensor(quantize_embedding)

#     return embedding, quantized_model, total_bits

def normal_compression(
    full_model,
    dataloader,
    encoder_model,
    decoder_model_org
):
    # Embedding
    embedding_list = []


    for batch_idx, (video, norm_idx, keyframe, backward_distance, frame_mask) in tqdm(enumerate(dataloader)):
        video = video.cuda()
        if type(full_model) is HDNeRV2:
            feature = encoder_model(video)
        else:
            feature = encoder_model(video[:,:,1:-1,:,:])
        embedding_list.append(feature.cpu().detach().numpy())

    # Quantization
    print('Quantization')
    quantize_embedding, _ = quantize_tensor_full(
        torch.Tensor(embedding_list), num_bits=8
    )

    quantized_model, quantized_model_state = quantize_model(decoder_model_org, num_bits=8)
    # Huffman Encoding
    print('Huffman Encoding')
    bits_per_param, total_bits_model, total_bits_embed = huffman_encoding(
        quantize_embedding, quantized_model_state
    )

    # Dequantization
    print('Dequantization')
    embedding = dequantize_tensor(quantize_embedding)

    return embedding, quantized_model, total_bits_model, total_bits_embed
