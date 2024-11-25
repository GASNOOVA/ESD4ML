# Imports
from functools import reduce
import numpy as np
import collections
from typing import List

### ENTER STUDENT CODE BELOW ###
# If required, add further imports here

### ENTER STUDENT CODE ABOVE ###

# Define custom types for tensors and layers
MyTensor = collections.namedtuple("MyTensor", ("idx", "name", "shape", "dtype", "is_const"))
MyLayer = collections.namedtuple("MyLayer", ("idx", "name", "inputs", "outputs"))

"""
Example Usage for FC layer (e.g. y=W'x+b):

>>> input = MyTensor(0, "x", [1, 1024], "int8", False)
>>> weights = MyTensor(1, "W", [10, 1024], "int8", True)
>>> bias = MyTensor(2, "b", [10], "int32", True)
>>> output = MyTensor(3, "y", [1, 10], "int8", False)
>>> fc = MyLayer(0, "FullyConnected", [input.idx, weights.idx, bias.idx], [output.idx])
"""


def estimate_conv2d_macs(in_shape: List[int], kernel_shape: List[int], out_shape: List[int]):
    """Calculate the estimated number of MACS to execute a Conv2D layer.

    Arguments
    ---------
    in_shape : list
        The shape of the NHWC input tensor [batch_size, input_h, input_w, input_c]
    kernel_shape : list
        The shape of the OHWI weight tensor [kernel_oc, kernel_h, kernel_w, kernel_ic]
    out_shape : list
        The shape of the NHWC output tensor [batch_size, output_h, output_w, output_c]


    Returns
    -------
    macs : int
        Estimated number of MAC operations for the given Conv2D layer

    Assumptions
    -----------
    - It is allowed to count #multiplications instead of #macs
    """

    # Hint: not every of the following values is required
    input_n, input_h, input_w, input_c = in_shape
    kernel_oc, kernel_h, kernel_w, kernel_ic = kernel_shape
    output_n, output_h, output_w, output_c = out_shape

    # Assertions
    assert input_n == output_n == 1  # Inference -> batch_size=1
    assert input_c == kernel_ic
    assert output_c == kernel_oc

    macs = 0

    ### ENTER STUDENT CODE BELOW ###

    ### ENTER STUDENT CODE ABOVE ###

    return macs


def estimate_depthwise_conv2d_macs(
    in_shape: List[int], kernel_shape: List[int], out_shape: List[int], channel_mult: int
):
    """Calculate the estimated number of MACS to execute a Depthwise Conv2D layer.

    Arguments
    ---------
    in_shape : list
        The shape of the NHWC input tensor [batch_size, input_h, input_w, input_c]
    kernel_shape : list
        The shape of the weight tensor [1, kernel_h, kernel_w, kernel_oc]
    out_shape : list
        The shape of the NHWC output tensor [batch_size, output_h, output_w, output_c]
    channel_mult : int
        The channel multiplier used to determine the number of output channels.
        See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D


    Returns
    -------
    macs : int
        Estimated number of MAC operations for the given Depthwise Conv2D layer

    Assumptions
    -----------
    - It is allowed to count #multiplications instead of #macs
    """

    # Hint: not every of the following values is required
    input_n, input_h, input_w, input_c = in_shape
    _, kernel_h, kernel_w, kernel_oc = kernel_shape
    output_n, output_h, output_w, output_c = out_shape

    # Assertions
    assert input_n == output_n == 1  # Inference -> batch_size=1
    assert output_c == kernel_oc == input_c * channel_mult

    macs = 0

    ### ENTER STUDENT CODE BELOW ###

    ### ENTER STUDENT CODE ABOVE ###

    return macs


def estimate_fully_connected_macs(
    in_shape: List[int], filter_shape: List[int], out_shape: List[int]
):
    """Calculate the estimated number of MACS to execute a Fully Connected layer.

    Arguments
    ---------
    in_shape : list
        The shape of the input tensor [input_h, input_w]
    filter_shape : list
        The shape of the weight tensor [filter_h, filter_w]
    out_shape : list
        The shape of the output tensor [output_h, output_w]


    Returns
    -------
    macs : int
        Estimated number of MAC operations for the given Fully Connected layer

    Assumptions
    -----------
    - It is allowed to count #multiplications instead of #macs
    """

    # Hint: not every of the following values is required
    input_h, input_w = in_shape
    filter_h, filter_w = filter_shape
    output_h, output_w = out_shape

    # Assertions
    assert input_w == filter_h
    assert output_w == filter_w
    assert input_h == output_h

    macs = 0

    ### ENTER STUDENT CODE BELOW ###

    ### ENTER STUDENT CODE ABOVE ###

    return macs


def estimate_rom(tensors: List[MyTensor]):
    """Calculate the estimated number of bytes required to store model weights in ROM.

    Arguments
    ---------
    tensors : list
        The tensors of the processed model (see definition of MyTensor type above)

    Returns
    -------
    rom_bytes : int
        Estimated number of bytes in ROM considering only model weights and biases

    Assumptions
    -----------
    - The considered model will be a fully quantized one in TFLite format
    - TFLite uses different datatypes for the biases.
    - Only constant tensors (weights, biases) contribute to the expected ROM usage (e.g. the model graph, metadata,... can be ignored)
    - A Reshape layer in TFLite has a constant shape tensor, which has to be considered as well.
    """

    rom_bytes = 0

    ### ENTER STUDENT CODE BELOW ###
    '''
    formula :
    rom_bytes = N_elements * size_per_weight
    N_weight = d1 * d2 * d3 *....* dn
    
    example :
    Reshape-Layer   [4] shape int32 -> 4*4=16
    Conv-Layer      [1,5,4,4] weights int8 -> 1*5*4*4*1=80
                    [4] bias int32 -> 4*4=16
    Reshape-Layer   [2] shape int32 -> 2*4=8
    FC-Layer        [4,20000] weight int8 -> 4*2000*1=8000
                    [4] bias int32 -> 4*4=16
    Total           8136 Byte
    '''
    
    for tensor in tensors:
        if tensor.is_const:
            if tensor.dtype == 'int8':
                size_per_weight = 1     # 1 byte for int8
            elif tensor.dtype == 'int32' or tensor.dtype == 'float32':
                size_per_weight = 4     # 4 byte for int32 and float32
            else:
                raise ValueError(f'Unsupported tensor data type: {tensor.dtype}')
            
            # calculate the total size of the tensors by multiplying all dimensions of its shape
            # multiply all the elements in tensor.shape sequentially
            num_elements = reduce(lambda x, y: x * y, tensor.shape)
            rom_bytes += num_elements * size_per_weight

    ### ENTER STUDENT CODE ABOVE ###

    return rom_bytes


def estimate_ram(tensors: List[MyTensor], layers: List[MyLayer]):
    """Calculate the estimated number of bytes required to store model tensors in RAM.

    Arguments
    ---------
    tensors : list
        The tensors of the processed model (see definition of MyTensor type above)
    layers : list
        The layers of the processed model (see definition of MyLayer type above)

    Returns
    -------
    ram_bytes : int
        Estimated RAM usage given ideal memory planning (e.g. buffers can be shared to reduce the memory footprint)

    Assumptions
    -----------
    - Only fully-sequential model architectures are considered (e.g. no branches/parallel paths are allowed)
    - Only intermediate tensors (activations) are considered for RAM usage (no temporary workspace buffers are used in the layers)
    - During the operation of a single layer, all of its input and output tensors have to be available
    - In-place operations are not allowed (e.g. the input and output buffer of a layer can not be the same)
    - The input and output tensors of the whole model can also be considered for memory planning

    """

    ram_bytes = 0

    ### ENTER STUDENT CODE BELOW ###
    '''
    formula :
    Buffer Requirements : each layer requires two buffers for in/output
    Optimal Memory Planning: reuse buffers between layers as much as possiable
    
    1. cauculate input and output buffers for each layer
    2. find the maximum of all layer
    
    example :
    Layer       input               output              buffer
    Reshape     [1,1960] int8       [1,49,40,1] int8    1960+1960=3920
    Conv        [1,49,40,4] int8    [1,25,20,4] int8    1960+2000=3960
    Reshape     [1,25,20,4] int8    [1,2000] int8       2000+2000=4000
    FC          [1,2000] int8       [1,4] int8          2000+8=2008
    
    ram_bytes = max(3920,3960,4000,2008) = 4000 Byte
    '''
    #layer_buffer = 0
    for layer in layers:
        input_size = sum(reduce(lambda x, y: x * y, tensors[t].shape) * 1 for t in layer.inputs)
        output_size = sum(reduce(lambda x, y: x * y, tensors[t].shape) * 1 for t in layer.outputs)
        layer_buffer = input_size + output_size
        ram_bytes = max(ram_bytes, layer_buffer)

    ### ENTER STUDENT CODE ABOVE ###

    return ram_bytes
