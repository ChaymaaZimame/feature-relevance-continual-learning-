import torch
from methods.lrp_runner import LRPRunner


# Test: reduce_to_per_output_neuron (Conv)
def test_reduce_conv():
    runner = LRPRunner.__new__(LRPRunner)

    # Shape: (B=2, C=2, H=2, W=2)
    R = torch.tensor([
        [  # Batch 0
            [  # Kanal 0
                # W0   W1
                [1.0, -1.0],   # H=0
                [2.0, -2.0],   # H=1
                # abs → [1,1,2,2] → sum = 6
            ],
            [  # Kanal 1
                [3.0, -3.0],
                [4.0, -4.0],
                # abs → [3,3,4,4] → sum = 14
            ],
        ],
        [  # Batch 1
            [  # Kanal 0
                [0.0, -1.0],
                [-6.0, 0.0],
                # abs → [0,1,6,0] → sum = 7
            ],
            [  # Kanal 1
                [3.0, 3.0],
                [-4.0, -1.0],
                # abs → [3,3,4,1] → sum = 11
            ],
        ],
    ])

    # Danach mean:
    # Kanal 0: (6 + 7) / 2 = 6.5
    # Kanal 1: (14 + 11) / 2 = 12.5

    result = runner.reduce_to_per_output_neuron(R)

    # abs → sum → mean über Batch
    expected = torch.tensor([6.5, 12.5]) 
    
    # https://docs.pytorch.org/docs/main/generated/torch.equal.html
    if torch.equal(result, expected):
        print("reduce_conv bestanden!")
    else:
        print("reduce_conv nicht bestanden!")
    #print(assert torch.equal(result, expected))


# Test: reduce_to_per_output_neuron (Linear)
def test_reduce_linear():
    runner = LRPRunner.__new__(LRPRunner)

    # Shape: (B=2, C=2)
    R = torch.tensor([ 
        # K0   K1              
        [1.0, -2.0], # Batch 0
        # abs → [1,2]
        [-5.0, 2.0] # Batch 1
        # abs → [5,2]
        
    ])
    
    # Danach mean:
    # Kanal 0: (1 + 5) / 2 = 3
    # Kanal 1: (2 + 2) / 2 = 2

    result = runner.reduce_to_per_output_neuron(R)

    # abs → mean über Batch
    expected = torch.tensor([3.0, 2.0])
    
    if torch.equal(result, expected):
        print("reduce_linear bestanden!")
    else:
        print("reduce_linear nicht bestanden!")

    #assert torch.allclose(result, expected)


# Test: normalize
def test_normalize():
    runner = LRPRunner.__new__(LRPRunner)

    s = torch.tensor([6.0, 3.0])
    result = runner.normalize(s)

    # 6 + 3 = 9 → mean = 4.5
    # s / mean = [6/4.5, 3/4.5] = [1.333..., 0.666...]
    #expected = s / s.mean()
    expected = torch.tensor([1.3333333333333333333333333333333, 0.6666666666666666666666666666667])

    if torch.allclose(result, expected):
        print("normalize bestanden!")
    else:
        print("normalize nicht bestanden!")
    #assert torch.allclose(result, expected)


# Test: scale_for_grad (BN / Bias)
def test_scale_bn():
    runner = LRPRunner.__new__(LRPRunner)

    s = torch.tensor([1.0, 2.0])
    result = runner.scale_for_grad(s, "bn")
    
    expected = torch.tensor([1.0, 2.0])
    
    if torch.equal(result, expected):
        print("scale_bn bestanden!")
    else:
        print("scale_bn nicht bestanden!")

    assert result.shape == (2,)
    #assert torch.allclose(result, s)


# Test: scale_for_grad (Linear weights)
def test_scale_linear_w():
    runner = LRPRunner.__new__(LRPRunner)

    s = torch.tensor([1.0, 2.0])
    result = runner.scale_for_grad(s, "linear_w")
    
    expected = torch.tensor([[1.0], [2.0]])
    
    if torch.equal(result, expected):
        print("scale_linear_w bestanden!")
    else:
        print("scale_linear_w nicht bestanden!")

    assert result.shape == (2, 1)


# Test: scale_for_grad (Conv weights)
def test_scale_conv():
    runner = LRPRunner.__new__(LRPRunner)

    s = torch.tensor([1.0, 2.0])
    result = runner.scale_for_grad(s, "conv")

    expected = torch.tensor([[[[1.0]]], [[[2.0]]]])
    
    if torch.equal(result, expected):
        print("scale_conv bestanden!")
    else:
        print("scale_conv nicht bestanden!")
    
    assert result.shape == (2, 1, 1, 1)
    
    
    
if __name__ == "__main__":
    test_reduce_conv()
    test_reduce_linear()
    test_normalize()
    test_scale_bn()
    test_scale_linear_w()
    test_scale_conv()
