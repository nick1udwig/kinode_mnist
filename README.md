# kinode_mnist

Proof-of-concept of [kinode-ml](https://github.com/hosted-fornet/kinode-ml) using MNIST with a Keras backend.

## Usage

Start `kinode-ml`.
Then, build and start `kinode_mnist`:

```bash
kit bs kinode_mnist
```

You should see somehting like:

```
output: [2.1645e-7, 5.189067e-6, 0.0012846275, 0.9980964, 1.017954e-9, 0.00017742209, 0.00042390113, 3.0028946e-8, 1.21770445e-5, 4.6000492e-9]
the given number was: 3
```

[Judge its accuracy for yourself!](https://github.com/EN10/KerasMNIST/blob/master/test3.png)

## Credits

Model and data from https://github.com/EN10/KerasMNIST

In particular:
* The model used is [TFKeras.h5](https://github.com/EN10/KerasMNIST/blob/master/TFKeras.h5) (see [here](https://github.com/hosted-fornet/kinode_mnist/blob/1a9aa7214eec1127667a6fd9933ca9b9020a1789/kinode_mnist/src/lib.rs#L14))
* The data used is [test3.png](https://github.com/EN10/KerasMNIST/blob/master/test3.png) (see [here](https://github.com/hosted-fornet/kinode_mnist/blob/1a9aa7214eec1127667a6fd9933ca9b9020a1789/kinode_mnist/src/lib.rs#L15))
* The run itself mirrors exactly [TFKpredict.py](https://github.com/EN10/KerasMNIST/blob/master/TFKpredict.py) (see [here](https://github.com/hosted-fornet/kinode_mnist/blob/1a9aa7214eec1127667a6fd9933ca9b9020a1789/kinode_mnist/src/lib.rs#L22-L30))
