
function im2col(input_data, filter_h, filter_w, stride=1, pad=0)
  H, W, C, N = size(input_data)
  out_h = Int(floor((H + 2*pad - filter_h)/stride + 1))
  out_w = Int(floor((W + 2*pad - filter_w)/stride + 1))

  img = input_data
  if pad != 0
    img = zeros(H+2*pad, W+2*pad, C, N)
    img[(1+pad):(H+pad), (1+pad):(W+pad), :, :] .= input_data
  end
  col = zeros(out_h, out_w, filter_h*filter_w, C, N)

  col_idx = 1
  for x in 1:filter_w
    x_max = x + stride*(out_w-1)
    for y in 1:filter_h
      y_max = y + stride*(out_h-1)
      col[:, :, col_idx, :, :] = img[y:stride:y_max, x:stride:x_max, :, :]
      col_idx+=1
    end
  end

  return reshape(permutedims(col,(1,2,5,3,4)), N*out_h*out_w, :)
end

function col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0)
  H, W, C, N = input_shape
  out_h = Int(floor((H + 2*pad - filter_h)/stride + 1))
  out_w = Int(floor((W + 2*pad - filter_w)/stride + 1))
  col = permutedims(reshape(col, (out_h, out_w, N, filter_h*filter_w, C)), (1,2,4,5,3))

  img = zeros(H + 2*pad + stride - 1, W + 2*pad + stride - 1, C, N)
  col_idx = 1
  for x in 1:filter_w
    x_max = x + stride*(out_w-1)
    for y in 1:filter_h
      y_max = y + stride*(out_h-1)
      img[y:stride:y_max, x:stride:x_max, :, :] .+= col[:, :, col_idx, :, :]
      col_idx+=1
    end
  end

  return img[(pad+1):(H+pad), (pad+1):(W+pad), :, :]
end

function Wx2im(Wx, OH, OW, FN, N)
  return permutedims(reshape(Wx, (OH, OW, N, FN)), (1, 2, 4, 3))
end

function im2Wx(im, FN)
  return reshape(permutedims(im, (1, 2, 4, 3)), (:, FN))
end

function W2col(W)
  FH,FW,C,FN = size(W)
  return reshape(W, (FH*FW*C, FN))
end

function col2W(colW, FH, FW, C, FN)
  return reshape(colW, (FH, FW, C, FN))
end

module Convolution
  mutable struct Convolution_st
    W::Array
    b::Array
    stride::Number
    pad::Number
    x::Array
    col::Array
    col_W::Array
    db::Array
    dW::Array
    forward
    backward
  end

  function new(W, b, stride=1, pad=0)
    tmp_st = Convolution_st(W, b, stride, pad,  [0], [0], [0], [0], [0], (x)->forward(tmp_st, x), (x)->backward(tmp_st, x))
    return tmp_st
  end

  function forward(self::Convolution_st, x)
    FH, FW, C, FN = size(self.W)
    H, W, C, N = size(x)
    out_h = Int(floor((H + 2*self.pad - FH)/self.stride + 1))
    out_w = Int(floor((W + 2*self.pad - FW)/self.stride + 1))

    col = Main.im2col(x, FH, FW, self.stride, self.pad)
    col_W = Main.W2col(self.W)

    self.x = x;
    self.col = col
    self.col_W = col_W

    return Main.Wx2im(col*col_W .+ self.b, out_h, out_w, FN, N)
  end

  function backward(self::Convolution_st, dout)
    FH, FW, C, FN = size(self.W)
    dout = Main.im2Wx(dout, FN)

    self.db = sum(dout, dims=1)
    self.dW = Main.col2W(self.col' * dout, FH, FW, C, FN)

    dcol = dout*self.col_W'
    return Main.col2im(dcol, size(self.x), FH, FW, self.stride, self.pad)
  end
end

module Pooling
  mutable struct Pooling_st
    x::Array
    arg_max::Array
    pool_h::Number
    pool_w::Number
    out_h::Number
    out_w::Number
    stride::Number
    pad::Number
    forward
    backward
  end

  function new(pool_h, pool_w, stride=1, pad=0)
    self = Pooling_st([], [], pool_h, pool_w, 0, 0, stride, pad, (x)->forward(self, x), (dout)->backward(self, dout))
    return self
  end

  function forward(self::Pooling_st, x)
    H, W, C, N = size(x)
    self.out_h = Int(floor((H + 2*self.pad - self.pool_h)/self.stride + 1))
    self.out_w = Int(floor((W + 2*self.pad - self.pool_w)/self.stride + 1))

    col = reshape(Main.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)', (self.pool_h*self.pool_w, :))

    self.x = x
    self.arg_max = (argmax.(eachcol(col))' .== 1:size(col, 1))
    return permutedims(reshape(maximum(col, dims=1), (C, self.out_h, self.out_w, N)), (2, 3, 1, 4))#, col
  end

  function backward(self::Pooling_st, dout)
    sizex = size(self.x)
    H, W, C, N = sizex
    pool_hw = self.pool_h*self.pool_w
    out_hw  = self.out_h*self.out_w

    dout = reshape(dout, (out_hw, 1, C, N))
    W = permutedims(reshape(self.arg_max, (pool_hw, C, out_hw, N)), (3,1,2,4))

    dcol = reshape(permutedims(dout .* W, (1,4,2,3)), (:, pool_hw*C))
    return Main.col2im(dcol, sizex, self.pool_h, self.pool_w, self.stride, self.pad)#, dcol, dout, W
  end

end

module SimpleConvNet
  layerOrder = ["Conv1", "Relu1", "Pool1", "Affine1", "Relu2", "Affine2"]

  export SimpleConvNet_st
  mutable struct SimpleConvNet_st
    params::Dict{String, Array}
    layers::Dict{String, Any}
    last_layer
    grads::Dict{String, Array}
    predict
    loss
    gradient
    accuracy
  end

  function new(input_dim=(28, 28, 1), conv_param=Dict("filter_num" => 30, "filter_size" => 5, "pad" => 0, "stride" => 1), hidden_size=100, output_size=10, weight_init_std=0.01)

    filter_num = conv_param["filter_num"]
    filter_size = conv_param["filter_size"]
    filter_pad = conv_param["pad"]
    filter_stride = conv_param["stride"]
    input_size = input_dim[1]
    conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    pool_output_size = Int(floor(filter_num * (conv_output_size/2)*(conv_output_size/2)))

    W1 = weight_init_std .* randn(filter_size, filter_size, input_dim[3], filter_num)
    b1 = zeros(1, filter_num)
    W2 = weight_init_std .* randn(pool_output_size, hidden_size)'
    b2 = zeros(hidden_size)
    W3 = weight_init_std .* randn(hidden_size, output_size)'
    b3 = zeros(output_size)
    self = SimpleConvNet_st(
      Dict(
        "W1" => W1,
        "b1" => b1,
        "W2" => W2,
        "b2" => b2,
        "W3" => W3,
        "b3" => b3),
      Dict(
        "Conv1" => Main.Convolution.new(W1,
                                        b1,
                                        conv_param["stride"],
                                        conv_param["pad"]),
        "Relu1"   => Main.Relu.new(),
        "Pool1"   => Main.Pooling.new(2, 2, 2),
        "Affine1" => Main.Affine.new(W2, b2),
        "Relu2"   => Main.Relu.new(),
        "Affine2" => Main.Affine.new(W3, b3)),
      Main.SoftmaxWithLoss.new(),
      Dict(
        "W1" => [],
        "b1" => [],
        "W2" => [],
        "W2" => [],
        "W3" => [],
        "b3" => []),
     (x)->predict(self, x),
     (x, t)->loss(self, x, t),
     (x, t)->gradient(self, x, t),
     (x, t, b=500)->accuracy(self, x, t, b))

     return self
  end

  export load_param
  function load_param(self, params)
    self.layers["Conv1"].W = params["W1"]
    self.layers["Conv1"].b = params["b1"]
    self.layers["Affine1"].W = params["W2"]
    self.layers["Affine1"].b = params["b2"]
    self.layers["Affine2"].W = params["W3"]
    self.layers["Affine2"].b = params["b3"]
  end

  function predict(self::SimpleConvNet_st, x::Array)
    for lname in SimpleConvNet.layerOrder
      layer = self.layers[lname]
      #println("lname=$(lname), x=$(size(x))")
      x = layer.forward(x)
    end
    return x
  end

  function loss(self::SimpleConvNet_st, x::Array, t)
    y = self.predict(x)
    return self.last_layer.forward(y, t)
  end

  function gradient(self::SimpleConvNet_st, x::Array, t)
    # forwad
    self.loss(x, t)

    # backward
    dout = 1
    dout = self.last_layer.backward(dout)

    for lname in reverse(layerOrder)
      layer = self.layers[lname]
      dout = layer.backward(dout)
    end

    # 設定
    self.grads["W1"] = self.layers["Conv1"].dW
    self.grads["b1"] = self.layers["Conv1"].db
    self.grads["W2"] = self.layers["Affine1"].dW
    self.grads["b2"] = self.layers["Affine1"].db
    self.grads["W3"] = self.layers["Affine2"].dW
    self.grads["b3"] = self.layers["Affine2"].db
    return self.grads
  end
  function accuracy(self::SimpleConvNet_st, x, t, batch_size=500)
    t = argmax.(eachrow(t))
    acc = 0.0

    for i in 1:(Int(size(x, 4) / batch_size))
      tx = x[:, :, :, (i-1)*batch_size+1:i*batch_size]
      tt = t[(i-1)*batch_size+1:i*batch_size]'
      y = self.predict(tx)
      y = reshape(argmax.(eachcol(y)), (1, size(y, 2)))
      acc += sum(y .== tt)
    end

    return acc / size(x, 4)
  end
end


# %% draft
out_h = 2
out_w = 2
filter_h = 2
filter_w = 2
col = zeros(out_h*out_w, filter_h, filter_w);
img = [1:3 4:6 7:9]

col[1, :, :] .= img[1:2, 1:2];
col[2, :, :] .= img[2:3, 1:2];
col[3, :, :] .= img[1:2, 2:3];
col[4, :, :] .= img[2:3, 2:3];
reshape(col, out_h*out_w, :)

# %% test
N = 2
C = 2
H = 3
W = 3
filter_h = 2
filter_w = 2
img2 = zeros(H,W, C, N)
img2[:, :, 1,1] .= img
img2[:, :, 2,1] .= img.+9
img2[:, :, 1,2] .= -img
img2[:, :, 2,2] .= -(img.+9)

col2 = im2col(img2, filter_h, filter_w)

col2imed = col2im(col2, (H,W,C,N), filter_h, filter_w)

# %% 重み
N = 2
H = 3
W = 3
C = 1
FN = 2
filter_h = 3
filter_w = 3
S = 1
P = 1
out_h = Int(floor((H+2*P-filter_h)/S)+1)
out_w = Int(floor((W+2*P-filter_w)/S)+1)
Wf = zeros(filter_h, filter_w, C, FN);
Wf[2, 2, 1, 1] = 1;
#Wf[:, :, 2, 1] = 0.11 .* ones(3,3);
#Wf[2, 2, 2, 1] = 0.12;
Wf[:, :, 1, 2] = -0.11 .* ones(3,3);
Wf[2, 2, 1, 2] = 1.88;
#Wf[:, :, 2, 2] = [-1 0 1; -2 0 2; -1 0 1];
Wf

colWf = W2col(Wf);
img3 = zeros(H,W, C, N);
img3[:, :, 1, 1] = img;
img3[:, :, 1, 2] = img .+ 9;
img3
col3 = im2col(img3,filter_h,filter_w, S,P)
size(col3)
affine = col3*colWf

img3_ = Wx2im(affine, out_h, out_w, FN, N)
size(img3_)
im2Wx(img3_, FN)

conv = Convolution.new(Wf, 0, S, P);
conv.forward(img3)
dx = conv.backward(ones(size(img3_)))
conv.dW
conv.db

# %%
using Images, ImageIO
using PyPlot

lena = load("dataset/lena_gray.png");
lenaAr = channelview(lena);
#lenaF = Array{Float32, 2}(lenaAr[1, :, :]);
imshow(lenaAr, cmap="gray")

# %% conv lena
H, W = size(lenaAr)
FN = 4
C = 1
FH = 3
FW = 3
N = 2
S = 1
P = 1
out_h = Int(floor((W+2*P-FH)/S)+1)
out_w = Int(floor((W+2*P-FW)/S)+1)
img4 = zeros(H,W, C, N);
img4[:, :, 1, 1] = lenaAr
img4[:, :, 1, 2] = lenaAr'
Wf2 = zeros(FH, FW, C, FN);
# Identity
Wf2[2, 2, 1, 1] = 1;
# avg
Wf2[:, :, 1, 2] = 0.11 .* ones(3,3);
Wf2[2, 2, 1, 2] = 0.12;
# unsharp
Wf2[:, :, 1, 3] = -0.11 .* ones(3,3);
Wf2[2, 2, 1, 3] = 1.88;
# edge
Wf2[:, :, 1, 4] = 0.5 .* [-1 0 1; -2 0 2; -1 0 1]';
Wf2[2, 2, 1, 4] = 1;
Wf2
conv = Convolution.new(Wf2, 0)
img4_ = conv.forward(img4)
size(img4_)
imshow(img4_[:, :, 1, 1], cmap="gray")
imshow(img4_[:, :, 2, 1], cmap="gray")
imshow(img4_[:, :, 3, 1], cmap="gray")
imshow(img4_[:, :, 4, 1], cmap="gray")
imshow(img4_[:, :, 1, 2], cmap="gray")
imshow(img4_[:, :, 2, 2], cmap="gray")
imshow(img4_[:, :, 3, 2], cmap="gray")
imshow(img4_[:, :, 4, 2], cmap="gray")

# %% MaxPool test
## %% init
N = 2
C = 2
H = 3
W = 3
pool_h = 3
pool_w = 3
img = [1:3 4:6 7:9]
img5 = zeros(H,W, C, N);
img5[:, :, 1,1] .= img;
img5[:, :, 2,1] .= img.+9;
img5[:, :, 1,2] .= -img;
img5[:, :, 2,2] .= -(img.+9);
img5

## %% forward
pool = Pooling.new(pool_h, pool_w);
#Main.im2col(img5, pool_h, pool_w, 1, 0)
img5[1, 2, 1, 1] = 10;
img5
img5_,col = pool.forward(img5);
img5_
col
pool.arg_max

## %% backward

dout = ones(size(img5_));
#dout[:,:, 1,1] = [1 2; 3 4];
#dout[:,:, 2,1] = 4 .+ [1 2; 3 4];
#dout[:,:, 1,2] = -dout[:,:, 1,1];
#dout[:,:, 2,2] = -dout[:,:, 2,1];
dout

dimg5, dcol, dout, W = pool.backward(dout)
dimg5

dcol# = reshape(permutedims(dout .* W, (1,4,2,3)), (:, 2*2*C))
dout .* W
dout
W

# %% Max pool lena
H, W, C, N = size(img4_)
C = 1
pool_h = 2
pool_w = 2
S = 2
P = 0
pool = Pooling.new(pool_h, pool_w, S, P);
img7 = lenaAr
img7 = reshape(lenaAr.*255, (256, 256, C, 1));
img7_, _ = pool.forward(img7);
dimg7_,_,_,_ = pool.backward(img7_ .* 0 .+ 1);
img7[1:10, 1:6, 1, 1]
img7_[1:10, 1:6, 1, 1]
dimg7_[1:10, 1:6, 1, 1]
imshow(img7_[:,:,1,1], cmap="gray")
imshow(dimg7_[:,:,1,1], cmap="gray")

img6 = img4_;
img6_, _ = pool.forward(img6);
dimg6_, _, _, _ = pool.backward(img6_ .* 0 .+ 1);

size(img6)
size(img6_)
imshow(img6_[:, :, 1, 1], cmap="gray")
imshow(img6_[:, :, 2, 1], cmap="gray")
imshow(img6_[:, :, 3, 1], cmap="gray")
imshow(img6_[:, :, 4, 1], cmap="gray")
imshow(dimg6_[:, :, 1, 1], cmap="gray")
imshow(dimg6_[:, :, 2, 1], cmap="gray")
imshow(dimg6_[:, :, 3, 1], cmap="gray")
imshow(dimg6_[:, :, 4, 1], cmap="gray")

# %% main

# %% imports
using PyPlot
include("./dataset/mnist.jl")
include("julia/lib7.jl"  )
struct List
  train_loss_list::Array
  train_acc_list::Array
  test_acc_list::Array
end

# %% Load
(x_train, t_train), (x_test, t_test) = MNIST.load_mnist(one_hot_label=true, normalize=true, flatten=false);
train_size = size(x_train, 4)

# %% test
n = 2
argmax(t_test[n, :])-1
argmax(network.predict(x_test[:, :, :, n:n]))[1]-1
imshow(x_test[:, :, 1, n])

#params = network.params
batch_size = 500*3
network = SimpleConvNet.new();
SimpleConvNet.load_param(network, params);
acc = network.accuracy(x_test[:, :, :, 1:batch_size], t_test[1:batch_size, :])

# %% parameter inits
iters_num = 10000
batch_size = 2000
learning_rate = 0.1
input_size = 784
hidden_size = 50
output_size = 10
iter_per_epoch = Int(max(train_size/batch_size, 1))

# %% AdaGrad
using .AdaGrad
opt = AdaGrad.new(learning_rate);
network = SimpleConvNet.new();
adalist = List(zeros(iters_num), [], [])

for i in 1:iters_num
  batch_mask = rand(1:train_size, batch_size);
  x_batch = x_train[:, :, :, batch_mask];
  t_batch = t_train[batch_mask, :]';

  grads = network.gradient(x_batch, t_batch)

  opt.update(network.params, grads)

  loss_ = network.loss(x_batch, t_batch)
  adalist.train_loss_list[i] = loss_

  if i % iter_per_epoch == 0
    train_acc = network.accuracy(x_train, t_train, 2000)
    test_acc = network.accuracy(x_test, t_test, 2000)
    push!(adalist.train_acc_list, train_acc)
    push!(adalist.test_acc_list, test_acc)
    println("iter: $i loss: $loss_ train acc: $train_acc test acc: $test_acc")
  end
end

# %% save
using JLD2, FileIO
save("./dataset/7_AdaGrad_Wb.jld2", "params", network.params, "train_acc_list", adalist.train_acc_list, "test_acc_list", adalist.test_acc_list, "train_loss_list", adalist.train_loss_list)

# %%
network = SimpleConvNet.new();
size()
train_acc = network.accuracy(x_train, t_train)
network.predict(x_batch)

Int(60000/100)*100
(3-1)*100+1
3*100
