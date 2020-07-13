include("julia/lib8.jl"  )
include("./dataset/mnist.jl")
using PyCall
using PyPlot
pickle = pyimport("pickle")

module DeepConvNet

  export DeepConvNet_st
  mutable struct DeepConvNet_st
    params::Dict{String, Array}
    layers::Array{Any}
    last_layer
    grads::Dict{String, Array}
    predict
    loss
    gradient
    accuracy
  end

  struct ConvParam
    filter_num::Number
    filter_size::Number
    pad::Number
    stride::Number
  end

  function new(input_dim=(28, 28, 1), conv_params=[ConvParam(16, 3, 1, 1),ConvParam(16, 3, 1, 1),ConvParam(32, 3, 1, 1),ConvParam(32, 3, 2, 1),ConvParam(64, 3, 1, 1),ConvParam(64, 3, 1, 1),], hidden_size=50, output_size=10; numpymode=false)
    pre_node_nums = [[1*3*3] map(cp->cp.filter_num*cp.filter_size^2, conv_params)' [hidden_size]]
    weight_init_scales = sqrt.(2.0 ./ pre_node_nums)

    params = Dict{String, Array}()
    pre_channel_num = input_dim[end]
    for idx in 1:length(conv_params)
      cp = conv_params[idx]
      params["W$(idx)"] = weight_init_scales[idx] .* randn(cp.filter_size, cp.filter_size, pre_channel_num, cp.filter_num)
      params["b$(idx)"] = zeros(1, cp.filter_num)
      pre_channel_num = cp.filter_num
    end

    params["W7"] = weight_init_scales[7] .* randn(64*4*4, hidden_size)'
    params["b7"] = zeros(hidden_size)
    params["W8"] = weight_init_scales[8] .* randn(hidden_size, output_size)'
    params["b8"] = zeros(output_size)

    layers = []
    for idx in 1:6
      insert!(layers, length(layers)+1, Main.Convolution.new(params["W$(idx)"], params["b$(idx)"], conv_params[idx].stride, conv_params[idx].pad))
      insert!(layers, length(layers)+1, Main.Relu.new())
      if iszero(idx % 2)
        insert!(layers, length(layers)+1, Main.Pooling.new(2, 2, 2))
      end
    end
    insert!(layers, length(layers)+1, Main.Affine.new(params["W7"], params["b7"], numpymode))
    insert!(layers, length(layers)+1, Main.Relu.new())
    insert!(layers, length(layers)+1, Main.Dropout.new(0.5))
    insert!(layers, length(layers)+1, Main.Affine.new(params["W8"], params["b8"], numpymode))
    insert!(layers, length(layers)+1, Main.Dropout.new(0.5))


    self = DeepConvNet_st(
      params,
      layers,
      Main.SoftmaxWithLoss.new(),
      Dict(
        "W1" => [],
        "b1" => [],
        "W2" => [],
        "W2" => [],
        "W3" => [],
        "b3" => []),
     (x, t=false)->predict(self, x, t),
     (x, t)->loss(self, x, t),
     (x, t)->gradient(self, x, t),
     (x, t, b=500)->accuracy(self, x, t, b))

     return self
  end

  export load_param
  function load_param(self, params)
    Windices = [0 2 5 7 10 12 15 18] .+ 1
    for i in 1:length(Windices)
      layer_idx = Windices[i]
      #println(typeof(self.layers[layer_idx]))
      #println(layer_idx)
      self.layers[layer_idx].W = params["W$(i)"]
      self.layers[layer_idx].b = params["b$(i)"]
    end
  end

  export load_pickle
  function load_pickle(self, pdata)
    newpdata = Dict{String,Array}()
    for i in 1:8
      wk = "W$(i)"
      bk = "b$(i)"
      w = pdata[wk]
      b = pdata[bk]
      if ndims(w) == 2
        newpdata[wk] = w'
        newpdata[bk] = reshape(b, (length(b), 1))
      else
        newpdata[wk] = permutedims(w, (3, 4, 2, 1))
        newpdata[bk] = reshape(b, (1, length(b)))
      end
    end
    load_param(self, newpdata)
  end

  function predict(self::DeepConvNet_st, x::Array, train_flg=false)
    #println("\n\nPredict")
    for layer in self.layers
      #println(match(r"\.(\w+)\.", string(typeof(layer)))[1])
      #print("$(size(x))->")
      #print("$(sum(x))->")
      if isa(layer, Main.Dropout.Dropout_st)
        x = layer.forward(x, train_flg)
      else
        x = layer.forward(x)
      end
      #println(sum(x))
      #println(size(x))
    end
    return x
  end

  function loss(self::DeepConvNet_st, x::Array, t)
    y = self.predict(x, true)
    return self.last_layer.forward(y, t)
  end

  function gradient(self::DeepConvNet_st, x::Array, t)
    # forwad
    self.loss(x, t)

    # backward
    dout = 1
    dout = self.last_layer.backward(dout)

    #i=20
    #println("\nBackward")
    for layer in reverse(self.layers)
      #if i==14
        #dout = permutedims(dout, (2, 1, 3, 4))
      #end
      #println(match(r"\.(\w+)\.", string(typeof(layer)))[1])
      #print("$(sum(dout))->")
      dout = layer.backward(dout)
      #println(sum(dout))
      #i-=1
    end

    Windices = [0 2 5 7 10 12 15 18] .+ 1
    for i in 1:length(Windices)
      layer_idx = Windices[i]
      self.grads["W$(i)"] = self.layers[layer_idx].dW
      self.grads["b$(i)"] = self.layers[layer_idx].db
    end
    return self.grads
  end

  function accuracy(self::DeepConvNet_st, x, t, batch_size=500)
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
function unpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end

struct List
  train_loss_list::Array
  train_acc_list::Array
  test_acc_list::Array
end
function Train(opt, network, iters_num, batch_size, acc_batch_size, learning_rate, iter_per_epoch)
  adalist = List(zeros(iters_num), [], [])

  for i in 1:iters_num
    batch_mask = rand(1:train_size, batch_size);
    x_batch = x_train[:, :, :, batch_mask];
    t_batch = t_train[batch_mask, :];

    grads = network.gradient(x_batch, t_batch)

    opt.update(network.params, grads)

    loss_ = network.loss(x_batch, t_batch)
    adalist.train_loss_list[i] = loss_
    println(loss_)

    if i % iter_per_epoch == 0
      train_acc = network.accuracy(x_train, t_train, acc_batch_size)
      test_acc = network.accuracy(x_test, t_test, acc_batch_size)
      push!(adalist.train_acc_list, train_acc)
      push!(adalist.test_acc_list, test_acc)
      println("iter: $i loss: $loss_ train acc: $train_acc test acc: $test_acc")
    end
  end
  return adalist
end

# %% Inference
network = DeepConvNet.new(numpymode=true);

# %% load pickle
pdata = unpickle("./ch08/deep_convnet_params.pkl");
DeepConvNet.load_pickle(network, pdata)
tmp = "W7"
size(pdata[tmp])#[1:5, 1:5]
size(network.layers[16].W)#.W[1:5, 1:5]

# %% Load
(x_train, t_train), (x_test, t_test) = MNIST.load_mnist(one_hot_label=true, flatten=false);
train_size = size(x_train, 4)

# %% test
# forward check
size(network.layers[16].W)
n = 3
argmax(t_test[n, :])-1
size(network.layers[16].W)
size(network.layers[16].b)
network.layers[16].W[1:3, 1:3]'
size(network.layers[16].forward(ones(4,4,64,1)))
argmax(network.predict(x_test[:, :, :, n:n]))[1]-1
maximum(x_test[:, :, :, n:n])
sum(network.layers[19].x)
network.layers[18].dropout_ratio
imshow(x_test[:, :, 1, n])
test_acc = network.accuracy(x_test, t_test, 500)

# backward check

y=network.predict(x_train[:, :, :, 1:1])
network.last_layer.y
network.last_layer.t
network.loss(x_train[:, :, :, 1:1], t_train[1:1, :])
network.last_layer.backward(1)
sum(network.last_layer.backward(1)[[4 6], 1])
network.gradient(x_train[:, :, :, 1:1], t_train[1:1, :]);
size(network.layers[14].mask)
sum(network.layers[14].mask, dims=(1,2))
network.layers[18].dropout_ratio

# %% main
# %% parameter inits
iters_num = 10000
acc_batch_size = 500
batch_size = 100
learning_rate = 0.001
input_size = 784
hidden_size = 50
output_size = 10
iter_per_epoch = Int(max(train_size/batch_size, 1))

# %% Train func
opt = AdaGrad.new(learning_rate);
network = DeepConvNet.new();
adalist = Train(opt, network, iters_num, batch_size, acc_batch_size, learning_rate, iter_per_epoch)

# %% AdaGrad
using .AdaGrad
opt = AdaGrad.new(learning_rate);
network = DeepConvNet.new();
adalist = List(zeros(iters_num), [], [])

for i in 1:iters_num
  batch_mask = rand(1:train_size, batch_size);
  x_batch = x_train[:, :, :, batch_mask];
  t_batch = t_train[batch_mask, :];

  grads = network.gradient(x_batch, t_batch)

  opt.update(network.params, grads)

  loss_ = network.loss(x_batch, t_batch)
  #network.predict(x_batch[:, :, :, 1:1])
  #println("L=$(loss_)")
  adalist.train_loss_list[i] = loss_

  if i % iter_per_epoch == 0
    train_acc = network.accuracy(x_train, t_train, 1000)
    test_acc = network.accuracy(x_test, t_test, 1000)
    push!(adalist.train_acc_list, train_acc)
    push!(adalist.test_acc_list, test_acc)
    println("iter: $i loss: $loss_ train acc: $train_acc test acc: $test_acc")
  end
end

# %% save
using JLD2, FileIO
save("./dataset/8_AdaGrad_Wb.jld2", "params", network.params, "train_acc_list", adalist.train_acc_list, "test_acc_list", adalist.test_acc_list, "train_loss_list", adalist.train_loss_list)

# %% load weights
params = load("./dataset/8_AdaGrad_Wb.jld2", "params")
network = SimpleConvNet.new();
SimpleConvNet.load_param(network, params);
