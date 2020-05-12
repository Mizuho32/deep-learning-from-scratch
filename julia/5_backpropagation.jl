# %% 5.4 単純なレイヤの実装
module BP
  export MulLayer
  mutable struct MulLayer
    x::Number
    y::Number
  end
  function MulLayer()
    return MulLayer(nothing, nothing)
  end
  function forward(self::MulLayer, x, y)
    self.x = x
    self.y = y
    out = x * y

    return out
  end
  function backward(self::MulLayer, dout)
    dx = dout * self.y
    dy = dout * self.x

    return dx, dy
end

  export AddLayer
  mutable struct AddLayer
    x::Number
    y::Number
  end
  function AddLayer()
    return AddLayer(0, 0)
  end
  function forward(self::AddLayer, x, y)
    self.x = x
    self.y = y
    out = x + y

    return out
  end
  function backward(self::AddLayer, dout)
    dx = dout * 1
    dy = dout * 1

    return dx, dy
  end

  export ReluLayer
  mutable struct ReluLayer
    mask::Array
  end
  function ReluLayer()
    return ReluLayer(BitArray(zeros(0, 0)))
  end
  function forward(self::ReluLayer, x)
    #println("x $(size(x)) mask $(size(self.mask))")
    self.mask = (x .<=  0)
    out = copy(x)
    out[self.mask] .= 0

    return out
  end
  function backward(self::ReluLayer, dout)
    dx = copy(dout)
    dx[self.mask] .= 0

    return dx
  end

  export SigmoidLayer
  mutable struct SigmoidLayer
    out::Array
  end
  function SigmoidLayer(dim)
    return SigmoidLayer(zeros(dim, 1))
  end
  function forward(self::SigmoidLayer, x)
    self.out = 1 ./ (1 .+ exp.(-x))

    return self.out
  end
  function backward(self::SigmoidLayer, dout)
    dx = dout .* (1.0 .- self.out) .* self.out

    return dx
  end


  export AffineLayer
  mutable struct AffineLayer
    W::Array
    b::Array
    x::Array
    dW::Array
    db::Array
  end
  function AffineLayer(W, b)
    return AffineLayer(W, b, zeros(size(W, 2), 0), 0 .*W, 0 .*b)
  end
  function forward(self::AffineLayer, x)
    self.x = x

    return self.W * x .+ self.b
  end
  function backward(self::AffineLayer, dout)
    dx = self.W' * dout
    self.dW .= dout * self.x'
    #println("db $(size(self.db)) sum(dout) $(size(sum(dout, dims=2)))")
    self.db .= vec(sum(dout, dims=2))

    return dx
  end

  export SoftmaxWithLossLayer
  mutable struct SoftmaxWithLossLayer
    loss::Number
    y::Array
    t::Array
  end
  function SoftmaxWithLossLayer()
    return SoftmaxWithLossLayer(0, zeros(0, 0), zeros(0, 0))
  end
  function forward(self::SoftmaxWithLossLayer, x, t)
    self.t = t
    self.y = softmax(x, 1)
    self.loss = cross_entropy_error(self.y, self.t)

    return self.loss
  end
  function backward(self::SoftmaxWithLossLayer, dout)
    dx = (self.y .- self.t) ./ size(dout, 2)

    return dx
  end
  function softmax(a, dims)
    c = maximum(a)
    return exp.(a .- c) ./ sum(exp.(a .- c), dims=dims)
end
  function cross_entropy_error(y, t)
    batch_size = size(y, 2)
    return -sum(t .* log.(y .+ 1e-7))/batch_size
  end
end

# %% 5.4.1 乗算レイヤの実装
using .BP

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = BP.MulLayer()
mul_tax_layer   = BP.MulLayer()

apple_price = BP.forward(mul_apple_layer, apple, apple_num)
price = BP.forward(mul_tax_layer, apple_price, tax)

println(price)

dprice = 1
dapple_price, dtax = BP.backward(mul_tax_layer, dprice)
dapple, dapple_num = BP.backward(mul_apple_layer, dapple_price)

println(dapple)
println(dapple_num)
println(dtax)

# %% 5.4.2 加算レイヤの実装

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = BP.MulLayer()
mul_orange_layer = BP.MulLayer()
add_apple_orange_layer = BP.AddLayer()
mul_tax_layer   = BP.MulLayer()

apple_price = BP.forward(mul_apple_layer, apple, apple_num)
orange_price = BP.forward(mul_orange_layer, orange, orange_num)
all_price = BP.forward(add_apple_orange_layer, apple_price, orange_price)
price = BP.forward(mul_tax_layer, all_price, tax)


dprice = 1
dall_price, dtax = BP.backward(mul_tax_layer, dprice)
dapple_price, dorange_price = BP.backward(add_apple_orange_layer, dall_price)
dorange, dorange_num = BP.backward(mul_orange_layer, dorange_price)
dapple, dapple_num = BP.backward(mul_apple_layer, dapple_price)

println(price)

println(dapple)
println(dapple_num)
println(dorange)
println(dorange_num)
println(dtax)


# %% 5.5 活性化関数レイヤの実装
relu_layer = BP.ReluLayer()

input = [1 -1 0; -1 1 0.1]';
BP.forward(relu_layer, input)
BP.backward(relu_layer, ones(3, 3))


sigmoid_layer = BP.SigmoidLayer(2)

BP.forward(sigmoid_layer, input)
BP.backward(sigmoid_layer, [1; 1])

N = 2
W = [0 0 0; 10 10 10]';
b = [1 2 3]';
affine_layer = BP.AffineLayer(W,b)

x = [2 3; 4 5]';
a = BP.forward(affine_layer, x)
BP.backward(affine_layer, [1 2 3; 4 5 6]')


t = [0 0 1; 0 1 0]';
softmax_layer = BP.SoftmaxWithLossLayer()
BP.forward(softmax_layer, a, t)
BP.backward(softmax_layer, [1 1 1; 1 1 1]')
softmax_layer.y
softmax_layer.t


# %% 5.7 逆誤差伝播の実装
module TLN2 # 2 Layer Net
  using ..BP
  layerOrder = ["Affine1" "Relu1" "Affine2"]

  params = Dict()
  layers = Dict()
  grads_n = Dict()
  grads_b = Dict()
  o = Dict()

  function init(input_size, hidden_size, output_size, w_init_std = 0.01)
    params["W1"] = w_init_std .* randn(input_size, hidden_size)'
    params["b1"] = zeros(hidden_size)
    params["W2"] = w_init_std .* randn(hidden_size, output_size)'
    params["b2"] = zeros(output_size)

    layers["Affine1"] = BP.AffineLayer(params["W1"], params["b1"])
    layers["Relu1"]   = BP.ReluLayer()
    layers["Affine2"] = BP.AffineLayer(params["W2"], params["b2"])
    o["lastLayer"] = BP.SoftmaxWithLossLayer()
    return;
  end

  function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
  end

  function softmax(a, dims)
    c = maximum(a)
    return exp.(a .- c) ./ sum(exp.(a .- c), dims=dims)
  end

  function predict(W1, W2, b1, b2, x)
    a1 = W1*x .+ b1
    z1 = max.(0, a1)
    #print("$z1")
    a2 = W2*z1 .+ b2
    return softmax(a2, 1)
  end

  function predict(x)
    W1, W2 = params["W1"], params["W2"]
    b1, b2 = params["b1"], params["b2"]

    return predict(W1, W2, b1, b2, x)
  end

  function predict2(x)
    for layername in layerOrder
      layer = layers[layername]
      #print("$layername $(size(x))->")
      x = BP.forward(layer, x)
      #print("$(size(x))")
      #if layername == "Relu1"
      #  print("$x")
      #end
    end

    return x
  end

  function loss(W1, W2, b1, b2, x, t)
    y = predict(W1, W2, b1, b2, x)
    return BP.cross_entropy_error(y, t)
  end

  function loss(x, t)
    W1, W2 = params["W1"], params["W2"]
    b1, b2 = params["b1"], params["b2"]
    return loss(W1, W2, b1, b2, x, t)
  end

  function loss2(x, t)
    y = predict2(x)
    return BP.forward(o["lastLayer"], y, t)
  end

  function accuracy(x, t)
    y = predict2(x)
    y = map(r->r[1], argmax(y, dims=1))
    t = map(r->r[1], argmax(t, dims=1))

    return sum(y .== t) / size(x, 2)
  end

  function numerical_diff(f, x)
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)
  end

  function numerical_gradient(f, x)
    grad = 0*x;
    for i in 1:length(x)
      x_ = copy(x)
      f_ = function(xvar)
        x_[i] = xvar
        return f(x_)
      end
      grad[i] = numerical_diff(f_, x[i])
    end
    return grad
  end

  function calc_grad(x, t)
    W1, W2 = params["W1"], params["W2"]
    b1, b2 = params["b1"], params["b2"]

    grads_n["W1"] = numerical_gradient(w->loss(w, W2, b1, b2, x,t), W1)
    grads_n["b1"] = numerical_gradient(w->loss(W1, W2, w, b2, x,t), b1)
    grads_n["W2"] = numerical_gradient(w->loss(W1, w, b1, b2, x,t), W2)
    grads_n["b2"] = numerical_gradient(w->loss(W1, W2, b1, w, x,t), b2)

    return grads_n
  end

  function gradient(x, t)
    loss2(x, t)

    dout = ones(size(t))
    dout = BP.backward(o["lastLayer"], dout)

    for layername in reverse(layerOrder, dims=2)
      #print("$layername $(size(dout))->")
      layer = layers[layername]
      dout = BP.backward(layer, dout)
      #print("$(size(dout))")
    end

    grads_b["W1"] = layers["Affine1"].dW
    grads_b["b1"] = layers["Affine1"].db
    grads_b["W2"] = layers["Affine2"].dW
    grads_b["b2"] = layers["Affine2"].db

    return grads_b
  end

end

# %% 5.7.3 誤差逆伝播の勾配確認
include("./dataset/mnist.jl")
using Statistics
using Dates
#%% load data
(x_train, t_train), (x_test, t_test) = MNIST.load_mnist(one_hot_label=true, normalize=true);
# %% main

iters_num = 1
batch_size = 100
learning_rate = 0.1
hid = 50

TLN2.init(784, hid, 10)

x_batch = x_train[1:batch_size, :]';
t_batch = t_train[1:batch_size, :]';

TLN2.predict(x_batch)
TLN2.softmax(TLN2.predict2(x_batch), 1)
TLN2.loss(x_batch, t_batch)

grad_numerical = TLN2.calc_grad(x_batch, t_batch)
grad_backprop  = TLN2.gradient(x_batch, t_batch)

for key in keys(grad_numerical)
  diff = mean(grad_backprop[key] .- grad_numerical[key])
  println("$key:$diff")
end
# %% 5.7.4 誤差逆伝播法を使った学習

train_loss_list = []
train_acc_list = []
test_acc_list = []
train_size = size(x_train, 1)

iters_num = 10000
batch_size = 100
learning_rate = 0.1
hid = 50

iter_per_epoch = Int(max(train_size/batch_size, 1))

TLN2.init(784, hid, 10)
start_time = now()

for i in 1:iters_num
  batch_mask = rand(1:train_size, batch_size);
  x_batch = x_train[batch_mask, :]';
  t_batch = t_train[batch_mask, :]';

  TLN2.softmax(TLN2.predict2(x_batch), 1)
  grad = TLN2.gradient(x_batch, t_batch)

  for key in keys(grad)
    TLN2.params[key] .-= learning_rate .* grad[key]
    #                ↑これがないとBP内の構造体の重み変数が更新されない。参照型怖い
  end

  loss_ = TLN2.loss(x_batch, t_batch)
  push!(train_loss_list, loss_)

  if i % iter_per_epoch == 0
    train_acc = TLN2.accuracy(x_train', t_train')
    test_acc = TLN2.accuracy(x_test', t_test')
    push!(train_acc_list, train_acc)
    push!(test_acc_list, test_acc)
    println("iter: $i loss: $loss_ train acc: $train_acc test acc: $test_acc")
  end
end

end_time = now()
println("Delta: $(end_time-start_time)")

# %% View
using PyPlot

# %% rates
fig, ax1 = subplots();
ax1.grid()
ax1.plot(1:length(train_loss_list), train_loss_list, label="loass")
ax2 = ax1.twinx()
ax2.plot(range(iter_per_epoch, length=length(train_acc_list), step=iter_per_epoch), train_acc_list, linestyle="-.", color="orange", linewidth=1, label="train")
ax2.plot(range(iter_per_epoch, length=length(test_acc_list), step=iter_per_epoch), test_acc_list, linestyle="-.", color="red", linewidth=1, label="test")
#ax2.legend(loc="center right")
fig.legend(loc="center")
fig.show()

# %% main

i = 1
x, t = x_test[i:i, :]', argmax(t_test[i, :])-1;
println(t)
y = map(c->c[1], argmax(TLN2.predict2(x), dims=1))[1]-1
img = reshape(x, (28, 28))';
label = t;
imshow(img, label=t)

# %% save
using JLD2, FileIO
save("./dataset/5_Wb.jld2", "params", TLN2.params, "train_acc_list", train_acc_list, "test_acc_list", test_acc_list, "train_loss_list", train_loss_list)
