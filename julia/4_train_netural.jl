# %% 4.2 損失関数

function cross_entropy_error(y, t)
  batch_size = size(y, 2)
  return -sum(t .* log.(y .+ 1e-7))/batch_size
end

t = ( 1:10 .== [3 3]);
y = [0.1 0.05 0.6 0.0 0.5 0.1 0.0 0.1 0.0 0.0; 0.1 0.05 0.1 0.0 0.05 0.1 0.0 0.6 0.5 0.0]';
cross_entropy_error(y, t)

#%% 4.3 数値微分

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

function f1(x)
  return 0.01*x^2 + 0.1*x
end

function f2(x)
  return sum(x.^2)
end

# %% main
numerical_diff(f1, 5)
numerical_diff(f1, 10)

numerical_diff(x0->f2([x0; 4]), 3)
numerical_diff(x1->f2([3; x1]), 4)

numerical_gradient(f2, [3.0; 4.0])
numerical_gradient(f2, [0.0; 2.0])
numerical_gradient(f2, [3.0; 0.0])

# %% 4.4.1 勾配法
function gradient_descent(f, x0, lr=0.01, step_num=100)
  x = x0
  for i in 1:step_num
    grad = numerical_gradient(f, x)
    x   -= lr*grad
  end
  return x
end

# %% main
gradient_descent(f2, [-3.0; 4.0], 0.1)
# %% 4.4.2 ニューラルネットワークに対する勾配

function init()
  return randn(2, 3)'
end

function predict(W, x)
  return W*x
end

function softmax(a)
  c = maximum(a)
  return exp.(a .- c) ./ sum(exp.(a .- c))
end

function loss(W, x, t)
  z = predict(W, x)
  y = softmax(z)
  loss_ = cross_entropy_error(y, t)

  return loss_
end

# %% main
W = init()
W = [ 0.47355232 0.9977393 0.84668094; 0.85557411 0.03563661 0.69422093]'
x = [0.6; 0.9]
p = predict(W, x)
argmax(p)
t = [0 0 1]'
loss(W, x, t)

f = w->loss(w,x,t)
dW = numerical_gradient(f, W)


# %% 4.5 学習アルゴリズムの実装
module TLN # 2 Layer Net
  params = Dict()
  grads = Dict()

  function init(input_size, hidden_size, output_size, w_init_std = 0.01)
    params["W1"] = w_init_std .* randn(input_size, hidden_size)'
    params["b1"] = zeros(hidden_size)
    params["W2"] = w_init_std .* randn(hidden_size, output_size)'
    params["b2"] = zeros(output_size)
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
    z1 = sigmoid(a1)
    a2 = W2*z1 .+ b2
    return softmax(a2, 1)
  end

  function predict(x)
    W1, W2 = params["W1"], params["W2"]
    b1, b2 = params["b1"], params["b2"]

    return predict(W1, W2, b1, b2, x)
  end

  function loss(W1, W2, b1, b2, x, t)
    y = predict(W1, W2, b1, b2, x)
    return Main.cross_entropy_error(y, t)
  end

  function loss(x, t)
    W1, W2 = params["W1"], params["W2"]
    b1, b2 = params["b1"], params["b2"]
    return loss(W1, W2, b1, b2, x, t)
  end

  function accuracy(x, t)
    y = predict(x)
    y = map(r->r[1], argmax(y, dims=1))
    t = map(r->r[1], argmax(t, dims=1))

    return sum(y .== t) / size(x, 2)
  end

  function calc_grad(x, t)
    W1, W2 = params["W1"], params["W2"]
    b1, b2 = params["b1"], params["b2"]

    grads["W1"] = Main.numerical_gradient(w->loss(w, W2, b1, b2, x,t), W1)
    grads["b1"] = Main.numerical_gradient(w->loss(W1, W2, w, b2, x,t), b1)
    grads["W2"] = Main.numerical_gradient(w->loss(W1, w, b1, b2, x,t), W2)
    grads["b2"] = Main.numerical_gradient(w->loss(W1, W2, b1, w, x,t), b2)

    return grads
  end
end

# %% main
hid = 10
TLN.init(784, hid, 10)
x = rand(784, hid);
y = TLN.predict(x);
t = rand(10, hid);

TLN.calc_grad(x, t)
size(TLN.grads["W2"])
size(TLN.params["W2"])

TLN.loss(x, t)
TLN.accuracy(x, t)

# %% 4.5.2 ミニバッチ学習の実装
include("./dataset/mnist.jl")

#%% load data
(x_train, t_train), (x_test, t_test) = MNIST.load_mnist(one_hot_label=true, normalize=true, );
t_train[1:3, :]
# %% main
train_loss_list = []
train_size = size(x_train, 1)

iters_num = 1
batch_size = 10
learning_rate = 0.1
hid = 50

TLN.init(784, hid, 10)

for i in 1:iters_num
  batch_mask = rand(1:train_size, batch_size);
  x_batch = x_train[batch_mask, :]';
  t_batch = t_train[batch_mask, :]';

  grad = TLN.calc_grad(x_batch, t_batch)

  for key in keys(grad)
    TLN.params[key] -= learning_rate .* grad[key]
  end

  loss_ = TLN.loss(x_batch, t_batch)
  push!(train_loss_list, loss_)
  println(loss_)
end
