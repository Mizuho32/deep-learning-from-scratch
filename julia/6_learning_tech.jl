# 6学習に関するテクニック
# %% 6.1 パラメータ更新

module SGD
  mutable struct SGD_st
    lr
    update
  end

  function new(lr=0.01)
    tmp_st = SGD_st(lr, nothing)
    tmp_st.update = (p,g)->update(tmp_st, p, g)
    return return tmp_st
  end

  function update(self::SGD_st, params::Dict, grads::Dict)
    for key in keys(params)
      params[key] .-= self.lr .* grads[key]
    end
  end
end

module Momentum
  mutable struct SGD_st
    lr
    momentum
    v
    update
  end

  function new(lr=0.01, momentum=0.9)
    tmp_st = SGD_st(lr, momentum, nothing, nothing)
    tmp_st.update = (p,g)->update(tmp_st, p, g)
    return return tmp_st
  end

  function update(self::SGD_st, params::Dict, grads::Dict)
    if self.v == nothing
      self.v = Dict()
      for key in keys(params)
        self.v[key] = zeros(size(params[key]))
      end
    end

    for key in keys(params)
      self.v[key] = self.momentum .* self.v[key] - self.lr .* grads[key]
      params[key] .-= self.lr .* grads[key]
    end
  end
end

module AdaGrad
  mutable struct SGD_st
    lr
    h
    update
  end

function new(lr=0.01)
    tmp_st = SGD_st(lr, nothing, nothing)
    tmp_st.update = (p,g)->update(tmp_st, p, g)
    return return tmp_st
  end

  function update(self::SGD_st, params::Dict, grads::Dict)
    if self.h == nothing
      self.h = Dict()
      for key in keys(params)
        self.h[key] = zeros(size(params[key]))
      end
    end

    for key in keys(params)
      self.h[key] += grads[key] .* grads[key]
      params[key] .-= self.lr .* grads[key] ./ (sqrt.(self.h[key]) .+ 1.e-7)
    end
  end
end

# %% main
using PyPlot
include("./dataset/mnist.jl")
include("julia/lib.jl"  )
struct List
  train_loss_list::Array
  train_acc_list::Array
  test_acc_list::Array
  #W::Dict{UInt8, Array}
  #b::Dict{UInt8, Array}
end
# %%
(x_train, t_train), (x_test, t_test) = MNIST.load_mnist(one_hot_label=true, normalize=true);
train_size = size(x_train, 1)
# %%
iters_num = 10000
batch_size = 100
learning_rate = 0.1
input_size = 784
hidden_size = 50
output_size = 10
iter_per_epoch = Int(max(train_size/batch_size, 1))

# %%
w_init_std = 0.01
W1seed = randn(input_size, hidden_size)';
W2seed = randn(hidden_size, output_size)';

# %% SGD
using .SGD
opt = SGD.new(learning_rate);
TLN2.init(input_size, hidden_size, output_size, w_init_std .* W1seed, w_init_std .* W2seed)
sgdlist = List(zeros(iters_num), [], [])

for i in 1:iters_num
  batch_mask = rand(1:train_size, batch_size);
  x_batch = x_train[batch_mask, :]';
  t_batch = t_train[batch_mask, :]';

  grad = TLN2.gradient(x_batch, t_batch)

  opt.update(TLN2.params, grad)

  loss_ = TLN2.loss(x_batch, t_batch)
  sgdlist.train_loss_list[i] = loss_

  if i % iter_per_epoch == 0
    train_acc = TLN2.accuracy(x_train', t_train')
    test_acc = TLN2.accuracy(x_test', t_test')
    push!(sgdlist.train_acc_list, train_acc)
    push!(sgdlist.test_acc_list, test_acc)
    println("iter: $i loss: $loss_ train acc: $train_acc test acc: $test_acc")
  end
end
# %% plot
fig, ax1 = subplots()
TLN2.plot(fig, ax1, iter_per_epoch, sgdlist.train_loss_list, sgdlist.train_acc_list,sgdlist.test_acc_list)



# %% AdaGrad
using .AdaGrad
opt = AdaGrad.new(learning_rate);
TLN2.init(input_size, hidden_size, output_size, w_init_std .* W1seed, w_init_std .* W2seed)
adalist = List(zeros(iters_num), [], [])

for i in 1:iters_num
  batch_mask = rand(1:train_size, batch_size);
  x_batch = x_train[batch_mask, :]';
  t_batch = t_train[batch_mask, :]';

  grad = TLN2.gradient(x_batch, t_batch)

  opt.update(TLN2.params, grad)

  loss_ = TLN2.loss(x_batch, t_batch)
  adalist.train_loss_list[i] = loss_

  if i % iter_per_epoch == 0
    train_acc = TLN2.accuracy(x_train', t_train')
    test_acc = TLN2.accuracy(x_test', t_test')
    push!(adalist.train_acc_list, train_acc)
    push!(adalist.test_acc_list, test_acc)
    println("iter: $i loss: $loss_ train acc: $train_acc test acc: $test_acc")
  end
end
# %% plot
fig, ax1 = subplots()
TLN2.plot(fig, ax1, iter_per_epoch, adalist.train_loss_list, adalist.train_acc_list,adalist.test_acc_list)



# %% AdaGrad + He init
using .AdaGrad
opt = AdaGrad.new(learning_rate);
HeW1 = sqrt(2/input_size)
HeW2 = sqrt(2/hidden_size)
TLN2.init(input_size, hidden_size, output_size, HeW1 .* W1seed, HeW2 .* W2seed)
adaHelist = List(zeros(iters_num), [], [])

for i in 1:iters_num
  batch_mask = rand(1:train_size, batch_size);
  x_batch = x_train[batch_mask, :]';
  t_batch = t_train[batch_mask, :]';

  grad = TLN2.gradient(x_batch, t_batch)

  opt.update(TLN2.params, grad)

  loss_ = TLN2.loss(x_batch, t_batch)
  adaHelist.train_loss_list[i] = loss_

  if i % iter_per_epoch == 0
    train_acc = TLN2.accuracy(x_train', t_train')
    test_acc = TLN2.accuracy(x_test', t_test')
    push!(adaHelist.train_acc_list, train_acc)
    push!(adaHelist.test_acc_list, test_acc)
    println("iter: $i loss: $loss_ train acc: $train_acc test acc: $test_acc")
  end
end
# %% plot
fig, ax1 = subplots()
TLN2.plot(fig, ax1, iter_per_epoch, adaHelist.train_loss_list, adaHelist.train_acc_list,adaHelist.test_acc_list)

# %% save
using JLD2, FileIO
save("./dataset/6_AdaGrad_Wb.jld2", "params", TLN2.params, "train_acc_list", adaHelist.train_acc_list, "test_acc_list", adaHelist.test_acc_list, "train_loss_list", adaHelist.train_loss_list)
