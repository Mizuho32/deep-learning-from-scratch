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

module TLN2 # 2 Layer Net
  using ..BP
  layerOrder = ["Affine1" "Relu1" "Affine2"]

  params = Dict()
  layers = Dict()
  grads_n = Dict()
  grads_b = Dict()
  o = Dict()

function init(input_size, hidden_size, output_size, W1, W2, w_init_std = 0.01)
    params["W1"] = W1
    params["b1"] = zeros(hidden_size)
    params["W2"] = W2
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

  function plot(fig, ax1, iter_per_epoch, train_loss_list, train_acc_list, test_acc_list)
    ax1.grid()
    ax1.plot(1:length(train_loss_list), train_loss_list, label="loss")
    ax2 = ax1.twinx()
    if !isempty(train_acc_list)
      ax2.plot(range(iter_per_epoch, length=length(train_acc_list), step=iter_per_epoch), train_acc_list, linestyle="-.", color="orange", linewidth=1, label="train")
    end
    if !isempty(test_acc_list)
      ax2.plot(range(iter_per_epoch, length=length(test_acc_list), step=iter_per_epoch), test_acc_list, linestyle="-.", color="red", linewidth=1, label="test")
    end
    fig.legend(loc="center")
    fig.show()
  end

end
