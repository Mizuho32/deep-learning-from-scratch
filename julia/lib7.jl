module Relu
  export Relu_st
  mutable struct Relu_st
    mask::Array
    forward
    backward
  end
  function new()
    self = Relu_st(BitArray(zeros(0, 0)), (x)->forward(self, x), (d)->backward(self, d))
    return self
  end
  function forward(self::Relu_st, x)
    #println("x $(size(x)) mask $(size(self.mask))")
    self.mask = (x .<=  0)
    out = copy(x)
    out[self.mask] .= 0

    return out
  end
  function backward(self::Relu_st, dout)
    dx = copy(dout)
    dx[self.mask] .= 0

    return dx
  end
end

module Sigmoid
  export Sigmoid_st
  mutable struct Sigmoid_st
    out::Array
    forward
    backward
  end
  function new(dim)
    self = Sigmoid_st(zeros(dim, 1), (x)->forward(self, x), (d)->backward(self, d))
    return self
  end
  function forward(self::Sigmoid_st, x)
    self.out = 1 ./ (1 .+ exp.(-x))

    return self.out
  end
  function backward(self::Sigmoid_st, dout)
    dx = dout .* (1.0 .- self.out) .* self.out

    return dx
  end
end

module Affine
  export Affine_st
  mutable struct Affine_st
    W::Array
    b::Array
    x::Array
    original_x_shape::Tuple
    dW::Array
    db::Array
    forward
    backward
  end
  function new(W, b)
    self = Affine_st(W, b, zeros(size(W, 2), 0), (), 0 .*W, 0 .*b, (x)->forward(self, x), (d)->backward(self, d))
    return self
  end
  function forward(self::Affine_st, x)
    self.original_x_shape = size(x)
    x = reshape(x, (:, self.original_x_shape[end]))
    self.x = x

    return self.W * x .+ self.b
  end
  function backward(self::Affine_st, dout)
    dx = self.W' * dout
    self.dW .= dout * self.x'
    self.db .= vec(sum(dout, dims=2))

    return reshape(dx, self.original_x_shape)
  end
end

module SoftmaxWithLoss
  export SoftmaxWithLoss_st
  mutable struct SoftmaxWithLoss_st
    loss::Number
    y::Array
    t::Array
    forward
    backward
  end
  function new()
    self = SoftmaxWithLoss_st(0, zeros(0, 0), zeros(0, 0), (x, t)->forward(self, x, t), (d)->backward(self, d))
    return self
  end
  function forward(self::SoftmaxWithLoss_st, x, t)
    self.t = t
    self.y = softmax(x, 1)
    self.loss = cross_entropy_error(self.y, self.t)

    return self.loss
  end
  function backward(self::SoftmaxWithLoss_st, dout)
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
