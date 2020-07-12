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
    self.mask = (x .<=  0)
    out = copy(x)
    out[self.mask] .= 0

    return out
  end
  function backward(self::Relu_st, dout)
    dout[self.mask] .= 0
    dx = dout

    return dx
  end
end

module Dropout
  export Dropout_st
  mutable struct Dropout_st
    dropout_ratio
    mask::Array
    forward
    backward
  end
  function new(dropout_ratio=0.5)
    self = Dropout_st(dropout_ratio, [], (x, t=true)->forward(self, x, t), (d)->backward(self, d))
    return self
  end
  function forward(self::Dropout_st, x, train_flg=true)
    if train_flg
      self.mask = rand(Float64, size(x)) .> self.dropout_ratio
      return x .* self.mask
    else
      return x .* (1.0 - self.dropout_ratio)
    end
  end
  function backward(self::Dropout_st, dout)
    return dout .* self.mask
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
    numpymode::Bool
  end
  function new(W, b, numpymode=false)
    self = Affine_st(W, b, zeros(size(W, 2), 0), (), 0 .*W, 0 .*b, (x)->forward(self, x), (d)->backward(self, d), numpymode)
    return self
  end
  function forward(self::Affine_st, x)
    self.original_x_shape = size(x)
    if ndims(x) > 2
      if self.numpymode
        # np:(N, C, H, W), jl:(H, W, C, N)
        x = reshape(permutedims(x, (2,1,3,4)), (:, self.original_x_shape[end]))
      else
        x = reshape(x, (:, self.original_x_shape[end]))
      end
    end
    self.x = x

    return self.W * x .+ self.b
  end
  function backward(self::Affine_st, dout)
    dx = self.W' * dout
    self.dW .= dout * self.x'
    self.db .= vec(sum(dout, dims=2))

    if length(self.original_x_shape)>2 && self.numpymode
      return permutedims(reshape(dx, self.original_x_shape), (2, 1, 3, 4))
    else
      return reshape(dx, self.original_x_shape)
    end
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
    self.loss = cross_entropy_error(self.y, self.t')

    return self.loss
  end
  function backward(self::SoftmaxWithLoss_st, dout)
    dx = (self.y .- self.t') ./ size(self.t, 1)

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
    return permutedims(reshape(maximum(col, dims=1), (C, self.out_h, self.out_w, N)), (2, 3, 1, 4))
  end

  function backward(self::Pooling_st, dout)
    sizex = size(self.x)
    H, W, C, N = sizex
    pool_hw = self.pool_h*self.pool_w
    out_hw  = self.out_h*self.out_w

    dout = reshape(dout, (out_hw, 1, C, N))
    W = permutedims(reshape(self.arg_max, (pool_hw, C, out_hw, N)), (3,1,2,4))

    dcol = reshape(permutedims(dout .* W, (1,4,2,3)), (:, pool_hw*C))
    return Main.col2im(dcol, sizex, self.pool_h, self.pool_w, self.stride, self.pad)
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
