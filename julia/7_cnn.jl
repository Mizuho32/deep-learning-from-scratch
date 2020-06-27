
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
    b::Number
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
    stride::Number
    pad::Number
    forward
    backward
  end

  function new(pool_h, pool_w, stride=1, pad=0)
    self = Pooling_st([], [], pool_h, pool_w, stride, pad, (x)->forward(self, x), (x, dout)->backward(self, x, dout))
    return self
  end

  function forward(self::Pooling_st, x)
    H, W, C, N = size(x)
    out_h = Int(floor((H + 2*self.pad - self.pool_h)/self.stride + 1))
    out_w = Int(floor((W + 2*self.pad - self.pool_w)/self.stride + 1))

    col = reshape(Main.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)', (self.pool_h*self.pool_w, :))

    return permutedims(reshape(maximum(col, dims=1), (C, out_h, out_w, N)), (2, 3, 1, 4)), col
  end

  function backward(self::Pooling_st, x, dout)
    dout = reshape(dout, (4, 1, size(dout, 3), size(dout, 4)) )
    W = zeros((4, 4, size(dout, 3), size(dout, 4)))
    for n in 1:size(dout, 4)
    for c in 1:size(dout, 3)
      tmpx = reshape(x[:, :, c, n], (3,3, 1, 1))
      arg_max = argmax(Main.im2col(tmpx, 2, 2, 1, 0), dims=2)
      for car in arg_max
        W[car[1], car[2], c, n] = 1
      end
    end
    end

    dcol = reshape(permutedims(dout .* W, (1,4,2,3)), (:, 2*2*size(dout, 3)))
    return Main.col2im(dcol, size(x), 2, 2, 1, 0), dcol, dout, W
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
out_h = Int(floor((W+2*P-filter_h)/S)+1)
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
pool_h = 2
pool_w = 2
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
img5_,col = pool.forward(img5);
img5_
col

## %% backward
dout = zeros(2, 2, C, N);
dout[:,:, 1,1] = [1 2; 3 4];
dout[:,:, 2,1] = 4 .+ [1 2; 3 4];
dout[:,:, 1,2] = -dout[:,:, 1,1];
dout[:,:, 2,2] = -dout[:,:, 2,1];

img5_test = zeros(3,3, C,N);
img5_test[:,:, :,:] = copy(img5[:, :, :, :]);
img5_test[1, 2, 1, 1] = 10;
img5_test
pool = Pooling.new(pool_h, pool_w);
dimg5, dcol, dout, W = pool.backward(img5_test, dout);
dimg5

dcol# = reshape(permutedims(dout .* W, (1,4,2,3)), (:, 2*2*C))
dout .* W
dout
W
