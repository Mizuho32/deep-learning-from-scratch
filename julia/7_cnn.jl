
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
      img[y:stride:y_max, x:stride:x_max, :, :] = col[:, :, col_idx, :, :]
      col_idx+=1
    end
  end

  return img[(pad+1):(H+pad), (pad+1):(W+pad), :, :]
end

function Wx2im(Wx, OH, OW, FN, N)
  return reshape(Wx, (OH, OW, FN, N))
end

function W2col(W)
  FH,FW,C,FN = size(W)
  return reshape(W, (FH*FW*C, FN))
end

export Convolution
mutable struct Convolution
  W::Array
  b::Array
  stride::Number
  pad::Number
end

function Convolution(self::Convolution, W, b, stride=1, pad=0)
  self.W = W
  self.b = b
  self.stride = stride
  self.pad = pad
end

function forward(self::Convolution, x)
  FH, FW, C, FN = shape(self.W)
  H, W, C, N = shape(x)
  out_h = Int(floor((H + 2*pad - FH)/stride + 1))
  out_w = Int(floor((W + 2*pad - FW)/stride + 1))

  col = im2col(x, FH, FW, self.stride, self.pad)
  col_W = W2col(self.W)
  return Wx2im(col*colW .+ self.b)
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
affine = col3*colWf
img3_ = Wx2im(affine, out_h, out_w, FN, N)

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
N = 1
S = 1
P = 1
out_h = Int(floor((W+2*P-filter_h)/S)+1)
out_w = Int(floor((W+2*P-filter_w)/S)+1)
img4 = zeros(H,W, C, N);
img4[:, :, 1, 1] = lenaAr
Wf2 = zeros(filter_h, filter_w, C, FN);
Wf2[2, 2, 1, 1] = 1;
Wf2[:, :, 1, 2] = 0.11 .* ones(3,3);
Wf2[2, 2, 1, 2] = 0.12;
Wf2[:, :, 1, 3] = -0.11 .* ones(3,3);
Wf2[2, 2, 1, 3] = 1.88;
Wf2[:, :, 1, 4] = 0.5 .* [-1 0 1; -2 0 2; -1 0 1]';
Wf2[2, 2, 1, 4] = 1;
colWf2 = W2col(Wf2);
col4 = im2col(img4,filter_h,filter_w, S,P)
affineL = col4*colWf2
img4_ = Wx2im(affineL, out_h, out_w, FN, N);
size(img4_)
imshow(img4_[:, :, 1, 1], cmap="gray")
imshow(img4_[:, :, 2, 1], cmap="gray")
imshow(img4_[:, :, 3, 1], cmap="gray")
imshow(img4_[:, :, 4, 1], cmap="gray")
