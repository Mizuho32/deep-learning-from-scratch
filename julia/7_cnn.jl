# %% Image
H = 2
W = 2
FN = 2
C = 2
FH = 2
FW = 2
N = 2
S = 1
P = 0

Fdim,Cdim,Hdim,Wdim,Sdim = 2,3,4,5,6

# %% filter
F1,F2 = [1 1; 1 1],[0 0; 0 0]'
F = zeros((FN, C, FH, FW));
F[1, 1,:,:] = F1
F[1, 2,:,:] = F2
F[2, 1,:,:] = F1 .* 0.5
F[2, 2,:,:] = F2

# %% image
As = permutedims(reshape(1:(H*W*C), H,W,C), (3,2,1))
img = zeros(N,1,C,H,W,1)
for n in 1:N
  img[n, 1, :, :, :, 1] = As[:, :, :] .+ (n-1)
end

# %% Filter
function stridedFilter(F, C, H, W, P, S)
  FN,_, FH,FW = size(F)
  OH = Int(floor((H+2P-FH)/S+1))
  OW = Int(floor((W+2P-FW)/S+1))
  filter = zeros(1, FN,C,H,W,OH*OW)
  preoutput = zeros(1, FN,1,OH,OW,OH*OW)
  for f in 1:FN
    for c in 1:C
      s = 1
      for w in 1:OW
        iw = (1+S*(w-1))
        for h in 1:OH
          ih = (1+S*(h-1))
          filter[1, f, c, ih:(ih+FH-1), iw:(iw+FW-1), s] .= F[f, c, :, :]
          #println(filter[f, c, :, :, s])
          preoutput[1,f,1,h,w,s] = 1
          #println(preoutput[f,1,:,:,s])
          s += 1
        end
      end
    end
  end
  return OH, OW, filter, preoutput
end

function conv(imgtensor, filter, preoutput, Cdim, Hdim, Wdim, Sdim)
  return sum(preoutput .* sum(sum(filter .* imgtensor, dims=(Hdim,Wdim)), dims=Cdim), dims=(Sdim));
end

# %% run
OH, OW, filter2, preoutput = stridedFilter(F, C, H, W, P, S);
size(filter2)
size(preoutput)
# %% Filter * Image
Fout = filter2[:, :, :, :, :, :] .* img;
size(Fout)

# %% 各ストライドの フィルタ*画像 を合計し、チャンネル足し合わせ
Fst = sum(sum(Fout, dims=(Hdim,Wdim)), dims=Cdim);

## %% outputに出力
out = sum(preoutput .* Fst, dims=(Sdim));
size(out)
out[1, 1, 1, :, :, 1]
out[1, 2, 1, :, :, 1]
out[2, 1, :, :, 1]
out[2, 2, :, :, 1]

# %% lena
using Images, ImageIO
using PyPlot

lena = load("dataset/lena_gray.png");
lenaAr = channelview(lena);
#lenaF = Array{Float32, 2}(lenaAr[1, :, :]);
size(lenaAr)
imshow(lenaAr, cmap="gray")

# %% conv lena
H, W = size(lenaAr)
FN = 2
C = 1
FH = 3
FW = 3
N = 1
S = 1
P = 0

F1,F2 = [0 0 0; 0 1 0; 0 0 0],[0 0; 0 0]'
F = zeros((FN, C, FH, FW));
F[1, 1,:,:] = F1
#F[1, 2,:,:] = F2
#F[2, 1,:,:] = F1 .* 0.5
#F[2, 2,:,:] = F2

imgtensor = zeros(N,1,C,H,W,1)
size(imgtensor)
for n in 1:N
  imgtensor[n, 1, :, :, :, 1] = lenaAr
end

_, _, filterL, preoutputL = stridedFilter(F, C, H, W, P, S);
outtensor = conv(imgtensor, filterL, preoutputL, Cdim, Hdim, Wdim, Sdim)
