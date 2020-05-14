# %% Image
H = 2
W = 2
FN = 2
C = 2
FH = 1
FW = 2
N = 2
S = 1
P = 0


Fdim,Cdim,Hdim,Wdim,Sdim = 1,2,3,4,5
F1,F2 = [1 1],[0 1]
F = zeros((2, FH, FW));
F[1,:,:] = F1
F[2,:,:] = F2

As = permutedims(reshape(1:(H*W*C), H,W,C), (3,2,1))
img = zeros(1,FN,H,W,1)
for c in 1:C
  img[1, c, :, :, 1] = As[c, :, :]
end

# %% Filter
function stridedFilter(F, C, H, W, P, S)
  FN,FH,FW = size(F)
  OH = Int(floor((H+2P-FH)/S+1))
  OW = Int(floor((W+2P-FW)/S+1))
  filter = zeros(FN,C,H,W,OH*OW)
  preoutput = zeros(FN,1,OH,OW,OH*OW)
  for f in 1:FN
    for c in 1:C
      s = 1
      for w in 1:OW
        iw = (1+S*(w-1))
        for h in 1:OH
          ih = (1+S*(h-1))
          filter[f, c, ih:(ih+FH-1), iw:(iw+FW-1), s] .= F[f, :, :]
          #println(filter[f, c, :, :, s])
          preoutput[f,1,h,w,s] = 1
          #println(preoutput[f,1,:,:,s])
          s += 1
        end
      end
    end
  end
  return OH, OW, filter, preoutput
end

# %% run
OH, OW, filter2, preoutput = stridedFilter(F, C, H, W, P, S);

# %% Filter * Image
Fout = filter2[:, :, :, :, :] .* img;

# %% 各ストライドの フィルタ*画像 を合計し、チャンネル足し合わせ
Fst = sum(sum(Fout, dims=(Hdim,Wdim)), dims=Cdim);

## %% outputに出力
out = sum(preoutput .* Fst, dims=(Sdim));
out[1, 1, :, :, 1]
# 14 22
out[2, 1, :, :, 1]
# 8 12
