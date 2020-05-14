# %% Image
H = 2
W = 2
FN = 2
C = 2
FH = 1
FW = 1
N = 2
As = permutedims(reshape(1:(H*W*C), H,W,C), (3,2,1))
As[1, :,:]
As[2, :,:]
img = zeros(1,FN,H,W,1)
img[1, 1, :, :, 1] = As[1, :, :]
img[1, 2, :, :, 1] = As[2, :, :]

# %% Filter
F1s1 = [1 1;0 0]
F1s2 = [0 0;1 1]
F2s1 = [0 1;0 1]
F2s2 = [1 0;1 0]
filter = zeros(2,2,2,2,2)
filter[1,1, :,:,1] = F1s1
filter[1,1, :,:,2] = F1s2
filter[1,2, :,:,1] = F1s1
filter[1,2, :,:,2] = F1s2

filter[2,1, :,:,1] = F2s1
filter[2,1, :,:,2] = F2s2
filter[2,2, :,:,1] = F2s1
filter[2,2, :,:,2] = F2s2

# %% Output
output = zeros(2,1,1,2,2)
output[1,1,1,:,1] = [1 0]
output[1,1,1,:,2] = [0 1]
output[2,1,1,:,1] = [1 0]
output[2,1,1,:,2] = [0 1]

# %% Filter * Image
Fout = filter[:, :, :, :, :] .* img;

Fout[1, 1, :, :, 1]
Fout[1, 1, :, :, 2]
Fout[1, 2, :, :, 1]
Fout[1, 2, :, :, 2]

Fout[2, 1, :, :, 1]
Fout[2, 1, :, :, 2]
Fout[2, 2, :, :, 1]
Fout[2, 2, :, :, 2]
size(Fout)

# %% ストライドを縮退し、チャンネル足し合わせ
Fst = sum(sum(Fout, dims=(3,4)), dims=2);
size(Fst)
Fst[1, 1, 1, 1, 1]
Fst[1, 1, 1, 1, 2]

Fst[2, 1, 1, 1, 1]
Fst[2, 1, 1, 1, 2]

## %% outputに出力
out = sum(output .* Fst, dims=(5));
out[1, 1, :, :, 1]
# 14 22
out[2, 1, :, :, 1]
# 20 16
