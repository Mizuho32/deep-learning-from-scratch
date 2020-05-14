
function im2col(input_data, filter_h, filter_w, stride=1, pad=0)
  H, W, C, N = size(input_data)
  out_h = Int(floor((H + 2*pad - filter_h)/stride + 1))
  out_w = Int((W + 2*pad - filter_w)/stride + 1)

  img = input_data
  if pad != 0
    img = zeros(N,C,H+2*pad, W+2*pad)
    img[:, :, (1+pad):(H+pad), (1+pad):(W+pad)] .= input_data
  end
  col = zeros(out_h*out_w, filter_h, filter_w, C, N)

  col_idx = 1
  for x in 1:filter_w
    x_shift = 1 + stride*(x-1)
    for y in 1:filter_h
      y_shift = 1 + stride*(y-1)
      col[col_idx, :, :, :, :] = img[y_shift:(y_shift+filter_h-1), x_shift:(x_shift+filter_w-1), :, :]
      col_idx+=1
    end
  end

  return reshape(permutedims(col,(1,5,2,3,4)), N*out_h*out_w, :)
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
img2 = zeros(3,3, 2, 2)
img2[:, :, 1,1] .= img
img2[:, :, 2,1] .= img.+9
img2[:, :, 1,2] .= -img
img2[:, :, 2,2] .= -(img.+9)
im2col(img2,2,2)
