module MNIST

using GZip
using JLD2
using FileIO

url_base = "http://yann.lecun.com/exdb/mnist/"
key_file = Dict(
    "train_img"=>"train-images-idx3-ubyte.gz",
    "train_label"=>"train-labels-idx1-ubyte.gz",
    "test_img"=>"t10k-images-idx3-ubyte.gz",
    "test_label"=>"t10k-labels-idx1-ubyte.gz"
)

dataset_dir = @__DIR__
save_file = "$dataset_dir/mnist.jld2"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784



function _download(file_name)
  file_path = "$dataset_dir/$file_name"

  if isfile(file_path)
        return
  end

  println("Downloading $file_name ... ")
  download(url_base * file_name, file_path)
  println("Done")
end

function download_mnist()
  for v in values(key_file)
   _download(v)
 end
end

function _load_label(file_name)
  file_path = "$dataset_dir/$file_name"

  println("Converting $file_name to Julia Array ...")
  io = GZip.open(file_path, "rb")
  labels = read(io)[9:end]
  close(io)
  println("Done")

  return labels
end


function _load_img(file_name)
  file_path = "$dataset_dir/$file_name"

  println("Converting $file_name to Julia Array ...")
  io = GZip.open(file_path, "rb")
  data = read(io)[17:end]
  #println(size(data))
  data = reshape(data, (img_size, :))'
  close(io)
  println("Done")

  return data
end


function _convert_numpy()
  dataset = Dict()
  dataset["train_img"] =  _load_img(key_file["train_img"])
  dataset["train_label"] = _load_label(key_file["train_label"])
  dataset["test_img"] = _load_img(key_file["test_img"])
  dataset["test_label"] = _load_label(key_file["test_label"])

  return dataset
end

function init_mnist()
  download_mnist()
  dataset = _convert_numpy()
  println("Creating JLD file ...")
  save(save_file, "dataset", dataset)
  println("Done!")
end

function _change_one_hot_label(X)
  return X .== (1:10)'
end

function load_mnist(;normalize=true, flatten=true, one_hot_label=false)
  """MNISTデータセットの読み込み

  Parameters
  ----------
  normalize : 画像のピクセル値を0.0~1.0に正規化する
  one_hot_label :
      one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
      one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
  flatten : 画像を一次元配列に平にするかどうか

  Returns
  -------
  (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
  """
  if !isfile(save_file)
      init_mnist()
  end

  dataset = load(save_file, "dataset")

  if normalize
    for key in ("train_img", "test_img")
      dataset[key] = Array{Float32}(dataset[key])./255.0f0
    end
  end

  if one_hot_label
    dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
    dataset["test_label"] = _change_one_hot_label(dataset["test_label"])
  end

  if !flatten
    for key in ("train_img", "test_img")
      dataset[key] = reshape(dataset[key].(:, 1, 28, 28))
    end
  end

  return (dataset["train_img"], dataset["train_label"]), (dataset["test_img"], dataset["test_label"])
end


end
