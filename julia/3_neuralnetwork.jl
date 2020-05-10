# %% using
include("./dataset/mnist.jl")

using PyCall
pickle = pyimport("pickle")

using PyPlot


# %% 3.2.3 Step

x = -5:0.1:5
y = x .>= 0

plot(x, y)

# %% 3.2.4 sigmoid

function sigmoid(x)
  return 1 ./ (1 .+ exp.(-x))
end

x = -5:0.1:5
y = sigmoid(x)
grid();
plot(x, y)
ylim(-0.1, 1.1);


# %% 3.3.3 ニューラルネットワークの内積
X = [1; 2]
W = [ 1 2
      3 4
      5 6 ]
Y = W*X

# %% 3.4.2 バイアス付き
X = [1; 0.5]
W1 = [0.1 0.3 0.5; 0.2 0.4 0.6]'
B1 = [0.1; 0.2; 0.3]
A1 = W1*X + B1
Z1 = sigmoid(A1)

W2 = [0.1 0.2 0.3; 0.4 0.5 0.6]
B2 = [0.1; 0.2]
A2 = W2*Z1 + B2
Z2 = sigmoid(A2)

function identity_function(x)
  return x
end

W3 = [0.1 0.2; 0.3 0.4]
B3 = [0.1; 0.2]
A3 = W3*Z2 + B3
Y = identity_function(A3)

# %% 3.4.3 実装まとめ
function init_network()
  network = Dict()
  network["W1"] = [0.1 0.3 0.5; 0.2 0.4 0.6]'
  network["b1"] = [0.1; 0.2; 0.3]
  network["W2"] = [0.1 0.2 0.3; 0.4 0.5 0.6]
  network["b2"] = [0.1; 0.2]
  network["W3"] = [0.1 0.2; 0.3 0.4]
  network["b3"] = [0.1; 0.2]
  return network
end

function forward(network, x)
  W1, W2, W3 = network["W1"], network["W2"], network["W3"]
  b1, b2, b3 = network["b1"], network["b2"], network["b3"]

  a1 = W1*X + b1
  z1 = sigmoid(a1)
  a2 = W2*z1 + b2
  z2 = sigmoid(a2)
  a3 = W3*z2 + b3
  return identity(a3)
end

network = init_network()
x = [1; 0.5]
y = forward(network, x)

# %% 出力層の計算

# %% 3.5.2 softmax

function softmax(a)
  c = maximum(a)
  return exp.(a .- c) ./ sum(exp.(a .- c))
end

softmax([1010 1000 990])

# %% 3.6 手書き文字認識

# %% load
(x_train, t_train), (x_test, t_test) = MNIST.load_mnist(normalize=false);
size(x_train)
size(t_train)
size(x_test)
size(t_test)

# %% imshow
i = 3
img = reshape(x_train[i, :], (28, 28))'
label = t_train[i]
imshow(img)

# %% 3.6.2 ニューラルネットワークの推論処理

# %% funcs
function get_data()
  (_, _), (x_test, t_test) = MNIST.load_mnist();
  return x_test, t_test
end

function init_network2()
  network = nothing
  filename = "ch03/sample_weight.pkl"
  @pywith pybuiltin("open")(filename,"rb") as f begin
    network = pickle.load(f)
  end
  return network
end

function predict(network, x)
  W1, W2, W3 = network["W1"]', network["W2"]', network["W3"]'
  b1, b2, b3 = network["b1"] , network["b2"] , network["b3"]

  a1 = W1*x .+ b1
  z1 = sigmoid(a1)
  a2 = W2*z1 .+ b2
  z2 = sigmoid(a2)
  a3 = W3*z2 .+ b3
  return softmax(a3)
end

# %% main

x,t = get_data();
network = init_network2()

accuracy_cnt = 0
for i in 1:size(x, 1)
  y = predict(network, x[i, :])
  p = argmax(y)-1
  if p==t[i]
    accuracy_cnt += 1
  end
end

println("Accuracy: $(accuracy_cnt/size(x, 1))")

# %% 3.6.3 バッチ処理

x,t = get_data();
network = init_network2()

batch_size = 100
accuracy_cnt = 0

for i in 1:batch_size:size(x, 1)
  x_batch = x[i:(i+batch_size-1), :]';
  y_batch = predict(network, x_batch);
  p = map(x->x[1]-1, argmax(y_batch, dims=1))
  accuracy_cnt += sum(p .== t[i:(i+batch_size-1)]')
end

println("Accuracy: $(accuracy_cnt/size(x, 1))")
