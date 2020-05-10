
# 2.3.1
# %% AND
function AND(x1, x2)
  w1,w2,theta = 0.5, 0.5, 0.7;
  tmp = x1*w1 + x2*w2;
  if tmp <= theta
    return 0;
  elseif tmp > theta
    return  1;
  end
end

function AND2(x1, x2)
  x = [x1 x2];
  w = [0.5 0.5]';
  b = -0.7;
  tmp = (x*w)[1] + b;
  if tmp <= 0
    return 0;
  else
    return 1;
  end
end

AND(0, 0)
AND(1, 0)
AND(0, 1)
AND(1, 1)

AND2(0, 0)
AND2(1, 0)
AND2(0, 1)
AND2(1, 1)

# %% NAND OR
function NAND(x1, x2)
  x = [x1; x2]
  w = [-0.5; -0.5]
  b = 0.7
  tmp = (x'*w)[1]+b
  if tmp <= 0
    return 0
  else
    return 1
  end
end

function OR(x1, x2)
  x = [x1; x2]
  w = [0.5; 0.5]
  b = -0.2
  tmp = (x'*w)[1] + b
  if tmp <= 0
    return 0
  else
    return 1
  end
end

NAND(0, 0)
NAND(0, 1)
NAND(1, 0)
NAND(1, 1)

OR(0, 0)
OR(0, 1)
OR(1, 0)
OR(1, 1)
