import numpy as np

f = open('raw/Blakeley_raw_data.csv', 'r')
f.readline()
# c = 0
rows = []
for line in f:
    line = line.strip()
    cols = line.split(',')
    # print(cols)
    assert len(cols) == 31
    # c += 1
    rows.append([float(c) for c in cols[1:]])

x = np.array(rows)
mat = x
min_count = 2
cs = np.sum(mat > min_count, axis=0)
print(cs.shape)
print(cs)
min_cell = 3
x_filt = mat[:, cs > min_cell]
print(x_filt.shape)

rs = np.sum(x_filt, axis=1, keepdims=True)
rs_med = np.median(rs)

print(rs)
print(rs.shape)
print(rs_med)

# print(rs.shape)
# print(rs_med.shape)

x_norm = x_filt * rs_med / rs
print("[!normalization] Complete.")

# print(x)

target = x_norm[891]
# print(target)
# 1,50124,101,3,1103,2385,47,0,4066,1520,554,0,199708,136,26,2,1295,7957,0,0,0,0,2,0,0,140,0,0,376,0
# exit(0)

# print(list(x_filt[:, 0]))

# print(c)

(23764, 30)

f = open('blakeley.csv', 'r')
f.readline()
# c = 0
rows = []
for line in f:
    line = line.strip()
    cols = line.split(',')
    # print(cols)
    # print(len(cols))
    assert len(cols) == 1001
    # c += 1
    rows.append([float(c) for c in cols[1:]])

x = np.array(rows)

(30, 1000)
# print(c)

print('!')

print(target)
target = np.log2(target + 1)
print(target)

real = x[:, 0]
print(real)

for i in range(target.shape[0]):
    print(i, target[i], real[i])
