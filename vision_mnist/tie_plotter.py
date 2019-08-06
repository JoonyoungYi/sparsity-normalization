import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, sharey=False)
plt.subplots_adjust(wspace=0.3)
fig.set_figwidth(5)
fig.set_figheight(1.2)

red = "#FFB82B"
red_blur = '#FF4841'
blue = "#0099FE"
blue_blur = '#5C2DAE'

s = 0.1
class_a = 1
class_b = 2

sparsity_normalization = False
dataset = 'fashion'

dense_outputs = np.load('vision_mnist/result-dense-outputs-{}-{}.npy'.format(
    sparsity_normalization, dataset).lower())
sparse_outputs = np.load('vision_mnist/result-sparse-outputs-{}-{}.npy'.format(
    sparsity_normalization, dataset).lower())
dense_ys = np.load('vision_mnist/result-dense-ys-{}-{}.npy'.format(
    sparsity_normalization, dataset).lower())
sparse_ys = np.load('vision_mnist/result-sparse-ys-{}-{}.npy'.format(
    sparsity_normalization, dataset).lower())

# print(dense_outputs.shape)
# print(sparse_outputs.shape)

dense_outputs_a = dense_outputs[dense_ys == class_a]
dense_outputs_b = dense_outputs[dense_ys == class_b]

sparse_outputs_a = sparse_outputs[sparse_ys == class_a]
sparse_outputs_b = sparse_outputs[sparse_ys == class_b]

ax = axs[0]
ax.set_title('w/o SN')
ax.scatter(
    dense_outputs_a[:, class_a],
    dense_outputs_a[:, class_b],
    s=s,
    color=red_blur)
ax.scatter(
    dense_outputs_b[:, class_a],
    dense_outputs_b[:, class_b],
    s=s,
    color=blue_blur)
ax.scatter(
    sparse_outputs_a[:, class_a], sparse_outputs_a[:, class_b], s=s, color=red)
ax.scatter(
    sparse_outputs_b[:, class_a],
    sparse_outputs_b[:, class_b],
    s=s,
    color=blue)

sparsity_normalization = True
dataset = 'fashion'

dense_outputs = np.load('vision_mnist/result-dense-outputs-{}-{}.npy'.format(
    sparsity_normalization, dataset).lower())
sparse_outputs = np.load('vision_mnist/result-sparse-outputs-{}-{}.npy'.format(
    sparsity_normalization, dataset).lower())
dense_ys = np.load('vision_mnist/result-dense-ys-{}-{}.npy'.format(
    sparsity_normalization, dataset).lower())
sparse_ys = np.load('vision_mnist/result-sparse-ys-{}-{}.npy'.format(
    sparsity_normalization, dataset).lower())

# print(dense_outputs.shape)
# print(sparse_outputs.shape)

dense_outputs_a = dense_outputs[dense_ys == class_a]
dense_outputs_b = dense_outputs[dense_ys == class_b]

sparse_outputs_a = sparse_outputs[sparse_ys == class_a]
sparse_outputs_b = sparse_outputs[sparse_ys == class_b]

ax = axs[1]
ax.set_title('w/ SN')
ax.scatter(
    dense_outputs_a[:, class_a],
    dense_outputs_a[:, class_b],
    s=s,
    color=red_blur,
    label='class A (dense group)')
ax.scatter(
    sparse_outputs_a[:, class_a],
    sparse_outputs_a[:, class_b],
    s=s,
    color=red,
    label='class A (sparse group)')
ax.scatter(
    dense_outputs_b[:, class_a],
    dense_outputs_b[:, class_b],
    s=s,
    color=blue_blur,
    label='class B (dense group)')
ax.scatter(
    sparse_outputs_b[:, class_a],
    sparse_outputs_b[:, class_b],
    s=s,
    color=blue,
    label='class B (sparse group)')

ax.legend(
    loc='upper center',
    bbox_to_anchor=(-0.2, -0.15),
    ncol=2,
    scatterpoints=1,
    markerscale=20,
    edgecolor='none')

plt.savefig(
    'vision_mnist/result-scatter-{}.png'.format(dataset).lower(),
    format='png',
    bbox_inches='tight',
    pad_inches=0)
plt.savefig(
    'vision_mnist/result-scatter-{}.pdf'.format(dataset).lower(),
    format='pdf',
    bbox_inches='tight',
    pad_inches=0)
plt.close()
