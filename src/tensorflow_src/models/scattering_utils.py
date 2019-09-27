
import tensorflow as tf
import numpy as np
import scipy.fftpack as fft

# All code for the filters bank directly adapted from https://github.com/edouardoyallon/pyscatwave
# Copyright (c) 2017, Eugene Belilovsky (INRIA), Edouard Oyallon (ENS) and Sergey Zagoruyko (ENPC)
# All rights reserved.


def filters_bank(M, N, J, L=8):
  filters = {}
  filters['psi'] = []

  offset_unpad = 0
  for j in range(J):
    for theta in range(L):
      psi = {}
      psi['j'] = j
      psi['theta'] = theta
      psi_signal = morlet_2d(
        M, N, 0.8 * 2**j,
        (int(L - L / 2 - 1) - theta) * np.pi / L,
        3.0 / 4.0 * np.pi / 2**j,
        offset=offset_unpad)
      # The 5 is here just to match the LUA implementation :)
      psi_signal_fourier = fft.fft2(psi_signal)
      for res in range(j + 1):
        psi_signal_fourier_res = crop_freq(psi_signal_fourier, res)
        psi[res] = tf.constant(
          np.stack((np.real(psi_signal_fourier_res),
                    np.imag(psi_signal_fourier_res)), axis=2))
        psi[res] = tf.div(
          psi[res], (M * N // 2**(2 * j)), name="psi_theta%s_j%s" % (theta, j))
      filters['psi'].append(psi)

  filters['phi'] = {}
  phi_signal = gabor_2d(M, N, 0.8 * 2**(J - 1), 0, 0, offset=offset_unpad)
  phi_signal_fourier = fft.fft2(phi_signal)
  filters['phi']['j'] = J
  for res in range(J):
    phi_signal_fourier_res = crop_freq(phi_signal_fourier, res)
    filters['phi'][res] = tf.constant(
      np.stack((np.real(phi_signal_fourier_res),
                np.imag(phi_signal_fourier_res)), axis=2))
    filters['phi'][res] = tf.div(
      filters['phi'][res], (M * N // 2 ** (2 * J)), name="phi_res%s" % res)
  return filters


def crop_freq(x, res):
  M = x.shape[0]
  N = x.shape[1]

  crop = np.zeros((M // 2 ** res, N // 2 ** res), np.complex64)

  mask = np.ones(x.shape, np.float32)
  len_x = int(M * (1 - 2 ** (-res)))
  start_x = int(M * 2 ** (-res - 1))
  len_y = int(N * (1 - 2 ** (-res)))
  start_y = int(N * 2 ** (-res - 1))
  mask[start_x:start_x + len_x,:] = 0
  mask[:, start_y:start_y + len_y] = 0
  x = np.multiply(x,mask)

  for k in range(int(M / 2 ** res)):
    for l in range(int(N / 2 ** res)):
      for i in range(int(2 ** res)):
        for j in range(int(2 ** res)):
          crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]
  return crop


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=None):
  """ This function generated a morlet"""
  wv = gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)
  wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
  K = np.sum(wv) / np.sum(wv_modulus)

  mor = wv - K * wv_modulus
  return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=None):
  gab = np.zeros((M, N), np.complex64)
  R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
  R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
  D = np.array([[1, 0], [0, slant * slant]])
  curv = np.dot(R, np.dot(D, R_inv)) / (2 * sigma * sigma)

  for ex in [-2, -1, 0, 1, 2]:
    for ey in [-2, -1, 0, 1, 2]:
      [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
      arg = -(curv[0, 0] * np.multiply(xx, xx) + \
              (curv[0, 1] + curv[1, 0]) * \
              np.multiply(xx, yy) + curv[ 1, 1] * \
              np.multiply(yy, yy)) + 1.j * \
              (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
      gab = gab + np.exp(arg)

  norm_factor = (2 * 3.1415 * sigma * sigma / slant)
  gab = gab / norm_factor

  if (fft_shift):
      gab = np.fft.fftshift(gab, axes=(0, 1))
  return gab


class Scattering:
  """Scattering module.
  Source:
    https://github.com/tdeboissiere/DeepLearningImplementations
  Runs scattering on an input image in NCHW format
  Input args:
      M, N: input image size
      J: number of layers
  """
  def __init__(self, M, N, J, name="scattering"):
    self.M, self.N, self.J = M, N, J

    self._prepare_padding_size([1, 1, M, N])

    # Create the filters
    filters = filters_bank(self.M_padded, self.N_padded, J)

    self.Psi = filters['psi']
    self.Phi = [filters['phi'][j] for j in range(J)]

  def _prepare_padding_size(self, s):
    M = s[-2]
    N = s[-1]
    self.M_padded = ((M + 2 ** (self.J)) // 2**self.J + 1) * 2**self.J
    self.N_padded = ((N + 2 ** (self.J)) // 2**self.J + 1) * 2**self.J
    s[-2] = self.M_padded
    s[-1] = self.N_padded
    self.padded_size_batch = [a for a in s]

  def _pad(self, x):
    """This function copies and view the real to complex
    """
    paddings = [[0, 0], [0, 0], [2 ** self.J, 2 ** self.J], [2 ** self.J, 2 ** self.J]]
    out_ = tf.pad(x, paddings, mode="REFLECT")
    out_ = tf.expand_dims(out_, axis=-1)
    output = tf.concat([out_, tf.zeros_like(out_)], axis=-1)
    return output

  def _unpad(self, in_):
    return in_[..., 1:-1, 1:-1]

  def __call__(self, x, reuse=False):

    x_shape = x.get_shape().as_list()
    x_h, x_w = x_shape[-2:]

    if x_w != self.N or x_h != self.M:
      raise RuntimeError("Tensor must be of spatial size (%i, %i)!" % (self.M, self.N))
    if len(x_shape) != 4:
      raise RuntimeError("Input tensor must be 4D")

    J = self.J
    phi = self.Phi
    psi = self.Psi
    n = 0

    pad = self._pad
    unpad = self._unpad

    S = []
    U_r = pad(x)

    U_0_c = compute_fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
    U_1_c = periodize(cdgmm(U_0_c, phi[0]), 2**J)
    U_J_r = compute_fft(U_1_c, 'C2R')

    S.append(unpad(U_J_r))
    n = n + 1

    for n1 in range(len(psi)):
      j1 = psi[n1]['j']
      U_1_c = cdgmm(U_0_c, psi[n1][0])
      if j1 > 0:
          U_1_c = periodize(U_1_c, k=2 ** j1)
      U_1_c = compute_fft(U_1_c, 'C2C', inverse=True)
      U_1_c = compute_fft(modulus(U_1_c), 'C2C')

      # Second low pass filter
      U_2_c = periodize(cdgmm(U_1_c, phi[j1]), k=2**(J - j1))
      U_J_r = compute_fft(U_2_c, 'C2R')
      S.append(unpad(U_J_r))
      n = n + 1

      for n2 in range(len(psi)):
        j2 = psi[n2]['j']
        if j1 < j2:
          U_2_c = periodize(cdgmm(U_1_c, psi[n2][j1]), k=2 ** (j2 - j1))
          U_2_c = compute_fft(U_2_c, 'C2C', inverse=True)
          U_2_c = compute_fft(modulus(U_2_c), 'C2C')

          # Third low pass filter
          U_2_c = periodize(cdgmm(U_2_c, phi[j2]), k=2 ** (J - j2))
          U_J_r = compute_fft(U_2_c, 'C2R')

          S.append(unpad(U_J_r))
          n = n + 1

    S = tf.concat(S, axis=1)
    return S


def stack_real_imag(x):
  stack_axis = len(x.get_shape().as_list())
  return tf.stack((tf.real(x), tf.imag(x)), axis=stack_axis)


def compute_fft(x, direction="C2C", inverse=False):
  if direction == 'C2R':
    inverse = True
  x_shape = x.get_shape().as_list()
  h, w = x_shape[-2], x_shape[-3]
  x_complex = tf.complex(x[..., 0], x[..., 1])
  if direction == 'C2R':
    out = tf.real(tf.ifft2d(x_complex)) * h * w
    return out
  else:
    if inverse:
      out = stack_real_imag(tf.ifft2d(x_complex)) * h * w
    else:
      out = stack_real_imag(tf.fft2d(x_complex))
    return out


def cdgmm(A, B):
  C_r = A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]
  C_i = A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]
  return tf.stack((C_r, C_i), -1)


def periodize(x, k):
  input_shape = x.get_shape().as_list()
  output_shape = [tf.shape(x)[0], input_shape[1], input_shape[2] // k, input_shape[3] // k]
  reshape_shape = [tf.shape(x)[0], input_shape[1],
                   input_shape[2] // output_shape[2], output_shape[2],
                   input_shape[3] // output_shape[3], output_shape[3]]
  x0 = x[..., 0]
  x1 = x[..., 1]
  y0 = tf.reshape(x0, tf.stack(reshape_shape))
  y1 = tf.reshape(x1, tf.stack(reshape_shape))
  y0 = tf.expand_dims(tf.reduce_mean(tf.reduce_mean(y0, axis=4), axis=2), axis=-1)
  y1 = tf.expand_dims(tf.reduce_mean(tf.reduce_mean(y1, axis=4), axis=2), axis=-1)
  out = tf.concat([y0, y1], axis=-1)
  return out


def modulus(x):
  input_shape = x.get_shape().as_list()
  out = tf.norm(x, axis=len(input_shape) - 1)
  out = tf.expand_dims(out, axis=-1)
  out = tf.concat([out, tf.zeros_like(out)], axis=-1)
  return out
