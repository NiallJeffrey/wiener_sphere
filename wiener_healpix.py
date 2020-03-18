import healpy as hp
import numpy as np


def iqu2teb(IQU, nside, lmax=None):
    alms = hp.map2alm(IQU, lmax=lmax, pol=True)
    return hp.alm2map(alms, nside=nside, lmax=lmax, pol=False)

def teb2iqu(TEB, nside, lmax=None):
    alms = hp.map2alm(TEB, lmax=lmax, pol=False)
    return hp.alm2map(alms, nside=nside, lmax=lmax, pol=True)



def messenger_1(data_vec, T_pixel, n_iter, s_cov_diag_grade, nside, noise_bar_diag, noise_diag):
    s = np.zeros(data_vec.shape, dtype='complex')

    T_harmonic_grade = np.ones(hp.map2alm(hp.ud_grade(data_vec.real, nside),
                                          lmax=nside * 3 - 1).shape) * T_pixel[0] / np.float(nside * nside)


    harmonic_operator = (s_cov_diag_grade / (s_cov_diag_grade + T_harmonic_grade))
    pixel_operator_signal = (noise_bar_diag / (noise_bar_diag + T_pixel))
    pixel_operator_data = (T_pixel / (T_pixel + noise_diag))

    for i in range(n_iter):
        t = pixel_operator_data * data_vec + pixel_operator_signal * s
        #     t = hp.ud_grade(t,512)

        t_alm1 = hp.map2alm(t.real, lmax=3 * nside - 1)
        t_alm2 = hp.map2alm(t.imag, lmax=3 * nside - 1)

        s1 = hp.alm2map(harmonic_operator * t_alm1, nside=nside, lmax=nside * 3 - 1, verbose=False)
        s2 = hp.alm2map(harmonic_operator * t_alm2, nside=nside, lmax=nside * 3 - 1, verbose=False)

        s = s1 + 1j * s2

        #     s = hp.ud_grade(s, 128)
        #     _ = hp.mollview(s.imag), plt.show()
        print(np.var(s))

    return s



def messenger_2(data_vec, s_cov_diag, T_ell, noise_diag, T_pixel, noise_bar_diag, nside, n_iter):
    data_vec_QU = np.concatenate([data_vec.real, data_vec.imag])
    s = np.zeros(data_vec_QU.shape, dtype='complex')

    convergence_test = [0.]

    harmonic_operator = s_cov_diag / (s_cov_diag + T_ell)
    pixel_operator_signal = (noise_bar_diag / (noise_bar_diag + T_pixel))
    pixel_operator_data = (T_pixel / (T_pixel + noise_diag))

    for i in range(n_iter):
        t = pixel_operator_data * data_vec_QU + pixel_operator_signal * s  # here t = concat[t_Q, t_U]
        t = np.real(t)
        t = [t[int(t.shape[0] / 2):] * 0., t[:int(t.shape[0] / 2)], t[int(t.shape[0] / 2):]]  # here t = {t_I = 0, t_Q, t_U}
        t = hp.ud_grade(t, nside)  # now upgrade

        t_alm = hp.map2alm(t, lmax=3 * (nside) - 1, pol=True)
        s = harmonic_operator * np.concatenate([t_alm[1], t_alm[2]])
        s = [s[int(s.shape[0] / 2):] * 0., s[:int(s.shape[0] / 2)], s[int(s.shape[0] / 2):]]

        print(np.var(s[0]), np.var(s[1]), np.var(s[2]))
        convergence_test.append(np.var(s[1]))

        s = hp.alm2map(s, nside=nside, lmax=nside * 3 - 1, verbose=False, pol=True)
        # s_qu = np.copy(s)
        s = np.concatenate([s[1], s[2]])

    return s