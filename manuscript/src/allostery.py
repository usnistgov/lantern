import numpy as np

# concentrations (K_A, K_I) are in umol/L; energies (delta_eps_AI, delta_eps_RA) are in k_B*T
K_A_0 = 139  # IPTG binding constant to lacI in active (DNA-binding) state
K_I_0 = 0.53  # IPTG binding constant to lacI in inactive (non-DNA-binding) state
delta_eps_AI_0 = 4.5  # Log of Allosteric Constant; free energy of inactive state - free energy of active state
R_0 = 150  # Number of lacI moleucules
n_0 = 2  # number of IPTG binding sites per complex = number of lacI molecules in each complex
delta_eps_RA_0 = (
    -15.3
)  # For active state of lacI, free energy of binding to operator - free energy of non-specific binding
N_NS_0 = 4.6e6  # number of non-specific binding sites (= "... length in base-pairs of the E. coli genome...")


def fold_change(
    c,
    K_A=K_A_0,
    K_I=K_I_0,
    delta_eps_AI=delta_eps_AI_0,
    R=R_0,
    n=n_0,
    delta_eps_RA=delta_eps_RA_0,
    N_NS=N_NS_0,
):
    a = (1 + c / K_A) ** n
    b = ((1 + c / K_I) ** n) * np.exp(-delta_eps_AI)
    d = R / N_NS * np.exp(-delta_eps_RA)
    f_ch = 1 / (1 + (a / (a + b)) * d)
    return f_ch


def ec50(
    K_A=K_A_0,
    K_I=K_I_0,
    delta_eps_AI=delta_eps_AI_0,
    R=R_0,
    n=n_0,
    delta_eps_RA=delta_eps_RA_0,
    N_NS=N_NS_0,
):
    a = 1 + R / N_NS * np.exp(-delta_eps_RA)
    b = a + (2 * np.exp(-delta_eps_AI) + a) * (K_A / K_I) ** n
    c = 2 * a + np.exp(-delta_eps_AI) + (K_A / K_I) ** n * np.exp(-delta_eps_AI)
    ret = K_A * ((K_A / K_I - 1) / (K_A / K_I - (b / c) ** (1 / n)) - 1)
    return ret


def g0(
    K_A=K_A_0,
    K_I=K_I_0,
    delta_eps_AI=delta_eps_AI_0,
    R=R_0,
    n=n_0,
    delta_eps_RA=delta_eps_RA_0,
    N_NS=N_NS_0,
):
    return fold_change(
        0,
        K_A=K_A,
        K_I=K_I,
        delta_eps_AI=delta_eps_AI,
        R=R,
        n=n,
        delta_eps_RA=delta_eps_RA,
        N_NS=N_NS,
    )


def ginf(
    K_A=K_A_0,
    K_I=K_I_0,
    delta_eps_AI=delta_eps_AI_0,
    R=R_0,
    n=n_0,
    delta_eps_RA=delta_eps_RA_0,
    N_NS=N_NS_0,
):
    return fold_change(
        1e8,
        K_A=K_A,
        K_I=K_I,
        delta_eps_AI=delta_eps_AI,
        R=R,
        n=n,
        delta_eps_RA=delta_eps_RA,
        N_NS=N_NS,
    )


def n_eff(
    K_A=K_A_0,
    K_I=K_I_0,
    delta_eps_AI=delta_eps_AI_0,
    R=R_0,
    n=n_0,
    delta_eps_RA=delta_eps_RA_0,
    N_NS=N_NS_0,
):
    ec = ec50(
        K_A=K_A,
        K_I=K_I,
        delta_eps_AI=delta_eps_AI,
        R=R,
        n=n,
        delta_eps_RA=delta_eps_RA,
        N_NS=N_NS,
    )
    g1 = ginf(
        K_A=K_A,
        K_I=K_I,
        delta_eps_AI=delta_eps_AI,
        R=R,
        n=n,
        delta_eps_RA=delta_eps_RA,
        N_NS=N_NS,
    )
    g2 = g0(
        K_A=K_A,
        K_I=K_I,
        delta_eps_AI=delta_eps_AI,
        R=R,
        n=n,
        delta_eps_RA=delta_eps_RA,
        N_NS=N_NS,
    )
    f_ch1 = fold_change(
        ec * 1.00001,
        K_A=K_A,
        K_I=K_I,
        delta_eps_AI=delta_eps_AI,
        R=R,
        n=n,
        delta_eps_RA=delta_eps_RA,
        N_NS=N_NS,
    )
    f_ch2 = fold_change(
        ec / 1.00001,
        K_A=K_A,
        K_I=K_I,
        delta_eps_AI=delta_eps_AI,
        R=R,
        n=n,
        delta_eps_RA=delta_eps_RA,
        N_NS=N_NS,
    )
    f_ch1 = (f_ch1 - g2) / (g1 - g2)
    f_ch2 = (f_ch2 - g2) / (g1 - g2)
    return np.abs((np.log(f_ch1) - np.log(f_ch2)) / np.log(1.00001))
