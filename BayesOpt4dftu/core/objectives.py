def objective_function_v1(delta_gap=0.0, delta_band=0.0, delta_mag=0.0,
                          alpha_gap=0.5, alpha_band=0.5, alpha_mag=0.0):
    return -alpha_gap * delta_gap ** 2 - alpha_band * delta_band ** 2 - alpha_mag * delta_mag ** 2
