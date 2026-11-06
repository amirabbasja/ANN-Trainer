from .fixed_constants import constant_bytes, constant_text


def s5_nonce():
    return constant_text('trainer_counter')


def s6_noise_seed_prefix():
    return constant_bytes('noise_seed_domain')


def s7_legacy_alias():
    return constant_text('legacyMethod')
