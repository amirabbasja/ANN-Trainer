from .fixed_constants import constant_bytes, constant_text


def s5_nonce():
    return constant_text(b'%FfBc)%eNC$;83')


def s6_noise_seed_prefix():
    return constant_bytes(b'*TTlt%ZS9n)zga6l!(d5&eXz+)WX8Uhy')


def s7_legacy_alias():
    return constant_text(b'*TTlt%g@Ht&j')
