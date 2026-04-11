#pragma once
// Run Hebrew G2P on a UTF-8 string, write IPA UTF-8 to out_buf.
void g2p_phonemize(const char *utf8_in, char *out_buf, int out_size);
