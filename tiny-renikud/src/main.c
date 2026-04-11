#include <stdio.h>
#include "model.h"

int main(void) {
    // Input is UTF-8; שלום עולם
    const char *text = "שלום עולם";
    char out[256];
    g2p_phonemize(text, out, sizeof(out));
    printf("Input:  %s\n", text);
    printf("Output: %s\n", out);
    return 0;
}
