/*
 * ternary_hash.c — Hash arbitrary byte sequences to 9-trit balanced ternary
 *
 * Bridges arbitrary inputs (filesystem paths, codons, strings, integers)
 * into the exact ternary space {-1,0,1}^9 the VAE was trained on.
 *
 * Design:
 *   1. FNV-1a hash → 64-bit integer (exact, deterministic, no floats)
 *   2. Balanced base-3 decomposition of hash mod 3^9 = 19683
 *   3. Map {0,1,2} → {-1,0,1} via offset -1
 *
 * This guarantees:
 *   - Same input always produces same trit sequence
 *   - Output covers the full 19683-space with near-uniform distribution
 *   - No floating point, no approximation
 *   - Compatible with ternary_algebra.h trit encoding
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o ternary_hash.so ternary_hash.c
 *
 * Python usage:
 *   import ctypes
 *   lib = ctypes.CDLL("src/c/ternary_hash.so")
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* 3^9 = 19683 — total ternary operation space */
#define TERNARY_SPACE 19683
#define TRIT_DIGITS   9

/* FNV-1a 64-bit constants */
#define FNV_OFFSET 14695981039346656037ULL
#define FNV_PRIME  1099511628211ULL

/*
 * fnv1a_64 — FNV-1a hash over arbitrary bytes
 * Exact integer arithmetic, no floating point.
 */
static uint64_t fnv1a_64(const uint8_t *data, size_t len) {
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)data[i];
        h *= FNV_PRIME;
    }
    return h;
}

/*
 * to_ternary_9 — Convert any 64-bit hash to 9-digit balanced ternary
 *
 * out[i] in {-1, 0, 1}, i=0 is least significant trit
 * Deterministic: same hash → same sequence
 */
void to_ternary_9(uint64_t hash, int8_t out[TRIT_DIGITS]) {
    uint64_t n = hash % TERNARY_SPACE;
    for (int i = 0; i < TRIT_DIGITS; i++) {
        out[i] = (int8_t)((int)(n % 3) - 1);  /* {0,1,2} → {-1,0,1} */
        n /= 3;
    }
}

/*
 * hash_bytes_to_ternary — Hash arbitrary bytes then convert to ternary
 * Main entry point for external callers.
 */
void hash_bytes_to_ternary(const uint8_t *data, size_t len, int8_t out[TRIT_DIGITS]) {
    uint64_t h = fnv1a_64(data, len);
    to_ternary_9(h, out);
}

/*
 * hash_string_to_ternary — Convenience: hash a null-terminated string
 */
void hash_string_to_ternary(const char *str, int8_t out[TRIT_DIGITS]) {
    hash_bytes_to_ternary((const uint8_t *)str, strlen(str), out);
}

/*
 * hash_uint64_to_ternary — Hash a uint64 directly (for integer inputs)
 */
void hash_uint64_to_ternary(uint64_t value, int8_t out[TRIT_DIGITS]) {
    /* Mix the integer through FNV to avoid trivial patterns */
    uint8_t buf[8];
    for (int i = 0; i < 8; i++) {
        buf[i] = (uint8_t)(value >> (i * 8));
    }
    hash_bytes_to_ternary(buf, 8, out);
}

/*
 * batch_hash_strings — Hash N null-terminated strings in one call
 * out: flat array of N*9 int8_t values
 */
void batch_hash_strings(const char **strings, size_t n, int8_t *out) {
    for (size_t i = 0; i < n; i++) {
        hash_string_to_ternary(strings[i], out + i * TRIT_DIGITS);
    }
}

/*
 * ternary_index — Convert 9-trit sequence to canonical index [0, 19682]
 * Inverse of to_ternary_9 (least significant trit first)
 */
uint32_t ternary_index(const int8_t trits[TRIT_DIGITS]) {
    uint32_t idx = 0;
    uint32_t base = 1;
    for (int i = 0; i < TRIT_DIGITS; i++) {
        idx += (uint32_t)((int)trits[i] + 1) * base;
        base *= 3;
    }
    return idx;
}
