/* GB10/GX10 DSpark CUDA modifications:
 * Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License. */

#include "ds4_dspark_sps.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void require_true(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "test_dspark_sps: %s\n", message);
        exit(1);
    }
}

static void write_profile(
        const char *path, uint64_t fingerprint, int duplicate,
        int omit_last) {
    FILE *fp = fopen(path, "wb");
    require_true(fp != NULL, "failed to create fixture");
    fprintf(fp, "DS4_DSPARK_SPS_V1 %016llx\n",
            (unsigned long long)fingerprint);
    for (uint32_t executor = 1u; executor <= 2u; executor++) {
        for (uint32_t rows = 2u; rows <= 12u; rows++) {
            if (omit_last && executor == 1u && rows == 12u) continue;
            fprintf(fp, "C %u 0 31 2 %u %.6f %u\n",
                    executor, rows,
                    executor * 0.010 + rows * 0.001,
                    executor * 10u + rows);
        }
    }
    if (duplicate) {
        fputs("C 1 0 31 2 2 0.012 9\n", fp);
    }
    require_true(fclose(fp) == 0, "failed to close fixture");
}

int main(void) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/ds4-sps-test-%ld.conf",
             (long)getpid());
    const uint64_t fingerprint = UINT64_C(0x123456789abcdef0);
    char err[256];
    ds4_dspark_sps_profile profile;
    double curve[DS4_DSPARK_SPS_MAX_ROWS + 1u] = {0};
    uint64_t samples[DS4_DSPARK_SPS_MAX_ROWS + 1u] = {0};

    write_profile(path, fingerprint, 0, 0);
    require_true(ds4_dspark_sps_profile_load(
                     path, fingerprint, &profile,
                     err, sizeof(err)) == 0,
                 "complete profile did not load");
    require_true(profile.record_count == 22u &&
                 profile.complete_groups == 2u,
                 "complete profile accounting is wrong");
    require_true(ds4_dspark_sps_profile_curve(
                     &profile, 1u, 0u, 31u, 2u,
                     curve, samples,
                     DS4_DSPARK_SPS_MAX_ROWS + 1u) == 1,
                 "complete curve was not exposed");
    require_true(curve[2] == 0.012 && curve[12] == 0.022 &&
                 samples[2] == 12u && samples[12] == 22u,
                 "curve values changed during parsing");
    memset(curve, 0, sizeof(curve));
    memset(samples, 0, sizeof(samples));
    require_true(ds4_dspark_sps_profile_curve(
                     &profile, 2u, 0u, 31u, 2u,
                     curve, samples,
                     DS4_DSPARK_SPS_MAX_ROWS + 1u) == 1,
                 "serial executor curve was not exposed");
    require_true(curve[2] == 0.022 && curve[12] == 0.032 &&
                 samples[2] == 22u && samples[12] == 32u,
                 "serial curve values changed during parsing");

    require_true(ds4_dspark_sps_profile_load(
                     path, fingerprint + 1u, &profile,
                     err, sizeof(err)) != 0,
                 "stale fingerprint was accepted");

    write_profile(path, fingerprint, 0, 1);
    require_true(ds4_dspark_sps_profile_load(
                     path, fingerprint, &profile,
                     err, sizeof(err)) == 0,
                 "valid partial profile should remain inspectable");
    memset(curve, 0, sizeof(curve));
    require_true(ds4_dspark_sps_profile_curve(
                     &profile, 1u, 0u, 31u, 2u,
                     curve, NULL,
                     DS4_DSPARK_SPS_MAX_ROWS + 1u) == 0 &&
                 ds4_dspark_sps_profile_curve(
                     &profile, 2u, 0u, 31u, 2u,
                     curve, NULL,
                     DS4_DSPARK_SPS_MAX_ROWS + 1u) == 1,
                 "partial curve must not leak into scheduling");

    write_profile(path, fingerprint, 1, 0);
    require_true(ds4_dspark_sps_profile_load(
                     path, fingerprint, &profile,
                     err, sizeof(err)) != 0,
                 "duplicate profile row was accepted");
    unlink(path);
    puts("dspark SPS profile tests: OK");
    return 0;
}
