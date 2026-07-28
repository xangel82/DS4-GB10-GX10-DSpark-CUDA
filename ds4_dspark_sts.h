/* GB10/GX10 DSpark CUDA modifications:
 * Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License. */

#ifndef DS4_DSPARK_STS_H
#define DS4_DSPARK_STS_H

#include <stddef.h>
#include <stdint.h>

enum {
    DS4_DSPARK_STS_POSITIONS = 5,
    DS4_DSPARK_STS_VERSION = 1,
};

typedef struct {
    double temperature[DS4_DSPARK_STS_POSITIONS];
    double cumulative_ece[DS4_DSPARK_STS_POSITIONS];
    uint64_t samples;
    uint32_t version;
    uint32_t loaded;
} ds4_dspark_sts_profile;

void ds4_dspark_sts_profile_init(ds4_dspark_sts_profile *profile);

/* Load the immutable artifact produced by tools/calibrate_dspark_sts.py.
 * Returns zero on success and leaves profile unchanged on failure. */
int ds4_dspark_sts_profile_load(
        ds4_dspark_sts_profile *profile,
        const char             *path,
        char                   *err,
        size_t                  errlen);

double ds4_dspark_sts_probability(
        const ds4_dspark_sts_profile *profile,
        uint32_t                      position,
        double                        confidence_logit);

#endif
