/* GB10/GX10 DSpark CUDA modifications:
 * Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License. */

#ifndef DS4_DSPARK_SPS_H
#define DS4_DSPARK_SPS_H

#include <stddef.h>
#include <stdint.h>

enum {
    DS4_DSPARK_SPS_PROFILE_VERSION = 1,
    DS4_DSPARK_SPS_MAX_REQUESTS = 4,
    DS4_DSPARK_SPS_MAX_PREFIX = 5,
    DS4_DSPARK_SPS_CONTEXT_BUCKETS = 32,
    DS4_DSPARK_SPS_EXECUTORS = 2,
    DS4_DSPARK_SPS_PATHS = 2,
    DS4_DSPARK_SPS_MAX_ROWS =
        DS4_DSPARK_SPS_MAX_REQUESTS *
        (DS4_DSPARK_SPS_MAX_PREFIX + 1),
    DS4_DSPARK_SPS_ENTRY_COUNT =
        DS4_DSPARK_SPS_EXECUTORS *
        DS4_DSPARK_SPS_PATHS *
        DS4_DSPARK_SPS_CONTEXT_BUCKETS *
        DS4_DSPARK_SPS_MAX_REQUESTS *
        (DS4_DSPARK_SPS_MAX_ROWS + 1),
    DS4_DSPARK_SPS_GROUP_COUNT =
        DS4_DSPARK_SPS_EXECUTORS *
        DS4_DSPARK_SPS_PATHS *
        DS4_DSPARK_SPS_CONTEXT_BUCKETS *
        DS4_DSPARK_SPS_MAX_REQUESTS,
};

typedef struct {
    double verify_seconds[DS4_DSPARK_SPS_ENTRY_COUNT];
    uint64_t samples[DS4_DSPARK_SPS_ENTRY_COUNT];
    uint8_t complete[DS4_DSPARK_SPS_GROUP_COUNT];
    uint64_t fingerprint;
    uint32_t record_count;
    uint32_t complete_groups;
} ds4_dspark_sps_profile;

/* Load an immutable offline SPS(B) profile. The file is accepted only when
 * its fingerprint matches and every record is unique and internally valid.
 * Incomplete groups may be present, but production lookup never exposes them.
 */
int ds4_dspark_sps_profile_load(
        const char *path,
        uint64_t expected_fingerprint,
        ds4_dspark_sps_profile *profile,
        char *err,
        size_t errlen);

/* Copy a complete verifier-seconds curve for rows R..R*(K+1). Returning zero
 * means the caller must use one other complete curve for the whole scheduling
 * decision; individual rows are never mixed with fallback estimates.
 */
int ds4_dspark_sps_profile_curve(
        const ds4_dspark_sps_profile *profile,
        uint32_t executor,
        uint32_t path_class,
        uint32_t context_bucket,
        uint32_t request_count,
        double *verify_seconds,
        uint64_t *samples,
        uint32_t value_count);

#endif
