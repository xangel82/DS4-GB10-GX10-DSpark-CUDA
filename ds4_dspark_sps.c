/* GB10/GX10 DSpark CUDA modifications:
 * Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License. */

#include "ds4_dspark_sps.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static void ds4_dspark_sps_error(
        char *err, size_t errlen, const char *message) {
    if (err && errlen) {
        snprintf(err, errlen, "%s", message ? message : "invalid profile");
    }
}

static int ds4_dspark_sps_coordinates_valid(
        uint32_t executor,
        uint32_t path_class,
        uint32_t context_bucket,
        uint32_t request_count,
        uint32_t rows) {
    return executor >= 1u &&
           executor <= DS4_DSPARK_SPS_EXECUTORS &&
           path_class < DS4_DSPARK_SPS_PATHS &&
           context_bucket < DS4_DSPARK_SPS_CONTEXT_BUCKETS &&
           request_count >= 1u &&
           request_count <= DS4_DSPARK_SPS_MAX_REQUESTS &&
           rows >= request_count &&
           rows <= request_count *
               (DS4_DSPARK_SPS_MAX_PREFIX + 1u);
}

static uint32_t ds4_dspark_sps_group_index(
        uint32_t executor,
        uint32_t path_class,
        uint32_t context_bucket,
        uint32_t request_count) {
    return (((executor - 1u) * DS4_DSPARK_SPS_PATHS + path_class) *
                DS4_DSPARK_SPS_CONTEXT_BUCKETS +
            context_bucket) *
               DS4_DSPARK_SPS_MAX_REQUESTS +
           (request_count - 1u);
}

static uint32_t ds4_dspark_sps_entry_index(
        uint32_t executor,
        uint32_t path_class,
        uint32_t context_bucket,
        uint32_t request_count,
        uint32_t rows) {
    return ds4_dspark_sps_group_index(
               executor, path_class, context_bucket, request_count) *
               (DS4_DSPARK_SPS_MAX_ROWS + 1u) +
           rows;
}

int ds4_dspark_sps_profile_load(
        const char *path,
        uint64_t expected_fingerprint,
        int force_fingerprint,
        ds4_dspark_sps_profile *profile,
        char *err,
        size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!path || !path[0] || !profile || expected_fingerprint == 0u) {
        ds4_dspark_sps_error(err, errlen, "invalid SPS profile arguments");
        return 1;
    }
    memset(profile, 0, sizeof(*profile));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (err && errlen) {
            snprintf(err, errlen, "%s", strerror(errno));
        }
        return 1;
    }

    char line[512];
    unsigned version = 0u;
    unsigned long long fingerprint = 0u;
    if (!fgets(line, sizeof(line), fp) ||
        sscanf(line, "DS4_DSPARK_SPS_V%u %llx",
               &version, &fingerprint) != 2 ||
        version != DS4_DSPARK_SPS_PROFILE_VERSION ||
        fingerprint == 0u ||
        (!force_fingerprint &&
         (uint64_t)fingerprint != expected_fingerprint)) {
        fclose(fp);
        ds4_dspark_sps_error(
                err, errlen, "stale or incompatible SPS profile");
        return 1;
    }

    uint8_t seen[DS4_DSPARK_SPS_ENTRY_COUNT];
    memset(seen, 0, sizeof(seen));
    bool valid = true;
    uint32_t line_number = 1u;
    while (fgets(line, sizeof(line), fp)) {
        line_number++;
        if (line[0] == '\0' || line[0] == '\n' || line[0] == '#') {
            continue;
        }
        unsigned executor = 0u;
        unsigned path_class = 0u;
        unsigned context_bucket = 0u;
        unsigned request_count = 0u;
        unsigned rows = 0u;
        unsigned long long samples = 0u;
        double seconds = 0.0;
        char trailing = '\0';
        if (sscanf(line, "C %u %u %u %u %u %lf %llu %c",
                   &executor, &path_class, &context_bucket,
                   &request_count, &rows, &seconds, &samples,
                   &trailing) != 7 ||
            !ds4_dspark_sps_coordinates_valid(
                    executor, path_class, context_bucket,
                    request_count, rows) ||
            !(seconds > 0.0) || !isfinite(seconds) ||
            samples == 0u) {
            valid = false;
            break;
        }
        const uint32_t index = ds4_dspark_sps_entry_index(
                executor, path_class, context_bucket,
                request_count, rows);
        if (seen[index]) {
            valid = false;
            break;
        }
        seen[index] = 1u;
        profile->verify_seconds[index] = seconds;
        profile->samples[index] = (uint64_t)samples;
        profile->record_count++;
    }
    if (fclose(fp) != 0) valid = false;
    if (!valid || profile->record_count == 0u) {
        memset(profile, 0, sizeof(*profile));
        if (err && errlen) {
            snprintf(err, errlen,
                     "malformed SPS profile near line %u", line_number);
        }
        return 1;
    }

    for (uint32_t executor = 1u;
         executor <= DS4_DSPARK_SPS_EXECUTORS; executor++) {
        for (uint32_t path_class = 0u;
             path_class < DS4_DSPARK_SPS_PATHS; path_class++) {
            for (uint32_t bucket = 0u;
                 bucket < DS4_DSPARK_SPS_CONTEXT_BUCKETS; bucket++) {
                for (uint32_t request_count = 1u;
                     request_count <= DS4_DSPARK_SPS_MAX_REQUESTS;
                     request_count++) {
                    bool complete = true;
                    const uint32_t max_rows =
                        request_count *
                        (DS4_DSPARK_SPS_MAX_PREFIX + 1u);
                    for (uint32_t rows = request_count;
                         rows <= max_rows; rows++) {
                        const uint32_t index =
                            ds4_dspark_sps_entry_index(
                                    executor, path_class, bucket,
                                    request_count, rows);
                        if (!seen[index]) {
                            complete = false;
                            break;
                        }
                    }
                    if (complete) {
                        const uint32_t group =
                            ds4_dspark_sps_group_index(
                                    executor, path_class, bucket,
                                    request_count);
                        profile->complete[group] = 1u;
                        profile->complete_groups++;
                    }
                }
            }
        }
    }
    profile->fingerprint = (uint64_t)fingerprint;
    return 0;
}

int ds4_dspark_sps_profile_curve(
        const ds4_dspark_sps_profile *profile,
        uint32_t executor,
        uint32_t path_class,
        uint32_t context_bucket,
        uint32_t request_count,
        double *verify_seconds,
        uint64_t *samples,
        uint32_t value_count) {
    if (!profile || !verify_seconds ||
        request_count == 0u ||
        request_count > DS4_DSPARK_SPS_MAX_REQUESTS ||
        value_count <=
            request_count * (DS4_DSPARK_SPS_MAX_PREFIX + 1u) ||
        !ds4_dspark_sps_coordinates_valid(
                executor, path_class, context_bucket,
                request_count, request_count)) {
        return 0;
    }
    const uint32_t group = ds4_dspark_sps_group_index(
            executor, path_class, context_bucket, request_count);
    if (!profile->complete[group]) return 0;

    const uint32_t max_rows =
        request_count * (DS4_DSPARK_SPS_MAX_PREFIX + 1u);
    for (uint32_t rows = request_count; rows <= max_rows; rows++) {
        const uint32_t index = ds4_dspark_sps_entry_index(
                executor, path_class, context_bucket,
                request_count, rows);
        verify_seconds[rows] = profile->verify_seconds[index];
        if (samples) samples[rows] = profile->samples[index];
    }
    return 1;
}
