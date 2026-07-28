/* GB10/GX10 DSpark CUDA modifications:
 * Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License. */

#include "ds4_dspark_sts.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void sts_set_error(char *err, size_t errlen, const char *message) {
    if (err && errlen) snprintf(err, errlen, "%s", message);
}

static char *sts_trim(char *text) {
    while (*text && isspace((unsigned char)*text)) text++;
    char *end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return text;
}

static int sts_parse_u64(const char *text, uint64_t *value) {
    if (!text || !value || !*text) return 1;
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    while (end && isspace((unsigned char)*end)) end++;
    if (errno || end == text || !end || *end != '\0') return 1;
    *value = (uint64_t)parsed;
    return 0;
}

static int sts_parse_vector(
        const char *text,
        double      value[DS4_DSPARK_STS_POSITIONS],
        double      min_value,
        double      max_value) {
    if (!text || !value) return 1;
    for (uint32_t i = 0; i < DS4_DSPARK_STS_POSITIONS; i++) {
        while (*text && (isspace((unsigned char)*text) ||
                         *text == ',' || *text == ';')) {
            text++;
        }
        if (!*text) return 1;
        errno = 0;
        char *end = NULL;
        const double parsed = strtod(text, &end);
        if (errno || end == text || !isfinite(parsed) ||
            parsed < min_value || parsed > max_value) {
            return 1;
        }
        value[i] = parsed;
        text = end;
    }
    while (*text && (isspace((unsigned char)*text) ||
                     *text == ',' || *text == ';')) {
        text++;
    }
    return *text ? 1 : 0;
}

void ds4_dspark_sts_profile_init(ds4_dspark_sts_profile *profile) {
    if (!profile) return;
    memset(profile, 0, sizeof(*profile));
    profile->version = DS4_DSPARK_STS_VERSION;
    for (uint32_t i = 0; i < DS4_DSPARK_STS_POSITIONS; i++) {
        profile->temperature[i] = 1.0;
    }
}

int ds4_dspark_sts_profile_load(
        ds4_dspark_sts_profile *profile,
        const char             *path,
        char                   *err,
        size_t                  errlen) {
    if (!profile || !path || !path[0]) {
        sts_set_error(err, errlen, "missing STS profile path");
        return 1;
    }
    FILE *fp = fopen(path, "r");
    if (!fp) {
        if (err && errlen) {
            snprintf(err, errlen, "cannot open STS profile: %s",
                     strerror(errno));
        }
        return 1;
    }

    ds4_dspark_sts_profile parsed;
    ds4_dspark_sts_profile_init(&parsed);
    int have_magic = 0;
    int have_positions = 0;
    int have_samples = 0;
    int have_temperatures = 0;
    int have_cumulative_ece = 0;
    char line[1024];
    uint32_t line_number = 0;
    int failed = 0;
    while (fgets(line, sizeof(line), fp)) {
        line_number++;
        if (!strchr(line, '\n') && !feof(fp)) {
            sts_set_error(err, errlen, "STS profile line is too long");
            failed = 1;
            break;
        }
        char *text = sts_trim(line);
        if (!*text || *text == '#') continue;
        if (!have_magic) {
            if (strcmp(text, "DS4_DSPARK_STS_V1") != 0) {
                sts_set_error(err, errlen, "invalid STS profile header");
                failed = 1;
            } else {
                have_magic = 1;
            }
            if (failed) break;
            continue;
        }

        char *equals = strchr(text, '=');
        if (!equals) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "invalid STS profile line %u", line_number);
            }
            failed = 1;
            break;
        }
        *equals = '\0';
        char *key = sts_trim(text);
        char *value = sts_trim(equals + 1);
        if (strcmp(key, "positions") == 0) {
            uint64_t positions = 0;
            if (have_positions ||
                sts_parse_u64(value, &positions) != 0 ||
                positions != DS4_DSPARK_STS_POSITIONS) {
                sts_set_error(err, errlen,
                              "STS profile positions must occur once and be 5");
                failed = 1;
                break;
            }
            have_positions = 1;
        } else if (strcmp(key, "samples") == 0) {
            if (have_samples ||
                sts_parse_u64(value, &parsed.samples) != 0 ||
                parsed.samples == 0) {
                sts_set_error(err, errlen,
                              "STS profile samples must occur once and be positive");
                failed = 1;
                break;
            }
            have_samples = 1;
        } else if (strcmp(key, "temperatures") == 0) {
            if (have_temperatures ||
                sts_parse_vector(value, parsed.temperature,
                                 0.05, 20.0) != 0) {
                sts_set_error(err, errlen,
                              "invalid or duplicate STS temperature vector");
                failed = 1;
                break;
            }
            have_temperatures = 1;
        } else if (strcmp(key, "cumulative_ece") == 0) {
            if (have_cumulative_ece ||
                sts_parse_vector(value, parsed.cumulative_ece,
                                 0.0, 1.0) != 0) {
                sts_set_error(err, errlen,
                              "invalid or duplicate STS cumulative ECE vector");
                failed = 1;
                break;
            }
            have_cumulative_ece = 1;
        } else {
            if (err && errlen) {
                snprintf(err, errlen,
                         "unknown STS profile key: %s", key);
            }
            failed = 1;
            break;
        }
    }
    if (ferror(fp) && !failed) {
        sts_set_error(err, errlen, "failed reading STS profile");
        failed = 1;
    }
    fclose(fp);
    if (!failed &&
        (!have_magic || !have_positions ||
         !have_samples || !have_temperatures ||
         !have_cumulative_ece)) {
        sts_set_error(err, errlen, "incomplete STS profile");
        failed = 1;
    }
    if (failed) return 1;
    parsed.loaded = 1;
    *profile = parsed;
    return 0;
}

double ds4_dspark_sts_probability(
        const ds4_dspark_sts_profile *profile,
        uint32_t                      position,
        double                        confidence_logit) {
    double temperature = 1.0;
    if (profile && position < DS4_DSPARK_STS_POSITIONS) {
        temperature = profile->temperature[position];
    }
    if (!isfinite(temperature) || temperature < 0.05 ||
        temperature > 20.0) {
        temperature = 1.0;
    }
    const double scaled = confidence_logit / temperature;
    if (scaled >= 0.0) {
        const double z = exp(-scaled);
        return 1.0 / (1.0 + z);
    }
    const double z = exp(scaled);
    return z / (1.0 + z);
}
