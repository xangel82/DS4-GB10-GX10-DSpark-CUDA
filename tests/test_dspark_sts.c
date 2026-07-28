/* Regression tests for immutable offline DSpark STS profiles. */

#include "ds4_dspark_sts.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void require_true(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "dspark STS regression: %s\n", message);
    exit(1);
}

int main(void) {
    ds4_dspark_sts_profile profile;
    ds4_dspark_sts_profile_init(&profile);
    require_true(profile.loaded == 0 &&
                 profile.temperature[0] == 1.0,
                 "default profile");

    char err[160] = {0};
    require_true(ds4_dspark_sts_profile_load(
                     &profile,
                     "tests/fixtures/dspark_sts_v1.conf",
                     err,
                     sizeof(err)) == 0,
                 err[0] ? err : "valid profile rejected");
    require_true(profile.loaded == 1 &&
                 profile.samples == 4096 &&
                 fabs(profile.temperature[0] - 0.75) < 1.0e-12 &&
                 fabs(profile.temperature[4] - 2.0) < 1.0e-12,
                 "loaded profile contents");

    const double p_fast =
        ds4_dspark_sts_probability(&profile, 0, 2.0);
    const double p_slow =
        ds4_dspark_sts_probability(&profile, 4, 2.0);
    require_true(p_fast > p_slow && p_slow > 0.5,
                 "temperature scaling direction");

    ds4_dspark_sts_profile unchanged = profile;
    require_true(ds4_dspark_sts_profile_load(
                     &profile,
                     "tests/fixtures/does-not-exist.conf",
                     err,
                     sizeof(err)) != 0,
                 "missing profile accepted");
    require_true(profile.samples == unchanged.samples &&
                 profile.temperature[2] == unchanged.temperature[2],
                 "failed load changed active profile");

    require_true(ds4_dspark_sts_profile_load(
                     &profile,
                     "profiles/dspark-sts-q2.conf",
                     err,
                     sizeof(err)) == 0,
                 err[0] ? err : "validated Q2 profile rejected");
    require_true(profile.loaded == 1 &&
                 profile.samples == 2535 &&
                 fabs(profile.temperature[0] - 0.982186182) < 1.0e-9 &&
                 fabs(profile.temperature[2] - 0.679465062) < 1.0e-9 &&
                 fabs(profile.temperature[4] - 0.750068834) < 1.0e-9,
                 "validated Q2 profile contents");

    puts("dspark offline STS regression: OK");
    return 0;
}
