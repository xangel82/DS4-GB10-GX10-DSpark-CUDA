#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_help.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory, performs a fixed greedy decode
 * run without allowing EOS, restores the snapshot, and continues to the next
 * frontier.  Snapshot save/restore time is intentionally outside both timing
 * windows.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *dspark_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    const char *expert_profile_path;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int dspark_draft_tokens;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    double step_mul;
    int *frontiers;
    size_t n_frontiers;
    const char *dump_frontier_logits_dir;
    ds4_dist_options dist;
    bool warm_weights;
    bool quality;
    bool cold_sweep;
    bool ssd_streaming;
    bool ssd_streaming_cold;
} bench_config;

typedef struct {
    uint64_t rss_bytes;
    uint64_t hwm_bytes;
} bench_process_memory;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_BENCH, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static void parse_frontiers(bench_config *c, const char *s, const char *opt) {
    if (!c || !s || !s[0]) {
        fprintf(stderr, "ds4-bench: %s requires a non-empty comma-separated list\n", opt);
        exit(2);
    }
    size_t count = 1;
    for (const char *p = s; *p; p++) if (*p == ',') count++;
    int *values = calloc(count, sizeof(values[0]));
    char *copy = malloc(strlen(s) + 1u);
    if (!values || !copy) {
        fprintf(stderr, "ds4-bench: out of memory parsing %s\n", opt);
        free(values);
        free(copy);
        exit(1);
    }
    strcpy(copy, s);
    size_t n = 0;
    char *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        const int v = parse_int(tok, opt);
        if (n != 0 && v <= values[n - 1]) {
            fprintf(stderr, "ds4-bench: %s values must be strictly increasing\n", opt);
            free(values);
            free(copy);
            exit(2);
        }
        values[n++] = v;
    }
    free(copy);
    if (n != count) {
        fprintf(stderr, "ds4-bench: malformed value for %s: %s\n", opt, s);
        free(values);
        exit(2);
    }
    free(c->frontiers);
    c->frontiers = values;
    c->n_frontiers = n;
    c->ctx_start = values[0];
    c->ctx_max = values[n - 1];
}

static bench_process_memory bench_process_memory_read(void) {
    bench_process_memory out = {0};
#if defined(__linux__)
    FILE *fp = fopen("/proc/self/status", "rb");
    if (!fp) return out;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long kib = 0;
        if (sscanf(line, "VmRSS: %llu kB", &kib) == 1) {
            out.rss_bytes = (uint64_t)kib * 1024u;
        } else if (sscanf(line, "VmHWM: %llu kB", &kib) == 1) {
            out.hwm_bytes = (uint64_t)kib * 1024u;
        }
    }
    fclose(fp);
#endif
    return out;
}

static uint64_t bench_token_hash_update(uint64_t hash, int token) {
    const uint32_t value = (uint32_t)token;
    for (uint32_t shift = 0; shift < 32u; shift += 8u) {
        hash ^= (value >> shift) & 0xffu;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-bench: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dspark")) {
            c.dspark_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dspark-draft")) {
            c.dspark_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.dspark_draft_tokens > 5) {
                fprintf(stderr, "ds4-bench: --dspark-draft must be in 1..5\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--frontiers")) {
            parse_frontiers(&c, need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cold-sweep")) {
            c.cold_sweep = true;
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-bench: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-bench: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-bench: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-bench: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    if (c.dspark_path && c.backend != DS4_BACKEND_CUDA) {
        fprintf(stderr, "ds4-bench: --dspark currently requires the CUDA backend\n");
        exit(2);
    }
    return c;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4-bench: failed to copy frontier logits at %d\n", frontier);
        free(logits);
        return 1;
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = ds4_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            ds4_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4-bench: failed to close %s\n", path);
        free(logits);
        return 1;
    }
    free(logits);
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill(backend,
                                                 ctx_size,
                                                 prefill_chunk);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-bench: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-bench: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-bench: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void maybe_warn_distributed_step_shape(const bench_config *cfg, ds4_session *session) {
    if (!cfg || !session || cfg->dist.role != DS4_DISTRIBUTED_COORDINATOR) return;
    uint32_t chunk = cfg->dist.prefill_chunk;
    if (chunk == 0) {
        const int cap = ds4_session_prefill_cap(session);
        if (cap > 0) chunk = (uint32_t)cap;
    }
    if (chunk == 0) return;
    if (cfg->step_mul == 1.0 &&
        cfg->step_incr > 0 &&
        (uint32_t)cfg->step_incr < chunk &&
        cfg->ctx_start < cfg->ctx_max)
    {
        fprintf(stderr,
                "ds4-bench: note: --step-incr=%d is smaller than distributed prefill chunk %u; "
                "suffix rows will not show multi-chunk pipeline overlap\n",
                cfg->step_incr,
                chunk);
    }
}

static int run_physical_rn_smoke(
        ds4_session *owner,
        const ds4_tokens *prompt) {
    if (!owner || !prompt || prompt->len < 515) return 1;
    uint32_t request_count = 2;
    const char *count_env = getenv("DS4_BENCH_PHYSICAL_RN_SMOKE");
    if (count_env && count_env[0]) {
        char *end = NULL;
        const unsigned long parsed = strtoul(count_env, &end, 10);
        if (end != count_env && *end == '\0' &&
            parsed >= 2u && parsed <= 4u) {
            request_count = (uint32_t)parsed;
        }
    }
    const int prefix_len = prompt->len > 4099 ? 4096 : prompt->len - 3;
    ds4_tokens prefix = {
        .v = prompt->v,
        .len = prefix_len,
        .cap = prefix_len,
    };
    char err[256] = {0};
    if (ds4_session_sync(owner, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-bench: physical R=%u owner prefill failed: %s\n",
                request_count,
                err);
        return 1;
    }

    ds4_session *sessions[4] = {owner, NULL, NULL, NULL};
    uint32_t allocated_sessions = 1;
    int rc = 0;
    for (uint32_t r = 1; r < request_count; r++) {
        if (ds4_session_create_shared(&sessions[r], owner) != 0) {
            fprintf(stderr,
                    "ds4-bench: physical R=%u lane %u allocation failed\n",
                    request_count, r);
            rc = 1;
            break;
        }
        allocated_sessions++;
        if (ds4_session_sync(
                    sessions[r], &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4-bench: physical R=%u lane %u prefill failed: %s\n",
                    request_count, r, err);
            rc = 1;
            break;
        }
    }

    int tops[4][2] = {{0}};
    int sequential_tops[4][2] = {{0}};
    float *logits[4] = {0};
    float *sequential_logits[4] = {0};
    float *row_logits[4] = {0};
    float *sequential_row_logits[4] = {0};
    uint64_t draft_rng[4] = {
        UINT64_C(0x1f123bb5a17d3c41),
        UINT64_C(0x62db1a716f92e5d3),
        UINT64_C(0x9e3779b97f4a7c15),
        UINT64_C(0xd1b54a32d192ed03),
    };
    ds4_physical_draft_request draft_request[4];
    ds4_tokens physical_tokens[4];
    memset(draft_request, 0, sizeof(draft_request));
    memset(physical_tokens, 0, sizeof(physical_tokens));
    for (uint32_t r = 0; rc == 0 && r < request_count; r++) {
        draft_request[r] = (ds4_physical_draft_request) {
            .session = sessions[r],
            .pending_token = prompt->v[prefix_len + (int)r],
            .proposal_count = 2,
            .temperature = 1.0f,
            .min_p = 0.0f,
            .rng = &draft_rng[r],
        };
    }
    if (rc == 0 && request_count > 1) {
        uint64_t rng_before[4] = {0};
        for (uint32_t r = 0; r < request_count; r++) {
            rng_before[r] = draft_rng[r];
        }
        const uint32_t last = request_count - 1u;
        draft_request[last].proposal_count = 0;
        const int invalid_rc = ds4_sessions_prepare_dspark_rn(
            draft_request, request_count, err, sizeof(err));
        draft_request[last].proposal_count = 2;
        if (invalid_rc == 0) {
            fprintf(stderr,
                    "ds4-bench: physical R=%u invalid DSpark request "
                    "was accepted\n",
                    request_count);
            rc = 1;
        }
        for (uint32_t r = 0; r < request_count; r++) {
            if (draft_rng[r] != rng_before[r]) {
                fprintf(stderr,
                        "ds4-bench: physical R=%u DSpark validation "
                        "advanced lane %u RNG\n",
                        request_count, r);
                rc = 1;
            }
        }
    }
    if (rc == 0 &&
        ds4_sessions_prepare_dspark_rn(
                draft_request, request_count, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-bench: physical R=%u DSpark preparation failed: %s\n",
                request_count, err);
        rc = 1;
    }
    for (uint32_t r = 0; rc == 0 && r < request_count; r++) {
        physical_tokens[r].len = prefix_len + 3;
        physical_tokens[r].cap = physical_tokens[r].len;
        physical_tokens[r].v = malloc(
                (size_t)physical_tokens[r].len *
                sizeof(physical_tokens[r].v[0]));
        if (!physical_tokens[r].v) {
            rc = 1;
            continue;
        }
        memcpy(physical_tokens[r].v, prompt->v,
               (size_t)prefix_len * sizeof(prompt->v[0]));
        physical_tokens[r].v[prefix_len] =
            draft_request[r].pending_token;
        physical_tokens[r].v[prefix_len + 1] =
            draft_request[r].draft_tokens[0];
        physical_tokens[r].v[prefix_len + 2] =
            draft_request[r].draft_tokens[1];
    }
    for (uint32_t r = 0; rc == 0 && r < request_count; r++) {
        logits[r] = malloc((size_t)129280 * sizeof(float));
        sequential_logits[r] =
            malloc((size_t)129280 * sizeof(float));
        row_logits[r] =
            malloc((size_t)3 * 129280 * sizeof(float));
        sequential_row_logits[r] =
            malloc((size_t)3 * 129280 * sizeof(float));
        if (!logits[r] || !sequential_logits[r] ||
            !row_logits[r] || !sequential_row_logits[r]) {
            rc = 1;
        }
    }
    ds4_physical_verify_request request[4];
    ds4_physical_verify_request sequential_request[4];
    memset(request, 0, sizeof(request));
    memset(sequential_request, 0, sizeof(sequential_request));
    for (uint32_t r = 0; r < request_count; r++) {
        request[r] = (ds4_physical_verify_request) {
            .session = sessions[r],
            .tokens = &physical_tokens[r],
            .start = (uint32_t)prefix_len,
            .rows = 3,
            .capture_prefixes = 2,
            .row_tops = tops[r],
            .row_logits = row_logits[r],
            .continuation_logits = logits[r],
        };
        sequential_request[r] = request[r];
        sequential_request[r].row_tops = sequential_tops[r];
        sequential_request[r].row_logits =
            sequential_row_logits[r];
        sequential_request[r].continuation_logits =
            sequential_logits[r];
    }

    /* Warm both launch shapes before timing. Frontier restoration makes these
     * calls observational and therefore repeatable. */
    if (rc == 0) {
        rc = ds4_sessions_verify_suffix_rn(
                &sequential_request[0], 1, err, sizeof(err));
    }
    if (rc == 0) {
        rc = ds4_sessions_verify_suffix_rn(
                request, request_count, err, sizeof(err));
    }
    double sequential_seconds = 0.0;
    double physical_seconds = 0.0;
    const uint32_t timing_runs = 3;
    for (uint32_t run = 0; rc == 0 && run < timing_runs; run++) {
        const double sequential_t0 = bench_now_sec();
        for (uint32_t r = 0; rc == 0 && r < request_count; r++) {
            rc = ds4_sessions_verify_suffix_rn(
                    &sequential_request[r], 1, err, sizeof(err));
        }
        sequential_seconds += bench_now_sec() - sequential_t0;

        const double physical_t0 = bench_now_sec();
        if (rc == 0) {
            rc = ds4_sessions_verify_suffix_rn(
                    request, request_count, err, sizeof(err));
        }
        physical_seconds += bench_now_sec() - physical_t0;
    }
    double squared = 0.0;
    double reference = 0.0;
    if (rc == 0) {
        for (uint32_t r = 0; r < request_count; r++) {
            for (uint32_t i = 0; i < 3u * 129280u; i++) {
                const double delta =
                    (double)row_logits[r][i] -
                    (double)sequential_row_logits[r][i];
                squared += delta * delta;
                reference +=
                    (double)sequential_row_logits[r][i] *
                    (double)sequential_row_logits[r][i];
            }
        }
    }
    const double rel_rmse =
        reference > 0.0 ? sqrt(squared / reference) : sqrt(squared);
    bool parity = rc == 0 && isfinite(rel_rmse) &&
                  rel_rmse <= 1.0e-5;
    for (uint32_t r = 0; parity && r < request_count; r++) {
        parity = tops[r][0] == sequential_tops[r][0] &&
                 tops[r][1] == sequential_tops[r][1] &&
                 memcmp(logits[r],
                        row_logits[r] + (size_t)2 * 129280,
                        (size_t)129280 * sizeof(float)) == 0;
    }

    ds4_physical_rejection_request sequential_rejection[4];
    ds4_physical_rejection_request physical_rejection[4];
    memset(sequential_rejection, 0, sizeof(sequential_rejection));
    memset(physical_rejection, 0, sizeof(physical_rejection));
    for (uint32_t r = 0; r < request_count; r++) {
        sequential_rejection[r] = (ds4_physical_rejection_request) {
            .accept_uniforms = draft_request[r].accept_uniforms,
            .residual_uniforms = draft_request[r].residual_uniforms,
            .temperature = 1.0f,
            .min_p = 0.0f,
        };
        physical_rejection[r] = sequential_rejection[r];
    }
    for (uint32_t r = 0; parity && r < request_count; r++) {
        ds4_physical_verify_txn *sequential_txn = NULL;
        if (ds4_sessions_verify_suffix_rn_begin(
                    &sequential_request[r], 1, &sequential_txn,
                    err, sizeof(err)) != 0 ||
            ds4_sessions_verify_suffix_rn_reject(
                    sequential_txn, &sequential_rejection[r],
                    err, sizeof(err)) != 0) {
            parity = false;
        }
        if (sequential_txn &&
            ds4_sessions_verify_suffix_rn_abort(
                    sequential_txn, err, sizeof(err)) != 0) {
            parity = false;
        }
    }
    ds4_physical_verify_txn *rejection_txn = NULL;
    if (parity &&
        ds4_sessions_verify_suffix_rn_begin(
                request, request_count, &rejection_txn,
                err, sizeof(err)) != 0) {
        parity = false;
    }
    if (parity &&
        ds4_sessions_verify_suffix_rn_reject(
                rejection_txn, physical_rejection,
                err, sizeof(err)) != 0) {
        parity = false;
    }
    for (uint32_t r = 0; parity && r < request_count; r++) {
        parity =
            physical_rejection[r].committed_drafts ==
                sequential_rejection[r].committed_drafts &&
            physical_rejection[r].correction_token ==
                sequential_rejection[r].correction_token;
    }
    if (rejection_txn &&
        ds4_sessions_verify_suffix_rn_abort(
                rejection_txn, err, sizeof(err)) != 0) {
        parity = false;
    }

    uint32_t keep_rows[4] = {1, 2, 3, 1};
    ds4_physical_verify_txn *txn = NULL;
    if (parity &&
        ds4_sessions_verify_suffix_rn_begin(
                request, request_count, &txn,
                err, sizeof(err)) != 0) {
        parity = false;
    }
    if (parity &&
        ds4_sessions_verify_suffix_rn_finish(
                txn, keep_rows, err, sizeof(err)) != 0) {
        txn = NULL;
        parity = false;
    } else {
        txn = NULL;
    }
    float *committed_logits =
        parity ? malloc((size_t)129280 * sizeof(float)) : NULL;
    if (parity && !committed_logits) parity = false;
    for (uint32_t r = 0; parity && r < request_count; r++) {
        const int expected_pos = prefix_len + (int)keep_rows[r];
        const ds4_tokens *committed = ds4_session_tokens(sessions[r]);
        parity = ds4_session_pos(sessions[r]) == expected_pos &&
                 committed && committed->len == expected_pos;
        for (uint32_t row = 0; parity && row < keep_rows[r]; row++) {
            parity =
                committed->v[prefix_len + (int)row] ==
                physical_tokens[r].v[prefix_len + (int)row];
        }
        parity = parity &&
            ds4_session_copy_logits(
                    sessions[r], committed_logits, 129280) == 129280 &&
            memcmp(committed_logits,
                   row_logits[r] +
                       (size_t)(keep_rows[r] - 1u) * 129280,
                   (size_t)129280 * sizeof(float)) == 0;
    }
    free(committed_logits);
    if (txn) {
        (void)ds4_sessions_verify_suffix_rn_abort(
                txn, NULL, 0);
    }
    if (!parity) {
        fprintf(stderr,
                "ds4-bench: physical R=%u parity failed "
                "rel-rmse=%.8f error=%s\n",
                request_count,
                rel_rmse,
                err);
        rc = 1;
    } else {
        double private_mib = 0.0;
        for (uint32_t r = 1; r < request_count; r++) {
            private_mib +=
                (double)ds4_session_private_device_bytes(sessions[r]) /
                1048576.0;
        }
        fprintf(stderr,
                "ds4-bench: physical R=%u verifier parity OK "
                "tops=[%d %d] rel-rmse=%.8f "
                "sequential=%.3fms physical=%.3fms speedup=%.3fx "
                "aggregate_rows_per_s=%.2f lane-private-total=%.2f MiB\n",
                request_count,
                tops[0][0], tops[0][1],
                rel_rmse,
                sequential_seconds * 1000.0 / timing_runs,
                physical_seconds * 1000.0 / timing_runs,
                physical_seconds > 0.0
                    ? sequential_seconds / physical_seconds : 0.0,
                physical_seconds > 0.0
                    ? (double)(request_count * 3u * timing_runs) /
                      physical_seconds : 0.0,
                private_mib);
        fprintf(stderr,
                "ds4-bench: physical R=%u transactional commit OK "
                "(device p/q parity, independent keep_rows=1..%u, "
                "no verifier replay)\n",
                request_count,
                request_count < 3u ? request_count : 3u);
    }
    for (uint32_t r = 0; r < request_count; r++) {
        free(logits[r]);
        free(sequential_logits[r]);
        free(row_logits[r]);
        free(sequential_row_logits[r]);
        ds4_tokens_free(&physical_tokens[r]);
    }
    for (uint32_t r = 1; r < allocated_sessions; r++) {
        ds4_session_free(sessions[r]);
    }
    ds4_session_rewind(owner, 0);
    return rc;
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .dspark_path = cfg.dspark_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .prefill_chunk = cfg.prefill_chunk,
        .dspark_draft_tokens = cfg.dspark_draft_tokens,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .expert_profile_path = cfg.expert_profile_path,
        .distributed = cfg.dist,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        return 2;
    }
    ds4_engine *engine = NULL;
    const double startup_t0 = bench_now_sec();
    if (ds4_engine_open(&engine, &opt) != 0) {
        free(cfg.frontiers);
        return 1;
    }
    const double startup_sec = bench_now_sec() - startup_t0;
    const bench_process_memory startup_memory = bench_process_memory_read();
    log_context_memory(cfg.backend, cfg.ctx_alloc, cfg.prefill_chunk);

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    ds4_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
    } else {
        ds4_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "ds4-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    if (getenv("DS4_BENCH_PHYSICAL_RN_SMOKE") != NULL &&
        run_physical_rn_smoke(session, &prompt) != 0) {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    maybe_warn_distributed_step_shape(&cfg, session);
    const bench_process_memory ready_memory = bench_process_memory_read();

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, "mode,ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes,decode_cycles,tokens_per_cycle,greedy_token_hash,startup_sec,startup_rss_bytes,startup_hwm_bytes,ready_rss_bytes,ready_hwm_bytes,prefill_rss_bytes,prefill_hwm_bytes,decode_rss_bytes,decode_hwm_bytes\n");
    fflush(out);

    const int eos = ds4_token_eos(engine);
    const bool distributed = cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR;
    ds4_session_snapshot snap = {0};
    char err[256];
    int previous = 0;
    int rc = 0;

    size_t frontier_i = 0;
    for (int frontier = cfg.n_frontiers ? cfg.frontiers[0] : cfg.ctx_start; ; ) {
        if (cfg.cold_sweep && frontier_i != 0) {
            ds4_session_free(session);
            session = NULL;
            if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
                fprintf(stderr, "ds4-bench: failed to create cold-sweep session at %d\n", frontier);
                rc = 1;
                break;
            }
        }
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        const double prefill_t0 = bench_now_sec();
        if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = cfg.cold_sweep ? frontier : frontier - previous;
        const bench_process_memory prefill_memory = bench_process_memory_read();

        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        if (cfg.gen_tokens > 0 && !distributed && !cfg.cold_sweep) {
            if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_t0 = bench_now_sec();
        int gen_done = 0;
        int decode_cycles = 0;
        uint64_t greedy_token_hash = UINT64_C(1469598103934665603);
        while (gen_done < cfg.gen_tokens) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const int token = ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (ds4_engine_has_dspark(engine) &&
                ds4_engine_dspark_draft_tokens(engine) > 0) {
                int accepted[17];
                int remaining = cfg.gen_tokens - gen_done;
                int n = ds4_session_eval_speculative_argmax(session,
                                                             token,
                                                             remaining,
                                                             -1,
                                                             accepted,
                                                             (int)(sizeof(accepted) / sizeof(accepted[0])),
                                                             err,
                                                             sizeof(err));
                if (n <= 0) {
                    fprintf(stderr, "ds4-bench: DSpark decode at frontier %d failed: %s\n",
                            frontier, n < 0 ? err : "no tokens accepted");
                    rc = 1;
                    break;
                }
                if (n > remaining) n = remaining;
                for (int i = 0; i < n; i++) {
                    greedy_token_hash = bench_token_hash_update(
                        greedy_token_hash, accepted[i]);
                }
                gen_done += n;
            } else {
                if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                greedy_token_hash = bench_token_hash_update(greedy_token_hash, token);
                gen_done++;
            }
            decode_cycles++;
        }
        const double gen_t1 = bench_now_sec();
        const bench_process_memory decode_memory = bench_process_memory_read();
        if (rc != 0) break;

        if (cfg.gen_tokens == 0) {
            /* Pure prefill benchmark: leave the live session at the frontier. */
        } else if (cfg.cold_sweep) {
            /* The next frontier starts from a new session. */
        } else if (distributed) {
            if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: distributed replay restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        } else {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_sec = gen_t1 - gen_t0;
        fprintf(out,
                "%s,%d,%d,%.2f,%d,%.2f,%llu,%d,%.4f,%016llx,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                cfg.cold_sweep ? "cold" : "append",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                gen_done,
                gen_sec > 0.0 ? (double)gen_done / gen_sec : 0.0,
                (unsigned long long)(distributed || cfg.cold_sweep ? 0 : snap.len),
                decode_cycles,
                decode_cycles > 0 ? (double)gen_done / (double)decode_cycles : 0.0,
                (unsigned long long)greedy_token_hash,
                startup_sec,
                (unsigned long long)startup_memory.rss_bytes,
                (unsigned long long)startup_memory.hwm_bytes,
                (unsigned long long)ready_memory.rss_bytes,
                (unsigned long long)ready_memory.hwm_bytes,
                (unsigned long long)prefill_memory.rss_bytes,
                (unsigned long long)prefill_memory.hwm_bytes,
                (unsigned long long)decode_memory.rss_bytes,
                (unsigned long long)decode_memory.hwm_bytes);
        fflush(out);

        previous = cfg.cold_sweep ? 0 : frontier;
        if (cfg.n_frontiers) {
            frontier_i++;
            if (frontier_i >= cfg.n_frontiers) break;
            frontier = cfg.frontiers[frontier_i];
        } else {
            if (frontier >= cfg.ctx_max) break;
            frontier = next_frontier(&cfg, frontier);
        }
    }

    if (out != stdout) fclose(out);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    free(cfg.frontiers);
    return rc;
}
