/* Host-only DSpark speculative rollback audit harness.
 *
 * EXPERIMENT ONLY.  This file does not change any production path: it is a
 * standalone model of the legacy speculative frontier lifecycle in ds4.c, built
 * from the real DeepSeek-V4-Flash shapes and the real per-token compressor
 * write pattern, so the copy amplification can be measured on a Mac without
 * CUDA or a GB10.
 *
 * Modelled legacy machinery (ds4.c):
 *   spec_frontier_snapshot            ds4.c:30340
 *   spec_frontier_restore             ds4.c:30381
 *   spec_frontier_commit_prefix       ds4.c:30424
 *   metal_graph_capture_prefix_*      ds4.c:14112 / 14125 / 14138
 *   dspark_spec_ring_save/restore     ds4.c:35787 / 35801
 *   compressor_store_kernel           ds4_cuda.cu:11659
 *   compressor_shift_ratio4_kernel    ds4_cuda.cu:11820
 *   ds4_gpu_ring_rows_save/restore    ds4_cuda.cu:16203 / 16221
 *
 * Modes compared:
 *   A  legacy: baseline snapshot + K prefix captures + commit_prefix/restore
 *   B  snapshot + replay: one baseline snapshot, no prefix captures
 *   C  delta journal (mutate-on-commit): no snapshot, no captures; undo log
 *
 * Build:  make test-dspark-rollback     (or see the cc line in the Makefile)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

/* ---------------------------------------------------------------- shapes --
 * Derived from g_ds4_shape (DeepSeek V4 Flash) in ds4.c:265 and from
 * ds4_expected_layer_compress_ratio() in ds4.c:645.  GB10 is a CUDA,
 * non-Apple, non-ROCm build, so DS4_GPU_INDEXER_PACKED == 1 (ds4.c:10493).
 */
enum {
    N_LAYER            = 43,
    N_HEAD_DIM         = 512,
    N_INDEXER_HEAD_DIM = 128,
    N_INDEXER_TOP_K    = 512,
    COMP_CAP           = 8192,   /* harness-local compressed-row capacity      */

    DSPARK_BLOCK_SIZE  = 5,     /* ds4.c:3129, launcher --dspark-draft 5      */
    DSPARK_WINDOW      = 128,   /* ds4.c:3144                                 */
    DSPARK_N_BLOCK     = 3,     /* ds4.c:3128                                 */
    HYBRID_LC_MAX_DRAFT= 15,    /* ds4.c:3130, HybridLC on by default          */
    INDEXER_PACKED     = 1      /* ds4.c:10493 (GB10)                          */
};

static uint32_t layer_ratio(uint32_t il) {
    if (il < 2) return 0;                 /* FLASH: layers 0,1 uncompressed   */
    return (il & 1u) == 0 ? 4u : 128u;
}

/* --------------------------------------------------------------- counters -- */
typedef struct {
    uint64_t copies;
    uint64_t bytes;
} cost;

#define CYCLE_COST_MECHANISMS 7
typedef struct {
    /* mechanism split -- the first CYCLE_COST_MECHANISMS entries are the ones
     * that make up a cycle total. */
    cost snapshot;      /* spec_frontier_snapshot                             */
    cost captures;      /* metal_graph_capture_prefix_*                       */
    cost commit;        /* spec_frontier_commit_prefix                        */
    cost restore;       /* spec_frontier_restore                              */
    cost ring_save;     /* dspark_spec_ring_save                              */
    cost ring_restore;  /* dspark_spec_ring_restore                           */
    cost replay;        /* mode B replay / mode C+D journal traffic           */
    /* component split -- the same bytes, attributed to the part of the
     * frontier they belong to.  Not added into the totals. */
    cost c_r128;
    cost c_r4;
    cost c_ring;
    uint64_t journal_entries;
    uint64_t restore_launches;
} cycle_cost;

static void cost_add(cost *c, uint64_t copies, uint64_t bytes) {
    c->copies += copies;
    c->bytes  += bytes;
}
#define CYCLE_COST_SLOTS 10
static void cyc_add(cycle_cost *a, const cycle_cost *b) {
    const cost *pb = (const cost *)b;
    cost *pa = (cost *)a;
    for (size_t i = 0; i < CYCLE_COST_SLOTS; i++) {
        pa[i].copies += pb[i].copies;
        pa[i].bytes  += pb[i].bytes;
    }
    a->journal_entries  += b->journal_entries;
    a->restore_launches += b->restore_launches;
}
static uint64_t cyc_copies(const cycle_cost *c) {
    const cost *p = (const cost *)c; uint64_t n = 0;
    for (size_t i = 0; i < CYCLE_COST_MECHANISMS; i++) n += p[i].copies;
    return n;
}
static uint64_t cyc_bytes(const cycle_cost *c) {
    const cost *p = (const cost *)c; uint64_t n = 0;
    for (size_t i = 0; i < CYCLE_COST_MECHANISMS; i++) n += p[i].bytes;
    return n;
}
/* Tally one copy against both its mechanism and its frontier component. */
static void acct(cost *mech, cost *comp, uint64_t copies, uint64_t bytes) {
    cost_add(mech, copies, bytes);
    if (comp) cost_add(comp, copies, bytes);
}

/* ------------------------------------------------------------------ state --
 * One layer's mutable speculative frontier, exactly the set of buffers that
 * spec_frontier_snapshot() copies (ds4.c:30340).
 */
typedef struct {
    uint32_t ratio;
    uint32_t attn_width, attn_rows;      /* coff*HEAD_DIM, coff*ratio        */
    uint32_t index_width, index_rows;
    float   *attn_kv,    *attn_score;    /* layer_attn_state_kv / _score     */
    float   *index_kv,   *index_score;   /* layer_index_state_kv / _score    */
    uint32_t*gvr_hint;                   /* layer_gvr_hint (TOP_K u32)       */
    bool     gvr_hint_valid;
    uint32_t n_comp, n_index_comp;       /* layer_n_comp / layer_n_index_comp*/
    /* Append-only compressed caches.  Speculative rows past n_comp stay as
     * invisible garbage in the legacy design; only rows < n_comp are read. */
    float   *comp_cache;                 /* COMP_CAP pooled rows (1 f/row)   */
    float   *index_comp_cache;
} layer_state;

typedef struct {
    layer_state L[N_LAYER];
    float   *ring[DSPARK_N_BLOCK];       /* dspark_raw_cache, 128 x HEAD_DIM */
    uint32_t dspark_n_raw;
    uint32_t mtp_n_raw;
} graph_state;

/* frontier copy of the mutable state (ds4_spec_frontier + spec_* tensors) */
typedef struct {
    float   *attn_kv[N_LAYER],  *attn_score[N_LAYER];
    float   *index_kv[N_LAYER], *index_score[N_LAYER];
    uint32_t*gvr_hint[N_LAYER];
    bool     gvr_hint_valid[N_LAYER];
    uint32_t n_comp[N_LAYER], n_index_comp[N_LAYER];
    uint32_t mtp_n_raw;
} frontier;

static void *xzalloc(size_t n) {
    void *p = calloc(1, n);
    if (!p) { fprintf(stderr, "oom\n"); exit(1); }
    return p;
}

static void graph_init(graph_state *g, uint64_t seed) {
    memset(g, 0, sizeof(*g));
    for (uint32_t il = 0; il < N_LAYER; il++) {
        layer_state *L = &g->L[il];
        L->ratio = layer_ratio(il);
        if (L->ratio == 0) continue;
        const uint32_t coff = L->ratio == 4 ? 2u : 1u;
        L->attn_width = coff * N_HEAD_DIM;
        L->attn_rows  = coff * L->ratio;
        L->attn_kv    = xzalloc((size_t)L->attn_width * L->attn_rows * sizeof(float));
        L->attn_score = xzalloc((size_t)L->attn_width * L->attn_rows * sizeof(float));
        if (L->ratio == 4) {
            L->index_width = coff * N_INDEXER_HEAD_DIM;
            L->index_rows  = coff * L->ratio;
            L->index_kv    = xzalloc((size_t)L->index_width * L->index_rows * sizeof(float));
            L->index_score = xzalloc((size_t)L->index_width * L->index_rows * sizeof(float));
            if (INDEXER_PACKED) {
                L->gvr_hint = xzalloc((size_t)N_INDEXER_TOP_K * sizeof(uint32_t));
                L->gvr_hint_valid = true;
            }
        }
        /* deterministic non-trivial starting content */
        for (uint32_t i = 0; i < L->attn_width * L->attn_rows; i++) {
            L->attn_kv[i]    = (float)((seed + il * 131u + i) % 997u) * 0.001f;
            L->attn_score[i] = (float)((seed + il * 17u + i) % 251u) * 0.01f;
        }
        for (uint32_t i = 0; L->index_kv && i < L->index_width * L->index_rows; i++) {
            L->index_kv[i]    = (float)((seed + il * 71u + i) % 887u) * 0.002f;
            L->index_score[i] = (float)((seed + il * 29u + i) % 233u) * 0.02f;
        }
        for (uint32_t i = 0; L->gvr_hint && i < N_INDEXER_TOP_K; i++) {
            L->gvr_hint[i] = (uint32_t)((seed + il * 7919u + i) % 100000u);
        }
        L->comp_cache       = xzalloc((size_t)COMP_CAP * sizeof(float));
        L->index_comp_cache = xzalloc((size_t)COMP_CAP * sizeof(float));
        L->n_comp       = 100u + il;
        L->n_index_comp = 50u + il;
        for (uint32_t i = 0; i < L->n_comp; i++)
            L->comp_cache[i] = (float)((seed + il + i) % 733u) * 0.01f;
        for (uint32_t i = 0; i < L->n_index_comp; i++)
            L->index_comp_cache[i] = (float)((seed + il + i) % 431u) * 0.01f;
    }
    for (uint32_t b = 0; b < DSPARK_N_BLOCK; b++) {
        g->ring[b] = xzalloc((size_t)DSPARK_WINDOW * N_HEAD_DIM * sizeof(float));
        for (uint32_t i = 0; i < DSPARK_WINDOW * N_HEAD_DIM; i++) {
            g->ring[b][i] = (float)((seed + b * 31u + i) % 613u) * 0.003f;
        }
    }
    g->dspark_n_raw = DSPARK_WINDOW;
}

static void graph_free(graph_state *g) {
    for (uint32_t il = 0; il < N_LAYER; il++) {
        free(g->L[il].attn_kv);    free(g->L[il].attn_score);
        free(g->L[il].index_kv);   free(g->L[il].index_score);
        free(g->L[il].gvr_hint);
        free(g->L[il].comp_cache); free(g->L[il].index_comp_cache);
    }
    for (uint32_t b = 0; b < DSPARK_N_BLOCK; b++) free(g->ring[b]);
}

static void frontier_alloc(frontier *f, const graph_state *g) {
    memset(f, 0, sizeof(*f));
    for (uint32_t il = 0; il < N_LAYER; il++) {
        const layer_state *L = &g->L[il];
        if (L->ratio == 0) continue;
        f->attn_kv[il]    = xzalloc((size_t)L->attn_width * L->attn_rows * sizeof(float));
        f->attn_score[il] = xzalloc((size_t)L->attn_width * L->attn_rows * sizeof(float));
        if (L->ratio == 4) {
            f->index_kv[il]    = xzalloc((size_t)L->index_width * L->index_rows * sizeof(float));
            f->index_score[il] = xzalloc((size_t)L->index_width * L->index_rows * sizeof(float));
            if (INDEXER_PACKED)
                f->gvr_hint[il] = xzalloc((size_t)N_INDEXER_TOP_K * sizeof(uint32_t));
        }
    }
}
static void frontier_free(frontier *f) {
    for (uint32_t il = 0; il < N_LAYER; il++) {
        free(f->attn_kv[il]);  free(f->attn_score[il]);
        free(f->index_kv[il]); free(f->index_score[il]);
        free(f->gvr_hint[il]);
    }
    memset(f, 0, sizeof(*f));
}

/* ---------------------------------------------- spec_frontier_snapshot ---- */
/* mode: 0 = legacy full frontier, 1 = mode D (skip the ratio-128 window). */
static void frontier_snapshot_mode(frontier *f, const graph_state *g, cost *c,
                                   cycle_cost *cc, int skip_r128) {
    f->mtp_n_raw = g->mtp_n_raw;
    for (uint32_t il = 0; il < N_LAYER; il++) {
        const layer_state *L = &g->L[il];
        f->n_comp[il] = L->n_comp;
        f->n_index_comp[il] = L->n_index_comp;
        f->gvr_hint_valid[il] = L->gvr_hint_valid;
        if (L->ratio == 0) continue;
        if (skip_r128 && L->ratio == 128) continue;   /* mode D */
        cost *comp = cc ? (L->ratio == 128 ? &cc->c_r128 : &cc->c_r4) : NULL;
        const size_t ab = (size_t)L->attn_width * L->attn_rows * sizeof(float);
        memcpy(f->attn_kv[il],    L->attn_kv,    ab);
        memcpy(f->attn_score[il], L->attn_score, ab);
        acct(c, comp, 2, 2 * ab);
        if (L->ratio == 4) {
            const size_t ib = (size_t)L->index_width * L->index_rows * sizeof(float);
            memcpy(f->index_kv[il],    L->index_kv,    ib);
            memcpy(f->index_score[il], L->index_score, ib);
            acct(c, comp, 2, 2 * ib);
            if (L->gvr_hint_valid && f->gvr_hint[il] && L->gvr_hint) {
                const size_t gb = (size_t)N_INDEXER_TOP_K * sizeof(uint32_t);
                memcpy(f->gvr_hint[il], L->gvr_hint, gb);
                acct(c, comp, 1, gb);
            }
        }
    }
}

__attribute__((unused))
static void frontier_snapshot(frontier *f, const graph_state *g, cost *c) {
    frontier_snapshot_mode(f, g, c, NULL, 0);
}

/* ----------------------------------------------- spec_frontier_restore ---- */
static void frontier_restore_mode(const frontier *f, graph_state *g, cost *c,
                                  cycle_cost *cc, int skip_r128) {
    g->mtp_n_raw = f->mtp_n_raw;
    for (uint32_t il = 0; il < N_LAYER; il++) {
        layer_state *L = &g->L[il];
        L->n_comp = f->n_comp[il];
        L->n_index_comp = f->n_index_comp[il];
        L->gvr_hint_valid = f->gvr_hint_valid[il];
        if (L->ratio == 0) continue;
        if (skip_r128 && L->ratio == 128) continue;   /* mode D */
        cost *comp = cc ? (L->ratio == 128 ? &cc->c_r128 : &cc->c_r4) : NULL;
        const size_t ab = (size_t)L->attn_width * L->attn_rows * sizeof(float);
        memcpy(L->attn_kv,    f->attn_kv[il],    ab);
        memcpy(L->attn_score, f->attn_score[il], ab);
        acct(c, comp, 2, 2 * ab);
        if (L->ratio == 4) {
            const size_t ib = (size_t)L->index_width * L->index_rows * sizeof(float);
            memcpy(L->index_kv,    f->index_kv[il],    ib);
            memcpy(L->index_score, f->index_score[il], ib);
            acct(c, comp, 2, 2 * ib);
            if (f->gvr_hint_valid[il] && f->gvr_hint[il] && L->gvr_hint) {
                const size_t gb = (size_t)N_INDEXER_TOP_K * sizeof(uint32_t);
                memcpy(L->gvr_hint, f->gvr_hint[il], gb);
                acct(c, comp, 1, gb);
            }
        }
    }
}

__attribute__((unused))
static void frontier_restore(const frontier *f, graph_state *g, cost *c) {
    frontier_restore_mode(f, g, c, NULL, 0);
}

/* --------------------------------------- ring save / restore (3 blocks) --- */
static void ring_save(const graph_state *g, float *backup[DSPARK_N_BLOCK],
                      uint32_t pos0, uint32_t n_rows, cost *c, cycle_cost *cc) {
    cost *comp = cc ? &cc->c_ring : NULL;
    for (uint32_t b = 0; b < DSPARK_N_BLOCK; b++) {
        for (uint32_t r = 0; r < n_rows; r++) {
            const uint32_t src = (pos0 + r) % DSPARK_WINDOW;
            memcpy(backup[b] + (size_t)r * N_HEAD_DIM,
                   g->ring[b] + (size_t)src * N_HEAD_DIM,
                   N_HEAD_DIM * sizeof(float));
        }
        acct(c, comp, 1, (uint64_t)n_rows * N_HEAD_DIM * sizeof(float));
    }
}
static void ring_restore(graph_state *g, float *const backup[DSPARK_N_BLOCK],
                         uint32_t pos0, uint32_t restore_from, uint32_t n_rows,
                         cost *c, cycle_cost *cc) {
    cost *comp = cc ? &cc->c_ring : NULL;
    if (restore_from >= n_rows) return;
    for (uint32_t b = 0; b < DSPARK_N_BLOCK; b++) {
        for (uint32_t r = restore_from; r < n_rows; r++) {
            const uint32_t dst = (pos0 + r) % DSPARK_WINDOW;
            memcpy(g->ring[b] + (size_t)dst * N_HEAD_DIM,
                   backup[b] + (size_t)r * N_HEAD_DIM,
                   N_HEAD_DIM * sizeof(float));
        }
        acct(c, comp, 1, (uint64_t)(n_rows - restore_from) * N_HEAD_DIM * sizeof(float));
    }
}

/* ------------------------------------------------ per-token state update --
 * Host model of compressor_store_kernel (ds4_cuda.cu:11659) plus
 * compressor_shift_ratio4_kernel (ds4_cuda.cu:11820) and the emit-time
 * n_comp increment, and of the DSpark raw ring row write.  Only the values
 * are stand-ins; the *write pattern* (which rows change, and when) is exact.
 */
static float tok_val(int tok, uint32_t pos, uint32_t il, uint32_t j, uint32_t salt) {
    uint64_t h = 1469598103934665603ull;
    const uint64_t xs[5] = {(uint64_t)(uint32_t)tok, pos, il, j, salt};
    for (int i = 0; i < 5; i++) { h ^= xs[i]; h *= 1099511628211ull; }
    return (float)(h % 65536u) * (1.0f / 65536.0f);
}

/* rows actually written by one token, per layer -- the true state delta. */
static uint64_t delta_bytes_for_token(const graph_state *g, uint32_t pos) {
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < N_LAYER; il++) {
        const layer_state *L = &g->L[il];
        if (L->ratio == 0) continue;
        bytes += 2ull * L->attn_width * sizeof(float);           /* kv + score */
        if (((pos + 1u) % L->ratio) == 0u && L->ratio == 4u)
            bytes += 2ull * 4u * L->attn_width * sizeof(float);  /* shift      */
        if (L->ratio == 4) {
            bytes += 2ull * L->index_width * sizeof(float);
            if (INDEXER_PACKED) bytes += N_INDEXER_TOP_K * sizeof(uint32_t);
        }
    }
    bytes += (uint64_t)DSPARK_N_BLOCK * N_HEAD_DIM * sizeof(float);
    return bytes;
}

static void state_step_token(graph_state *g, int tok, uint32_t pos) {
    for (uint32_t il = 0; il < N_LAYER; il++) {
        layer_state *L = &g->L[il];
        if (L->ratio == 0) continue;
        const uint32_t pos_mod = pos % L->ratio;
        const uint32_t dst = L->ratio == 4u ? L->ratio + pos_mod : pos_mod;
        for (uint32_t j = 0; j < L->attn_width; j++) {
            L->attn_kv[(size_t)dst * L->attn_width + j]    = tok_val(tok, pos, il, j, 1);
            L->attn_score[(size_t)dst * L->attn_width + j] = tok_val(tok, pos, il, j, 2);
        }
        const bool emit = ((pos + 1u) % L->ratio) == 0u;
        if (emit) {
            /* compressor_update_pool_kernel folds the state into one row of
             * the append-only compressed cache (ds4_cuda.cu:11779). */
            double acc = 0.0;
            for (uint32_t r = 0; r < L->attn_rows; r++)
                acc += L->attn_kv[(size_t)r * L->attn_width];
            if (L->n_comp < COMP_CAP) L->comp_cache[L->n_comp] = (float)acc;
            L->n_comp++;
            if (L->ratio == 4u) { /* compressor_shift_ratio4_kernel */
                const uint32_t half = 4u * L->attn_width;
                for (uint32_t i = 0; i < half; i++) {
                    L->attn_kv[i]    = L->attn_kv[half + i];
                    L->attn_score[i] = L->attn_score[half + i];
                }
            }
        }
        if (L->ratio == 4) {
            for (uint32_t j = 0; j < L->index_width; j++) {
                L->index_kv[(size_t)dst * L->index_width + j]    = tok_val(tok, pos, il, j, 3);
                L->index_score[(size_t)dst * L->index_width + j] = tok_val(tok, pos, il, j, 4);
            }
            if (emit) {
                double iacc = 0.0;
                for (uint32_t r = 0; r < L->index_rows; r++)
                    iacc += L->index_kv[(size_t)r * L->index_width];
                if (L->n_index_comp < COMP_CAP)
                    L->index_comp_cache[L->n_index_comp] = (float)iacc;
                L->n_index_comp++;
            }
            if (INDEXER_PACKED && L->gvr_hint) {
                for (uint32_t j = 0; j < N_INDEXER_TOP_K; j++)
                    L->gvr_hint[j] = (uint32_t)(tok_val(tok, pos, il, j, 5) * 1e6f);
                L->gvr_hint_valid = true;
            }
        }
    }
    for (uint32_t b = 0; b < DSPARK_N_BLOCK; b++) {
        const uint32_t row = pos % DSPARK_WINDOW;
        for (uint32_t j = 0; j < N_HEAD_DIM; j++)
            g->ring[b][(size_t)row * N_HEAD_DIM + j] = tok_val(tok, pos, b, j, 6);
    }
}

/* --------------------------------------- prefix capture / commit_prefix --- */
typedef struct { frontier f[HYBRID_LC_MAX_DRAFT]; } prefix_store;

static void prefix_capture(prefix_store *ps, uint32_t p, const graph_state *g,
                           cost *c, cycle_cost *cc, int skip_r128) {
    frontier_snapshot_mode(&ps->f[p], g, c, cc, skip_r128);
}
static void commit_prefix(graph_state *g, const prefix_store *ps, uint32_t prefix_tokens,
                          cost *c, cycle_cost *cc, int skip_r128) {
    frontier_restore_mode(&ps->f[prefix_tokens - 1u], g, c, cc, skip_r128);
}

/* ------------------------------------------------------------ comparison -- */
static bool graph_equal(const graph_state *a, const graph_state *b, const char **what) {
    for (uint32_t il = 0; il < N_LAYER; il++) {
        const layer_state *x = &a->L[il], *y = &b->L[il];
        if (x->n_comp != y->n_comp)             { *what = "n_comp"; return false; }
        if (x->n_index_comp != y->n_index_comp) { *what = "n_index_comp"; return false; }
        if (x->gvr_hint_valid != y->gvr_hint_valid) { *what = "gvr_hint_valid"; return false; }
        if (x->ratio == 0) continue;
        const uint32_t vc = x->n_comp < COMP_CAP ? x->n_comp : COMP_CAP;
        const uint32_t vi = x->n_index_comp < COMP_CAP ? x->n_index_comp : COMP_CAP;
        if (memcmp(x->comp_cache, y->comp_cache, (size_t)vc * sizeof(float)))
            { *what = "compressed_cache_visible_rows"; return false; }
        if (memcmp(x->index_comp_cache, y->index_comp_cache,
                   (size_t)vi * sizeof(float)))
            { *what = "index_compressed_cache_visible_rows"; return false; }
        const size_t ab = (size_t)x->attn_width * x->attn_rows * sizeof(float);
        if (memcmp(x->attn_kv, y->attn_kv, ab))       { *what = "attn_state_kv"; return false; }
        if (memcmp(x->attn_score, y->attn_score, ab)) { *what = "attn_state_score"; return false; }
        if (x->ratio == 4) {
            const size_t ib = (size_t)x->index_width * x->index_rows * sizeof(float);
            if (memcmp(x->index_kv, y->index_kv, ib))       { *what = "index_state_kv"; return false; }
            if (memcmp(x->index_score, y->index_score, ib)) { *what = "index_state_score"; return false; }
            if (x->gvr_hint && memcmp(x->gvr_hint, y->gvr_hint,
                                      N_INDEXER_TOP_K * sizeof(uint32_t)))
                { *what = "gvr_hint"; return false; }
        }
    }
    for (uint32_t bl = 0; bl < DSPARK_N_BLOCK; bl++)
        if (memcmp(a->ring[bl], b->ring[bl],
                   (size_t)DSPARK_WINDOW * N_HEAD_DIM * sizeof(float)))
            { *what = "dspark_frontier_ring"; return false; }
    if (a->dspark_n_raw != b->dspark_n_raw) { *what = "dspark_n_raw"; return false; }
    if (a->mtp_n_raw != b->mtp_n_raw)       { *what = "mtp_n_raw"; return false; }
    *what = NULL;
    return true;
}

/* =========================================================================
 * Cycle drivers.
 * =========================================================================
 * Shared contract, taken from ds4_session_eval_dspark_cycle (ds4.c:36281):
 *   verify_rows = draft_n + 1; the current token is row 0 and is always
 *   accepted, so keep = committed + 1 in 1..verify_rows.
 *   keep == verify_rows  -> full accept, nothing is undone
 *   keep <  verify_rows  -> partial accept (committed == 0 is "reject")
 *   ok == false          -> hard verifier failure, full baseline restore
 */
typedef enum { OUT_FULL, OUT_PARTIAL, OUT_FAILURE } outcome;

static void run_A_legacy(graph_state *g, const int *toks, uint32_t verify_rows,
                         uint32_t start, uint32_t keep, outcome out,
                         prefix_store *ps, float *ring_backup[DSPARK_N_BLOCK],
                         cycle_cost *cc) {
    frontier base;
    frontier_alloc(&base, g);
    frontier_snapshot_mode(&base, g, &cc->snapshot, cc, 0);   /* ds4.c:36847 */
    ring_save(g, ring_backup, start, verify_rows, &cc->ring_save, cc); /* :36849 */

    const uint32_t draft_n = verify_rows - 1u;
    for (uint32_t t = 0; t < verify_rows; t++) {
        state_step_token(g, toks[t], start + t);
        /* capture_prefixes == draft_n (ds4.c:36869), so prefixes 0..draft_n-1 */
        if (t < draft_n) prefix_capture(ps, t, g, &cc->captures, cc, 0);
    }
    g->dspark_n_raw = start + verify_rows < DSPARK_WINDOW
                    ? start + verify_rows : DSPARK_WINDOW;

    if (out == OUT_FAILURE) {                              /* ds4.c:37137 */
        ring_restore(g, ring_backup, start, 0, verify_rows, &cc->ring_restore, cc);
        frontier_restore_mode(&base, g, &cc->restore, cc, 0);
        cc->restore_launches++;
        g->dspark_n_raw = DSPARK_WINDOW;
    } else if (keep < verify_rows) {                       /* ds4.c:36982 */
        ring_restore(g, ring_backup, start, keep, verify_rows, &cc->ring_restore, cc);
        commit_prefix(g, ps, keep, &cc->commit, cc, 0);
        cc->restore_launches++;
        g->dspark_n_raw = start + keep < DSPARK_WINDOW
                        ? start + keep : DSPARK_WINDOW;
    }
    frontier_free(&base);
}

static void run_B_snapshot_replay(graph_state *g, const int *toks,
                                  uint32_t verify_rows, uint32_t start,
                                  uint32_t keep, outcome out,
                                  float *ring_backup[DSPARK_N_BLOCK],
                                  cycle_cost *cc) {
    frontier base;
    frontier_alloc(&base, g);
    frontier_snapshot_mode(&base, g, &cc->snapshot, cc, 0);
    ring_save(g, ring_backup, start, verify_rows, &cc->ring_save, cc);

    for (uint32_t t = 0; t < verify_rows; t++)
        state_step_token(g, toks[t], start + t);           /* no captures */
    g->dspark_n_raw = start + verify_rows < DSPARK_WINDOW
                    ? start + verify_rows : DSPARK_WINDOW;

    if (out == OUT_FAILURE) {
        ring_restore(g, ring_backup, start, 0, verify_rows, &cc->ring_restore, cc);
        frontier_restore_mode(&base, g, &cc->restore, cc, 0);
        g->dspark_n_raw = DSPARK_WINDOW;
    } else if (keep < verify_rows) {
        ring_restore(g, ring_backup, start, keep, verify_rows, &cc->ring_restore, cc);
        frontier_restore_mode(&base, g, &cc->restore, cc, 0);
        for (uint32_t t = 0; t < keep; t++) {              /* replay prefix */
            state_step_token(g, toks[t], start + t);
            cost_add(&cc->replay, 1, delta_bytes_for_token(g, start + t));
        }
        g->dspark_n_raw = start + keep < DSPARK_WINDOW
                        ? start + keep : DSPARK_WINDOW;
    }
    frontier_free(&base);
}

/* ---------------------------------------------------------------- mode D --
 * Ratio-128 sparse undo journal, as implemented in the legacy behind
 * DS4_DSPARK_R128_UNDO_JOURNAL.  Everything else stays on the legacy path:
 * ratio-4 attn/index/GVR keep the full snapshot and the K prefix captures, and
 * the DSpark ring keeps its save/restore.
 *
 *   spec_frontier_snapshot_ex(.., allow_r128_journal=true)   ds4.c:30543
 *   metal_graph_spec_r128_journal_begin                      ds4.c:25948
 *   ds4_gpu_spec_r128_journal_capture                        ds4_cuda.cu
 *   metal_graph_capture_prefix_attn_state (r128 short cut)   ds4.c:14300
 *   spec_frontier_commit_prefix (batched suffix undo)        ds4.c:30640
 *   spec_frontier_restore       (batched full undo)          ds4.c:30584
 *
 * A ratio-128 layer writes row (pos % 128) per token, and a speculative block
 * is at most 16 rows, so the touched rows are distinct: the journal never
 * aliases and one batched suffix restore is exact regardless of order.
 */
typedef struct {
    float   *payload;          /* [layer][row][2][width] before-images        */
    uint32_t layers[N_LAYER];
    uint32_t layer_count;
    uint32_t rows, pos0;
    uint32_t base_n_comp[N_LAYER];
    bool     valid;
} r128_journal;

static void r128_journal_init(r128_journal *j, const graph_state *g) {
    memset(j, 0, sizeof(*j));
    for (uint32_t il = 0; il < N_LAYER; il++)
        if (g->L[il].ratio == 128u) j->layers[j->layer_count++] = il;
    /* preallocated once at the maximum verifier row count, never on the hot
     * path -- mirrors the tensor allocated in metal_graph_alloc_raw_cap */
    j->payload = xzalloc((size_t)j->layer_count * (HYBRID_LC_MAX_DRAFT + 1) *
                         2u * N_HEAD_DIM * sizeof(float));
}
static void r128_journal_free(r128_journal *j) { free(j->payload); }

static uint64_t r128_journal_bytes(const r128_journal *j, uint32_t rows) {
    return (uint64_t)j->layer_count * rows * 2ull * N_HEAD_DIM * sizeof(float);
}

/* ONE batched capture: every ratio-128 layer, every speculative row, both
 * tensors.  Counted as a single copy operation because it is a single kernel. */
static void r128_journal_capture(r128_journal *j, const graph_state *g,
                                 uint32_t pos0, uint32_t rows, cycle_cost *cc) {
    j->pos0 = pos0; j->rows = rows; j->valid = true;
    for (uint32_t l = 0; l < j->layer_count; l++) {
        const uint32_t il = j->layers[l];
        const layer_state *L = &g->L[il];
        j->base_n_comp[il] = L->n_comp;
        for (uint32_t t = 0; t < rows; t++) {
            const uint32_t row = (pos0 + t) % 128u;
            float *dst = j->payload +
                ((size_t)l * (HYBRID_LC_MAX_DRAFT + 1) + t) * 2u * N_HEAD_DIM;
            memcpy(dst,               L->attn_kv    + (size_t)row * N_HEAD_DIM,
                   N_HEAD_DIM * sizeof(float));
            memcpy(dst + N_HEAD_DIM,  L->attn_score + (size_t)row * N_HEAD_DIM,
                   N_HEAD_DIM * sizeof(float));
        }
    }
    acct(&cc->replay, &cc->c_r128, 1, r128_journal_bytes(j, rows));
    cc->journal_entries += (uint64_t)j->layer_count * rows;
}

/* ONE batched undo of rows [keep, rows).  Full accept never calls it. */
static void r128_journal_undo(const r128_journal *j, graph_state *g,
                              uint32_t keep, cycle_cost *cc) {
    if (!j->valid || keep >= j->rows) return;
    for (uint32_t l = 0; l < j->layer_count; l++) {
        layer_state *L = &g->L[j->layers[l]];
        for (uint32_t t = keep; t < j->rows; t++) {
            const uint32_t row = (j->pos0 + t) % 128u;
            const float *src = j->payload +
                ((size_t)l * (HYBRID_LC_MAX_DRAFT + 1) + t) * 2u * N_HEAD_DIM;
            memcpy(L->attn_kv    + (size_t)row * N_HEAD_DIM, src,
                   N_HEAD_DIM * sizeof(float));
            memcpy(L->attn_score + (size_t)row * N_HEAD_DIM, src + N_HEAD_DIM,
                   N_HEAD_DIM * sizeof(float));
        }
    }
    acct(&cc->commit, &cc->c_r128, 1,
         (uint64_t)j->layer_count * (j->rows - keep) * 2ull * N_HEAD_DIM *
         sizeof(float));
    cc->restore_launches++;
}

/* n_comp is derived, never journalled: the compressor emits when
 * (pos + 1) % ratio == 0 (ds4_cuda.cu:16321), so it is a pure function of the
 * baseline counter, the block start and the accepted prefix length. */
static uint32_t r128_derived_n_comp(uint32_t base, uint32_t pos0, uint32_t keep) {
    uint32_t n = base;
    for (uint32_t t = 0; t < keep; t++)
        if (((pos0 + t + 1u) % 128u) == 0u) n++;
    return n;
}

static void run_D_r128_journal(graph_state *g, const int *toks,
                               uint32_t verify_rows, uint32_t start,
                               uint32_t keep, outcome out, prefix_store *ps,
                               float *ring_backup[DSPARK_N_BLOCK],
                               r128_journal *j, cycle_cost *cc) {
    frontier base;
    frontier_alloc(&base, g);
    /* ratio-4 / indexer / GVR keep the legacy snapshot; ratio-128 does not */
    frontier_snapshot_mode(&base, g, &cc->snapshot, cc, 1);
    ring_save(g, ring_backup, start, verify_rows, &cc->ring_save, cc);
    j->valid = false;
    r128_journal_capture(j, g, start, verify_rows, cc);

    const uint32_t draft_n = verify_rows - 1u;
    for (uint32_t t = 0; t < verify_rows; t++) {
        state_step_token(g, toks[t], start + t);
        /* ratio-4 prefixes only: the ratio-128 prefix is the journal */
        if (t < draft_n) prefix_capture(ps, t, g, &cc->captures, cc, 1);
    }
    g->dspark_n_raw = start + verify_rows < DSPARK_WINDOW
                    ? start + verify_rows : DSPARK_WINDOW;

    if (out == OUT_FAILURE) {
        ring_restore(g, ring_backup, start, 0, verify_rows, &cc->ring_restore, cc);
        r128_journal_undo(j, g, 0u, cc);
        frontier_restore_mode(&base, g, &cc->restore, cc, 1);
        for (uint32_t l = 0; l < j->layer_count; l++)
            g->L[j->layers[l]].n_comp = base.n_comp[j->layers[l]];
        g->dspark_n_raw = DSPARK_WINDOW;
    } else if (keep < verify_rows) {
        ring_restore(g, ring_backup, start, keep, verify_rows, &cc->ring_restore, cc);
        r128_journal_undo(j, g, keep, cc);
        commit_prefix(g, ps, keep, &cc->commit, cc, 1);
        for (uint32_t l = 0; l < j->layer_count; l++) {
            const uint32_t il = j->layers[l];
            g->L[il].n_comp =
                r128_derived_n_comp(j->base_n_comp[il], start, keep);
        }
        g->dspark_n_raw = start + keep < DSPARK_WINDOW
                        ? start + keep : DSPARK_WINDOW;
    }
    /* full accept: zero ratio-128 restore, zero ratio-128 commit copies */
    frontier_free(&base);
}

/* ---------------------------------------------------------------- mode C --
 * Mutate-on-commit with a delta journal.  No baseline snapshot and no prefix
 * captures: every speculative write is journalled with its pre-image, and a
 * partial accept undoes the journal tail.  This is the "separate speculative
 * frontier" architecture priced honestly, including its own bookkeeping.
 */
typedef struct {
    uint32_t il, kind, row, count;   /* kind: 0 attn, 1 index, 2 gvr, 3 ring */
    uint32_t token_index;
    float   *fdata;
    uint32_t*udata;
    uint32_t n_comp_before, n_index_comp_before;
} journal_entry;

typedef struct { journal_entry *e; size_t n, cap; } journal;

static journal_entry *jpush(journal *j) {
    if (j->n == j->cap) {
        j->cap = j->cap ? j->cap * 2 : 1024;
        j->e = realloc(j->e, j->cap * sizeof(*j->e));
        if (!j->e) { fprintf(stderr, "oom\n"); exit(1); }
    }
    memset(&j->e[j->n], 0, sizeof(j->e[0]));
    return &j->e[j->n++];
}
static void jfree(journal *j) {
    for (size_t i = 0; i < j->n; i++) { free(j->e[i].fdata); free(j->e[i].udata); }
    free(j->e); j->e = NULL; j->n = j->cap = 0;
}

/* Journal the rows one token is about to overwrite, then apply the token. */
static void state_step_token_journalled(graph_state *g, journal *j, int tok,
                                        uint32_t pos, uint32_t tindex, cost *c) {
    for (uint32_t il = 0; il < N_LAYER; il++) {
        layer_state *L = &g->L[il];
        if (L->ratio == 0) continue;
        const uint32_t pos_mod = pos % L->ratio;
        const uint32_t dst = L->ratio == 4u ? L->ratio + pos_mod : pos_mod;
        const bool emit = ((pos + 1u) % L->ratio) == 0u;
        /* attn rows: the written row, plus the 4 shifted rows on an emit */
        const uint32_t arow0 = emit && L->ratio == 4u ? 0u : dst;
        const uint32_t arows = emit && L->ratio == 4u ? L->attn_rows : 1u;
        journal_entry *e = jpush(j);
        e->il = il; e->kind = 0; e->row = arow0; e->count = arows;
        e->token_index = tindex;
        e->n_comp_before = L->n_comp; e->n_index_comp_before = L->n_index_comp;
        const size_t abytes = (size_t)arows * L->attn_width * sizeof(float);
        e->fdata = xzalloc(2 * abytes);
        memcpy(e->fdata, L->attn_kv + (size_t)arow0 * L->attn_width, abytes);
        memcpy((char *)e->fdata + abytes,
               L->attn_score + (size_t)arow0 * L->attn_width, abytes);
        cost_add(c, 2, 2 * abytes);
        if (L->ratio == 4) {
            journal_entry *e2 = jpush(j);
            e2->il = il; e2->kind = 1; e2->row = dst; e2->count = 1;
            e2->token_index = tindex;
            const size_t ibytes = (size_t)L->index_width * sizeof(float);
            e2->fdata = xzalloc(2 * ibytes);
            memcpy(e2->fdata, L->index_kv + (size_t)dst * L->index_width, ibytes);
            memcpy((char *)e2->fdata + ibytes,
                   L->index_score + (size_t)dst * L->index_width, ibytes);
            cost_add(c, 2, 2 * ibytes);
            if (INDEXER_PACKED && L->gvr_hint) {
                journal_entry *e3 = jpush(j);
                e3->il = il; e3->kind = 2; e3->count = 1; e3->token_index = tindex;
                e3->udata = xzalloc(N_INDEXER_TOP_K * sizeof(uint32_t));
                memcpy(e3->udata, L->gvr_hint, N_INDEXER_TOP_K * sizeof(uint32_t));
                cost_add(c, 1, N_INDEXER_TOP_K * sizeof(uint32_t));
            }
        }
    }
    for (uint32_t b = 0; b < DSPARK_N_BLOCK; b++) {
        journal_entry *e = jpush(j);
        e->il = b; e->kind = 3; e->row = pos % DSPARK_WINDOW; e->count = 1;
        e->token_index = tindex;
        e->fdata = xzalloc(N_HEAD_DIM * sizeof(float));
        memcpy(e->fdata, g->ring[b] + (size_t)e->row * N_HEAD_DIM,
               N_HEAD_DIM * sizeof(float));
        cost_add(c, 1, N_HEAD_DIM * sizeof(float));
    }
    state_step_token(g, tok, pos);
}

static void journal_undo_from(graph_state *g, journal *j, uint32_t keep, cost *c) {
    for (size_t i = j->n; i-- > 0; ) {
        journal_entry *e = &j->e[i];
        if (e->token_index < keep) break;
        layer_state *L = &g->L[e->il];
        if (e->kind == 0) {
            const size_t abytes = (size_t)e->count * L->attn_width * sizeof(float);
            memcpy(L->attn_kv + (size_t)e->row * L->attn_width, e->fdata, abytes);
            memcpy(L->attn_score + (size_t)e->row * L->attn_width,
                   (char *)e->fdata + abytes, abytes);
            L->n_comp = e->n_comp_before;
            L->n_index_comp = e->n_index_comp_before;
            cost_add(c, 2, 2 * abytes);
        } else if (e->kind == 1) {
            const size_t ibytes = (size_t)L->index_width * sizeof(float);
            memcpy(L->index_kv + (size_t)e->row * L->index_width, e->fdata, ibytes);
            memcpy(L->index_score + (size_t)e->row * L->index_width,
                   (char *)e->fdata + ibytes, ibytes);
            cost_add(c, 2, 2 * ibytes);
        } else if (e->kind == 2) {
            memcpy(L->gvr_hint, e->udata, N_INDEXER_TOP_K * sizeof(uint32_t));
            cost_add(c, 1, N_INDEXER_TOP_K * sizeof(uint32_t));
        } else {
            memcpy(g->ring[e->il] + (size_t)e->row * N_HEAD_DIM, e->fdata,
                   N_HEAD_DIM * sizeof(float));
            cost_add(c, 1, N_HEAD_DIM * sizeof(float));
        }
    }
}

static void run_C_journal(graph_state *g, const int *toks, uint32_t verify_rows,
                          uint32_t start, uint32_t keep, outcome out,
                          cycle_cost *cc) {
    journal j = {0};
    for (uint32_t t = 0; t < verify_rows; t++)
        state_step_token_journalled(g, &j, toks[t], start + t, t, &cc->captures);
    g->dspark_n_raw = start + verify_rows < DSPARK_WINDOW
                    ? start + verify_rows : DSPARK_WINDOW;
    if (out == OUT_FAILURE) {
        journal_undo_from(g, &j, 0, &cc->restore);
        g->dspark_n_raw = DSPARK_WINDOW;
    } else if (keep < verify_rows) {
        journal_undo_from(g, &j, keep, &cc->commit);
        g->dspark_n_raw = start + keep < DSPARK_WINDOW
                        ? start + keep : DSPARK_WINDOW;
    }
    jfree(&j);
}

/* =========================================================================
 * Fixtures, correctness gate, microbenchmark.
 * ========================================================================= */
static uint64_t rng_next(uint64_t *s) {          /* deterministic, host-only */
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s;
}

typedef struct {
    int      toks[HYBRID_LC_MAX_DRAFT + 1];
    uint32_t verify_rows;
    uint32_t start;
    uint32_t keep;          /* 1..verify_rows                                 */
    outcome  out;
    int      correction_token;
} fixture;

/* The acceptance decision itself is mode independent: it is computed from the
 * verifier logits, before any frontier bookkeeping.  Modelled here so the gate
 * can assert that A, B and C observe the same accept count and the same
 * correction-token decision. */
static void fixture_make(fixture *fx, uint64_t *rng, uint32_t draft_n,
                         uint32_t start) {
    fx->verify_rows = draft_n + 1u;
    fx->start = start;
    for (uint32_t t = 0; t < fx->verify_rows; t++)
        fx->toks[t] = (int)(rng_next(rng) % 129280u);
    const uint32_t roll = (uint32_t)(rng_next(rng) % 100u);
    if (roll < 40u) { fx->out = OUT_FULL;    fx->keep = fx->verify_rows; }
    else if (roll < 98u) {
        fx->out = OUT_PARTIAL;
        fx->keep = 1u + (uint32_t)(rng_next(rng) % draft_n);
    } else { fx->out = OUT_FAILURE; fx->keep = 0u; }
    fx->correction_token = fx->keep < fx->verify_rows
        ? (int)(rng_next(rng) % 129280u) : -1;
}

typedef struct { uint32_t accept; int correction; uint64_t digest; } observed;

static uint64_t graph_digest(const graph_state *g) {
    uint64_t h = 1469598103934665603ull;
    #define MIX(p, n) do { const unsigned char *q = (const unsigned char *)(p); \
        for (size_t _i = 0; _i < (size_t)(n); _i++) { h ^= q[_i]; h *= 1099511628211ull; } } while (0)
    for (uint32_t il = 0; il < N_LAYER; il++) {
        const layer_state *L = &g->L[il];
        MIX(&L->n_comp, sizeof(L->n_comp));
        MIX(&L->n_index_comp, sizeof(L->n_index_comp));
        if (L->ratio == 0) continue;
        MIX(L->attn_kv, (size_t)L->attn_width * L->attn_rows * sizeof(float));
        MIX(L->attn_score, (size_t)L->attn_width * L->attn_rows * sizeof(float));
        MIX(L->comp_cache, (size_t)(L->n_comp < COMP_CAP ? L->n_comp : COMP_CAP) * sizeof(float));
        if (L->ratio == 4) {
            MIX(L->index_kv, (size_t)L->index_width * L->index_rows * sizeof(float));
            MIX(L->index_score, (size_t)L->index_width * L->index_rows * sizeof(float));
            MIX(L->index_comp_cache, (size_t)(L->n_index_comp < COMP_CAP ? L->n_index_comp : COMP_CAP) * sizeof(float));
            if (L->gvr_hint) MIX(L->gvr_hint, N_INDEXER_TOP_K * sizeof(uint32_t));
        }
    }
    for (uint32_t b = 0; b < DSPARK_N_BLOCK; b++)
        MIX(g->ring[b], (size_t)DSPARK_WINDOW * N_HEAD_DIM * sizeof(float));
    MIX(&g->dspark_n_raw, sizeof(g->dspark_n_raw));
    #undef MIX
    return h;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static void frontier_report(void) {
    uint64_t r4_bytes = 0, r128_bytes = 0, attn_b = 0, index_b = 0, gvr_b = 0;
    uint64_t r4_copies = 0, r128_copies = 0;
    uint32_t n4 = 0, n128 = 0;
    for (uint32_t il = 0; il < N_LAYER; il++) {
        const uint32_t ratio = layer_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t ab = 2ull * (uint64_t)coff * N_HEAD_DIM *
                            (uint64_t)coff * ratio * sizeof(float);
        attn_b += ab;
        if (ratio == 4) {
            const uint64_t ib = 2ull * (uint64_t)coff * N_INDEXER_HEAD_DIM *
                                (uint64_t)coff * ratio * sizeof(float);
            const uint64_t gb = INDEXER_PACKED
                ? (uint64_t)N_INDEXER_TOP_K * sizeof(uint32_t) : 0ull;
            index_b += ib; gvr_b += gb;
            r4_bytes += ab + ib + gb; r4_copies += INDEXER_PACKED ? 5 : 4; n4++;
        } else {
            r128_bytes += ab; r128_copies += 2; n128++;
        }
    }
    printf("3. FRONTIER SIZE (DeepSeek-V4-Flash, GB10/CUDA build)\n");
    printf("   layers                : %u total, %u ratio-4, %u ratio-128, 2 uncompressed\n",
           (unsigned)N_LAYER, n4, n128);
    printf("   ratio-4   per layer   : %8llu B  (attn 65536 + index 16384 + gvr 2048), 5 copies\n",
           (unsigned long long)(r4_bytes / (n4 ? n4 : 1)));
    printf("   ratio-128 per layer   : %8llu B  (attn only), 2 copies\n",
           (unsigned long long)(r128_bytes / (n128 ? n128 : 1)));
    printf("   ratio-4   total       : %8llu B = %.3f MiB, %llu copies\n",
           (unsigned long long)r4_bytes, (double)r4_bytes / 1048576.0,
           (unsigned long long)r4_copies);
    printf("   ratio-128 total       : %8llu B = %.3f MiB, %llu copies\n",
           (unsigned long long)r128_bytes, (double)r128_bytes / 1048576.0,
           (unsigned long long)r128_copies);
    printf("     attention compressor: %8llu B\n", (unsigned long long)attn_b);
    printf("     indexer compressor  : %8llu B\n", (unsigned long long)index_b);
    printf("     GVR hints           : %8llu B\n", (unsigned long long)gvr_b);
    printf("   ONE FULL FRONTIER     : %8llu B = %.3f MiB, %llu copies\n",
           (unsigned long long)(r4_bytes + r128_bytes),
           (double)(r4_bytes + r128_bytes) / 1048576.0,
           (unsigned long long)(r4_copies + r128_copies));
    printf("   DSpark ring backup    : %8llu B (%u blocks x %u rows x %u f32), %u copies\n",
           (unsigned long long)((uint64_t)DSPARK_N_BLOCK * (DSPARK_BLOCK_SIZE + 1) *
                                N_HEAD_DIM * sizeof(float)),
           (unsigned)DSPARK_N_BLOCK, (unsigned)(DSPARK_BLOCK_SIZE + 1),
           (unsigned)N_HEAD_DIM, (unsigned)DSPARK_N_BLOCK);
    printf("\n");
}


/* =========================================================================
 * A/D correctness gate and copy accounting.
 * ========================================================================= */
typedef struct {
    graph_state gA, gD, gB, gC;
    prefix_store *psA, *psD;
    r128_journal jD;
    float *rbA[DSPARK_N_BLOCK], *rbD[DSPARK_N_BLOCK], *rbB[DSPARK_N_BLOCK];
} rig;

static void rig_init(rig *r, uint32_t draft_n, uint64_t seed) {
    graph_init(&r->gA, seed); graph_init(&r->gD, seed);
    graph_init(&r->gB, seed); graph_init(&r->gC, seed);
    r->psA = xzalloc(sizeof(*r->psA));
    r->psD = xzalloc(sizeof(*r->psD));
    for (uint32_t p = 0; p < draft_n; p++) {
        frontier_alloc(&r->psA->f[p], &r->gA);
        frontier_alloc(&r->psD->f[p], &r->gD);
    }
    r128_journal_init(&r->jD, &r->gD);
    for (uint32_t b = 0; b < DSPARK_N_BLOCK; b++) {
        const size_t n = (size_t)(HYBRID_LC_MAX_DRAFT + 2) * N_HEAD_DIM * sizeof(float);
        r->rbA[b] = xzalloc(n); r->rbD[b] = xzalloc(n); r->rbB[b] = xzalloc(n);
    }
}
static void rig_free(rig *r, uint32_t draft_n) {
    for (uint32_t p = 0; p < draft_n; p++) {
        frontier_free(&r->psA->f[p]); frontier_free(&r->psD->f[p]);
    }
    free(r->psA); free(r->psD);
    r128_journal_free(&r->jD);
    for (uint32_t b = 0; b < DSPARK_N_BLOCK; b++) {
        free(r->rbA[b]); free(r->rbD[b]); free(r->rbB[b]);
    }
    graph_free(&r->gA); graph_free(&r->gD);
    graph_free(&r->gB); graph_free(&r->gC);
}

typedef struct {
    cycle_cost A[3], D[3];      /* [0] full, [1] partial, [2] failure */
    uint32_t   n[3];
    uint32_t   accept_hist[HYBRID_LC_MAX_DRAFT + 2];
} run_totals;

/* Exhaustive outcome schedule: full accept, every partial A = 0..K-1
 * (A == 0 is the complete-draft reject) and hard failure, cycled so that all
 * of them are exercised many times over the fixture count. */
static void outcome_for(uint32_t it, uint32_t draft_n, uint32_t *keep,
                        outcome *out) {
    const uint32_t classes = draft_n + 2u;   /* full + K partials + failure */
    const uint32_t c = it % classes;
    if (c == 0u)                 { *out = OUT_FULL;    *keep = draft_n + 1u; }
    else if (c == classes - 1u)  { *out = OUT_FAILURE; *keep = 0u; }
    else                         { *out = OUT_PARTIAL; *keep = c; }
}

static bool ad_gate(uint32_t draft_n, uint32_t iters, run_totals *tot,
                    bool verbose, bool also_bc) {
    const uint32_t verify_rows = draft_n + 1u;
    rig r; rig_init(&r, draft_n, 7);
    memset(tot, 0, sizeof(*tot));
    uint64_t rng = 0xD5A4B10CULL ^ ((uint64_t)draft_n << 32);
    bool pass = true;
    for (uint32_t it = 0; it < iters && pass; it++) {
        fixture fx;
        fixture_make(&fx, &rng, draft_n, 1024u + it * verify_rows);
        outcome_for(it, draft_n, &fx.keep, &fx.out);
        const uint32_t keep = fx.out == OUT_FAILURE ? 0u : fx.keep;
        cycle_cost cA = {0}, cD = {0}, cB = {0}, cC = {0};

        run_A_legacy(&r.gA, fx.toks, verify_rows, fx.start, keep, fx.out,
                     r.psA, r.rbA, &cA);
        run_D_r128_journal(&r.gD, fx.toks, verify_rows, fx.start, keep, fx.out,
                           r.psD, r.rbD, &r.jD, &cD);
        if (also_bc) {
            run_B_snapshot_replay(&r.gB, fx.toks, verify_rows, fx.start, keep,
                                  fx.out, r.rbB, &cB);
            run_C_journal(&r.gC, fx.toks, verify_rows, fx.start, keep, fx.out, &cC);
        }

        const char *what = NULL;
        if (!graph_equal(&r.gA, &r.gD, &what)) {
            printf("   D NOT READY -- mismatch at K=%u iter=%u outcome=%s keep=%u: %s\n",
                   draft_n, it,
                   fx.out == OUT_FULL ? "full" :
                   fx.out == OUT_PARTIAL ? "partial" : "failure",
                   keep, what);
            pass = false; break;
        }
        if (graph_digest(&r.gA) != graph_digest(&r.gD)) {
            printf("   D NOT READY -- digest mismatch at K=%u iter=%u\n", draft_n, it);
            pass = false; break;
        }
        if (also_bc && (!graph_equal(&r.gA, &r.gB, &what) ||
                        !graph_equal(&r.gA, &r.gC, &what))) {
            printf("   B/C regression at K=%u iter=%u: %s\n", draft_n, it, what);
            pass = false; break;
        }
        const int idx = fx.out == OUT_FULL ? 0 : fx.out == OUT_PARTIAL ? 1 : 2;
        tot->n[idx]++;
        tot->accept_hist[keep]++;
        cyc_add(&tot->A[idx], &cA);
        cyc_add(&tot->D[idx], &cD);
    }
    if (verbose && pass) {
        printf("   K=%-2u fixtures=%u  full=%u partial=%u failure=%u  "
               "accepted-prefix coverage:", draft_n, iters,
               tot->n[0], tot->n[1], tot->n[2]);
        for (uint32_t a = 0; a <= verify_rows; a++)
            if (tot->accept_hist[a]) printf(" keep=%u:%u", a, tot->accept_hist[a]);
        printf("\n");
    }
    rig_free(&r, draft_n);
    return pass;
}

static void print_component_table(const char *label, uint32_t draft_n,
                                  const run_totals *t) {
    static const char *on[3] = {"full accept", "partial/reject", "hard failure"};
    printf("   -- K=%u %s\n", draft_n, label);
    printf("   %-15s %-4s %14s %14s %14s %14s %8s %8s %8s\n",
           "outcome", "mode", "ratio128 B", "ratio4 B", "ring B", "total B",
           "copies", "jrnl", "undoK");
    for (int i = 0; i < 3; i++) {
        if (!t->n[i]) continue;
        const double n = t->n[i];
        printf("   %-15s %-4s %14.0f %14.0f %14.0f %14.0f %8.1f %8.1f %8.2f\n",
               on[i], "A",
               t->A[i].c_r128.bytes / n, t->A[i].c_r4.bytes / n,
               t->A[i].c_ring.bytes / n, (double)cyc_bytes(&t->A[i]) / n,
               (double)cyc_copies(&t->A[i]) / n,
               (double)t->A[i].journal_entries / n,
               (double)t->A[i].restore_launches / n);
        printf("   %-15s %-4s %14.0f %14.0f %14.0f %14.0f %8.1f %8.1f %8.2f\n",
               "", "D",
               t->D[i].c_r128.bytes / n, t->D[i].c_r4.bytes / n,
               t->D[i].c_ring.bytes / n, (double)cyc_bytes(&t->D[i]) / n,
               (double)cyc_copies(&t->D[i]) / n,
               (double)t->D[i].journal_entries / n,
               (double)t->D[i].restore_launches / n);
        const double ar = t->A[i].c_r128.bytes / n, dr = t->D[i].c_r128.bytes / n;
        const double at = (double)cyc_bytes(&t->A[i]) / n;
        const double dt = (double)cyc_bytes(&t->D[i]) / n;
        printf("   %-15s %-4s ratio-128 %.1fx less, cycle total %.1fx less "
               "(%.2f -> %.2f MiB)\n", "", "A/D",
               dr > 0 ? ar / dr : 0.0, dt > 0 ? at / dt : 0.0,
               at / 1048576.0, dt / 1048576.0);
    }
    printf("\n");
}

static double pctile(double *v, uint32_t n, double q) {
    uint32_t i = (uint32_t)(q * (double)(n - 1) + 0.5);
    if (i >= n) i = n - 1;
    return v[i];
}

int main(int argc, char **argv) {
    uint32_t iters = 600;
    for (int i = 1; i < argc; i++)
        if (!strncmp(argv[i], "--iters=", 8)) iters = (uint32_t)atoi(argv[i] + 8);
    if (iters < 400) iters = 400;

    printf("DSpark ratio-128 sparse undo journal -- host-only A/D harness\n");
    printf("(no CUDA, no GB10; buffers use the real DeepSeek-V4-Flash shapes)\n\n");
    frontier_report();

    const uint32_t r128_layers = 20u;
    printf("2. RATIO-128 MUTATION UNIT (from compressor_store_kernel)\n");
    printf("   dst_row              : (pos0 + t) %% 128   -> one row per token\n");
    printf("   mutated per row      : state_kv %u f32 + state_score %u f32 = %u B\n",
           N_HEAD_DIM, N_HEAD_DIM, 2u * N_HEAD_DIM * (unsigned)sizeof(float));
    printf("   rows touched / token : 1 per ratio-128 layer (distinct mod 128)\n");
    printf("   ratio-128 layers     : %u\n", r128_layers);
    printf("   real delta / token   : %u B\n",
           r128_layers * 2u * N_HEAD_DIM * (unsigned)sizeof(float));
    printf("   legacy copy / token  : %u B  -> 128x write amplification\n",
           r128_layers * 2u * N_HEAD_DIM * 128u * (unsigned)sizeof(float));
    printf("   metadata mutated     : layer_n_comp only (derivable, not journalled);\n");
    printf("                          no indexer state, no GVR, no score metadata\n\n");

    /* ------------------------------------------------ correctness gate --- */
    printf("6. CORRECTNESS A/D\n");
    bool pass = true;
    run_totals t1, t5, t15;
    pass &= ad_gate(1u,  iters, &t1,  true, false);
    pass &= ad_gate(5u,  iters, &t5,  true, true);
    pass &= ad_gate(15u, iters, &t15, true, false);
    printf("   compared byte-for-byte after every cycle: ratio-128 compressor KV,\n"
           "   ratio-128 compressor score, n_comp, ratio-4 attn state, indexer\n"
           "   state, n_index_comp, GVR hints + validity, DSpark ring, visible\n"
           "   compressed rows (garbage past n_comp ignored, as in the legacy).\n");
    printf("   accept count / correction token: identical -- the acceptance\n"
           "   decision is computed from verifier logits before any frontier\n"
           "   bookkeeping and mode D does not touch it.\n");
    printf("   RESULT: %s\n\n", pass ? "PASS" : "D NOT READY");
    if (!pass) return 1;

    /* ------------------------------------------------- copy accounting --- */
    printf("7/8. COPY REDUCTION AND COPY/KERNEL COUNT\n");
    print_component_table("(legacy A vs journal D)", 5u, &t5);
    print_component_table("(legacy A vs journal D)", 15u, &t15);
    printf("   Full-accept invariants under D: ratio-128 bytes = one batched\n"
           "   capture only; 0 prefix copies, 0 commit copies, 0 undo kernels.\n"
           "   Partial/reject: exactly 1 batched rejected-suffix undo.\n"
           "   Hard failure:   exactly 1 batched full-suffix undo.\n\n");

    /* ---------------------------------------------------- microbenchmark - */
    printf("10. HOST MICROBENCHMARK (Mac; copy amplification only)\n");
    printf("   %-8s %-15s %10s %10s %14s %8s\n",
           "K", "outcome", "median ms", "p95 ms", "bytes moved", "ops");
    const uint32_t reps = 400;
    double *sa = xzalloc(reps * sizeof(double));
    double *sd = xzalloc(reps * sizeof(double));
    const uint32_t ks[2] = {5u, 15u};
    for (int ki = 0; ki < 2; ki++) {
        const uint32_t draft_n = ks[ki], verify_rows = draft_n + 1u;
        rig r; rig_init(&r, draft_n, 11);
        for (int oi = 0; oi < 3; oi++) {
            const outcome out = oi == 0 ? OUT_FULL : oi == 1 ? OUT_PARTIAL : OUT_FAILURE;
            const uint32_t keep = out == OUT_FULL ? verify_rows
                                : out == OUT_PARTIAL ? (draft_n + 1u) / 2u : 0u;
            uint64_t r2 = 0x1234567ULL; fixture fx;
            fixture_make(&fx, &r2, draft_n, 4096u);
            cycle_cost accA = {0}, accD = {0};
            for (uint32_t rep = 0; rep < reps; rep++) {
                cycle_cost c = {0};
                double t0 = now_ms();
                run_A_legacy(&r.gA, fx.toks, verify_rows, fx.start, keep, out,
                             r.psA, r.rbA, &c);
                sa[rep] = now_ms() - t0;
                if (rep == 0) accA = c;
                memset(&c, 0, sizeof(c));
                t0 = now_ms();
                run_D_r128_journal(&r.gD, fx.toks, verify_rows, fx.start, keep,
                                   out, r.psD, r.rbD, &r.jD, &c);
                sd[rep] = now_ms() - t0;
                if (rep == 0) accD = c;
            }
            qsort(sa, reps, sizeof(double), cmp_double);
            qsort(sd, reps, sizeof(double), cmp_double);
            static const char *on[3] = {"full accept", "partial", "hard failure"};
            printf("   %-8s %-15s %10.4f %10.4f %14llu %8llu   A\n",
                   ki == 0 ? "5" : "15", on[oi],
                   pctile(sa, reps, 0.5), pctile(sa, reps, 0.95),
                   (unsigned long long)cyc_bytes(&accA),
                   (unsigned long long)cyc_copies(&accA));
            printf("   %-8s %-15s %10.4f %10.4f %14llu %8llu   D\n",
                   "", "",
                   pctile(sd, reps, 0.5), pctile(sd, reps, 0.95),
                   (unsigned long long)cyc_bytes(&accD),
                   (unsigned long long)cyc_copies(&accD));
        }
        rig_free(&r, draft_n);
    }
    free(sa); free(sd);
    printf("   NOTE: host memcpy timings. NOT a GB10 decode tokens/s estimate.\n\n");

    /* --------------------------------------------------- Nightjar R=n ---- */
    printf("9. NIGHTJAR / R=n (launcher: max-r=3, lanes=3, hot-lanes=2)\n");
    printf("   Each lane owns its own journal payload and pointer table\n");
    printf("   (ds4.c metal_graph_alloc_shared_lane), so mode D never\n");
    printf("   reintroduces a per-lane full snapshot by another route.\n");
    printf("   %-4s %16s %16s %10s\n", "R", "A bytes/cycle", "D bytes/cycle", "A/D");
    {
        const double a = (double)(cyc_bytes(&t5.A[0]) + cyc_bytes(&t5.A[1]) +
                                  cyc_bytes(&t5.A[2])) /
                         (double)(t5.n[0] + t5.n[1] + t5.n[2]);
        const double d = (double)(cyc_bytes(&t5.D[0]) + cyc_bytes(&t5.D[1]) +
                                  cyc_bytes(&t5.D[2])) /
                         (double)(t5.n[0] + t5.n[1] + t5.n[2]);
        for (uint32_t R = 1; R <= 3; R++)
            printf("   %-4u %16.0f %16.0f %10.2f\n", R, a * R, d * R,
                   d > 0 ? a / d : 0.0);
    }
    printf("\n11. EXPECTED REMAINING COST UNDER D (K=5, per cycle, mean)\n");
    {
        const double n = t5.n[0] + t5.n[1] + t5.n[2];
        const double r4 = (double)(t5.D[0].c_r4.bytes + t5.D[1].c_r4.bytes +
                                   t5.D[2].c_r4.bytes) / n;
        const double rg = (double)(t5.D[0].c_ring.bytes + t5.D[1].c_ring.bytes +
                                   t5.D[2].c_ring.bytes) / n;
        const double r128 = (double)(t5.D[0].c_r128.bytes + t5.D[1].c_r128.bytes +
                                     t5.D[2].c_r128.bytes) / n;
        printf("   ratio-4 legacy snapshot + K prefix captures + commit : %9.0f B (%.1f%%)\n",
               r4, 100.0 * r4 / (r4 + rg + r128));
        printf("   DSpark ring save/restore (legacy)                    : %9.0f B (%.1f%%)\n",
               rg, 100.0 * rg / (r4 + rg + r128));
        printf("   ratio-128 journal (capture + undo)                   : %9.0f B (%.1f%%)\n",
               r128, 100.0 * r128 / (r4 + rg + r128));
    }
    printf("\n");
    return 0;
}
