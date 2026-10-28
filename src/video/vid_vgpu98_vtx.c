/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the vertex input of command set 2.0 (the project's
 *          spec/gpu-commands.md, "Vertex input"): the declaration, the
 *          streams, the index buffer and the inline buffer, and what a
 *          draw reads from them.
 *
 *          The host GPU's vertex shader (vgpu98/vgpu98.vert) decodes the
 *          declaration itself; the renderer (vid_vgpu98_r3d.c) hands it
 *          a copy of the bytes the draw's vertices occupy in each stream.
 *          A draw's vertices are numbered within a window, the smallest
 *          run of vertex numbers holding every vertex it uses: an indexed
 *          draw's indices become window numbers here, so the host GPU
 *          never sees the guest's index values, and a vertex the card has
 *          no number for (below 0, or from VGPU98_VERTEX_LIMIT up) takes a
 *          window number past the window and reads as 0.
 *
 *          Runs on the command processor's thread only.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <86box/vid_vgpu98_card.h>
#include <86box/vid_vgpu98_gpu.h>
#include <86box/vid_vgpu98_dev.h>

static uint32_t
type_bytes(uint32_t type)
{
    switch (type) {
        case VGPU98_DT_D3DCOLOR:
        case VGPU98_DT_UBYTE4:
        case VGPU98_DT_SHORT2:
            return 4;
        case VGPU98_DT_SHORT4:
            return 8;
        default:
            return 4 * type; /* FLOAT1-FLOAT4 */
    }
}

int
vgpu98_vin_init(vgpu98_vin_t *v, vgpu98_gpu_t *gpu)
{
    memset(v, 0, sizeof(*v));
    v->gpu = gpu;
    v->inl = calloc(1, VGPU98_INLINE_BYTES);
    return v->inl != NULL;
}

void
vgpu98_vin_close(vgpu98_vin_t *v)
{
    free(v->inl);
    free(v->local);
    free(v->gidx);
    free(v->gv);
    v->inl       = NULL;
    v->local     = NULL;
    v->local_cap = 0;
    v->gidx      = NULL;
    v->gidx_cap  = 0;
    v->gv        = NULL;
    v->gv_cap    = 0;
}

/* A buffer of at least `bytes`, grown as needed; NULL if it cannot be had. */
static uint8_t *
scratch(uint8_t **p, uint32_t *cap, uint32_t bytes)
{
    if (bytes > *cap) {
        uint8_t *n = realloc(*p, bytes);

        if (!n)
            return NULL;
        *p   = n;
        *cap = bytes;
    }
    return *p;
}

void
vgpu98_vin_reset(vgpu98_vin_t *v)
{
    v->have_decl    = 0;
    v->program      = 0;
    v->weights      = 0;
    v->streams_used = 0;
    v->n_elements   = 0;
    memset(v->slot, 0, sizeof(v->slot));
    memset(v->stream, 0, sizeof(v->stream));
    memset(&v->index, 0, sizeof(v->index));
}

/* --- DECLARE, STREAM, INDICES, INLINE ------------------------------------------- */

static int
pkt_declare(vgpu98_vin_t *v, const uint32_t *p, uint32_t len, uint32_t profile)
{
    const uint32_t count    = p[0];
    const int      fixed8   = (profile & VGPU98_PROFILE_FIXED8) != 0;
    const int      programs = (profile & VGPU98_PROFILE_PROGRAMS) != 0;
    uint32_t       seen[VGPU98_DU_LAST22 + 1];
    uint32_t       inputs    = 0; /* 2.2: INPUT elements */
    int            positions = 0;
    int            program   = 0;
    int            second    = 0; /* tweening's second position or normal (2.1) */
    uint32_t       weights   = 0;
    uint32_t       used      = 0;
    uint32_t       extent[VGPU98_MAX_STREAMS];
    uint32_t       i;
    struct {
        int      on;
        uint32_t stream;
        uint32_t type;
        uint32_t offset;
    } slot[VGPU98_SLOTS];

    if (!count || (count > VGPU98_DECL_MAX))
        return VGPU98_ERR_ARGUMENT;
    if (len < 1 + 2 * count)
        return VGPU98_ERR_LENGTH;
    memset(seen, 0, sizeof(seen));
    memset(extent, 0, sizeof(extent));
    memset(slot, 0, sizeof(slot));
    for (i = 0; i < count; i++) {
        const uint32_t d0     = p[1 + 2 * i];
        const uint32_t d1     = p[2 + 2 * i];
        const uint32_t stream = VGPU98_DECL_STREAM(d0);
        const uint32_t type   = VGPU98_DECL_TYPE(d0);
        const uint32_t use    = VGPU98_DECL_USE(d0);
        const uint32_t index  = VGPU98_DECL_INDEX(d0);
        int            s;
        int            ok;
        int            is21;

        if ((d0 & 0xf0) || (d1 >> 16) || (d1 & 3) || (use > VGPU98_DU_LAST22) ||
            (index > ((use == VGPU98_DU_INPUT) ? 15u : 7u)) || !type || (type > VGPU98_DT_LAST8)) {
            vgpu98_gpu_fail_item(v->gpu, 1 + 2 * i);
            return VGPU98_ERR_ARGUMENT;
        }
        /* Command set 2.1's types and uses, and its second position and
           normal, need PROFILE.FIXED8 (checked once the element is one 2.1
           takes: one it refuses too is ERROR_CODE 3 on either). */
        is21 = (type > VGPU98_DT_LAST7) || (use > VGPU98_DU_LAST7) ||
               ((index == 1) && ((use == VGPU98_DU_POSITION) || (use == VGPU98_DU_NORMAL)));
        /* 2.2's INPUT, a vertex program's input register vn (slot n), any
           type, needs PROFILE.PROGRAMS. */
        if ((use == VGPU98_DU_INPUT) && !programs) {
            vgpu98_gpu_fail_item(v->gpu, 1 + 2 * i);
            return VGPU98_ERR_PROFILE;
        }
        switch (use) {
            case VGPU98_DU_INPUT:
                ok = 1;
                s  = (int) index;
                inputs++;
                break;
            case VGPU98_DU_POSITION:
                ok = (type == VGPU98_DT_FLOAT3) && (index <= 1);
                if (index) {
                    s = VGPU98_SLOT_POSITION2;
                    second = 1;
                } else {
                    s       = VGPU98_SLOT_POSITION;
                    program = 1;
                    positions++;
                }
                break;
            case VGPU98_DU_POSITIONT:
                ok = (type == VGPU98_DT_FLOAT4) && !index;
                s  = VGPU98_SLOT_POSITION;
                positions++;
                break;
            case VGPU98_DU_BLENDWEIGHT:
                ok      = (type >= VGPU98_DT_FLOAT1) && (type <= VGPU98_DT_FLOAT4) && !index;
                s       = VGPU98_SLOT_WEIGHT;
                weights = type;
                break;
            case VGPU98_DU_NORMAL:
                ok = (type == VGPU98_DT_FLOAT3) && (index <= 1);
                s  = index ? VGPU98_SLOT_NORMAL2 : VGPU98_SLOT_NORMAL;
                if (index)
                    second = 1;
                break;
            case VGPU98_DU_COLOR:
                ok = (type == VGPU98_DT_D3DCOLOR) && (index <= 1);
                s  = VGPU98_SLOT_DIFFUSE + (int) index;
                break;
            case VGPU98_DU_PSIZE:
                ok = (type == VGPU98_DT_FLOAT1) && !index;
                s  = VGPU98_SLOT_PSIZE;
                break;
            case VGPU98_DU_BLENDINDICES:
                ok = (type == VGPU98_DT_UBYTE4) && !index;
                s  = VGPU98_SLOT_INDICES;
                break;
            default: /* TEXCOORD */
                ok = ((type >= VGPU98_DT_FLOAT1) && (type <= VGPU98_DT_FLOAT4)) || (type == VGPU98_DT_SHORT2) ||
                     (type == VGPU98_DT_SHORT4);
                s  = VGPU98_SLOT_TEX0 + (int) index;
                break;
        }
        if (!ok || (seen[use] & (1u << index)) || (positions > 1)) {
            vgpu98_gpu_fail_item(v->gpu, 1 + 2 * i);
            return VGPU98_ERR_ARGUMENT;
        }
        if (is21 && !fixed8) {
            vgpu98_gpu_fail_item(v->gpu, 1 + 2 * i);
            return VGPU98_ERR_PROFILE;
        }
        seen[use] |= 1u << index;
        slot[s].on     = 1;
        slot[s].stream = stream;
        slot[s].type   = type;
        slot[s].offset = d1;
        used |= 1u << stream;
        if (d1 + type_bytes(type) > extent[stream])
            extent[stream] = d1 + type_bytes(type);
    }
    /* 2.2: a declaration of INPUT elements has nothing else, and needs no
       position: it feeds a vertex program (program 2). Otherwise one
       position; a second position or normal only with an untransformed one
       (tweening). */
    if (inputs) {
        if (inputs != count)
            return VGPU98_ERR_ARGUMENT;
        program = 2;
    } else if ((positions != 1) || (second && !program))
        return VGPU98_ERR_ARGUMENT;

    v->have_decl    = 1;
    v->program      = program;
    v->weights      = weights;
    v->streams_used = used;
    v->n_elements   = count;
    memcpy(v->decl, p + 1, 2 * count * sizeof(uint32_t));
    for (i = 0; i < VGPU98_SLOTS; i++) {
        v->slot[i].on     = slot[i].on;
        v->slot[i].stream = slot[i].stream;
        v->slot[i].type   = slot[i].type;
        v->slot[i].offset = slot[i].offset;
    }
    for (i = 0; i < VGPU98_MAX_STREAMS; i++)
        v->stream[i].extent = extent[i];
    return 0;
}

static int
pkt_stream(vgpu98_vin_t *v, const uint32_t *p, uint32_t profile)
{
    const uint32_t s = p[0];

    if ((s >= VGPU98_MAX_STREAMS) || (p[1] > VGPU98_SRC_GPUVM) || (p[2] & 3) || (p[4] > VGPU98_MAX_STRIDE) || (p[4] & 3))
        return VGPU98_ERR_ARGUMENT;
    if ((p[1] == VGPU98_SRC_GPUVM) && !(profile & VGPU98_PROFILE_GPUVM))
        return VGPU98_ERR_PROFILE;
    if ((p[1] == VGPU98_SRC_INLINE) && ((uint64_t) p[2] + p[3] > VGPU98_INLINE_BYTES))
        return VGPU98_ERR_ARGUMENT;
    v->stream[s].set     = 1;
    v->stream[s].source  = p[1];
    v->stream[s].address = p[2];
    v->stream[s].size    = p[3];
    v->stream[s].stride  = p[4];
    v->stream[s].version = p[5];
    return 0;
}

static int
pkt_indices(vgpu98_vin_t *v, const uint32_t *p, uint32_t profile)
{
    const uint32_t isize = p[0] & VGPU98_IDX_SIZE_MASK;
    const int      gpuvm = (p[0] & VGPU98_IDX_GPUVM) != 0;

    if ((p[0] & ~(VGPU98_IDX_SIZE_MASK | VGPU98_IDX_GPUVM)) || ((isize != 2) && (isize != 4)) || (p[1] % isize))
        return VGPU98_ERR_ARGUMENT;
    if (gpuvm && !(profile & VGPU98_PROFILE_GPUVM))
        return VGPU98_ERR_PROFILE;
    if (!gpuvm && ((uint64_t) p[1] + p[2] > VGPU98_INLINE_BYTES))
        return VGPU98_ERR_ARGUMENT;
    v->index.set     = 1;
    v->index.isize   = isize;
    v->index.source  = gpuvm ? VGPU98_SRC_GPUVM : VGPU98_SRC_INLINE;
    v->index.address = p[1];
    v->index.size    = p[2];
    v->index.version = p[3];
    return 0;
}

int
vgpu98_vin_packet(vgpu98_vin_t *v, uint32_t op, const uint32_t *p, uint32_t len, uint32_t profile)
{
    switch (op) {
        case VGPU98_OP_DECLARE:
            return pkt_declare(v, p, len, profile);
        case VGPU98_OP_STREAM:
            return pkt_stream(v, p, profile);
        case VGPU98_OP_INDICES:
            return pkt_indices(v, p, profile);
        case VGPU98_OP_INLINE: {
            const uint32_t off   = p[0];
            const uint64_t bytes = (uint64_t) (len - 1) * 4;

            if ((off & 3) || ((uint64_t) off + bytes > VGPU98_INLINE_BYTES))
                return VGPU98_ERR_ARGUMENT;
            memcpy(v->inl + off, p + 1, (size_t) bytes);
            return 0;
        }
        default:
            return VGPU98_ERR_OPCODE;
    }
}

/* --- DRAW ------------------------------------------------------------------- */

/* The most bytes of its streams at GPU addresses one draw reads: a host
   limit far above any real draw (the vertices past it read as 0). */
#define GPUVM_DRAW_BYTES (64u << 20)

int
vgpu98_vin_check(vgpu98_vin_t *v, const uint32_t *p, uint32_t len, vgpu98_vin_draw_t *d)
{
    const uint32_t control = p[0];
    const uint32_t prim    = control & 0xff;
    const uint32_t start   = p[1];
    const uint32_t count   = p[2];
    const int32_t  base    = (int32_t) p[3];
    const int      edges   = (control & VGPU98_DRAW_EDGES) != 0;
    uint32_t       ntri    = 0;
    uint32_t       s;

    if ((control & ~(0xffu | VGPU98_DRAW_INDEXED | VGPU98_DRAW_EDGES)) || (prim < VGPU98_PRIM_POINTS) ||
        (prim > VGPU98_PRIM_TRIANGLE_FAN) || (edges && (prim < VGPU98_PRIM_TRIANGLES)) || (count > VGPU98_DRAW_MAX_COUNT))
        return VGPU98_ERR_ARGUMENT;
    if (!v->have_decl)
        return VGPU98_ERR_ARGUMENT;
    for (s = 0; s < VGPU98_MAX_STREAMS; s++) {
        if (((v->streams_used >> s) & 1) && !v->stream[s].set)
            return VGPU98_ERR_ARGUMENT;
    }
    if ((control & VGPU98_DRAW_INDEXED) && !v->index.set)
        return VGPU98_ERR_ARGUMENT;
    if (prim == VGPU98_PRIM_TRIANGLES)
        ntri = count / 3;
    else if (prim >= VGPU98_PRIM_TRIANGLE_STRIP)
        ntri = (count >= 3) ? count - 2 : 0;
    if (len < 4 + (edges ? (ntri + 7) / 8 : 0))
        return VGPU98_ERR_LENGTH;

    memset(d, 0, sizeof(*d));
    d->prim    = prim;
    d->count   = count;
    d->start   = start;
    d->base    = base;
    d->indexed = (control & VGPU98_DRAW_INDEXED) != 0;
    d->edges   = edges ? (p + 4) : NULL;
    d->ntri    = ntri;

    if (!d->indexed) {
        /* Vertices start to start + count - 1: the window itself. */
        d->first  = start;
        d->nverts = count;
        if (start >= VGPU98_VERTEX_LIMIT)
            d->nverts = 0;
        else if ((uint64_t) start + count > VGPU98_VERTEX_LIMIT)
            d->nverts = VGPU98_VERTEX_LIMIT - start;
        d->outside = d->nverts < count;
    }
    return 0;
}

int
vgpu98_vin_draw(vgpu98_vin_t *v, const uint32_t *p, uint32_t len, vgpu98_vin_draw_t *d)
{
    int err;

    if ((err = vgpu98_vin_check(v, p, len, d)))
        return err;
    return vgpu98_vin_window(v, d);
}

int
vgpu98_vin_window(vgpu98_vin_t *v, vgpu98_vin_draw_t *d)
{
    const uint32_t start = d->start;
    const uint32_t count = d->count;
    const int32_t  base  = d->base;

    if (!d->indexed)
        return 0;
    /* Indexed: the index values first (an index past the end of its buffer
       reads 0), then the window of the vertex numbers base + index the card
       has, and every index as a window number. */
    {
        const uint32_t isize = v->index.isize;
        const uint64_t have  = v->index.size / isize; /* indices in the buffer */
        const uint8_t *src   = NULL;                  /* index `start`'s bytes */
        uint32_t       n_in  = 0;                     /* the draw's indices the buffer holds, from `start` */
        int64_t        lo    = INT64_MAX;
        int64_t        hi    = -1;
        uint32_t       i;

        if (count > v->local_cap) {
            uint32_t *n = realloc(v->local, (size_t) count * sizeof(uint32_t));

            if (!n)
                return VGPU98_ERR_ARGUMENT;
            v->local     = n;
            v->local_cap = count;
        }
        if (start < have) {
            n_in = (have - start < count) ? (uint32_t) (have - start) : count;
            if (v->index.source == VGPU98_SRC_INLINE)
                src = v->inl + v->index.address + (uint64_t) start * isize;
            else if (v->index_res)
                src = v->index_res + (uint64_t) start * isize; /* resident: in the renderer's arena */
            else {
                /* From guest RAM through the page table. */
                const uint64_t at    = (uint64_t) v->index.address + (uint64_t) start * isize;
                const uint32_t bytes = n_in * isize;
                int            err;

                if (at + bytes > 0x100000000ull)
                    return VGPU98_ERR_UNMAPPED;
                if (!(src = scratch(&v->gidx, &v->gidx_cap, bytes)))
                    return VGPU98_ERR_ARGUMENT;
                if ((err = vgpu98_gpu_gpuvm_read(v->gpu, (uint32_t) at, bytes, v->gidx, VGPU98_GPUVM_INDICES)))
                    return err;
            }
        }
        for (i = 0; i < count; i++) {
            uint32_t x = 0;
            int64_t  n;

            if (i < n_in) {
                const uint8_t *b = src + (size_t) i * isize;

                x = (isize == 2) ? (uint32_t) (b[0] | (b[1] << 8))
                                 : (uint32_t) (b[0] | (b[1] << 8) | (b[2] << 16) | ((uint32_t) b[3] << 24));
            }
            v->local[i] = x;
            n           = (int64_t) base + x;
            if ((n >= 0) && (n < VGPU98_VERTEX_LIMIT)) {
                if (n < lo)
                    lo = n;
                if (n > hi)
                    hi = n;
            }
        }
        if (hi < 0) {
            d->first  = 0;
            d->nverts = 0;
        } else {
            d->first  = lo;
            d->nverts = (uint32_t) (hi - lo + 1);
        }
        for (i = 0; i < count; i++) {
            const int64_t n = (int64_t) base + v->local[i];

            if ((n >= 0) && (n < VGPU98_VERTEX_LIMIT))
                v->local[i] = (uint32_t) (n - d->first);
            else {
                v->local[i] = d->nverts;
                d->outside  = 1;
            }
        }
        d->local = v->local;
    }
    return 0;
}

/* The bytes of one stream the draw's window needs, [*lo, *hi) from the
   stream's address; 0 if none. */
static int
stream_range(const vgpu98_vin_t *v, const vgpu98_vin_draw_t *d, uint32_t s, uint64_t *lo, uint64_t *hi)
{
    const uint64_t stride = v->stream[s].stride;

    if (!((v->streams_used >> s) & 1) || !d->nverts)
        return 0;
    if (!stride) {
        /* Every vertex reads the same bytes. */
        *lo = 0;
        *hi = v->stream[s].extent;
    } else {
        *lo = (uint64_t) d->first * stride;
        *hi = ((uint64_t) d->first + d->nverts - 1) * stride + v->stream[s].extent;
    }
    if (*hi > v->stream[s].size)
        *hi = v->stream[s].size;
    return *hi > *lo;
}

/* The window's vertices whose bytes of stream s the draw's copy lacks (past
   the stream's end, or past the host's limits): they read as 0. */
static uint32_t
copy_short(const vgpu98_vin_t *v, const vgpu98_vin_draw_t *d, uint32_t s)
{
    const uint64_t stride = v->stream[s].stride;
    const uint64_t extent = v->stream[s].extent;
    const uint64_t have   = (uint64_t) v->copy[s].words * 4;
    uint64_t       fit;

    if (!((v->streams_used >> s) & 1) || !d->nverts)
        return 0;
    if (have < extent)
        return d->nverts;
    if (!stride)
        return 0;
    fit = (have - extent) / stride + 1;
    return (fit >= d->nverts) ? 0 : (uint32_t) (d->nverts - fit);
}

int
vgpu98_vin_layout(vgpu98_vin_t *v, const vgpu98_vin_draw_t *d, uint32_t max_words, uint32_t *words)
{
    uint64_t ulo = UINT64_MAX;
    uint64_t uhi = 0;
    uint64_t lo;
    uint64_t hi;
    uint32_t total;
    uint32_t gpuvm_left = GPUVM_DRAW_BYTES / 4;
    uint32_t s;

    /* Every stream in the inline buffer shares one copy of the bytes they
       span together: at most the 16 MB buffer, whatever the streams. */
    memset(v->copy, 0, sizeof(v->copy));
    for (s = 0; s < VGPU98_MAX_STREAMS; s++) {
        if ((v->stream[s].source != VGPU98_SRC_INLINE) || !stream_range(v, d, s, &lo, &hi))
            continue;
        if (v->stream[s].address + lo < ulo)
            ulo = v->stream[s].address + lo;
        if (v->stream[s].address + hi > uhi)
            uhi = v->stream[s].address + hi;
    }
    v->inl_lo    = 0;
    v->inl_words = 0;
    if (uhi > ulo) {
        v->inl_lo    = (uint32_t) ulo;
        v->inl_words = (uint32_t) ((uhi - ulo) / 4);
        if (v->inl_words > max_words)
            v->inl_words = max_words;
    }
    total = v->inl_words;
    for (s = 0; s < VGPU98_MAX_STREAMS; s++) {
        uint64_t at;
        uint64_t n;

        if (!stream_range(v, d, s, &lo, &hi))
            continue;
        if (v->stream[s].source == VGPU98_SRC_INLINE) {
            at = (v->stream[s].address + lo - v->inl_lo) / 4;
            n  = (hi - lo) / 4;
            if (at >= v->inl_words)
                continue;
            if (n > v->inl_words - at)
                n = v->inl_words - at;
        } else if ((v->res_mask >> s) & 1) {
            /* Resident: the arena holds the whole buffer; nothing copied. */
            at = 0;
            n  = (hi - lo) / 4;
            v->copy[s].lo    = lo;
            v->copy[s].at    = 0;
            v->copy[s].words = (uint32_t) n;
            continue;
        } else {
            /* A stream at a GPU address: its bytes from guest RAM, after the
               copies before it. */
            const uint64_t addr = (uint64_t) v->stream[s].address + lo;
            int            err;

            at = total;
            n  = (hi - lo) / 4;
            if (n > max_words - total)
                n = max_words - total;
            if (n > gpuvm_left)
                n = gpuvm_left;
            if (!n)
                continue;
            if (addr + n * 4 > 0x100000000ull)
                return VGPU98_ERR_UNMAPPED;
            /* After the streams read before it, as in the draw's copy. */
            if (!scratch(&v->gv, &v->gv_cap, (total - v->inl_words + (uint32_t) n) * 4))
                return VGPU98_ERR_ARGUMENT;
            if ((err = vgpu98_gpu_gpuvm_read(v->gpu, (uint32_t) addr, (uint32_t) n * 4, v->gv + (size_t) (total - v->inl_words) * 4,
                                             VGPU98_GPUVM_VERTICES)))
                return err;
            total += (uint32_t) n;
            gpuvm_left -= (uint32_t) n;
        }
        v->copy[s].lo    = lo;
        v->copy[s].at    = (uint32_t) at;
        v->copy[s].words = (uint32_t) n;
    }
    v->gv_words = total - v->inl_words;
    *words      = total;
    v->past        = 0;
    v->past_stream = 0;
    for (s = 0; s < VGPU98_MAX_STREAMS; s++) {
        const uint32_t n = copy_short(v, d, s);

        if (n > v->past) {
            v->past        = n;
            v->past_stream = s;
        }
    }
    return 0;
}

/* The bytes of stream s's copy, from its first (copy[s].lo from the stream's
   address). */
static const uint8_t *
copy_bytes(const vgpu98_vin_t *v, uint32_t s)
{
    if ((v->stream[s].source != VGPU98_SRC_INLINE) && ((v->res_mask >> s) & 1))
        return v->res_map + ((size_t) v->res_word[s] * 4 + (size_t) v->copy[s].lo);
    if (v->stream[s].source == VGPU98_SRC_INLINE)
        return v->inl + v->inl_lo + (size_t) v->copy[s].at * 4;
    return v->copy[s].words ? v->gv + (size_t) (v->copy[s].at - v->inl_words) * 4 : v->gv;
}

void
vgpu98_vin_copy(const vgpu98_vin_t *v, uint32_t *dst, uint32_t slots[VGPU98_SLOTS][4])
{
    uint32_t s;

    if (v->inl_words)
        memcpy(dst, v->inl + v->inl_lo, (size_t) v->inl_words * 4);
    if (v->gv_words)
        memcpy(dst + v->inl_words, v->gv, (size_t) v->gv_words * 4);
    for (s = 0; s < VGPU98_SLOTS; s++) {
        const vgpu98_vin_copy_t *c;

        if (!v->slot[s].on) {
            memset(slots[s], 0, 4 * sizeof(uint32_t));
            continue;
        }
        /* Window vertex 0's element: the stream's copy starts at the
           window's first vertex (or, with stride 0, at the bytes every vertex
           reads). A stream with nothing copied ends where it starts. A
           resident one's words are the arena's, from its own. */
        c           = &v->copy[v->slot[s].stream];
        slots[s][1] = v->stream[v->slot[s].stream].stride / 4;
        if ((v->stream[v->slot[s].stream].source != VGPU98_SRC_INLINE) && ((v->res_mask >> v->slot[s].stream) & 1)) {
            const uint32_t at = v->res_word[v->slot[s].stream] + (uint32_t) (c->lo / 4);

            slots[s][0] = at + v->slot[s].offset / 4;
            slots[s][2] = v->slot[s].type | VGPU98_SLOT_RES;
            slots[s][3] = at + c->words;
            continue;
        }
        slots[s][0] = c->at + v->slot[s].offset / 4;
        slots[s][2] = v->slot[s].type;
        slots[s][3] = c->at + c->words;
    }
}

/* A float of the little-endian bytes at p. */
static float
rdf(const uint8_t *p)
{
    uint32_t u = p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t) p[3] << 24);
    float    f;

    memcpy(&f, &u, 4);
    return f;
}

static float
grid_clamp(float f)
{
    if (!(f > -4194304.0f))
        return -4194304.0f;
    return (f > 4194304.0f) ? 4194304.0f : f;
}

static void
bounds_add(float *b, float x, float y)
{
    x = grid_clamp(x);
    y = grid_clamp(y);
    if (x < b[0])
        b[0] = x;
    if (y < b[1])
        b[1] = y;
    if (x > b[2])
        b[2] = x;
    if (y > b[3])
        b[3] = y;
}

int
vgpu98_vin_bounds(const vgpu98_vin_t *v, const vgpu98_vin_draw_t *d, float *b)
{
    const uint32_t           s      = v->slot[VGPU98_SLOT_POSITION].stream;
    const vgpu98_vin_copy_t *c      = &v->copy[s];
    const uint32_t           first  = v->slot[VGPU98_SLOT_POSITION].offset / 4;
    const uint32_t           stride = v->stream[s].stride / 4;
    const uint32_t           comps  = v->slot[VGPU98_SLOT_POSITION].type; /* FLOAT3 or FLOAT4 */
    const uint8_t           *src    = copy_bytes(v, s);
    uint32_t                 with;
    uint32_t                 i;

    if (!d->nverts && !d->outside)
        return 0;
    b[0] = b[1] = 4194304.0f;
    b[2] = b[3] = -4194304.0f;

    /* As the vertex shader reads them: the window's vertices whose position
       lies in the copy, then (0, 0) for the rest and for vertices outside
       the window. */
    with = 0;
    if (first + comps <= c->words)
        with = stride ? (c->words - first - comps) / stride + 1 : d->nverts;
    if (with > d->nverts)
        with = d->nverts;
    for (i = 0; i < with; i++) {
        const uint8_t *e = src + ((uint64_t) first + (uint64_t) i * stride) * 4;

        bounds_add(b, rdf(e), rdf(e + 4));
    }
    if ((with < d->nverts) || d->outside)
        bounds_add(b, 0.0f, 0.0f);
    return 1;
}

uint32_t
vgpu98_vin_nprims(const vgpu98_vin_draw_t *d)
{
    switch (d->prim) {
        case VGPU98_PRIM_POINTS:
            return d->count;
        case VGPU98_PRIM_LINES:
            return d->count / 2;
        case VGPU98_PRIM_LINE_STRIP:
            return d->count ? d->count - 1 : 0;
        default:
            return d->ntri;
    }
}

/* The window number of the draw's vertex i. */
static int32_t
wn(const vgpu98_vin_draw_t *d, uint32_t i)
{
    return (int32_t) (d->indexed ? d->local[i] : i);
}

void
vgpu98_vin_prims(const vgpu98_vin_draw_t *d, int lastpixel, int32_t *out)
{
    const uint32_t n = vgpu98_vin_nprims(d);
    uint32_t       i;

    for (i = 0; i < n; i++, out += 4) {
        const uint32_t e = d->edges ? ((d->edges[i / 8] >> (4 * (i & 7))) & 7) : 7;

        switch (d->prim) {
            case VGPU98_PRIM_POINTS:
                out[0] = wn(d, i);
                out[1] = out[2] = out[3] = 0;
                break;
            case VGPU98_PRIM_LINES:
                out[0] = wn(d, 2 * i);
                out[1] = wn(d, 2 * i + 1);
                out[2] = lastpixel;
                out[3] = 0;
                break;
            case VGPU98_PRIM_LINE_STRIP:
                /* Shared vertices are drawn once: only the last line may
                   draw its last pixel. */
                out[0] = wn(d, i);
                out[1] = wn(d, i + 1);
                out[2] = lastpixel && (i + 1 == n);
                out[3] = 0;
                break;
            case VGPU98_PRIM_TRIANGLES:
                out[0] = wn(d, 3 * i);
                out[1] = wn(d, 3 * i + 1);
                out[2] = wn(d, 3 * i + 2);
                out[3] = (int32_t) e;
                break;
            case VGPU98_PRIM_TRIANGLE_STRIP:
                /* (i, i+1, i+2), then (i, i+2, i+1). */
                out[0] = wn(d, i);
                out[1] = wn(d, (i & 1) ? i + 2 : i + 1);
                out[2] = wn(d, (i & 1) ? i + 1 : i + 2);
                out[3] = (int32_t) e;
                break;
            default:
                /* Fans: (i+1, i+2, 0), the centre last. */
                out[0] = wn(d, i + 1);
                out[1] = wn(d, i + 2);
                out[2] = wn(d, 0);
                out[3] = (int32_t) e;
                break;
        }
    }
}

uint32_t
vgpu98_vin_element(const vgpu98_vin_draw_t *d, uint32_t i, uint32_t k)
{
    switch (d->prim) {
        case VGPU98_PRIM_POINTS:
            return i;
        case VGPU98_PRIM_LINES:
            return 2 * i + (k & 1);
        case VGPU98_PRIM_LINE_STRIP:
            return i + (k & 1);
        case VGPU98_PRIM_TRIANGLES:
            return 3 * i + k;
        case VGPU98_PRIM_TRIANGLE_STRIP:
            /* (i, i+1, i+2), then (i, i+2, i+1), as vgpu98_vin_prims */
            return !k ? i : ((k == 1) == !(i & 1)) ? i + 1 : i + 2;
        default:
            /* Fans: (i+1, i+2, 0) */
            return (k == 2) ? 0 : i + 1 + k;
    }
}

int32_t
vgpu98_vin_corner(const vgpu98_vin_draw_t *d, uint32_t i, uint32_t k)
{
    return wn(d, vgpu98_vin_element(d, i, k));
}

uint32_t
vgpu98_vin_raw(const vgpu98_vin_t *v, uint32_t s, int32_t i, uint32_t from, uint8_t *buf, uint32_t max)
{
    const vgpu98_vin_copy_t *c;
    uint32_t                 stride;
    uint64_t                 at;
    uint32_t                 n;

    if ((s >= VGPU98_MAX_STREAMS) || !v->stream[s].set || (i < 0))
        return 0;
    c      = &v->copy[s];
    stride = v->stream[s].stride;
    if (from >= stride)
        return 0;
    at = (uint64_t) (uint32_t) i * stride + from;
    n  = (stride - from < max) ? stride - from : max;
    if (!stride || (at >= (uint64_t) c->words * 4))
        return 0;
    if (at + n > (uint64_t) c->words * 4)
        n = (uint32_t) ((uint64_t) c->words * 4 - at);
    memcpy(buf, copy_bytes(v, s) + at, n);
    return n;
}

int
vgpu98_vin_fetch(const vgpu98_vin_t *v, uint32_t slot, int32_t i, float *out)
{
    const uint32_t           s     = v->slot[slot].stream;
    const vgpu98_vin_copy_t *c     = &v->copy[s];
    const uint32_t           type  = v->slot[slot].type;
    const uint32_t           words = (type <= VGPU98_DT_FLOAT4) ? type : (type == VGPU98_DT_SHORT4) ? 2 : 1;
    const uint64_t           at    = (uint64_t) v->slot[slot].offset / 4 + (uint64_t) (uint32_t) i * (v->stream[s].stride / 4);
    const uint8_t           *e;
    uint32_t                 k;

    out[0] = out[1] = out[2] = 0.0f;
    out[3]                   = 1.0f;
    if (!v->slot[slot].on || !type || (i < 0) || (at + words > c->words))
        return 0;
    e = copy_bytes(v, s) + at * 4;
    switch (type) {
        case VGPU98_DT_D3DCOLOR: /* B, G, R, A */
            out[0] = e[2] / 255.0f;
            out[1] = e[1] / 255.0f;
            out[2] = e[0] / 255.0f;
            out[3] = e[3] / 255.0f;
            return 4;
        case VGPU98_DT_UBYTE4:
            for (k = 0; k < 4; k++)
                out[k] = (float) e[k];
            return 4;
        case VGPU98_DT_SHORT2:
        case VGPU98_DT_SHORT4:
            for (k = 0; k < ((type == VGPU98_DT_SHORT2) ? 2u : 4u); k++)
                out[k] = (float) (int16_t) (e[2 * k] | (e[2 * k + 1] << 8));
            return (type == VGPU98_DT_SHORT2) ? 2 : 4;
        default:
            for (k = 0; k < type; k++)
                out[k] = rdf(e + 4 * k);
            return (int) type;
    }
}
