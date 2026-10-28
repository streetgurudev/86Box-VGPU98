/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the Glide device, command set 2.4
 *          (PROFILE.GLIDE). The Glide DLLs' sessions, their buffers on
 *          the host GPU, Glide's state, its draws, texture memory and
 *          frame buffer transfers, and the screen while a session shows
 *          (spec/glide.md and
 *          spec/gpu-commands.md).
 *
 *          A device of its own: nothing here touches the 3D renderer's
 *          state, images or video memory's coherency. It records into the
 *          renderer's command buffers (vid_vgpu98_vk.h), ending its own
 *          render pass whenever the renderer records or submits, and
 *          hands each swapped frame to the display (vid_vgpu98_disp.c),
 *          which shows it instead of video memory until the session ends.
 *          A command it cannot take is dropped and counted; it never stops
 *          the ring.
 *
 *          Runs on the command processor's thread only.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <86box/thread.h>
#include <86box/vid_vgpu98_card.h>
#include <86box/vid_vgpu98_gpu.h>
#include <86box/vid_vgpu98_glcmd.h>
#include <86box/vid_vgpu98_glcpu.h>
#include <86box/vid_vgpu98_dev.h>

#ifndef VGPU98_NO_VULKAN

#    include <png.h>
#    include <86box/86box.h>
#    include <86box/path.h>
#    include <86box/plat.h>
#    include <86box/vid_vgpu98_vk.h>
#    include "vgpu98_glide_vert.h"
#    include "vgpu98_glide_frag.h"

#    define DROP_LOGGED 16   /* commands dropped named in the log; the rest counted */
#    define LFB_STRIDE  8192 /* a frame buffer transfer's rows at most: a Voodoo3's 2048 32-bit pixels */
#    define MAX_PIPES   256  /* pipelines kept: blending, the masks and the depth test make them */
#    define POOL_SETS   2048 /* descriptor sets a slot's pool holds: one a DRAW */
#    define TEX_MAX     2048 /* textures decoded and kept, a session */
#    define TEX_LIVE    4096 /* and in all */
#    define POOL_BLOCK  ((VkDeviceSize) 32 << 20) /* the textures' memory, taken from the host GPU in blocks of this */
#    define POOL_BLOCKS 16
#    define POOL_FREES  1024 /* free ranges a block keeps */
#    define REPORT_MS   5000 /* the Glide report's period (Diagnostics "Glide") */
#    define CMDS_KEPT   4096 /* commands kept for a capture's frame */
#    define CAPTURES    8    /* a session's at most */
#    define SHOT_CAPTURES 10 /* and of 86Box screenshots, apart */
#    define CHECK_EVERY 257  /* one texture found again in this many checked against TMU memory */
#    define CHECKS_MAX  128  /* a report period's at most */
#    define EXAMPLES    6    /* the report's examples of oddities, two of a kind at most */
#    define GONE_KEPT   512  /* dropped textures remembered, to count decodes of unchanged bytes */
#    define CAPTURE_MS  10000 /* a session's first capture this long after its open */
#    define TMU_PAGES   (VGPU98_GL_TMU_BYTES / 4096)

/* An oddity's example for the report: the unit,
   its texture, the first vertex's texel s and t and its 1/w and the
   vertex's, and the DRAW's combines and blend. */
typedef struct gl_example_t {
    const char *what;
    uint32_t    unit;
    uint32_t    mem, address, info, avg;
    uint32_t    vertices;
    float       s, t, tw, w;
    uint32_t    cc, ac, tc0, tc1, blend;
} gl_example_t;

/* The Glide report's counters, since the
   last report. */
typedef struct gl_stats_t {
    uint64_t swaps, draws, triangles, states, clears, lfb_reads, lfb_writes, dropped;
    uint64_t texdata_bytes, palettes, decoded, found, tex_dropped;
    uint64_t blended, depth_z, depth_w, fogged, textured;
    uint64_t bad_w; /* vertices with 1/w at or below 0, or not a number */
    uint32_t frame_worst_ms;
    uint32_t decoded_by_format[16];
    struct {
        uint32_t cc, ac, tc0, tc1;
        uint64_t n;
    } combine[8];
    uint32_t n_combine;
    /* What the texture units a DRAW read sampled:
       each unit's DRAWs with one texel, at the smallest level, its 1/w its
       own, its 1/w bad; TMU0 passing TMU1 through, and with TMU1's own 1/w. */
    uint64_t one_texel[2], smallest[2], own_w[2], tmu_bad_w[2];
    /* DRAWs reading TMU0's texel alone, TMU1's
       alone (TMU0 not active), both; TMU1's alone with its own 1/w. Depth:
       tested (by function), writing, with a bias, with a 1/w past the W
       buffer's far end. Textures decoded again from a dropped one's bytes. */
    uint64_t units[4], alone_own_w;
    uint64_t depth_tested, depth_fn[8], depth_writing, depth_bias, w_far;
    uint64_t unchanged;
    uint64_t checked, stale; /* textures found again checked against TMU memory; changed */
    struct {
        uint32_t blend; /* VGPU98_GS_BLEND, or FFFFFFFFh for none */
        uint64_t n;
    } blend[8];
    uint32_t     n_blend;
    gl_example_t ex[EXAMPLES];
    uint32_t     n_ex;
    uint32_t     ex_kind[4];
} gl_stats_t;

/* A command kept for a capture: its opcode, length and first dwords. */
typedef struct gl_cmd_t {
    uint32_t op, len;
    uint32_t p[6];
} gl_cmd_t;

/* The findings, each said once. */
#    define FIND_BLANK   0x01
#    define FIND_BAD_W   0x02
#    define FIND_MISSING 0x04
#    define FIND_OVERLAP 0x08
#    define FIND_STALE   0x10
#    define TEX_BYTES   ((VkDeviceSize) 256 << 20) /* their host images at most, a session */

/* glide.frag's uniform block (std140). */
typedef struct gl_ubo_t {
    uint32_t comb[4];   /* colour combine, alpha combine, TMU0's and TMU1's texture combine */
    uint32_t misc[4];   /* texture inverts, the constant ARGB, chroma (RGB | on << 24), alpha test */
    uint32_t pix[4];    /* depth, fog mode, fog colour, F_* */
    uint32_t fog[16];   /* the fog table */
    uint32_t tex[2][4]; /* each TMU: T_* and clamps, its levels, the LOD bias (a float), detail */
} gl_ubo_t;

#    define T_TEXTURE    1u
#    define T_EVEN       2u
#    define T_ODD        4u
#    define T_TRILINEAR  8u
#    define T_MIN_LINEAR 32u
#    define T_MAG_LINEAR 64u
#    define T_LOD_DITHER 128u

/* glide.vert's push constants. */
typedef struct gl_push_t {
    float    width, height;
    uint32_t lower_left;
    uint32_t first;
} gl_push_t;

#    define F_ITRGB 4u
#    define F_BLEND 8u
#    define F_DEPTH 16u

typedef struct gl_image_t {
    VkImage        image;
    VkDeviceMemory mem; /* its own memory, or none for a texture's (in a pool block) */
    VkImageView    view;
    int            pooled;
    uint32_t       block;
    VkDeviceSize   at, bytes;
} gl_image_t;

/* The textures' memory: a few large blocks from the host GPU (it allows
   4096 allocations at least, shared with the renderer), each with its free
   ranges; a texture's range comes back once the slot it was dropped in has
   been begun again (the host GPU done with it, its image destroyed). */
typedef struct gl_range_t {
    VkDeviceSize at, n;
} gl_range_t;

typedef struct gl_block_t {
    VkDeviceMemory mem;
    uint32_t       type;
    VkDeviceSize   size;
    gl_range_t    *free;
    uint32_t       n_free;
} gl_block_t;

typedef struct gl_pending_t {
    uint32_t     block, slot, gen;
    VkDeviceSize at, n;
} gl_pending_t;

/* The pixels last written into one of the staging surface's areas by a
   READ, kept to take back only what the program changed (WRITE's
   CHANGED). */
typedef struct gl_snap_t {
    uint8_t  *bytes;  /* as written, at their place in the area */
    size_t    size;   /* bytes from the area's start */
    uint32_t  offset; /* the area's video memory offset */
    uint32_t  stride;
    uint32_t  format;
    uint32_t  buffer;
    uint32_t  x0, x1, y0, y1; /* the rectangle written, in the program's rows */
    uint16_t *native; /* those pixels' buffer values (565 or depth), w x h of the session */
    uint16_t *native2; /* the depth values too, for a format with both */
    int       valid;
} gl_snap_t;

/* A texture decoded from a TMU's memory into a host image, its levels from
   the largest (those of a split texture's parity alone filled). */
typedef struct gl_tex_t {
    uint32_t     tmu, address, info; /* where it is and what it is (TEX_ADDRESS, TEX_INFO) */
    uint64_t     pal;                /* the palette's hash, for a palette format */
    uint32_t     lo, hi;             /* the TMU memory its texels came from */
    uint64_t     used;               /* the session's DRAW that last used it */
    VkDeviceSize bytes;
    gl_image_t   im;
    /* Diagnostics "Glide": its largest level's
       average colour (0xRRGGBB), its TMU memory's hash when decoded, its
       DRAWs in the report's period. */
    uint32_t     avg;
    uint64_t     hash;
    int          hashed;
    uint32_t     draws;
} gl_tex_t;

typedef struct gl_session_t {
    uint32_t      id; /* 0: free */
    uint32_t      w, h;
    uint32_t      n_color; /* 1-3 */
    int           aux;
    gl_image_t    color[3];
    gl_image_t    depth;
    VkFramebuffer fb[3];
    uint32_t      front; /* color[] shown */
    uint32_t      staging;       /* the DLL's staging surface in video memory: transfers stay inside it */
    uint32_t      staging_bytes;
    uint32_t      reg[VGPU98_GS_COUNT];
    uint32_t      gamma[256];
    gl_snap_t     snap[2]; /* the surface's write area (its start), and its read area */
    uint64_t      swaps;
    /* Texture memory: each TMU's, its palette, the textures
       decoded from it */
    uint8_t      *tmem[2];
    uint32_t      palette[2][256];
    uint64_t      pal_hash[2];
    gl_tex_t     *tex;
    uint32_t      n_tex;
    VkDeviceSize  tex_bytes;
    uint64_t      draw_serial;
    uint32_t      last[2]; /* the texture each TMU used last, to find it again at once */
    /* Diagnostics: the TMUs' pages a download wrote, the highest byte
       written, and the session's captures */
    uint8_t       written[2][TMU_PAGES / 8];
    uint32_t      tmem_high[2];
    uint32_t      open_ms;
    uint32_t      captures;
    int           capture_due; /* a finding: a capture at the next swap */
    int           captured_late;
    uint32_t      shot_captures; /* of 86Box screenshots, apart */
} gl_session_t;

struct vgpu98_glide_t {
    vgpu98_gpu_t        *gpu;
    vgpu98_r3d_t        *r3d;
    vgpu98_vk_t         *vk;
    vgpu98_disp_t       *disp;
    const vgpu98_vram_t *vram;

    gl_session_t  s[VGPU98_GL_SESSIONS];
    gl_session_t *shown; /* the session the screen shows */

    VkRenderPass pass[2]; /* colour alone; colour and depth */

    /* Drawing: glide.vert and glide.frag, their layout, the pipelines made
       so far, each slot's descriptor pool (reset when the slot is begun
       again), and a white texture for a TMU with none. */
    VkShaderModule        vs, fs;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      layout;
    VkDescriptorPool      dpool[VGPU98_R3D_SLOTS];
    uint32_t              dpool_gen[VGPU98_R3D_SLOTS];
    struct {
        uint32_t   key;
        VkPipeline pipe;
    } pipes[MAX_PIPES];
    uint32_t   n_pipes;
    VkPipeline bound_pipe; /* in the command buffer since the renderer last recorded */
    gl_image_t white;
    VkSampler  sampler_white;

    /* The textures' memory blocks, and the ranges waiting to come back */
    gl_block_t    blocks[POOL_BLOCKS];
    uint32_t      n_blocks;
    gl_pending_t *pending;
    uint32_t      n_pending, cap_pending;
    int        pipes_full_told;

    /* The render pass open in the renderer's command buffer, if any */
    int           pass_open;
    gl_session_t *pass_s;
    uint32_t      pass_buf;

    /* A front buffer drawn into since it was shown (vgpu98_glide_tick) */
    gl_session_t *front_dirty;
    uint32_t      front_vblank; /* the vertical blank it was last shown at */

    /* Counters */
    uint64_t dropped;
    uint32_t drop_logged;
    uint64_t draws, triangles;
    uint64_t tex_decoded, tex_dropped, tex_bytes_down;
    uint32_t tex_live;
    int      tex_fail_told;

    /* Diagnostics "Glide": the report's counters, the findings
       said, the last commands for a capture's frame */
    int        logging;
    int        capture_path_told;
    /* Report periods in a row with swaps and clears but nothing drawn: a
       blank screen after three (15 s), not a loading screen's few. */
    uint32_t   blank_run;
    uint64_t   blank_swaps;
    double     blank_secs;
    gl_stats_t st;
    uint32_t   report_ms;
    uint32_t   last_swap_ms;
    uint32_t   told;
    uint32_t   capture_serial;
    gl_cmd_t  *cmds;
    uint32_t   n_cmds;     /* kept so far */
    uint32_t   frame_from; /* the first of the frame being drawn */
    atomic_int shot_asked; /* an 86Box screenshot: a capture at the next swap */
    struct {
        uint32_t mem, address, info;
        uint64_t pal, hash;
    } gone[GONE_KEPT]; /* textures dropped lately, as decoded */
    uint32_t gone_at;
    uint32_t   check_tick; /* textures found again, for the cache's sampled check */
    int      fallback_told;
    int      pipeline_told;
};

static void
gl_log(vgpu98_glide_t *gl, const char *fmt, ...)
{
    char    line[256];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    vgpu98_gpu_log(gl->gpu, VGPU98_LOG_EVENTS, "VGPU98 Glide: %s\n", line);
}

/* A command dropped: counted, and the first ones named. */
static void
drop(vgpu98_glide_t *gl, uint32_t op, const char *why, ...)
{
    gl->dropped++;
    gl->st.dropped++;
    if (gl->drop_logged < DROP_LOGGED) {
        char    text[160];
        va_list ap;

        va_start(ap, why);
        vsnprintf(text, sizeof(text), why, ap);
        va_end(ap);
        gl->drop_logged++;
        gl_log(gl, "command %02Xh dropped: %s%s", op, text,
               (gl->drop_logged == DROP_LOGGED) ? " (more are counted, not named)" : "");
    }
}

/* A finding past a real problem's threshold: said once, worded "possible",
   and the session captured at its next swap (Diagnostics "Glide"). */
static void
finding(vgpu98_glide_t *gl, gl_session_t *s, uint32_t kind, const char *fmt, ...)
{
    char    text[200];
    va_list ap;

    if (gl->told & kind)
        return;
    gl->told |= kind;
    va_start(ap, fmt);
    vsnprintf(text, sizeof(text), fmt, ap);
    va_end(ap);
    gl_log(gl, "possible %s", text);
    if (s)
        s->capture_due = 1;
}

static gl_session_t *
find(vgpu98_glide_t *gl, uint32_t id)
{
    int i;

    if (!id)
        return NULL;
    for (i = 0; i < VGPU98_GL_SESSIONS; i++) {
        if (gl->s[i].id == id)
            return &gl->s[i];
    }
    return NULL;
}

static int  capture(vgpu98_glide_t *gl, gl_session_t *s, const char *why, int counted);
static void draw_noted(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t count, const int *has);
static void example_add(gl_stats_t *st, uint32_t kind, const char *what, uint32_t unit, const gl_tex_t *x,
                        const gl_session_t *s, uint32_t vertices, float es, float et, float tw, float w);
static int  blending(const gl_session_t *s);
static const char *const format_names[16];

/* A session's front buffer drawn into: shown at the next tick. */
static void
front_written(vgpu98_glide_t *gl, gl_session_t *s)
{
    if (gl->front_dirty != s) {
        gl->front_dirty = s;
        vgpu98_gpu_glide_front(gl->gpu, 1);
    }
}

/* --- Recording ---------------------------------------------------------------- */

static void
mem_barrier(vgpu98_glide_t *gl, VkCommandBuffer c)
{
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };

    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    gl->vk->vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                                 NULL, 0, NULL);
}

/* The renderer's hook: the render pass ends before it records or submits. */
static void
pass_end(void *priv)
{
    vgpu98_glide_t *gl = (vgpu98_glide_t *) priv;
    VkCommandBuffer c;

    /* The renderer records next (its pipeline in the buffer), or a new
       buffer begins: ours is bound again. */
    gl->bound_pipe = VK_NULL_HANDLE;
    if (!gl->pass_open)
        return;
    c = vgpu98_r3d_glide_rec(gl->r3d);
    gl->vk->vkCmdEndRenderPass(c);
    mem_barrier(gl, c);
    gl->pass_open = 0;
    gl->pass_s    = NULL;
}

/* Outside any render pass: the command buffer for copies and barriers. */
static VkCommandBuffer
outside(vgpu98_glide_t *gl)
{
    VkCommandBuffer c = vgpu98_r3d_glide_cmd(gl->r3d);

    pass_end(gl);
    return c;
}

/* The colour buffer a register names (GR_BUFFER_*): the front, the back or
   the third; -1 for none. */
static int
color_index(const gl_session_t *s, uint32_t buffer)
{
    switch (buffer) {
        case VGPU98_GL_FRONTBUFFER:
            return (int) s->front;
        case VGPU98_GL_BACKBUFFER:
            return (int) ((s->front + ((s->n_color > 1) ? 1 : 0)) % s->n_color);
        case VGPU98_GL_TRIPLEBUFFER:
            return (s->n_color > 2) ? (int) ((s->front + 2) % s->n_color) : -1;
        default:
            return -1;
    }
}

/* In the session's render pass on colour buffer `ci`: the command buffer. */
static VkCommandBuffer
inside(vgpu98_glide_t *gl, gl_session_t *s, uint32_t ci)
{
    VkCommandBuffer       c;
    VkRenderPassBeginInfo bi = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };

    if (gl->pass_open && (gl->pass_s == s) && (gl->pass_buf == ci))
        return vgpu98_r3d_glide_rec(gl->r3d);
    c                        = outside(gl);
    bi.renderPass            = gl->pass[s->aux ? 1 : 0];
    bi.framebuffer           = s->fb[ci];
    bi.renderArea.extent.width  = s->w;
    bi.renderArea.extent.height = s->h;
    gl->vk->vkCmdBeginRenderPass(c, &bi, VK_SUBPASS_CONTENTS_INLINE);
    gl->pass_open = 1;
    gl->pass_s    = s;
    gl->pass_buf  = ci;
    return c;
}

/* --- The textures' memory -------------------------------------------------------------- */

/* n bytes aligned in a block's free ranges, first fit. */
static int
range_take(gl_block_t *b, VkDeviceSize n, VkDeviceSize align, VkDeviceSize *out)
{
    uint32_t i;

    if (!align)
        align = 1;
    for (i = 0; i < b->n_free; i++) {
        gl_range_t        *f     = &b->free[i];
        const VkDeviceSize start = (f->at + align - 1) / align * align;
        VkDeviceSize       head, tail;

        if (start + n > f->at + f->n)
            continue;
        head = start - f->at;
        tail = f->at + f->n - (start + n);
        if (head && tail) {
            if (b->n_free == POOL_FREES)
                continue;
            memmove(&b->free[i + 2], &b->free[i + 1], (b->n_free - i - 1) * sizeof(gl_range_t));
            b->free[i + 1].at = start + n;
            b->free[i + 1].n  = tail;
            f->n              = head;
            b->n_free++;
        } else if (head)
            f->n = head;
        else if (tail) {
            f->at = start + n;
            f->n  = tail;
        } else {
            memmove(&b->free[i], &b->free[i + 1], (b->n_free - i - 1) * sizeof(gl_range_t));
            b->n_free--;
        }
        *out = start;
        return 1;
    }
    return 0;
}

/* A range back into its block's free ranges, joined to its neighbours. */
static void
range_give(gl_block_t *b, VkDeviceSize at, VkDeviceSize n)
{
    uint32_t i = 0;

    while ((i < b->n_free) && (b->free[i].at < at))
        i++;
    if (i && (b->free[i - 1].at + b->free[i - 1].n == at)) {
        b->free[i - 1].n += n;
        if ((i < b->n_free) && (at + n == b->free[i].at)) {
            b->free[i - 1].n += b->free[i].n;
            memmove(&b->free[i], &b->free[i + 1], (b->n_free - i - 1) * sizeof(gl_range_t));
            b->n_free--;
        }
        return;
    }
    if ((i < b->n_free) && (at + n == b->free[i].at)) {
        b->free[i].at = at;
        b->free[i].n += n;
        return;
    }
    if (b->n_free == POOL_FREES) {
        static int told;

        if (!told++)
            pclog("VGPU98 Glide: texture memory in too many pieces: some is not reused until 86Box closes\n");
        return; /* this one is lost until the device closes */
    }
    memmove(&b->free[i + 1], &b->free[i], (b->n_free - i) * sizeof(gl_range_t));
    b->free[i].at = at;
    b->free[i].n  = n;
    b->n_free++;
}

/* The ranges whose slot has been begun again since they were dropped. */
static void
pool_reap(vgpu98_glide_t *gl, uint32_t slot, uint32_t gen)
{
    uint32_t i;

    for (i = 0; i < gl->n_pending;) {
        const gl_pending_t *q = &gl->pending[i];

        if ((q->slot == slot) && (q->gen != gen)) {
            range_give(&gl->blocks[q->block], q->at, q->n);
            gl->pending[i] = gl->pending[--gl->n_pending];
        } else
            i++;
    }
}

/* Memory for an image from the blocks, a new block made when none has
   room. */
static int
pool_bind(vgpu98_glide_t *gl, gl_image_t *im)
{
    vgpu98_vk_t         *vk = gl->vk;
    VkMemoryRequirements req;
    uint32_t             i, gen, slot;

    slot = vgpu98_r3d_glide_slot(gl->r3d, &gen);
    pool_reap(gl, slot, gen);
    vk->vkGetImageMemoryRequirements(vk->dev, im->image, &req);
    for (i = 0;; i++) {
        gl_block_t *b;

        if (i == gl->n_blocks) {
            VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            int                  type;

            if (gl->n_blocks == POOL_BLOCKS)
                return 0;
            type = vgpu98_vk_memory_type(vk, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (type < 0)
                type = vgpu98_vk_memory_type(vk, req.memoryTypeBits, 0);
            b = &gl->blocks[gl->n_blocks];
            memset(b, 0, sizeof(*b));
            b->size            = (req.size > POOL_BLOCK) ? req.size : POOL_BLOCK;
            b->free            = malloc(POOL_FREES * sizeof(gl_range_t));
            ai.allocationSize  = b->size;
            ai.memoryTypeIndex = (uint32_t) type;
            if ((type < 0) || !b->free || (vk->vkAllocateMemory(vk->dev, &ai, NULL, &b->mem) != VK_SUCCESS)) {
                free(b->free);
                memset(b, 0, sizeof(*b));
                return 0;
            }
            b->type       = (uint32_t) type;
            b->free[0].at = 0;
            b->free[0].n  = b->size;
            b->n_free     = 1;
            gl->n_blocks++;
        }
        b = &gl->blocks[i];
        if (!(req.memoryTypeBits & (1u << b->type)) || !range_take(b, req.size, req.alignment, &im->at))
            continue;
        if (vk->vkBindImageMemory(vk->dev, im->image, b->mem, im->at) != VK_SUCCESS) {
            range_give(b, im->at, req.size);
            return 0;
        }
        im->pooled = 1;
        im->block  = i;
        im->bytes  = req.size;
        return 1;
    }
}

/* An image's range, back once the slot recording now has been begun again. */
static void
pool_drop(vgpu98_glide_t *gl, const gl_image_t *im)
{
    gl_pending_t *q;
    uint32_t      gen;

    if (gl->n_pending == gl->cap_pending) {
        const uint32_t n   = gl->cap_pending ? gl->cap_pending * 2 : 256;
        gl_pending_t  *mem = realloc(gl->pending, n * sizeof(gl_pending_t));

        if (!mem)
            return; /* lost until the device closes */
        gl->pending     = mem;
        gl->cap_pending = n;
    }
    q        = &gl->pending[gl->n_pending++];
    q->block = im->block;
    q->at    = im->at;
    q->n     = im->bytes;
    q->slot  = vgpu98_r3d_glide_slot(gl->r3d, &gen);
    q->gen   = gen;
}

static void
pool_close(vgpu98_glide_t *gl)
{
    uint32_t i;

    for (i = 0; i < gl->n_blocks; i++) {
        if (gl->blocks[i].mem)
            gl->vk->vkFreeMemory(gl->vk->dev, gl->blocks[i].mem, NULL);
        free(gl->blocks[i].free);
    }
    gl->n_blocks = 0;
    free(gl->pending);
    gl->pending   = NULL;
    gl->n_pending = gl->cap_pending = 0;
}

/* --- Sessions ------------------------------------------------------------------ */

static void
image_free(vgpu98_glide_t *gl, gl_image_t *im)
{
    if (im->view)
        vgpu98_r3d_glide_dead(gl->r3d, 1, (uint64_t) (uintptr_t) im->view);
    if (im->image)
        vgpu98_r3d_glide_dead(gl->r3d, 0, (uint64_t) (uintptr_t) im->image);
    if (im->mem)
        vgpu98_r3d_glide_dead(gl->r3d, 2, (uint64_t) (uintptr_t) im->mem);
    if (im->pooled)
        pool_drop(gl, im);
    memset(im, 0, sizeof(*im));
}

static int
image_make(vgpu98_glide_t *gl, gl_image_t *im, VkFormat format, uint32_t w, uint32_t h, uint32_t levels,
           VkImageUsageFlags usage, VkImageAspectFlags aspect, int pooled)
{
    vgpu98_vk_t          *vk = gl->vk;
    VkImageCreateInfo     ic = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    VkImageMemoryBarrier  ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    VkCommandBuffer       c;

    memset(im, 0, sizeof(*im));
    ic.imageType     = VK_IMAGE_TYPE_2D;
    ic.format        = format;
    ic.extent.width  = w;
    ic.extent.height = h;
    ic.extent.depth  = 1;
    ic.mipLevels     = levels;
    ic.arrayLayers   = 1;
    ic.samples       = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ic.usage         = usage;
    ic.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vk->vkCreateImage(vk->dev, &ic, NULL, &im->image) != VK_SUCCESS) {
        im->image = VK_NULL_HANDLE;
        return 0;
    }
    if (pooled ? !pool_bind(gl, im) : !vgpu98_r3d_glide_image_memory(gl->r3d, im->image, &im->mem, NULL)) {
        vk->vkDestroyImage(vk->dev, im->image, NULL);
        memset(im, 0, sizeof(*im));
        return 0;
    }
    vi.image                       = im->image;
    vi.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    vi.format                      = format;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.levelCount = levels;
    vi.subresourceRange.layerCount = 1;
    if (vk->vkCreateImageView(vk->dev, &vi, NULL, &im->view) != VK_SUCCESS) {
        im->view = VK_NULL_HANDLE;
        image_free(gl, im);
        return 0;
    }
    /* GENERAL from the start, as every image of the renderer's. */
    c                              = outside(gl);
    ib.srcAccessMask               = 0;
    ib.dstAccessMask               = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib.oldLayout                   = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout                   = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    ib.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    ib.image                       = im->image;
    ib.subresourceRange.aspectMask = aspect;
    ib.subresourceRange.levelCount = levels;
    ib.subresourceRange.layerCount = 1;
    vk->vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
                             NULL, 1, &ib);
    return 1;
}

static void
snap_free(gl_snap_t *sn)
{
    free(sn->bytes);
    free(sn->native);
    free(sn->native2);
    memset(sn, 0, sizeof(*sn));
}

/* The session the screen shows, or none: the card hides the hardware
   cursor while one does, as a Voodoo3's Glide turns its cursor off. */
static void
shown_set(vgpu98_glide_t *gl, gl_session_t *s)
{
    gl->shown = s;
    vgpu98_gpu_glide_screen(gl->gpu, s != NULL);
}

static void
session_end(vgpu98_glide_t *gl, gl_session_t *s)
{
    uint32_t i;

    if (!s->id)
        return;
    if (gl->pass_open && (gl->pass_s == s))
        outside(gl);
    if (gl->shown == s) {
        shown_set(gl, NULL);
        vgpu98_disp_glide_end(gl->disp);
    }
    if (gl->front_dirty == s) {
        gl->front_dirty = NULL;
        vgpu98_gpu_glide_front(gl->gpu, 0);
    }
    for (i = 0; i < 3; i++) {
        if (s->fb[i])
            vgpu98_r3d_glide_dead(gl->r3d, 3, (uint64_t) (uintptr_t) s->fb[i]);
        image_free(gl, &s->color[i]);
    }
    image_free(gl, &s->depth);
    snap_free(&s->snap[0]);
    snap_free(&s->snap[1]);
    for (i = 0; i < s->n_tex; i++)
        image_free(gl, &s->tex[i].im);
    gl->tex_live -= s->n_tex;
    free(s->tex);
    free(s->tmem[0]);
    free(s->tmem[1]);
    memset(s, 0, sizeof(*s));
}

/* Glide's state at a window's open, before the program sets any: as
   grGlideInit leaves a Voodoo. */
static void
state_defaults(gl_session_t *s, int lower_left)
{
    uint32_t i;

    memset(s->reg, 0, sizeof(s->reg));
    s->reg[VGPU98_GS_CLIP_MIN]      = 0;
    s->reg[VGPU98_GS_CLIP_MAX]      = s->w | (s->h << 16);
    s->reg[VGPU98_GS_RENDER_BUFFER] = (s->n_color > 1) ? VGPU98_GL_BACKBUFFER : VGPU98_GL_FRONTBUFFER;
    s->reg[VGPU98_GS_MASKS]         = VGPU98_GS_MASK_RGB;
    s->reg[VGPU98_GS_ORIGIN]        = lower_left ? 1 : 0;
    /* The rest as a Voodoo3's Glide sets them at the open (the DLL sends
       them all; another client drawing first finds them so). */
    s->reg[VGPU98_GS_COLOR_COMBINE] = VGPU98_GS_COMBINE(VGPU98_GL_FN_SCALE_OTHER, VGPU98_GL_FAC_ONE, VGPU98_GL_LOCAL_ITERATED,
                                                        VGPU98_GL_OTHER_ITERATED, 0);
    s->reg[VGPU98_GS_ALPHA_COMBINE] = VGPU98_GS_COMBINE(VGPU98_GL_FN_SCALE_OTHER, VGPU98_GL_FAC_ONE, VGPU98_GL_LOCAL_CONSTANT,
                                                        VGPU98_GL_OTHER_CONSTANT, 0);
    s->reg[VGPU98_GS_CONSTANT]      = 0xffffffffu;
    s->reg[VGPU98_GS_BLEND]         = VGPU98_GL_BLEND_ONE | (VGPU98_GL_BLEND_ZERO << 8) | (VGPU98_GL_BLEND_ONE << 16) |
                                      (VGPU98_GL_BLEND_ZERO << 24);
    s->reg[VGPU98_GS_ALPHA_TEST]    = VGPU98_GL_CMP_ALWAYS;
    s->reg[VGPU98_GS_DEPTH]         = VGPU98_GL_DEPTH_DISABLE | (VGPU98_GL_CMP_LESS << 8);
    s->reg[VGPU98_GS_DITHER]        = 2;
    s->reg[VGPU98_GS_LFB_CONSTANT]  = 0xff;
    for (i = 0; i < 2; i++) {
        s->reg[VGPU98_GS_TMU(i) + VGPU98_GS_TEX_CLAMP]  = 1 | (1 << 8);
        s->reg[VGPU98_GS_TMU(i) + VGPU98_GS_TEX_DETAIL] = 0 | (1 << 8) | (255 << 16);
    }
    for (i = 0; i < 256; i++)
        s->gamma[i] = i * 0x010101u;
}

static void
pkt_open(vgpu98_glide_t *gl, const uint32_t *p, uint32_t len)
{
    uint32_t      id, w, h, flags, n_color;
    gl_session_t *s;
    uint32_t      i;
    int           ok = 1;

    if (len < VGPU98_GL_LEN_OPEN) {
        drop(gl, VGPU98_GL_OPEN, "%u dwords", len);
        return;
    }
    id      = p[0];
    w       = p[1] & 0xffff;
    h       = p[1] >> 16;
    flags   = p[2];
    n_color = flags & VGPU98_GL_OPEN_BUFFERS_MASK;
    if (!id || !w || !h || (w > VGPU98_GL_MAX_SIZE) || (h > VGPU98_GL_MAX_SIZE) || !n_color ||
        ((uint64_t) p[3] + p[4] > gl->vram->size)) {
        drop(gl, VGPU98_GL_OPEN, "session %x, %ux%u, %u colour buffers, staging %x+%x", id, w, h, n_color, p[3], p[4]);
        return;
    }
    /* The same session opened again (a program's second grSstWinOpen):
       the old one goes. */
    s = find(gl, id);
    if (s)
        session_end(gl, s);
    for (i = 0; (i < VGPU98_GL_SESSIONS) && !s; i++) {
        if (!gl->s[i].id)
            s = &gl->s[i];
    }
    if (!s) {
        /* All taken: one not on the screen, with the fewest swaps (a
           process that ended without closing), gives way. */
        for (i = 0; i < VGPU98_GL_SESSIONS; i++) {
            if ((&gl->s[i] != gl->shown) && (!s || (gl->s[i].swaps < s->swaps)))
                s = &gl->s[i];
        }
        gl_log(gl, "%u sessions open: session %x ends for session %x", VGPU98_GL_SESSIONS, s->id, id);
        session_end(gl, s);
    }
    memset(s, 0, sizeof(*s));
    s->w             = w;
    s->h             = h;
    s->n_color       = n_color;
    s->aux           = (flags & VGPU98_GL_OPEN_AUX) != 0;
    s->staging       = p[3];
    s->staging_bytes = p[4];
    /* Texture memory: zero, as a board's after its reset reads. */
    s->tmem[0] = calloc(1, VGPU98_GL_TMU_BYTES);
    s->tmem[1] = calloc(1, VGPU98_GL_TMU_BYTES);
    s->tex     = calloc(TEX_MAX, sizeof(gl_tex_t));
    ok         = s->tmem[0] && s->tmem[1] && s->tex;
    for (i = 0; ok && (i < n_color); i++)
        ok = image_make(gl, &s->color[i], VK_FORMAT_R5G6B5_UNORM_PACK16, w, h, 1,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, 0);
    if (ok && s->aux)
        ok = image_make(gl, &s->depth, VK_FORMAT_D16_UNORM, w, h, 1,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_DEPTH_BIT, 0);
    for (i = 0; ok && (i < n_color); i++) {
        VkFramebufferCreateInfo fi = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        VkImageView             att[2];

        att[0]            = s->color[i].view;
        att[1]            = s->depth.view;
        fi.renderPass     = gl->pass[s->aux ? 1 : 0];
        fi.attachmentCount = s->aux ? 2 : 1;
        fi.pAttachments   = att;
        fi.width          = w;
        fi.height         = h;
        fi.layers         = 1;
        ok = gl->vk->vkCreateFramebuffer(gl->vk->dev, &fi, NULL, &s->fb[i]) == VK_SUCCESS;
        if (!ok)
            s->fb[i] = VK_NULL_HANDLE;
    }
    s->id = id; /* so session_end frees what was made */
    if (!ok) {
        gl_log(gl, "session %x: no host images for %ux%u with %u colour buffers%s", id, w, h, n_color,
               s->aux ? " and a depth buffer" : "");
        session_end(gl, s);
        return;
    }
    state_defaults(s, (flags & VGPU98_GL_OPEN_LOWER_LEFT) != 0);
    s->open_ms = plat_get_ticks();
    gl->told        = 0; /* the findings and the first dropped commands said again for a new window */
    gl->drop_logged = 0;
    memset(gl->gone, 0, sizeof(gl->gone)); /* a new window's decodes against its own drops alone */
    gl->gone_at = 0;
    gl->blank_run   = 0;
    gl->blank_swaps = 0;
    gl->blank_secs  = 0.0;
    /* Every buffer starts black, the depth buffer at its farthest. */
    {
        VkCommandBuffer         c = outside(gl);
        VkClearColorValue       cv;
        VkClearDepthStencilValue dv;
        VkImageSubresourceRange rc = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkImageSubresourceRange rd = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

        memset(&cv, 0, sizeof(cv));
        memset(&dv, 0, sizeof(dv));
        dv.depth = 1.0f;
        mem_barrier(gl, c);
        for (i = 0; i < n_color; i++)
            gl->vk->vkCmdClearColorImage(c, s->color[i].image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rc);
        if (s->aux)
            gl->vk->vkCmdClearDepthStencilImage(c, s->depth.image, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &rd);
        mem_barrier(gl, c);
    }
    gl_log(gl, "session %x opened: %ux%u, %u colour buffers%s", id, w, h, n_color, s->aux ? ", a depth buffer" : "");
}

static void
pkt_close(vgpu98_glide_t *gl, const uint32_t *p)
{
    gl_session_t *s = find(gl, p[0]);

    if (!s) {
        drop(gl, VGPU98_GL_CLOSE, "no session %x", p[0]);
        return;
    }
    gl_log(gl, "session %x closed after %llu swaps", s->id, (unsigned long long) s->swaps);
    session_end(gl, s);
}

static void
pkt_state(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t len)
{
    const uint32_t first = p[1];
    const uint32_t n     = len - 2;

    if ((first >= VGPU98_GS_COUNT) || (n > VGPU98_GS_COUNT - first)) {
        drop(gl, VGPU98_GL_STATE, "registers %u-%u", first, first + n - 1);
        return;
    }
    memcpy(&s->reg[first], &p[2], n * sizeof(uint32_t));
    gl->st.states++;
}

/* The clip window in the image's rows: empty is 0 wide. */
static VkRect2D
clip_rect(const gl_session_t *s)
{
    uint32_t x0 = s->reg[VGPU98_GS_CLIP_MIN] & 0xffff;
    uint32_t y0 = s->reg[VGPU98_GS_CLIP_MIN] >> 16;
    uint32_t x1 = s->reg[VGPU98_GS_CLIP_MAX] & 0xffff;
    uint32_t y1 = s->reg[VGPU98_GS_CLIP_MAX] >> 16;
    VkRect2D r;

    if (x1 > s->w)
        x1 = s->w;
    if (y1 > s->h)
        y1 = s->h;
    if (x0 > x1)
        x0 = x1;
    if (y0 > y1)
        y0 = y1;
    if (s->reg[VGPU98_GS_ORIGIN]) {
        /* Lower left: the program's row 0 is the image's last. */
        const uint32_t t = s->h - y1;

        y1 = s->h - y0;
        y0 = t;
    }
    r.offset.x      = (int32_t) x0;
    r.offset.y      = (int32_t) y0;
    r.extent.width  = x1 - x0;
    r.extent.height = y1 - y0;
    return r;
}

static void
pkt_clear(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t len)
{
    const uint32_t    masks = s->reg[VGPU98_GS_MASKS];
    const int         ci    = color_index(s, s->reg[VGPU98_GS_RENDER_BUFFER]);
    VkClearAttachment att[2];
    VkClearRect       cr;
    uint32_t          n = 0;

    if (len < VGPU98_GL_LEN_CLEAR) {
        drop(gl, VGPU98_GL_CLEAR, "%u dwords", len);
        return;
    }
    if (ci < 0) {
        drop(gl, VGPU98_GL_CLEAR, "render buffer %u", s->reg[VGPU98_GS_RENDER_BUFFER]);
        return;
    }
    memset(att, 0, sizeof(att));
    if ((masks & VGPU98_GS_MASK_RGB) && (ci == (int) s->front))
        front_written(gl, s);
    if (masks & VGPU98_GS_MASK_RGB) {
        att[n].aspectMask                  = VK_IMAGE_ASPECT_COLOR_BIT;
        att[n].colorAttachment             = 0;
        att[n].clearValue.color.float32[0] = (float) ((p[1] >> 16) & 0xff) / 255.0f;
        att[n].clearValue.color.float32[1] = (float) ((p[1] >> 8) & 0xff) / 255.0f;
        att[n].clearValue.color.float32[2] = (float) (p[1] & 0xff) / 255.0f;
        att[n].clearValue.color.float32[3] = (float) (p[2] & 0xff) / 255.0f;
        n++;
    }
    /* The aux buffer with its mask on and depth buffering on, its raw
       16-bit value as given (the aux buffer as alpha is deferred). */
    if (s->aux && (masks & VGPU98_GS_MASK_DEPTH) && (s->reg[VGPU98_GS_DEPTH] & 0xff)) {
        att[n].aspectMask              = VK_IMAGE_ASPECT_DEPTH_BIT;
        att[n].clearValue.depthStencil.depth = (float) (p[3] & 0xffff) / 65535.0f;
        n++;
    }
    gl->st.clears++;
    cr.rect           = clip_rect(s);
    cr.baseArrayLayer = 0;
    cr.layerCount     = 1;
    if (!n || !cr.rect.extent.width || !cr.rect.extent.height)
        return;
    gl->vk->vkCmdClearAttachments(inside(gl, s, (uint32_t) ci), n, att, 1, &cr);
}

/* --- The screen ------------------------------------------------------------------ */

/* Reads `aspect` of an image's rows [y, y + h) (the image's), columns
   [x, x + w), into `out` (w x h 16-bit values) from the host GPU: what was
   drawn before is done. 0 with 3D stopped. */
static int
read_image(vgpu98_glide_t *gl, VkImage image, VkImageAspectFlags aspect, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
           uint16_t *out)
{
    VkBuffer          buf;
    VkDeviceSize      off;
    VkBufferImageCopy bc;
    VkCommandBuffer   c;
    uint8_t          *map;

    map = vgpu98_r3d_glide_alloc(gl->r3d, 0, (VkDeviceSize) w * h * 2, 4, &buf, &off);
    if (!map)
        return 0;
    c = outside(gl);
    mem_barrier(gl, c);
    memset(&bc, 0, sizeof(bc));
    bc.bufferOffset                = off;
    bc.bufferRowLength             = w;
    bc.imageSubresource.aspectMask = aspect;
    bc.imageSubresource.layerCount = 1;
    bc.imageOffset.x               = (int32_t) x;
    bc.imageOffset.y               = (int32_t) y;
    bc.imageExtent.width           = w;
    bc.imageExtent.height          = h;
    bc.imageExtent.depth           = 1;
    gl->vk->vkCmdCopyImageToBuffer(c, image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &bc);
    mem_barrier(gl, c);
    if (!vgpu98_r3d_glide_submit(gl->r3d, 1))
        return 0;
    memcpy(out, map, (size_t) w * h * 2);
    return 1;
}

/* Without the display on the host GPU, a frame goes into video memory where
   the screen shows it, when the mode there is the session's at 16 bpp. */
static void
show_in_vram(vgpu98_glide_t *gl, gl_session_t *s)
{
    int       enabled;
    uint32_t  start, pitch, height, bpp, y;
    uint16_t *px;

    vgpu98_gpu_visible(gl->gpu, &enabled, &start, &pitch, &height, &bpp);
    if (!enabled || (bpp != 16) || (pitch < s->w * 2) || (height < s->h) ||
        ((uint64_t) start + (uint64_t) pitch * s->h > gl->vram->size)) {
        if (!gl->fallback_told) {
            gl->fallback_told = 1;
            gl_log(gl, "the screen is not a %ux%u 16-bit mode on the host GPU: the session's frames are not shown",
                   s->w, s->h);
        }
        return;
    }
    px = malloc((size_t) s->w * s->h * 2);
    if (!px)
        return;
    if (read_image(gl, s->color[s->front].image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, s->w, s->h, px)) {
        for (y = 0; y < s->h; y++)
            memcpy(gl->vram->data + start + (size_t) y * pitch, px + (size_t) y * s->w, (size_t) s->w * 2);
        vgpu98_gpu_wrote(gl->gpu, start, pitch * s->h);
    }
    free(px);
}

/* The session's front buffer to the screen. */
static void
show(vgpu98_glide_t *gl, gl_session_t *s)
{
    VkCommandBuffer   c;
    VkBufferImageCopy bc;

    if (!gl->disp || !vgpu98_disp_showing(gl->disp) || !vgpu98_disp_glide_buffer(gl->disp)) {
        show_in_vram(gl, s);
        shown_set(gl, s);
        return;
    }
    c = outside(gl);
    mem_barrier(gl, c);
    memset(&bc, 0, sizeof(bc));
    bc.bufferRowLength             = s->w;
    bc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bc.imageSubresource.layerCount = 1;
    bc.imageExtent.width           = s->w;
    bc.imageExtent.height          = s->h;
    bc.imageExtent.depth           = 1;
    gl->vk->vkCmdCopyImageToBuffer(c, s->color[s->front].image, VK_IMAGE_LAYOUT_GENERAL,
                                   (VkBuffer) (uintptr_t) vgpu98_disp_glide_buffer(gl->disp), 1, &bc);
    mem_barrier(gl, c);
    /* The display composes from the buffer only while it is not being
       written. */
    {
        const int step = vgpu98_gpu_step(gl->gpu, VGPU98_STEP_LAYER);

        vgpu98_disp_layer_lock(gl->disp);
        vgpu98_gpu_step(gl->gpu, step);
    }
    if (vgpu98_r3d_glide_submit(gl->r3d, 1))
        vgpu98_disp_glide_frame(gl->disp, s->w, s->h, s->gamma);
    vgpu98_disp_layer_unlock(gl->disp);
    shown_set(gl, s);
}

static void
pkt_swap(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t len, uint32_t generation)
{
    /* The V-sync overrides on the interval the DLL asked for. */
    const uint32_t interval =
        vgpu98_gpu_vsync_interval(gl->gpu, VGPU98_VSYNC_GLIDE_SHIFT, (len >= VGPU98_GL_LEN_SWAP) ? p[1] : 1);

    outside(gl);
    /* What the front buffer held goes behind: the swap shows the new one. */
    if (gl->front_dirty == s) {
        gl->front_dirty = NULL;
        vgpu98_gpu_glide_front(gl->gpu, 0);
    }
    /* The back buffer becomes the front, at the interval's vertical blank,
       as a Voodoo swaps. */
    if (s->n_color > 1)
        s->front = (s->front + 1) % s->n_color;
    vgpu98_gpu_wait_vblanks(gl->gpu, (interval > 4) ? 4 : interval, generation);
    show(gl, s);
    gl->front_vblank = vgpu98_gpu_vblanks(gl->gpu);
    s->swaps++;
    {
        const uint32_t now = plat_get_ticks();

        if (gl->st.swaps && ((uint32_t) (now - gl->last_swap_ms) > gl->st.frame_worst_ms))
            gl->st.frame_worst_ms = now - gl->last_swap_ms;
        gl->last_swap_ms = now;
        gl->st.swaps++;
        /* A capture of the frame just shown: at a finding, and once 10
           seconds into the session. */
        if (gl->logging && (s->captures < CAPTURES) &&
            (s->capture_due || (!s->captured_late && ((uint32_t) (now - s->open_ms) >= CAPTURE_MS)))) {
            const int due = s->capture_due;

            if (capture(gl, s, due ? "a finding" : "10 seconds into the session", 1)) {
                if (!due)
                    s->captured_late = 1;
                s->capture_due = 0;
            }
        }
        /* An 86Box screenshot at Glide: the frame just shown, ten a session
           apart from the automatic ones. */
        if (gl->logging && atomic_load(&gl->shot_asked)) {
            if (s->shot_captures >= SHOT_CAPTURES)
                atomic_store(&gl->shot_asked, 0);
            else if (capture(gl, s, "a screenshot (Action > Take screenshot)", 0)) {
                atomic_store(&gl->shot_asked, 0);
                s->shot_captures++;
            }
        }
        gl->frame_from = gl->n_cmds;
    }
}

static void
pkt_gamma(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t len)
{
    if (len < VGPU98_GL_LEN_GAMMA) {
        drop(gl, VGPU98_GL_GAMMA, "%u dwords", len);
        return;
    }
    memcpy(s->gamma, &p[1], sizeof(s->gamma));
    if ((gl->shown == s) && gl->disp && vgpu98_disp_glide_on(gl->disp)) {
        /* The frame on the screen again, through the new table. */
        vgpu98_disp_layer_lock(gl->disp);
        vgpu98_disp_glide_frame(gl->disp, s->w, s->h, s->gamma);
        vgpu98_disp_layer_unlock(gl->disp);
    }
}

void
vgpu98_glide_tick(vgpu98_glide_t *gl)
{
    gl_session_t *s;
    uint32_t      vb;

    if (!gl || !gl->front_dirty)
        return;
    /* Once a vertical blank: the next one wakes the thread for it. */
    vb = vgpu98_gpu_vblanks(gl->gpu);
    if (vb == gl->front_vblank)
        return;
    s                = gl->front_dirty;
    gl->front_dirty  = NULL;
    gl->front_vblank = vb;
    vgpu98_gpu_glide_front(gl->gpu, 0);
    if (!vgpu98_r3d_broken(gl->r3d))
        show(gl, s);
}

/* --- The frame buffer (LFB_READ, LFB_WRITE) ------------------------------------- */

/* Bytes of a pixel in a Glide write format; 0 for one the device does not
   know. */
static uint32_t
lfb_bytes(uint32_t f)
{
    switch (f) {
        case VGPU98_GL_LFB_565:
        case VGPU98_GL_LFB_555:
        case VGPU98_GL_LFB_1555:
        case VGPU98_GL_LFB_ZA16:
            return 2;
        case VGPU98_GL_LFB_888:
        case VGPU98_GL_LFB_8888:
        case VGPU98_GL_LFB_565_DEPTH:
        case VGPU98_GL_LFB_555_DEPTH:
        case VGPU98_GL_LFB_1555_DEPTH:
            return 4;
        default:
            return 0;
    }
}

static int
lfb_has_color(uint32_t f)
{
    return f != VGPU98_GL_LFB_ZA16;
}

static int
lfb_has_depth(uint32_t f)
{
    return (f == VGPU98_GL_LFB_ZA16) || (f == VGPU98_GL_LFB_565_DEPTH) || (f == VGPU98_GL_LFB_555_DEPTH) ||
           (f == VGPU98_GL_LFB_1555_DEPTH);
}

/* A 32-bit colour as the program writes it (grLfbWriteColorSwizzle, then
   grLfbWriteColorFormat) to 0x00RRGGBB, and back. */
static uint32_t
argb_in(uint32_t v, uint32_t flags)
{
    if (flags & VGPU98_GL_LFB_SWAP_BYTES)
        v = ((v & 0xff) << 24) | ((v & 0xff00) << 8) | ((v >> 8) & 0xff00) | (v >> 24);
    if (flags & VGPU98_GL_LFB_SWAP_WORDS)
        v = (v << 16) | (v >> 16);
    switch ((flags >> VGPU98_GL_LFB_COLOR_SHIFT) & 3) {
        case 1: /* ABGR */
            return ((v & 0xff) << 16) | (v & 0xff00) | ((v >> 16) & 0xff);
        case 2: /* RGBA */
            return v >> 8;
        case 3: /* BGRA */
            return (((v >> 8) & 0xff) << 16) | (((v >> 16) & 0xff) << 8) | (v >> 24);
        default: /* ARGB */
            return v & 0xffffff;
    }
}

static uint32_t
argb_out(uint32_t rgb, uint32_t flags)
{
    const uint32_t r = (rgb >> 16) & 0xff, g = (rgb >> 8) & 0xff, b = rgb & 0xff;
    uint32_t       v;

    switch ((flags >> VGPU98_GL_LFB_COLOR_SHIFT) & 3) {
        case 1:
            v = 0xff000000u | (b << 16) | (g << 8) | r;
            break;
        case 2:
            v = (r << 24) | (g << 16) | (b << 8) | 0xff;
            break;
        case 3:
            v = (b << 24) | (g << 16) | (r << 8) | 0xff;
            break;
        default:
            v = 0xff000000u | rgb;
            break;
    }
    if (flags & VGPU98_GL_LFB_SWAP_WORDS)
        v = (v << 16) | (v >> 16);
    if (flags & VGPU98_GL_LFB_SWAP_BYTES)
        v = ((v & 0xff) << 24) | ((v & 0xff00) << 8) | ((v >> 8) & 0xff00) | (v >> 24);
    return v;
}

static uint32_t
rgb565_to_888(uint16_t v)
{
    const uint32_t r = v >> 11, g = (v >> 5) & 63, b = v & 31;

    return (((r << 3) | (r >> 2)) << 16) | (((g << 2) | (g >> 4)) << 8) | ((b << 3) | (b >> 2));
}

static uint16_t
rgb888_to_565(uint32_t rgb)
{
    return (uint16_t) ((((rgb >> 16) & 0xf8) << 8) | (((rgb >> 8) & 0xfc) << 3) | ((rgb & 0xff) >> 3));
}

/* A buffer pixel (565 colour, 16-bit depth) as the program sees it in
   format f; `px` points at the pixel's bytes. */
static void
lfb_put(uint8_t *px, uint32_t f, uint32_t flags, uint16_t color, uint16_t depth)
{
    uint32_t v;
    uint16_t c16;

    switch (f) {
        case VGPU98_GL_LFB_565:
            c16 = color;
            memcpy(px, &c16, 2);
            return;
        case VGPU98_GL_LFB_555:
        case VGPU98_GL_LFB_1555:
            c16 = (uint16_t) (((color >> 1) & 0x7fe0) | (color & 0x1f) | ((f == VGPU98_GL_LFB_1555) ? 0x8000 : 0));
            memcpy(px, &c16, 2);
            return;
        case VGPU98_GL_LFB_ZA16:
            memcpy(px, &depth, 2);
            return;
        case VGPU98_GL_LFB_888:
        case VGPU98_GL_LFB_8888:
            v = argb_out(rgb565_to_888(color), flags);
            memcpy(px, &v, 4);
            return;
        case VGPU98_GL_LFB_565_DEPTH:
            v = color | ((uint32_t) depth << 16);
            memcpy(px, &v, 4);
            return;
        case VGPU98_GL_LFB_555_DEPTH:
        case VGPU98_GL_LFB_1555_DEPTH:
            v = ((color >> 1) & 0x7fe0) | (color & 0x1f) | ((f == VGPU98_GL_LFB_1555_DEPTH) ? 0x8000 : 0) |
                ((uint32_t) depth << 16);
            memcpy(px, &v, 4);
            return;
        default:
            return;
    }
}

/* A pixel the program wrote in format f, as 565 colour and 16-bit depth. */
static void
lfb_get(const uint8_t *px, uint32_t f, uint32_t flags, uint16_t *color, uint16_t *depth)
{
    uint32_t v = 0;
    uint16_t c16;

    switch (f) {
        case VGPU98_GL_LFB_565:
            memcpy(color, px, 2);
            return;
        case VGPU98_GL_LFB_555:
        case VGPU98_GL_LFB_1555:
            memcpy(&c16, px, 2);
            *color = (uint16_t) (((c16 << 1) & 0xffc0) | ((c16 >> 4) & 0x20) | (c16 & 0x1f));
            return;
        case VGPU98_GL_LFB_ZA16:
            memcpy(depth, px, 2);
            return;
        case VGPU98_GL_LFB_888:
        case VGPU98_GL_LFB_8888:
            memcpy(&v, px, 4);
            *color = rgb888_to_565(argb_in(v, flags));
            return;
        case VGPU98_GL_LFB_565_DEPTH:
            memcpy(&v, px, 4);
            *color = (uint16_t) v;
            *depth = (uint16_t) (v >> 16);
            return;
        case VGPU98_GL_LFB_555_DEPTH:
        case VGPU98_GL_LFB_1555_DEPTH:
            memcpy(&v, px, 4);
            c16    = (uint16_t) v;
            *color = (uint16_t) (((c16 << 1) & 0xffc0) | ((c16 >> 4) & 0x20) | (c16 & 0x1f));
            *depth = (uint16_t) (v >> 16);
            return;
        default:
            return;
    }
}

/* LFB's arguments, checked: the rectangle inside the session; the rows,
   at most a Voodoo3's, inside the staging surface the session's OPEN gave. */
typedef struct lfb_args_t {
    uint32_t buffer, x, y, w, h, format, flags, offset, stride, bytes;
    int      ci; /* the colour buffer, -1 for the aux buffer alone */
} lfb_args_t;

static int
lfb_args(vgpu98_glide_t *gl, gl_session_t *s, uint32_t op, const uint32_t *p, uint32_t len, lfb_args_t *a)
{
    if (len < VGPU98_GL_LEN_LFB) {
        drop(gl, op, "%u dwords", len);
        return 0;
    }
    a->buffer = p[1];
    a->x      = p[2] & 0xffff;
    a->y      = p[2] >> 16;
    a->w      = p[3] & 0xffff;
    a->h      = p[3] >> 16;
    a->format = p[4] & VGPU98_GL_LFB_FORMAT_MASK;
    a->flags  = p[4];
    a->offset = p[5];
    a->stride = p[6];
    a->bytes  = lfb_bytes(a->format);
    a->ci     = (a->buffer == VGPU98_GL_AUXBUFFER) ? -1 : color_index(s, a->buffer);
    if (!a->bytes || ((a->ci < 0) && (a->buffer != VGPU98_GL_AUXBUFFER)) ||
        ((a->buffer == VGPU98_GL_AUXBUFFER) && (!s->aux || (a->format != VGPU98_GL_LFB_ZA16))) ||
        (lfb_has_color(a->format) && (a->ci < 0)) || !a->w || !a->h || (a->x + a->w > s->w) || (a->y + a->h > s->h) ||
        (a->stride > LFB_STRIDE) || ((uint64_t) a->stride < (uint64_t) (a->x + a->w) * a->bytes) ||
        (a->offset < s->staging) ||
        ((uint64_t) a->offset + (uint64_t) a->stride * (a->y + a->h) > (uint64_t) s->staging + s->staging_bytes) ||
        ((uint64_t) s->staging + s->staging_bytes > gl->vram->size)) {
        drop(gl, op, "buffer %u, %u,%u %ux%u, format %x, offset %x, stride %u", a->buffer, a->x, a->y, a->w, a->h,
             a->format, a->offset, a->stride);
        return 0;
    }
    if (lfb_has_depth(a->format) && !s->aux) {
        drop(gl, op, "a depth format without a depth buffer");
        return 0;
    }
    return 1;
}

/* The image rows of the program's rows [y, y + h): the first, and whether
   they run backwards (lower-left origin). */
static uint32_t
lfb_image_y(const gl_session_t *s, const lfb_args_t *a)
{
    return (a->flags & VGPU98_GL_LFB_LOWER_LEFT) ? s->h - (a->y + a->h) : a->y;
}

/* The snapshot of the staging area a transfer uses: the write area is at
   the surface's start, the read area after it. */
static gl_snap_t *
lfb_snap(gl_session_t *s, const lfb_args_t *a)
{
    return &s->snap[(a->offset == s->staging) ? 0 : 1];
}

static void
pkt_lfb_read(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t len)
{
    lfb_args_t a;
    uint16_t  *color = NULL, *depth = NULL;
    gl_snap_t *sn;
    uint32_t   r, i;
    size_t     area;

    if (!lfb_args(gl, s, VGPU98_GL_LFB_READ, p, len, &a))
        return;
    gl->st.lfb_reads++;
    sn   = lfb_snap(s, &a);
    area = (size_t) a.stride * s->h;
    if (lfb_has_color(a.format))
        color = malloc((size_t) a.w * a.h * 2);
    if (lfb_has_depth(a.format))
        depth = malloc((size_t) a.w * a.h * 2);
    if ((lfb_has_color(a.format) && !color) || (lfb_has_depth(a.format) && !depth))
        goto out;
    if ((color && !read_image(gl, s->color[a.ci].image, VK_IMAGE_ASPECT_COLOR_BIT, a.x, lfb_image_y(s, &a), a.w, a.h, color)) ||
        (depth && !read_image(gl, s->depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, a.x, lfb_image_y(s, &a), a.w, a.h, depth)))
        goto out;

    /* The snapshot: what goes into the staging rows, and the buffer's
       values, for the whole area at the transfer's stride. */
    if ((sn->size != area) || !sn->bytes || !sn->native || !sn->native2) {
        snap_free(sn);
        sn->size    = area;
        sn->bytes   = malloc(area);
        sn->native  = malloc((size_t) s->w * s->h * 2);
        sn->native2 = malloc((size_t) s->w * s->h * 2);
        if (!sn->bytes || !sn->native || !sn->native2) {
            snap_free(sn);
            goto out;
        }
    }
    sn->offset = a.offset;
    sn->stride = a.stride;
    sn->format = a.format;
    sn->buffer = a.buffer;
    sn->x0     = a.x;
    sn->x1     = a.x + a.w;
    sn->y0     = a.y;
    sn->y1     = a.y + a.h;
    sn->valid  = 1;
    for (r = 0; r < a.h; r++) {
        /* The program's row a.y + r is the readback's row r, or (lower
           left) its row a.h - 1 - r. */
        const uint32_t src = (a.flags & VGPU98_GL_LFB_LOWER_LEFT) ? a.h - 1 - r : r;
        const size_t   at  = (size_t) (a.y + r) * a.stride + (size_t) a.x * a.bytes;
        uint8_t       *dst = gl->vram->data + a.offset + at;

        for (i = 0; i < a.w; i++) {
            const uint16_t c = color ? color[(size_t) src * a.w + i] : 0;
            const uint16_t d = depth ? depth[(size_t) src * a.w + i] : 0;

            lfb_put(dst + (size_t) i * a.bytes, a.format, a.flags, c, d);
            sn->native[(size_t) (a.y + r) * s->w + a.x + i]  = c;
            sn->native2[(size_t) (a.y + r) * s->w + a.x + i] = d;
        }
        memcpy(sn->bytes + at, dst, (size_t) a.w * a.bytes);
    }
    vgpu98_gpu_wrote(gl->gpu, a.offset + a.y * a.stride, a.h * a.stride);
out:
    free(color);
    free(depth);
}

static void
pkt_lfb_write(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t len)
{
    lfb_args_t         a;
    gl_snap_t         *sn;
    uint16_t          *color, *depth;
    uint8_t           *map;
    VkBufferImageCopy *rc = NULL, *rd = NULL;
    uint32_t           n_c = 0, n_d = 0, r, i;
    VkBuffer           buf = VK_NULL_HANDLE;
    VkDeviceSize       off = 0, row, half;
    int                changed_only;

    if (!lfb_args(gl, s, VGPU98_GL_LFB_WRITE, p, len, &a))
        return;
    gl->st.lfb_writes++;
    if ((a.flags & VGPU98_GL_LFB_PIPELINE) && !gl->pipeline_told) {
        gl->pipeline_told = 1;
        gl_log(gl, "a frame buffer write through the pixel pipeline is taken as a plain write (not done yet)");
    }
    sn = lfb_snap(s, &a);
    /* Only the pixels changed since the READ that filled the rows: the
       others keep the buffer's values exactly (a format narrower than the
       buffer's would otherwise change them). */
    changed_only = (a.flags & VGPU98_GL_LFB_CHANGED) && sn->valid && (sn->offset == a.offset) && (sn->stride == a.stride) &&
                   (sn->format == a.format) && (sn->buffer == a.buffer);
    /* One allocation for the colour and the depth rows, each row padded to
       4 bytes (a depth copy's offset must be a multiple of 4). */
    row  = ((VkDeviceSize) a.w * 2 + 3) & ~(VkDeviceSize) 3;
    half = row * a.h;
    map  = vgpu98_r3d_glide_alloc(gl->r3d, 0, half * 2, 4, &buf, &off);
    rc   = calloc(a.h, sizeof(*rc));
    rd   = calloc(a.h, sizeof(*rd));
    if (!map || !rc || !rd)
        goto out;
    for (r = 0; r < a.h; r++) {
        const uint32_t py       = a.y + r;
        const size_t   at       = (size_t) py * a.stride + (size_t) a.x * a.bytes;
        const uint8_t *src      = gl->vram->data + a.offset + at;
        const uint32_t iy       = (a.flags & VGPU98_GL_LFB_LOWER_LEFT) ? s->h - 1 - py : py;
        const int      row_snap = changed_only && (py >= sn->y0) && (py < sn->y1) && (a.x >= sn->x0) && (a.x + a.w <= sn->x1);

        if (row_snap && !memcmp(src, sn->bytes + at, (size_t) a.w * a.bytes))
            continue; /* the row as the READ left it */
        color = (uint16_t *) (map + row * r);
        depth = (uint16_t *) (map + half + row * r);
        for (i = 0; i < a.w; i++) {
            uint16_t c = 0, d = 0;

            if (row_snap && !memcmp(src + (size_t) i * a.bytes, sn->bytes + at + (size_t) i * a.bytes, a.bytes)) {
                c = sn->native[(size_t) py * s->w + a.x + i];
                d = sn->native2[(size_t) py * s->w + a.x + i];
            } else
                lfb_get(src + (size_t) i * a.bytes, a.format, a.flags, &c, &d);
            color[i] = c;
            depth[i] = d;
        }
        if (lfb_has_color(a.format)) {
            rc[n_c].bufferOffset                = off + row * r;
            rc[n_c].bufferRowLength             = (uint32_t) (row / 2);
            rc[n_c].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            rc[n_c].imageSubresource.layerCount = 1;
            rc[n_c].imageOffset.x               = (int32_t) a.x;
            rc[n_c].imageOffset.y               = (int32_t) iy;
            rc[n_c].imageExtent.width           = a.w;
            rc[n_c].imageExtent.height          = 1;
            rc[n_c].imageExtent.depth           = 1;
            n_c++;
        }
        if (lfb_has_depth(a.format)) {
            rd[n_d].bufferOffset                = off + half + row * r;
            rd[n_d].bufferRowLength             = (uint32_t) (row / 2);
            rd[n_d].imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            rd[n_d].imageSubresource.layerCount = 1;
            rd[n_d].imageOffset.x               = (int32_t) a.x;
            rd[n_d].imageOffset.y               = (int32_t) iy;
            rd[n_d].imageExtent.width           = a.w;
            rd[n_d].imageExtent.height          = 1;
            rd[n_d].imageExtent.depth           = 1;
            n_d++;
        }
    }
    if (n_c || n_d) {
        VkCommandBuffer c = outside(gl);

        mem_barrier(gl, c);
        if (n_c)
            gl->vk->vkCmdCopyBufferToImage(c, buf, s->color[a.ci].image, VK_IMAGE_LAYOUT_GENERAL, n_c, rc);
        if (n_d)
            gl->vk->vkCmdCopyBufferToImage(c, buf, s->depth.image, VK_IMAGE_LAYOUT_GENERAL, n_d, rd);
        mem_barrier(gl, c);
        if (n_c && (a.ci == (int) s->front))
            front_written(gl, s);
    }
    /* The rows go on as the program left them: a second unlock takes
       nothing again. */
    if (changed_only)
        sn->valid = 0;
out:
    free(rc);
    free(rd);
}

/* --- Textures (TEXDATA, PALETTE, the TMUs' sources) --------------------------------- */

static void
tex_drop(vgpu98_glide_t *gl, gl_session_t *s, uint32_t i, int remember)
{
    uint32_t k;

    /* At Glide, a texture a download dropped is remembered: a decode of
       the same bytes later is one a content-keyed cache would have kept.
       Room's evictions are not: any cache of
       this size has them. */
    if (remember && gl->logging && s->tex[i].hashed) {
        k = gl->gone_at++ % GONE_KEPT;

        gl->gone[k].mem     = s->tex[i].tmu;
        gl->gone[k].address = s->tex[i].address;
        gl->gone[k].info    = s->tex[i].info;
        gl->gone[k].pal     = s->tex[i].pal;
        gl->gone[k].hash    = s->tex[i].hash;
    }
    s->tex_bytes -= s->tex[i].bytes;
    image_free(gl, &s->tex[i].im);
    gl->tex_live--;
    gl->st.tex_dropped++;
    s->tex[i] = s->tex[--s->n_tex];
    memset(&s->tex[s->n_tex], 0, sizeof(gl_tex_t));
    /* The last texture moved into the slot: a TMU's last follows it. */
    for (k = 0; k < 2; k++) {
        if (s->last[k] == s->n_tex)
            s->last[k] = i;
    }
    gl->tex_dropped++;
}

/* Bytes into a TMU's memory: the textures decoded from them go. */
static void
pkt_texdata(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t len)
{
    uint32_t mem, at, n, i;

    if (len < VGPU98_GL_LEN_TEXDATA) {
        drop(gl, VGPU98_GL_TEXDATA, "%u dwords", len);
        return;
    }
    mem = p[1];
    at  = p[2];
    n   = p[3];
    if ((mem > VGPU98_GL_MEM_TMU1) || (at > VGPU98_GL_TMU_BYTES) || (n > VGPU98_GL_TMU_BYTES - at) ||
        ((uint64_t) n > (uint64_t) (len - VGPU98_GL_LEN_TEXDATA) * 4)) {
        drop(gl, VGPU98_GL_TEXDATA, "memory %u, %u bytes at %x, %u dwords", mem, n, at, len);
        return;
    }
    memcpy(s->tmem[mem] + at, &p[VGPU98_GL_LEN_TEXDATA], n);
    gl->tex_bytes_down += n;
    gl->st.texdata_bytes += n;
    if (n) {
        uint32_t pg;

        for (pg = at / 4096; pg <= (at + n - 1) / 4096; pg++)
            s->written[mem][pg / 8] |= (uint8_t) (1u << (pg & 7));
        if (at + n > s->tmem_high[mem])
            s->tmem_high[mem] = at + n;
    }
    for (i = 0; i < s->n_tex;) {
        const gl_tex_t *t = &s->tex[i];

        if ((t->tmu == mem) && (t->lo < at + n) && (at < t->hi)) {
            /* Part of a texture a TMU samples now, by a download not of it
               alone: two textures in the same memory. */
            if (((at < t->lo) || (at + n > t->hi)) && (n < t->hi - t->lo)) {
                uint32_t k;

                for (k = 0; k < 2; k++) {
                    if (s->reg[VGPU98_GS_TMU(k) + VGPU98_GS_TEX_SOURCE] && (s->tex[i].address == s->reg[VGPU98_GS_TMU(k) + VGPU98_GS_TEX_ADDRESS]) &&
                        (s->tex[i].info == s->reg[VGPU98_GS_TMU(k) + VGPU98_GS_TEX_INFO]))
                        finding(gl, s, FIND_OVERLAP,
                                "texture overlap: a download to TMU %u at %x-%x wrote over part of the texture TMU %u samples at %x-%x",
                                mem, at, at + n, k, t->lo, t->hi);
                }
            }
            tex_drop(gl, s, i, 1);
        } else
            i++;
    }
}

static uint64_t
pal_hash(const uint32_t *pal)
{
    uint64_t h = 14695981039346656037ull;
    uint32_t i;

    for (i = 0; i < 256; i++)
        h = (h ^ pal[i]) * 1099511628211ull;
    return h ? h : 1;
}

/* Palette entries to a TMU, or both (2): textures of a palette format are
   found by the palette's contents, so one decoded for another stays. */
static void
pkt_palette(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t len)
{
    uint32_t tmu, first, count, t;

    if (len < VGPU98_GL_LEN_PALETTE) {
        drop(gl, VGPU98_GL_PALETTE, "%u dwords", len);
        return;
    }
    tmu   = p[1];
    first = p[2] & 0xffff;
    count = p[2] >> 16;
    if ((tmu > 2) || !count || (first + count > 256) || (count > len - VGPU98_GL_LEN_PALETTE)) {
        drop(gl, VGPU98_GL_PALETTE, "TMU %u, entries %u+%u, %u dwords", tmu, first, count, len);
        return;
    }
    for (t = 0; t < 2; t++) {
        if ((tmu != 2) && (tmu != t))
            continue;
        memcpy(&s->palette[t][first], &p[VGPU98_GL_LEN_PALETTE], count * sizeof(uint32_t));
        s->pal_hash[t] = pal_hash(s->palette[t]);
    }
    gl->st.palettes++;
}

static int
tex_paletted(int format)
{
    return (format == VGPU98_GL_TEX_P_8) || (format == VGPU98_GL_TEX_AP_88) || (format == 0x6);
}

/* A TMU's texture as its registers describe it: valid, in the device's
   numbers. */
static int
tex_desc(const gl_session_t *s, uint32_t t, vgpu98_gltex_t *g, uint32_t *mem)
{
    const uint32_t info = s->reg[VGPU98_GS_TMU(t) + VGPU98_GS_TEX_INFO];

    g->format  = (int) VGPU98_GS_TEXINFO_FORMAT(info);
    g->small   = (int) VGPU98_GS_TEXINFO_SMALL(info);
    g->large   = (int) VGPU98_GS_TEXINFO_LARGE(info);
    g->aspect  = VGPU98_GS_TEXINFO_ASPECT(info);
    g->evenodd = (int) VGPU98_GS_TEXINFO_EVENODD(info);
    *mem       = VGPU98_GS_TEXINFO_MEM(info);
    return (g->format <= 0xe) && (g->format != 0x7) && (g->small <= g->large) && (g->large <= 8) && (g->aspect >= -3) &&
           (g->aspect <= 3) && (g->evenodd >= 1) && (g->evenodd <= 3) && (*mem <= VGPU98_GL_MEM_TMU1);
}

/* A level's texels from TMU memory, those past its end zero. */
static void
tex_read(const gl_session_t *s, uint32_t mem, uint32_t at, uint32_t n, uint8_t *out)
{
    const uint32_t have = (at < VGPU98_GL_TMU_BYTES) ? VGPU98_GL_TMU_BYTES - at : 0;
    const uint32_t take = (n < have) ? n : have;

    if (take)
        memcpy(out, s->tmem[mem] + at, take);
    if (take < n)
        memset(out + take, 0, n - take);
}

/* The TMU memory a texture's levels came from, hashed (FNV-1a 64). */
static uint64_t
tex_hash(const gl_session_t *s, uint32_t mem, uint32_t lo, uint32_t hi)
{
    uint64_t h = 0xcbf29ce484222325ull;
    uint32_t i;

    if (hi > VGPU98_GL_TMU_BYTES)
        hi = VGPU98_GL_TMU_BYTES;
    for (i = lo; i < hi; i++)
        h = (h ^ s->tmem[mem][i]) * 0x100000001b3ull;
    return h;
}

/* Decodes a TMU's texture into a new host image. NULL when it cannot. */
static gl_tex_t *
tex_decode(vgpu98_glide_t *gl, gl_session_t *s, uint32_t t, const vgpu98_gltex_t *g, uint32_t mem, uint64_t pal)
{
    const uint32_t     address = s->reg[VGPU98_GS_TMU(t) + VGPU98_GS_TEX_ADDRESS];
    const uint32_t     levels  = (uint32_t) (g->large - g->small + 1);
    uint32_t           w0, h0, w, h, i, lo = 0xffffffffu, hi = 0;
    VkDeviceSize       texels = 0, at = 0, off = 0;
    VkBuffer           buf = VK_NULL_HANDLE;
    uint8_t           *map, *raw;
    VkBufferImageCopy  bc[9];
    uint32_t           n_bc = 0;
    gl_tex_t          *x;
    VkCommandBuffer    c;
    int                lod;
    uint64_t           sum[3] = { 0, 0, 0 }, sum_n = 0;

    /* Room: the least lately used go first. */
    while ((s->n_tex == TEX_MAX) || (s->n_tex && ((s->tex_bytes > TEX_BYTES) || (gl->tex_live >= TEX_LIVE)))) {
        uint32_t k, oldest = 0;

        for (k = 1; k < s->n_tex; k++) {
            if (s->tex[k].used < s->tex[oldest].used)
                oldest = k;
        }
        if (s->tex[oldest].used == s->draw_serial)
            break; /* every one in this DRAW */
        tex_drop(gl, s, oldest, 0);
    }
    if ((s->n_tex == TEX_MAX) || (gl->tex_live >= TEX_LIVE))
        return NULL;
    for (lod = g->large; lod >= g->small; lod--) {
        if (vgpu98_gl_of_parity(g, lod)) {
            vgpu98_gl_level_size(lod, g->aspect, &w, &h);
            texels += (VkDeviceSize) w * h;
        }
    }
    raw = malloc(256 * 256 * 2);
    map = raw ? vgpu98_r3d_glide_alloc(gl->r3d, 0, texels * 4, 16, &buf, &off) : NULL;
    if (!map) {
        free(raw);
        return NULL;
    }
    x = &s->tex[s->n_tex];
    memset(x, 0, sizeof(*x));
    vgpu98_gl_level_size(g->large, g->aspect, &w0, &h0);
    if (!image_make(gl, &x->im, VK_FORMAT_R8G8B8A8_UNORM, w0, h0, levels,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 1)) {
        free(raw);
        return NULL;
    }
    for (lod = g->large, i = 0; lod >= g->small; lod--, i++) {
        uint32_t lv_at, lv_bytes;

        if (!vgpu98_gl_of_parity(g, lod))
            continue;
        vgpu98_gl_level_size(lod, g->aspect, &w, &h);
        lv_at    = address + vgpu98_gl_level_offset(g, lod);
        lv_bytes = vgpu98_gl_level_bytes(g, lod);
        tex_read(s, mem, lv_at, lv_bytes, raw);
        if (lv_at < lo)
            lo = lv_at;
        if (lv_at + lv_bytes > hi)
            hi = lv_at + lv_bytes;
        vgpu98_gl_decode(g->format, raw, w, h, s->palette[mem], (uint32_t *) (map + at * 4));
        /* Diagnostics: the largest level's average colour. */
        if (gl->logging && !sum_n) {
            const uint8_t *px = map + at * 4;
            uint32_t       k;

            for (k = 0; k < w * h; k++) {
                sum[0] += px[k * 4];
                sum[1] += px[k * 4 + 1];
                sum[2] += px[k * 4 + 2];
            }
            sum_n = (uint64_t) w * h;
        }
        memset(&bc[n_bc], 0, sizeof(bc[0]));
        bc[n_bc].bufferOffset                = off + at * 4;
        bc[n_bc].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bc[n_bc].imageSubresource.mipLevel   = i;
        bc[n_bc].imageSubresource.layerCount = 1;
        bc[n_bc].imageExtent.width           = w;
        bc[n_bc].imageExtent.height          = h;
        bc[n_bc].imageExtent.depth           = 1;
        n_bc++;
        at += (VkDeviceSize) w * h;
    }
    free(raw);
    c = outside(gl);
    mem_barrier(gl, c);
    if (g->evenodd != 3) {
        /* A split texture fills its parity's levels: the others read 0. */
        VkClearColorValue       cv;
        VkImageSubresourceRange rc = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1 };

        memset(&cv, 0, sizeof(cv));
        gl->vk->vkCmdClearColorImage(c, x->im.image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rc);
        mem_barrier(gl, c);
    }
    gl->vk->vkCmdCopyBufferToImage(c, buf, x->im.image, VK_IMAGE_LAYOUT_GENERAL, n_bc, bc);
    mem_barrier(gl, c);
    x->tmu     = mem;
    x->address = address;
    x->info    = s->reg[VGPU98_GS_TMU(t) + VGPU98_GS_TEX_INFO];
    x->pal     = pal;
    x->lo      = lo;
    x->hi      = hi;
    x->bytes   = texels * 4 * 4 / 3 + 1024;
    if (gl->logging && sum_n && (lo < hi)) {
        x->avg    = (uint32_t) ((sum[0] / sum_n) << 16 | (sum[1] / sum_n) << 8 | (sum[2] / sum_n));
        x->hash   = tex_hash(s, mem, lo, hi);
        x->hashed = 1;
        for (i = 0; i < GONE_KEPT; i++) {
            if ((gl->gone[i].hash == x->hash) && (gl->gone[i].address == address) && (gl->gone[i].info == x->info) &&
                (gl->gone[i].mem == mem) && (gl->gone[i].pal == pal)) {
                gl->st.unchanged++;
                gl->gone[i].info = 0xffffffffu; /* counted once */
                break;
            }
        }
    }
    s->tex_bytes += x->bytes;
    s->n_tex++;
    gl->tex_decoded++;
    gl->tex_live++;
    gl->st.decoded++;
    gl->st.decoded_by_format[g->format & 15]++;
    /* Sampled where no download wrote: the program's textures elsewhere. */
    if (lo < hi) {
        uint32_t pg, any = 0;

        for (pg = lo / 4096; (pg <= (hi - 1) / 4096) && (pg < TMU_PAGES); pg++)
            any |= s->written[mem][pg / 8] & (1u << (pg & 7));
        if (!any)
            finding(gl, s, FIND_MISSING, "missing texture: TMU %u samples %x-%x of its memory, where no download wrote", mem, lo, hi);
    }
    return x;
}

/* The image a TMU samples in this DRAW: its texture, found or decoded; the
   white image for a TMU with none. */
static VkImageView
tex_view(vgpu98_glide_t *gl, gl_session_t *s, uint32_t t)
{
    vgpu98_gltex_t g;
    uint32_t       mem, address, info, i;
    uint64_t       pal;
    gl_tex_t      *x = NULL;

    if (!s->reg[VGPU98_GS_TMU(t) + VGPU98_GS_TEX_SOURCE] || !tex_desc(s, t, &g, &mem))
        return gl->white.view;
    address = s->reg[VGPU98_GS_TMU(t) + VGPU98_GS_TEX_ADDRESS];
    info    = s->reg[VGPU98_GS_TMU(t) + VGPU98_GS_TEX_INFO];
    pal     = tex_paletted(g.format) ? s->pal_hash[mem] : 0;
    i       = s->last[t];
    if ((i < s->n_tex) && (s->tex[i].address == address) && (s->tex[i].info == info) && (s->tex[i].pal == pal) &&
        (s->tex[i].tmu == mem))
        x = &s->tex[i];
    for (i = 0; !x && (i < s->n_tex); i++) {
        if ((s->tex[i].address == address) && (s->tex[i].info == info) && (s->tex[i].pal == pal) && (s->tex[i].tmu == mem))
            x = &s->tex[i];
    }
    if (x) {
        gl->st.found++;
        /* At Glide, one hit in CHECK_EVERY checked against TMU memory: a
           decode whose bytes changed without a download dropping it is a
           cache defect. */
        if (gl->logging && x->hashed && !(++gl->check_tick % CHECK_EVERY) && (gl->st.checked < CHECKS_MAX)) {
            const uint64_t h = tex_hash(s, x->tmu, x->lo, x->hi);

            gl->st.checked++;
            if (h != x->hash) {
                uint32_t w, hh;

                x->hash = h; /* counted once */
                gl->st.stale++;
                example_add(&gl->st, 3, "stale texture", t, x, s, 0, 0.0f, 0.0f, 0.0f, 0.0f);
                vgpu98_gl_level_size((int) VGPU98_GS_TEXINFO_LARGE(x->info), VGPU98_GS_TEXINFO_ASPECT(x->info), &w, &hh);
                finding(gl, s, FIND_STALE,
                        "stale texture: one found again in TMU %u's memory at %x (%s %ux%u) no longer matches it: the memory "
                        "changed without a download dropping it",
                        x->tmu, x->address, format_names[VGPU98_GS_TEXINFO_FORMAT(x->info) & 15], w, hh);
            }
        }
    } else
        x = tex_decode(gl, s, t, &g, mem, pal);
    if (!x) {
        if (!gl->tex_fail_told) {
            gl->tex_fail_told = 1;
            gl_log(gl, "a texture could not be made on the host GPU (no memory, or %u textures): it draws white", TEX_MAX);
        }
        return gl->white.view;
    }
    x->used    = s->draw_serial;
    s->last[t] = (uint32_t) (x - s->tex);
    return x->im.view;
}

/* A TMU's word of the uniform block's: T_*, its clamps, its levels. */
static void
tex_ubo(const gl_session_t *s, uint32_t t, int has_texture, uint32_t *out)
{
    const uint32_t *r = &s->reg[VGPU98_GS_TMU(t)];
    vgpu98_gltex_t  g;
    uint32_t        mem, flags = 0, lodmin, lodmax;
    const uint32_t  filter = r[VGPU98_GS_TEX_FILTER];
    const uint32_t  mipmap = r[VGPU98_GS_TEX_MIPMAP];

    out[0] = out[1] = out[2] = out[3] = 0;
    out[2] = r[VGPU98_GS_TEX_LOD_BIAS];
    out[3] = r[VGPU98_GS_TEX_DETAIL];
    if (!has_texture || !tex_desc(s, t, &g, &mem)) {
        out[1] = (8u << 8) | (8u << 12); /* one level, the white texel */
        return;
    }
    flags |= T_TEXTURE;
    if (g.evenodd == 1)
        flags |= T_EVEN;
    else if (g.evenodd == 2)
        flags |= T_ODD;
    if ((mipmap >> 8) & 1)
        flags |= T_TRILINEAR;
    if (filter & 1)
        flags |= T_MIN_LINEAR;
    if ((filter >> 8) & 1)
        flags |= T_MAG_LINEAR;
    if ((mipmap & 0xff) == 2)
        flags |= T_LOD_DITHER;
    flags |= (r[VGPU98_GS_TEX_CLAMP] & 3u) << 8;
    flags |= ((r[VGPU98_GS_TEX_CLAMP] >> 8) & 3u) << 10;
    /* Glide 2's level numbers (0 the 256 level): the largest the texture
       has, and with mipmapping on its smallest. */
    lodmin = 8u - (uint32_t) g.large;
    lodmax = ((mipmap & 0xff) != 0) ? 8u - (uint32_t) g.small : lodmin;
    out[0] = flags;
    out[1] = lodmin | (lodmax << 4) | ((uint32_t) (8 - ((g.aspect < 0) ? -g.aspect : 0)) << 8) |
             ((uint32_t) (8 - ((g.aspect > 0) ? g.aspect : 0)) << 12);
}

/* Which TMUs' texels a DRAW reads: TMU0's when the colour or alpha combine
   (or ITRGB lighting) takes the texture, TMU1's when TMU0's combine takes
   its "other" too. A TMU's texture is decoded only then. */
static void
tex_reads(const gl_session_t *s, int *reads)
{
    const uint32_t cc  = s->reg[VGPU98_GS_COLOR_COMBINE], ac = s->reg[VGPU98_GS_ALPHA_COMBINE];
    const uint32_t tc  = s->reg[VGPU98_GS_TMU(0) + VGPU98_GS_TEX_COMBINE];
    const uint32_t ccf = VGPU98_GS_COMBINE_FACTOR(cc) & 7, acf = VGPU98_GS_COMBINE_FACTOR(ac) & 7;
    uint32_t       k;

    reads[0] = (VGPU98_GS_COMBINE_OTHER(cc) == VGPU98_GL_OTHER_TEXTURE) || (ccf == 4) || (ccf == 5) ||
               (VGPU98_GS_COMBINE_OTHER(ac) == VGPU98_GL_OTHER_TEXTURE) || (acf == 4) || s->reg[VGPU98_GS_ITRGB_LIGHTING];
    reads[1] = 0;
    if (reads[0]) {
        /* TMU0's rgb (low half) and alpha (high half) functions and factors. */
        for (k = 0; k < 2; k++) {
            const uint32_t fn  = (tc >> (k * 16)) & 0xff;
            const uint32_t fac = (tc >> (k * 16 + 8)) & 7;

            if (((fn >= 0x3) && (fn <= 0x8)) || (fac == 2))
                reads[1] = 1;
        }
    }
}

/* --- Drawing (DRAW) ----------------------------------------------------------------- */

/* Glide's blend factors as Vulkan's: as a source, and as a destination
   (codes 2 and 6 mean the other side's colour there). PREFOG_COLOR as a
   destination is the source colour before fog: the fogged one stands in. */
static VkBlendFactor
blend_factor(uint32_t f, int dst)
{
    switch (f) {
        case VGPU98_GL_BLEND_ZERO:
            return VK_BLEND_FACTOR_ZERO;
        case VGPU98_GL_BLEND_SRC_ALPHA:
            return VK_BLEND_FACTOR_SRC_ALPHA;
        case VGPU98_GL_BLEND_SRC_COLOR:
            return dst ? VK_BLEND_FACTOR_SRC_COLOR : VK_BLEND_FACTOR_DST_COLOR;
        case VGPU98_GL_BLEND_DST_ALPHA:
            return VK_BLEND_FACTOR_DST_ALPHA;
        case VGPU98_GL_BLEND_ONE:
            return VK_BLEND_FACTOR_ONE;
        case VGPU98_GL_BLEND_ONE_MINUS_SRC_ALPHA:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case VGPU98_GL_BLEND_ONE_MINUS_SRC_COLOR:
            return dst ? VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR : VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case VGPU98_GL_BLEND_ONE_MINUS_DST_ALPHA:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case VGPU98_GL_BLEND_ALPHA_SATURATE:
            return dst ? VK_BLEND_FACTOR_SRC_COLOR : VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        default:
            return dst ? VK_BLEND_FACTOR_ZERO : VK_BLEND_FACTOR_ONE;
    }
}

static const VkCompareOp compare_ops[8] = {
    VK_COMPARE_OP_NEVER,   VK_COMPARE_OP_LESS,      VK_COMPARE_OP_EQUAL,         VK_COMPARE_OP_LESS_OR_EQUAL,
    VK_COMPARE_OP_GREATER, VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_ALWAYS,
};

/* Does the session blend: anything but ONE, ZERO for its colour. */
static int
blending(const gl_session_t *s)
{
    const uint32_t b = s->reg[VGPU98_GS_BLEND];

    return ((b & 0xff) != VGPU98_GL_BLEND_ONE) || (((b >> 8) & 0xff) != VGPU98_GL_BLEND_ZERO);
}

/* Depth buffering: on with an aux buffer and a mode. */
static int
depth_on(const gl_session_t *s)
{
    return s->aux && ((s->reg[VGPU98_GS_DEPTH] & 0xff) != VGPU98_GL_DEPTH_DISABLE);
}

/* What a pipeline is made of: its render pass (an aux buffer or not), the
   colour mask, the depth test, its function and the depth mask, and the
   colour's blend factors. */
static uint32_t
pipe_key(const gl_session_t *s)
{
    const uint32_t masks = s->reg[VGPU98_GS_MASKS];
    const int      depth = depth_on(s);
    uint32_t       k     = s->aux ? 1u : 0u;

    k |= (masks & VGPU98_GS_MASK_RGB) ? 2u : 0u;
    if (depth) {
        k |= 4u;
        k |= (masks & VGPU98_GS_MASK_DEPTH) ? 8u : 0u;
        k |= ((s->reg[VGPU98_GS_DEPTH] >> 8) & 7u) << 4;
    }
    if (blending(s))
        k |= 0x80u | ((s->reg[VGPU98_GS_BLEND] & 0xfu) << 8) | (((s->reg[VGPU98_GS_BLEND] >> 8) & 0xfu) << 12);
    return k;
}

static VkPipeline
pipe_make(vgpu98_glide_t *gl, uint32_t key)
{
    vgpu98_vk_t                           *vk = gl->vk;
    VkPipelineShaderStageCreateInfo        st[2];
    VkPipelineVertexInputStateCreateInfo   vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkPipelineViewportStateCreateInfo      vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineMultisampleStateCreateInfo   ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    VkPipelineDepthStencilStateCreateInfo  ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    VkPipelineColorBlendAttachmentState    ba;
    VkPipelineColorBlendStateCreateInfo    cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    static const VkDynamicState            dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo       dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    VkGraphicsPipelineCreateInfo           ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    VkPipeline                             pipe = VK_NULL_HANDLE;
    VkResult                               res;

    memset(st, 0, sizeof(st));
    st[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = gl->vs;
    st[0].pName  = "main";
    st[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = gl->fs;
    st[1].pName  = "main";
    ia.topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;
    rs.polygonMode   = VK_POLYGON_MODE_FILL;
    rs.cullMode      = VK_CULL_MODE_NONE; /* the DLL culls, as a Voodoo3's Glide does */
    rs.frontFace     = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth     = 1.0f;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ds.depthTestEnable  = (key & 4u) ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = (key & 8u) ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp   = compare_ops[(key >> 4) & 7u];
    memset(&ba, 0, sizeof(ba));
    ba.blendEnable         = (key & 0x80u) ? VK_TRUE : VK_FALSE;
    ba.srcColorBlendFactor = blend_factor((key >> 8) & 0xfu, 0);
    ba.dstColorBlendFactor = blend_factor((key >> 12) & 0xfu, 1);
    ba.colorBlendOp        = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    ba.alphaBlendOp        = VK_BLEND_OP_ADD;
    ba.colorWriteMask      = (key & 2u) ? (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT) : 0;
    cb.attachmentCount     = 1;
    cb.pAttachments        = &ba;
    dy.dynamicStateCount   = 2;
    dy.pDynamicStates      = dyn;
    ci.stageCount          = 2;
    ci.pStages             = st;
    ci.pVertexInputState   = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState   = &ms;
    ci.pDepthStencilState  = (key & 1u) ? &ds : NULL;
    ci.pColorBlendState    = &cb;
    ci.pDynamicState       = &dy;
    ci.layout              = gl->layout;
    ci.renderPass          = gl->pass[(key & 1u) ? 1 : 0];
    res = vk->vkCreateGraphicsPipelines(vk->dev, vgpu98_r3d_glide_pcache(gl->r3d), 1, &ci, NULL, &pipe);
    if (res != VK_SUCCESS) {
        vgpu98_r3d_glide_fail(gl->r3d, "making a Glide pipeline", (int) res);
        return VK_NULL_HANDLE;
    }
    vgpu98_r3d_glide_new_pipeline(gl->r3d);
    return pipe;
}

/* The session's pipeline now: one of those made, or a new one. */
static VkPipeline
pipe_get(vgpu98_glide_t *gl, const gl_session_t *s)
{
    const uint32_t key = pipe_key(s);
    VkPipeline     pipe;
    uint32_t       i;

    for (i = 0; i < gl->n_pipes; i++) {
        if (gl->pipes[i].key == key)
            return gl->pipes[i].pipe;
    }
    if (gl->n_pipes == MAX_PIPES) {
        /* Never in practice (the key has a few hundred values in use at
           most): the oldest goes, once the host GPU is done with it. */
        if (!gl->pipes_full_told) {
            gl->pipes_full_told = 1;
            gl_log(gl, "%u pipelines made: the oldest are made again when needed", MAX_PIPES);
        }
        if (!vgpu98_r3d_glide_submit(gl->r3d, 1))
            return VK_NULL_HANDLE;
        vgpu98_vk_wait_idle(gl->vk);
        gl->vk->vkDestroyPipeline(gl->vk->dev, gl->pipes[0].pipe, NULL);
        memmove(&gl->pipes[0], &gl->pipes[1], (MAX_PIPES - 1) * sizeof(gl->pipes[0]));
        gl->n_pipes--;
        gl->bound_pipe = VK_NULL_HANDLE;
    }
    pipe = pipe_make(gl, key);
    if (!pipe)
        return VK_NULL_HANDLE;
    gl->pipes[gl->n_pipes].key  = key;
    gl->pipes[gl->n_pipes].pipe = pipe;
    gl->n_pipes++;
    return pipe;
}

/* The uniform block from the session's registers; has[t]: the TMU has
   its texture bound. */
static void
ubo_fill(const gl_session_t *s, gl_ubo_t *u, const int *has)
{
    const uint32_t *r = s->reg;
    uint32_t        t;

    memset(u, 0, sizeof(*u));
    u->comb[0] = r[VGPU98_GS_COLOR_COMBINE];
    u->comb[1] = r[VGPU98_GS_ALPHA_COMBINE];
    u->comb[2] = r[VGPU98_GS_TMU(0) + VGPU98_GS_TEX_COMBINE];
    u->comb[3] = r[VGPU98_GS_TMU(1) + VGPU98_GS_TEX_COMBINE];
    u->misc[0] = (r[VGPU98_GS_TMU(0) + VGPU98_GS_TEX_INVERT] & 3u) | ((r[VGPU98_GS_TMU(1) + VGPU98_GS_TEX_INVERT] & 3u) << 2);
    u->misc[1] = r[VGPU98_GS_CONSTANT];
    u->misc[2] = (r[VGPU98_GS_CHROMA_VALUE] & 0xffffffu) | (r[VGPU98_GS_CHROMA] ? 0x1000000u : 0);
    u->misc[3] = r[VGPU98_GS_ALPHA_TEST];
    u->pix[0]  = r[VGPU98_GS_DEPTH];
    u->pix[1]  = r[VGPU98_GS_FOG];
    u->pix[2]  = r[VGPU98_GS_FOG_COLOR];
    u->pix[3]  = (r[VGPU98_GS_DITHER] & 3u) | (r[VGPU98_GS_ITRGB_LIGHTING] ? F_ITRGB : 0) | (blending(s) ? F_BLEND : 0) |
                (depth_on(s) ? F_DEPTH : 0);
    memcpy(u->fog, &r[VGPU98_GS_FOG_TABLE], sizeof(u->fog));
    for (t = 0; t < 2; t++)
        tex_ubo(s, t, has[t], u->tex[t]);
}

/* A descriptor set from the slot's pool, the pool reset when the slot was
   begun again (its last use done). VK_NULL_HANDLE when the pool is full. */
static VkDescriptorSet
set_alloc(vgpu98_glide_t *gl, uint32_t slot, uint32_t gen)
{
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorSet             set;

    if (slot >= VGPU98_R3D_SLOTS)
        return VK_NULL_HANDLE;
    if (gl->dpool_gen[slot] != gen) {
        gl->vk->vkResetDescriptorPool(gl->vk->dev, gl->dpool[slot], 0);
        gl->dpool_gen[slot] = gen;
    }
    ai.descriptorPool     = gl->dpool[slot];
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &gl->dsl;
    if (gl->vk->vkAllocateDescriptorSets(gl->vk->dev, &ai, &set) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return set;
}

static void
pkt_draw(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t len)
{
    const VkPhysicalDeviceLimits *lim = &gl->vk->props.limits;
    const uint32_t                kind  = (len >= VGPU98_GL_LEN_DRAW) ? p[1] : 0;
    const uint32_t                count = (len >= VGPU98_GL_LEN_DRAW) ? p[2] : 0;
    const int                     ci    = color_index(s, s->reg[VGPU98_GS_RENDER_BUFFER]);
    const VkDeviceSize            vbytes = (VkDeviceSize) count * VGPU98_GL_VERTEX_DWORDS * 4;
    VkRect2D                      sc;
    VkPipeline                    pipe;
    VkCommandBuffer               c;
    VkDescriptorSet               set = VK_NULL_HANDLE;
    VkBuffer                      vbuf = VK_NULL_HANDLE, ubuf = VK_NULL_HANDLE;
    VkDeviceSize                  voff = 0, uoff = 0;
    void                         *vmap = NULL, *umap = NULL;
    VkDescriptorBufferInfo        bi[2];
    VkDescriptorImageInfo         ii[2];
    VkWriteDescriptorSet          w[4];
    VkViewport                    vp;
    gl_push_t                     pc;
    VkImageView                   views[2];
    int                           has[2];
    int                           tries, k;

    if ((len < VGPU98_GL_LEN_DRAW) || (kind != VGPU98_GL_TRIANGLES) || !count || (count % 3) ||
        (count > (len - VGPU98_GL_LEN_DRAW) / VGPU98_GL_VERTEX_DWORDS)) {
        drop(gl, VGPU98_GL_DRAW, "%u dwords, kind %u, %u vertices", len, kind, count);
        return;
    }
    if (ci < 0) {
        drop(gl, VGPU98_GL_DRAW, "render buffer %u", s->reg[VGPU98_GS_RENDER_BUFFER]);
        return;
    }
    sc = clip_rect(s);
    if (!sc.extent.width || !sc.extent.height)
        return;
    pipe = pipe_get(gl, s);
    if (!pipe)
        return;
    /* The TMUs' textures first: decoding one records its upload outside
       the render pass. */
    s->draw_serial++;
    {
        int reads[2];

        tex_reads(s, reads);
        for (k = 0; k < 2; k++) {
            views[k] = reads[k] ? tex_view(gl, s, (uint32_t) k) : gl->white.view;
            has[k]   = views[k] != gl->white.view;
        }
    }
    /* The vertices, the uniform block and the descriptor set in one slot:
       a slot filled between them begins the next, and they go again there. */
    for (tries = 0; tries < 3; tries++) {
        uint32_t gen0, gen1, slot0, slot1;

        slot0 = vgpu98_r3d_glide_slot(gl->r3d, &gen0);
        vmap  = vgpu98_r3d_glide_alloc(gl->r3d, 1, vbytes, lim->minStorageBufferOffsetAlignment ? lim->minStorageBufferOffsetAlignment : 16,
                                       &vbuf, &voff);
        umap  = vmap ? vgpu98_r3d_glide_alloc(gl->r3d, 2, sizeof(gl_ubo_t),
                                              lim->minUniformBufferOffsetAlignment ? lim->minUniformBufferOffsetAlignment : 16,
                                              &ubuf, &uoff)
                     : NULL;
        if (!vmap || !umap)
            return; /* 3D stopped */
        slot1 = vgpu98_r3d_glide_slot(gl->r3d, &gen1);
        if ((slot0 != slot1) || (gen0 != gen1))
            continue;
        set = set_alloc(gl, slot1, gen1);
        if (set)
            break;
        /* The slot's pool is full: the next slot. */
        if (!vgpu98_r3d_glide_submit(gl->r3d, 0))
            return;
    }
    if (!set) {
        drop(gl, VGPU98_GL_DRAW, "no room in a slot for %u vertices", count);
        return;
    }
    memcpy(vmap, &p[VGPU98_GL_LEN_DRAW], (size_t) vbytes);
    ubo_fill(s, (gl_ubo_t *) umap, has);
    bi[0].buffer = ubuf;
    bi[0].offset = uoff;
    bi[0].range  = sizeof(gl_ubo_t);
    bi[1].buffer = vbuf;
    bi[1].offset = voff;
    bi[1].range  = vbytes;
    for (k = 0; k < 2; k++) {
        ii[k].sampler     = gl->sampler_white; /* texels are fetched: the sampler is not used */
        ii[k].imageView   = views[k];
        ii[k].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    memset(w, 0, sizeof(w));
    for (k = 0; k < 4; k++) {
        w[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet          = set;
        w[k].dstBinding      = (uint32_t) k;
        w[k].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo    = &bi[0];
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo    = &bi[1];
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[2].pImageInfo     = &ii[0];
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[3].pImageInfo     = &ii[1];
    gl->vk->vkUpdateDescriptorSets(gl->vk->dev, 4, w, 0, NULL);

    c = inside(gl, s, (uint32_t) ci);
    if (gl->bound_pipe != pipe) {
        gl->vk->vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        gl->bound_pipe = pipe;
    }
    gl->vk->vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, gl->layout, 0, 1, &set, 0, NULL);
    vp.x        = 0.0f;
    vp.y        = 0.0f;
    vp.width    = (float) s->w;
    vp.height   = (float) s->h;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    gl->vk->vkCmdSetViewport(c, 0, 1, &vp);
    gl->vk->vkCmdSetScissor(c, 0, 1, &sc);
    pc.width      = (float) s->w;
    pc.height     = (float) s->h;
    pc.lower_left = s->reg[VGPU98_GS_ORIGIN] ? 1u : 0u;
    pc.first      = 0;
    gl->vk->vkCmdPushConstants(c, gl->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
    gl->vk->vkCmdDraw(c, count, 1, 0, 0);
    gl->draws++;
    gl->triangles += count / 3;
    draw_noted(gl, s, p, count, has);
    if (((s->reg[VGPU98_GS_MASKS] & VGPU98_GS_MASK_RGB) != 0) && (ci == (int) s->front))
        front_written(gl, s);
}

/* glide.vert and glide.frag's layout, each slot's pool, the white texture.
   0 if the host GPU refused one of them. */
static int
draw_objects(vgpu98_glide_t *gl)
{
    vgpu98_vk_t                    *vk = gl->vk;
    VkShaderModuleCreateInfo        smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkDescriptorSetLayoutBinding    b[4];
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPushConstantRange             pcr;
    VkPipelineLayoutCreateInfo      pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkDescriptorPoolSize            ps[3];
    VkDescriptorPoolCreateInfo      dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    VkSamplerCreateInfo             sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    int                             i;

    smi.codeSize = sizeof(vgpu98_glide_vert);
    smi.pCode    = vgpu98_glide_vert;
    if (vk->vkCreateShaderModule(vk->dev, &smi, NULL, &gl->vs) != VK_SUCCESS)
        return 0;
    smi.codeSize = sizeof(vgpu98_glide_frag);
    smi.pCode    = vgpu98_glide_frag;
    if (vk->vkCreateShaderModule(vk->dev, &smi, NULL, &gl->fs) != VK_SUCCESS)
        return 0;
    memset(b, 0, sizeof(b));
    for (i = 0; i < 4; i++) {
        b[i].binding         = (uint32_t) i;
        b[i].descriptorCount = 1;
    }
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[0].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[1].stageFlags     = VK_SHADER_STAGE_VERTEX_BIT;
    b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[2].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[3].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
    dli.bindingCount    = 4;
    dli.pBindings       = b;
    if (vk->vkCreateDescriptorSetLayout(vk->dev, &dli, NULL, &gl->dsl) != VK_SUCCESS)
        return 0;
    pcr.stageFlags             = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.offset                 = 0;
    pcr.size                   = sizeof(gl_push_t);
    pli.setLayoutCount         = 1;
    pli.pSetLayouts            = &gl->dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges    = &pcr;
    if (vk->vkCreatePipelineLayout(vk->dev, &pli, NULL, &gl->layout) != VK_SUCCESS)
        return 0;
    ps[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ps[0].descriptorCount = POOL_SETS;
    ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[1].descriptorCount = POOL_SETS;
    ps[2].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[2].descriptorCount = POOL_SETS * 2;
    dpi.maxSets           = POOL_SETS;
    dpi.poolSizeCount     = 3;
    dpi.pPoolSizes        = ps;
    for (i = 0; i < VGPU98_R3D_SLOTS; i++) {
        if (vk->vkCreateDescriptorPool(vk->dev, &dpi, NULL, &gl->dpool[i]) != VK_SUCCESS)
            return 0;
        gl->dpool_gen[i] = 0xffffffffu;
    }
    sci.magFilter    = VK_FILTER_NEAREST;
    sci.minFilter    = VK_FILTER_NEAREST;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod       = 0.0f;
    if (vk->vkCreateSampler(vk->dev, &sci, NULL, &gl->sampler_white) != VK_SUCCESS)
        return 0;
    if (!image_make(gl, &gl->white, VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0))
        return 0;
    {
        VkCommandBuffer         c = outside(gl);
        VkClearColorValue       cv;
        VkImageSubresourceRange rc = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        for (i = 0; i < 4; i++)
            cv.float32[i] = 1.0f;
        mem_barrier(gl, c);
        vk->vkCmdClearColorImage(c, gl->white.image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rc);
        mem_barrier(gl, c);
    }
    return 1;
}

static void
draw_objects_free(vgpu98_glide_t *gl)
{
    vgpu98_vk_t *vk = gl->vk;
    uint32_t     i;

    for (i = 0; i < gl->n_pipes; i++)
        vk->vkDestroyPipeline(vk->dev, gl->pipes[i].pipe, NULL);
    gl->n_pipes = 0;
    for (i = 0; i < VGPU98_R3D_SLOTS; i++) {
        if (gl->dpool[i])
            vk->vkDestroyDescriptorPool(vk->dev, gl->dpool[i], NULL);
    }
    if (gl->white.view)
        vk->vkDestroyImageView(vk->dev, gl->white.view, NULL);
    if (gl->white.image)
        vk->vkDestroyImage(vk->dev, gl->white.image, NULL);
    if (gl->white.mem)
        vk->vkFreeMemory(vk->dev, gl->white.mem, NULL);
    if (gl->sampler_white)
        vk->vkDestroySampler(vk->dev, gl->sampler_white, NULL);
    if (gl->layout)
        vk->vkDestroyPipelineLayout(vk->dev, gl->layout, NULL);
    if (gl->dsl)
        vk->vkDestroyDescriptorSetLayout(vk->dev, gl->dsl, NULL);
    if (gl->fs)
        vk->vkDestroyShaderModule(vk->dev, gl->fs, NULL);
    if (gl->vs)
        vk->vkDestroyShaderModule(vk->dev, gl->vs, NULL);
}

/* --- Diagnostics "Glide" -------------------------- */

static const char *const fn_names[] = {
    "ZERO", "LOCAL", "LOCAL_ALPHA", "SCALE_OTHER", "SCALE_OTHER_ADD_LOCAL", "SCALE_OTHER_ADD_LOCAL_ALPHA",
    "SCALE_OTHER_MINUS_LOCAL", "BLEND", "SCALE_OTHER_MINUS_LOCAL_ADD_LOCAL_ALPHA", "BLEND_LOCAL",
};
static const char *const factor_names[16] = {
    "ZERO", "LOCAL", "OTHER_ALPHA", "LOCAL_ALPHA", "TEXTURE_ALPHA", "TEXTURE_RGB", "6", "7",
    "ONE", "ONE_MINUS_LOCAL", "ONE_MINUS_OTHER_ALPHA", "ONE_MINUS_LOCAL_ALPHA", "ONE_MINUS_TEXTURE_ALPHA",
    "ONE_MINUS_TEXTURE_RGB", "14", "15",
};
static const char *const local_names[4] = { "ITERATED", "CONSTANT", "DEPTH", "3" };
static const char *const other_names[4] = { "ITERATED", "TEXTURE", "CONSTANT", "3" };
static const char *const format_names[16] = {
    "332", "YIQ", "A8", "I8", "AI44", "P8", "P8 6666", "7", "8332", "AYIQ", "565", "1555", "4444", "AI88", "AP88", "15",
};

static const char *
fn_name(uint32_t fn)
{
    if (fn == 0x10)
        return "SCALE_MINUS_LOCAL_ADD_LOCAL_ALPHA";
    return (fn < sizeof(fn_names) / sizeof(fn_names[0])) ? fn_names[fn] : "?";
}

/* A combine register in Glide's names: "SCALE_OTHER by ONE of ITERATED, TEXTURE". */
static void
combine_name(char *buf, size_t size, uint32_t r)
{
    snprintf(buf, size, "%s by %s of %s, %s%s", fn_name(VGPU98_GS_COMBINE_FN(r)), factor_names[VGPU98_GS_COMBINE_FACTOR(r) & 15],
             local_names[VGPU98_GS_COMBINE_LOCAL(r) & 3], other_names[VGPU98_GS_COMBINE_OTHER(r) & 3],
             VGPU98_GS_COMBINE_INVERT(r) ? ", inverted" : "");
}

/* A DRAW counted: its kinds of drawing, its combine set-up, its vertices'
   1/w. */
/* An oddity's example (two of a kind, EXAMPLES a report at most). */
static void
example_add(gl_stats_t *st, uint32_t kind, const char *what, uint32_t unit, const gl_tex_t *x, const gl_session_t *s,
            uint32_t vertices, float es, float et, float tw, float w)
{
    gl_example_t *e;

    if ((st->n_ex >= EXAMPLES) || (kind >= 4) || (st->ex_kind[kind] >= 2))
        return;
    st->ex_kind[kind]++;
    e           = &st->ex[st->n_ex++];
    e->what     = what;
    e->unit     = unit;
    e->mem      = x->tmu;
    e->address  = x->address;
    e->info     = x->info;
    e->avg      = x->avg;
    e->vertices = vertices;
    e->s        = es;
    e->t        = et;
    e->tw       = tw;
    e->w        = w;
    e->cc       = s->reg[VGPU98_GS_COLOR_COMBINE];
    e->ac       = s->reg[VGPU98_GS_ALPHA_COMBINE];
    e->tc0      = s->reg[VGPU98_GS_TMU(0) + VGPU98_GS_TEX_COMBINE];
    e->tc1      = s->reg[VGPU98_GS_TMU(1) + VGPU98_GS_TEX_COMBINE];
    e->blend    = blending(s) ? s->reg[VGPU98_GS_BLEND] : 0xffffffffu;
}

/* What the texture units a DRAW reads sample:
   for each, whether every vertex takes one texel of its largest level,
   whether its size on the screen (its triangles' texel and screen areas,
   with the LOD bias) takes its smallest of several levels, whether its 1/w
   is its own or bad; and TMU0 passing TMU1 through. Counts and examples,
   not warnings: programs do each of them on purpose too. */
/* A TMU whose own combine uses its texel, as 3dfx's grTexCombine marks it
   (the DLLs' glc_tmu_active). */
static int
tmu_active(const gl_session_t *s, uint32_t t)
{
    const uint32_t c = s->reg[VGPU98_GS_TMU(t) + VGPU98_GS_TEX_COMBINE];
    uint32_t       k;

    for (k = 0; k < 2; k++) {
        const uint32_t fn = (c >> (k * 16)) & 0xff, fac = (c >> (k * 16 + 8)) & 7;

        if ((fac == VGPU98_GL_FAC_LOCAL) || (fac == VGPU98_GL_FAC_LOCAL_ALPHA) ||
            ((fn >= VGPU98_GL_FN_LOCAL) && (fn <= VGPU98_GL_FN_SCALE_MINUS_LOCAL_ADD_LOCAL) && (fn != VGPU98_GL_FN_SCALE_OTHER)) ||
            (fn == VGPU98_GL_FN_SCALE_MINUS_LOCAL_ADD_LOCAL_ALPHA))
            return 1;
    }
    return 0;
}

static void
draw_coords(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t count, const int *has)
{
    gl_stats_t *st     = &gl->st;
    const int   active0 = tmu_active(s, 0);
    const int   u0      = has[0] && active0;          /* TMU0's texel reaches the output */
    const int   u1      = has[1] && tmu_active(s, 1); /* TMU1's */
    const int   alone   = u1 && !active0;             /* TMU1's alone: TMU0 not active */
    uint32_t    k, i, j;

    if (u0 && u1)
        st->units[2]++;
    else if (u1)
        st->units[1]++;
    else if (u0)
        st->units[0]++;
    else if (has[0] || has[1])
        st->units[3]++;
    for (k = 0; k < 2; k++) {
        vgpu98_gltex_t g;
        uint32_t       mem, w0, h0;
        gl_tex_t      *x;
        float          scale, bias, s_lo = 0.0f, s_hi = 0.0f, t_lo = 0.0f, t_hi = 0.0f, tex_area = 0.0f, scr_area = 0.0f;
        float          es = 0.0f, et = 0.0f, etw = 0.0f, ew = 0.0f;
        int            own = 0, bad = 0, any = 0, one, smallest;

        if (!has[k] || (s->last[k] >= s->n_tex) || !tex_desc(s, k, &g, &mem))
            continue;
        x = &s->tex[s->last[k]];
        x->draws++;
        vgpu98_gl_level_size(g.large, g.aspect, &w0, &h0);
        scale = (float) ((w0 > h0) ? w0 : h0) / 256.0f; /* texels a unit of Glide's 256 range */
        memcpy(&bias, &s->reg[VGPU98_GS_TMU(k) + VGPU98_GS_TEX_LOD_BIAS], 4);
        for (j = 0; j + 3 <= count; j += 3) {
            float tx[3] = { 0.0f, 0.0f, 0.0f }, ty[3] = { 0.0f, 0.0f, 0.0f }, sx[3], sy[3];
            int   ok = 1;

            for (i = 0; i < 3; i++) {
                const float *v  = (const float *) &p[VGPU98_GL_LEN_DRAW + (j + i) * VGPU98_GL_VERTEX_DWORDS];
                const float  tw = v[VGPU98_GLV_OOW0 + 3 * k], w = v[VGPU98_GLV_OOW];

                sx[i] = v[VGPU98_GLV_X];
                sy[i] = v[VGPU98_GLV_Y];
                if (!(tw > 0.0f) || (tw > 1e30f)) {
                    bad = 1;
                    ok  = 0;
                    continue;
                }
                if (fabsf(tw - w) > 1e-4f * fabsf(w))
                    own = 1;
                tx[i] = v[VGPU98_GLV_SOW0 + 3 * k] / tw * scale;
                ty[i] = v[VGPU98_GLV_TOW0 + 3 * k] / tw * scale;
                if (!any) {
                    s_lo = s_hi = es = tx[i];
                    t_lo = t_hi = et = ty[i];
                    etw             = tw;
                    ew              = w;
                    any             = 1;
                } else {
                    s_lo = fminf(s_lo, tx[i]);
                    s_hi = fmaxf(s_hi, tx[i]);
                    t_lo = fminf(t_lo, ty[i]);
                    t_hi = fmaxf(t_hi, ty[i]);
                }
            }
            if (ok) {
                tex_area += fabsf((tx[1] - tx[0]) * (ty[2] - ty[0]) - (tx[2] - tx[0]) * (ty[1] - ty[0]));
                scr_area += fabsf((sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]));
            }
        }
        one      = any && !bad && ((s_hi - s_lo) < 1.0f) && ((t_hi - t_lo) < 1.0f);
        smallest = (g.large > g.small) && (scr_area > 0.0f) && (tex_area > 0.0f) &&
                   ((0.5f * log2f(tex_area / scr_area) + bias) >= (float) (g.large - g.small));
        if (one) {
            st->one_texel[k]++;
            example_add(st, 0, "one texel", k, x, s, count, es, et, etw, ew);
        }
        if (smallest)
            st->smallest[k]++;
        if (own)
            st->own_w[k]++;
        if (bad) {
            st->tmu_bad_w[k]++;
            example_add(st, 1, "1/w bad", k, x, s, count, es, et, etw, ew);
        }
        if (alone && (k == 1) && own) {
            st->alone_own_w++;
            example_add(st, 2, "its own 1/w, TMU0 not active", k, x, s, count, es, et, etw, ew);
        }
    }
}

static void
draw_noted(vgpu98_glide_t *gl, gl_session_t *s, const uint32_t *p, uint32_t count, const int *has)
{
    const uint32_t dmode = s->reg[VGPU98_GS_DEPTH] & 0xff;
    /* 1/w read: a TMU textured, W-buffering, table fog on 1/w. */
    const int w_read = has[0] || has[1] ||
                       (depth_on(s) && ((dmode == VGPU98_GL_DEPTH_WBUFFER) || (dmode == VGPU98_GL_DEPTH_WBUFFER_TO_BIAS))) ||
                       ((s->reg[VGPU98_GS_FOG] & 0xff) == 2);
    gl_stats_t    *st = &gl->st;
    const uint32_t cc = s->reg[VGPU98_GS_COLOR_COMBINE], ac = s->reg[VGPU98_GS_ALPHA_COMBINE];
    const uint32_t tc = s->reg[VGPU98_GS_TMU(0) + VGPU98_GS_TEX_COMBINE];
    const uint32_t tc1 = s->reg[VGPU98_GS_TMU(1) + VGPU98_GS_TEX_COMBINE];
    const uint32_t bl  = blending(s) ? s->reg[VGPU98_GS_BLEND] : 0xffffffffu;
    uint32_t       i;

    st->draws++;
    st->triangles += count / 3;
    if (blending(s))
        st->blended++;
    if (depth_on(s)) {
        const uint32_t mode = s->reg[VGPU98_GS_DEPTH] & 0xff;

        if ((mode == VGPU98_GL_DEPTH_WBUFFER) || (mode == VGPU98_GL_DEPTH_WBUFFER_TO_BIAS))
            st->depth_w++;
        else
            st->depth_z++;
    }
    if (s->reg[VGPU98_GS_FOG] & 0xff)
        st->fogged++;
    if (has[0] || has[1])
        st->textured++;
    for (i = 0; i < st->n_combine; i++) {
        if ((st->combine[i].cc == cc) && (st->combine[i].ac == ac) && (st->combine[i].tc0 == tc) && (st->combine[i].tc1 == tc1))
            break;
    }
    if (i == st->n_combine) {
        if (st->n_combine < sizeof(st->combine) / sizeof(st->combine[0])) {
            st->combine[i].cc  = cc;
            st->combine[i].ac  = ac;
            st->combine[i].tc0 = tc;
            st->combine[i].tc1 = tc1;
            st->combine[i].n   = 0;
            st->n_combine++;
        } else
            i = 0xffffffffu;
    }
    if (i != 0xffffffffu)
        st->combine[i].n++;
    /* The blend set-ups. */
    for (i = 0; i < st->n_blend; i++) {
        if (st->blend[i].blend == bl)
            break;
    }
    if (i == st->n_blend) {
        if (st->n_blend < sizeof(st->blend) / sizeof(st->blend[0])) {
            st->blend[i].blend = bl;
            st->blend[i].n     = 0;
            st->n_blend++;
        } else
            i = 0xffffffffu;
    }
    if (i != 0xffffffffu)
        st->blend[i].n++;
    if (gl->logging)
        draw_coords(gl, s, p, count, has);
    if (gl->logging && w_read) {
        /* The W buffer's far end: past 2^-16 its 16-bit W is FFFFh. */
        const int wbuf = depth_on(s) && ((dmode == VGPU98_GL_DEPTH_WBUFFER) || (dmode == VGPU98_GL_DEPTH_WBUFFER_TO_BIAS));
        int       far  = 0;

        for (i = 0; i < count; i++) {
            float w;

            memcpy(&w, &p[VGPU98_GL_LEN_DRAW + i * VGPU98_GL_VERTEX_DWORDS + VGPU98_GLV_OOW], 4);
            if (!(w > 0.0f) || (w > 1e30f))
                st->bad_w++;
            else if (wbuf && (w < 1.52587890625e-05f))
                far = 1;
        }
        st->w_far += far;
    }
    /* Depth. */
    if (gl->logging && depth_on(s)) {
        const uint32_t d = s->reg[VGPU98_GS_DEPTH];

        st->depth_tested++;
        st->depth_fn[(d >> 8) & 7]++;
        if (s->reg[VGPU98_GS_MASKS] & VGPU98_GS_MASK_DEPTH)
            st->depth_writing++;
        if ((int16_t) (d >> 16))
            st->depth_bias++;
    }
}

static unsigned
pct(uint64_t part, uint64_t whole)
{
    return whole ? (unsigned) ((part * 100 + whole / 2) / whole) : 0;
}

/* A blend set-up in Glide's names: "SRC_ALPHA, ONE_MINUS_SRC_ALPHA", the
   alpha's after when they differ; "none". */
static void
blend_name(char *buf, size_t size, uint32_t b)
{
    static const char *const src[16] = { "ZERO", "SRC_ALPHA", "DST_COLOR", "DST_ALPHA", "ONE", "ONE_MINUS_SRC_ALPHA",
                                         "ONE_MINUS_DST_COLOR", "ONE_MINUS_DST_ALPHA", "8", "9", "10", "11", "12", "13", "14",
                                         "ALPHA_SATURATE" };
    static const char *const dst[16] = { "ZERO", "SRC_ALPHA", "SRC_COLOR", "DST_ALPHA", "ONE", "ONE_MINUS_SRC_ALPHA",
                                         "ONE_MINUS_SRC_COLOR", "ONE_MINUS_DST_ALPHA", "8", "9", "10", "11", "12", "13", "14",
                                         "PREFOG_COLOR" };

    if (b == 0xffffffffu)
        snprintf(buf, size, "none");
    else if ((b >> 16) == (b & 0xffff))
        snprintf(buf, size, "%s, %s", src[b & 15], dst[(b >> 8) & 15]);
    else
        snprintf(buf, size, "%s, %s (alpha %s, %s)", src[b & 15], dst[(b >> 8) & 15], src[(b >> 16) & 15], dst[(b >> 24) & 15]);
}

/* A texture in a few words: "565 32x32 at 1a2000 (average c08040)". */
static void
tex_text(char *buf, size_t size, uint32_t info, uint32_t address, uint32_t avg)
{
    uint32_t w, h;

    vgpu98_gl_level_size((int) VGPU98_GS_TEXINFO_LARGE(info), VGPU98_GS_TEXINFO_ASPECT(info), &w, &h);
    snprintf(buf, size, "%s %ux%u at %x (average %06x)", format_names[VGPU98_GS_TEXINFO_FORMAT(info) & 15], w, h, address, avg);
}

/* The Glide report's lines: the blends, what the
   texture units sampled and its examples, the busiest textures, the cache's
   checks. */
static void
report_samples(vgpu98_glide_t *gl, gl_stats_t *st)
{
    char     line[400], a[96], b[96], c[96], d[96], e[96];
    size_t   n = 0;
    uint32_t i, k, m;

    if (st->n_blend) {
        for (k = 0; k < 3; k++) {
            uint32_t best = 0xffffffffu;

            for (i = 0; i < st->n_blend; i++) {
                if (st->blend[i].n && ((best == 0xffffffffu) || (st->blend[i].n > st->blend[best].n)))
                    best = i;
            }
            if (best == 0xffffffffu)
                break;
            blend_name(a, sizeof(a), st->blend[best].blend);
            n += (size_t) snprintf(line + n, sizeof(line) - n, "%s%u%% %s", k ? "; " : "", pct(st->blend[best].n, st->draws), a);
            st->blend[best].n = 0;
        }
        vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE, "VGPU98 Glide:   blends: %s\n", line);
    }
    if (st->textured) {
        vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE,
                       "VGPU98 Glide:   texture units: %llu DRAWs TMU0's texel alone, %llu TMU1's alone (TMU0 not active), %llu both, "
                       "%llu neither active; %llu of TMU1's alone with its own 1/w\n",
                       (unsigned long long) st->units[0], (unsigned long long) st->units[1], (unsigned long long) st->units[2],
                       (unsigned long long) st->units[3], (unsigned long long) st->alone_own_w);
        vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE,
                       "VGPU98 Glide:   texture coordinates: TMU0 %llu one texel, %llu at the smallest level, %llu 1/w its own, "
                       "%llu 1/w bad; TMU1 %llu one texel, %llu at the smallest level, %llu 1/w its own, %llu 1/w bad\n",
                       (unsigned long long) st->one_texel[0], (unsigned long long) st->smallest[0],
                       (unsigned long long) st->own_w[0], (unsigned long long) st->tmu_bad_w[0],
                       (unsigned long long) st->one_texel[1], (unsigned long long) st->smallest[1],
                       (unsigned long long) st->own_w[1], (unsigned long long) st->tmu_bad_w[1]);
    }
    if (st->draws) {
        static const char *const fns[8] = { "NEVER", "LESS", "EQUAL", "LEQUAL", "GREATER", "NOTEQUAL", "GEQUAL", "ALWAYS" };
        uint32_t                 b1 = 8, b2 = 8;

        for (i = 0; i < 8; i++) {
            if (!st->depth_fn[i])
                continue;
            if ((b1 == 8) || (st->depth_fn[i] > st->depth_fn[b1])) {
                b2 = b1;
                b1 = i;
            } else if ((b2 == 8) || (st->depth_fn[i] > st->depth_fn[b2]))
                b2 = i;
        }
        a[0] = 0;
        if ((b1 < 8) && (b2 < 8))
            snprintf(a, sizeof(a), " (%s %u%%, %s %u%%)", fns[b1], pct(st->depth_fn[b1], st->depth_tested), fns[b2],
                     pct(st->depth_fn[b2], st->depth_tested));
        else if (b1 < 8)
            snprintf(a, sizeof(a), " (%s %u%%)", fns[b1], pct(st->depth_fn[b1], st->depth_tested));
        vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE,
                       "VGPU98 Glide:   depth: %u%% of the draws tested%s, %u%% writing, %llu with a depth bias, %llu with a 1/w "
                       "past the W buffer's far end\n",
                       pct(st->depth_tested, st->draws), a, pct(st->depth_writing, st->draws),
                       (unsigned long long) st->depth_bias, (unsigned long long) st->w_far);
    }
    for (i = 0; i < st->n_ex; i++) {
        const gl_example_t *x = &st->ex[i];
        char                at[160];

        tex_text(a, sizeof(a), x->info, x->address, x->avg);
        combine_name(b, sizeof(b), x->cc);
        combine_name(c, sizeof(c), x->ac);
        snprintf(d, sizeof(d), "TMU0 %s by %s; TMU1 %s by %s", fn_name(x->tc0 & 0xff), factor_names[(x->tc0 >> 8) & 15],
                 fn_name(x->tc1 & 0xff), factor_names[(x->tc1 >> 8) & 15]);
        blend_name(e, sizeof(e), x->blend);
        if (x->vertices)
            snprintf(at, sizeof(at), "; texel s %.2f t %.2f, its 1/w %.6g, the vertex's %.6g (%u vertices)", x->s, x->t, x->tw,
                     x->w, x->vertices);
        else
            at[0] = 0;
        vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE, "VGPU98 Glide:   e.g. TMU%u %s: TMU %u memory's %s%s; colour %s; alpha %s; %s; blend %s\n",
                       x->unit, x->what, x->mem, a, at, b, c, d, e);
    }
    /* Each TMU memory's three most drawn textures of the period; their counts
       start again. */
    for (m = 0; m < 2; m++) {
        const gl_tex_t *top[3] = { NULL, NULL, NULL };
        uint32_t        q;

        for (q = 0; q < VGPU98_GL_SESSIONS; q++) {
            const gl_session_t *s = &gl->s[q];

            if (!s->id)
                continue;
            for (i = 0; i < s->n_tex; i++) {
                const gl_tex_t *x = &s->tex[i];

                if ((x->tmu != m) || !x->draws)
                    continue;
                for (k = 0; k < 3; k++) {
                    if (!top[k] || (x->draws > top[k]->draws)) {
                        if (k < 2)
                            memmove(&top[k + 1], &top[k], (2 - k) * sizeof(top[0]));
                        top[k] = x;
                        break;
                    }
                }
            }
        }
        if (!top[0])
            continue;
        n = 0;
        for (k = 0; (k < 3) && top[k]; k++) {
            tex_text(a, sizeof(a), top[k]->info, top[k]->address, top[k]->avg);
            n += (size_t) snprintf(line + n, sizeof(line) - n, "%s%s, %u draws", k ? "; " : "", a, top[k]->draws);
        }
        vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE, "VGPU98 Glide:   busiest textures in TMU %u's memory: %s\n", m, line);
    }
    for (k = 0; k < VGPU98_GL_SESSIONS; k++) {
        for (i = 0; i < gl->s[k].n_tex; i++)
            gl->s[k].tex[i].draws = 0;
    }
    if (st->checked)
        vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE, "VGPU98 Glide:   textures checked against TMU memory: %llu, %llu changed\n",
                       (unsigned long long) st->checked, (unsigned long long) st->stale);
}

/* Every 5 seconds while Glide works, at Diagnostics "Glide": what was
   drawn and how, the textures, the frame buffer, the commands dropped; the
   findings of the period. */
/* An 86Box screenshot at Diagnostics "Glide" (the emulation's thread): a
   capture of the frame shown at the next swap. */
void
vgpu98_glide_screenshot(vgpu98_glide_t *gl)
{
    if (gl)
        atomic_store(&gl->shot_asked, 1);
}

void
vgpu98_glide_diag(vgpu98_glide_t *gl)
{
    const uint32_t now = plat_get_ticks();
    gl_stats_t    *st;
    uint32_t       i, k;

    if (!gl)
        return;
    st          = &gl->st;
    gl->logging = vgpu98_gpu_logging(gl->gpu, VGPU98_LOG_GLIDE);
    if (!gl->report_ms) {
        gl->report_ms = now;
        return;
    }
    if ((uint32_t) (now - gl->report_ms) < REPORT_MS)
        return;
    if (st->swaps || st->draws || st->texdata_bytes || st->lfb_reads || st->lfb_writes || st->dropped) {
        const double secs = (now - gl->report_ms) / 1000.0;
        char         formats[160];
        uint32_t     kept = 0, high[2] = { 0, 0 };
        uint64_t     kept_bytes = 0;
        size_t       at = 0;

        vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE,
                       "VGPU98 Glide: %.1f s: %llu swaps (worst frame %u ms), %llu draws of %llu triangles, %llu state changes, "
                       "%llu clears; frame buffer %llu reads, %llu writes; %llu commands dropped\n",
                       secs, (unsigned long long) st->swaps, st->frame_worst_ms, (unsigned long long) st->draws,
                       (unsigned long long) st->triangles, (unsigned long long) st->states, (unsigned long long) st->clears,
                       (unsigned long long) st->lfb_reads, (unsigned long long) st->lfb_writes, (unsigned long long) st->dropped);
        if (st->draws) {
            vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE,
                           "VGPU98 Glide:   of the draws %u%% textured, %u%% blended, %u%% depth Z, %u%% depth W, %u%% fogged\n",
                           pct(st->textured, st->draws), pct(st->blended, st->draws), pct(st->depth_z, st->draws),
                           pct(st->depth_w, st->draws), pct(st->fogged, st->draws));
            /* The busiest combine set-ups, three at most. */
            for (k = 0; k < 3; k++) {
                uint32_t best = 0xffffffffu;
                char     c1[96], c2[96], c3[96];

                for (i = 0; i < st->n_combine; i++) {
                    if (st->combine[i].n && ((best == 0xffffffffu) || (st->combine[i].n > st->combine[best].n)))
                        best = i;
                }
                if (best == 0xffffffffu)
                    break;
                combine_name(c1, sizeof(c1), st->combine[best].cc);
                combine_name(c2, sizeof(c2), st->combine[best].ac);
                snprintf(c3, sizeof(c3), "%s by %s; TMU1 %s by %s", fn_name(st->combine[best].tc0 & 0xff),
                         factor_names[(st->combine[best].tc0 >> 8) & 15], fn_name(st->combine[best].tc1 & 0xff),
                         factor_names[(st->combine[best].tc1 >> 8) & 15]);
                vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE, "VGPU98 Glide:   %u%% of the draws: colour %s; alpha %s; TMU0 %s\n",
                               pct(st->combine[best].n, st->draws), c1, c2, c3);
                st->combine[best].n = 0;
            }
        }
        formats[0] = 0;
        for (i = 0; i < 16; i++) {
            if (st->decoded_by_format[i] && (at + 24 < sizeof(formats)))
                at += (size_t) snprintf(formats + at, sizeof(formats) - at, "%s%s %u", at ? ", " : "", format_names[i],
                                        st->decoded_by_format[i]);
        }
        for (i = 0; i < VGPU98_GL_SESSIONS; i++) {
            if (gl->s[i].id) {
                kept += gl->s[i].n_tex;
                kept_bytes += gl->s[i].tex_bytes;
                for (k = 0; k < 2; k++) {
                    if (gl->s[i].tmem_high[k] > high[k])
                        high[k] = gl->s[i].tmem_high[k];
                }
            }
        }
        if (st->texdata_bytes || st->decoded || st->found)
            vgpu98_gpu_log(gl->gpu, VGPU98_LOG_GLIDE,
                           "VGPU98 Glide:   textures: %llu KB downloaded, %llu palettes; %llu decoded%s%s%s (%llu of them a dropped "
                           "one's bytes again), %llu found again, %llu dropped; %u kept (%llu KB); TMU memory written to %u KB and "
                           "%u KB\n",
                           (unsigned long long) (st->texdata_bytes >> 10), (unsigned long long) st->palettes,
                           (unsigned long long) st->decoded, formats[0] ? " (" : "", formats, formats[0] ? ")" : "",
                           (unsigned long long) st->unchanged, (unsigned long long) st->found, (unsigned long long) st->tex_dropped,
                           kept,
                           (unsigned long long) (kept_bytes >> 10), high[0] >> 10, high[1] >> 10);
        report_samples(gl, st);
        /* Findings past real problems' thresholds. */
        /* Clears alone count too: a program whose triangles never come
           still clears every frame. Three periods
           in a row of them (15 s), so a black loading screen's few seconds
           are not a finding (said once a window, it would hide a later one). */
        if ((st->swaps >= 10) && !st->draws && !st->lfb_writes && !st->clears) {
            finding(gl, gl->shown, FIND_BLANK, "blank screen: %llu swaps in %.0f s with nothing drawn",
                    (unsigned long long) st->swaps, secs);
            gl->blank_run = 0;
        } else if ((st->swaps >= 10) && !st->draws && !st->lfb_writes) {
            gl->blank_swaps += st->swaps;
            gl->blank_secs += secs;
            if (++gl->blank_run >= 3) {
                finding(gl, gl->shown, FIND_BLANK, "blank screen: %llu swaps in %.0f s with nothing drawn but clears",
                        (unsigned long long) gl->blank_swaps, gl->blank_secs);
                gl->blank_run = 0;
            }
        } else
            gl->blank_run = 0;
        if (!gl->blank_run) {
            gl->blank_swaps = 0;
            gl->blank_secs  = 0.0;
        }
        if (st->bad_w >= 100)
            finding(gl, gl->shown, FIND_BAD_W,
                    "unclipped geometry: %llu vertices in %.0f s with 1/w at or below 0, or not a number (a Voodoo draws them wrong too)",
                    (unsigned long long) st->bad_w, secs);
    } else {
        /* A period without Glide's work ends a run of blank ones: three in
           a row means three. */
        gl->blank_run   = 0;
        gl->blank_swaps = 0;
        gl->blank_secs  = 0.0;
    }
    memset(st, 0, sizeof(*st));
    gl->report_ms = now;
}

/* A command kept for the capture's frame. */
static void
cmd_note(vgpu98_glide_t *gl, uint32_t op, const uint32_t *p, uint32_t len)
{
    gl_cmd_t *c;
    uint32_t  i;

    if (!gl->cmds)
        gl->cmds = calloc(CMDS_KEPT, sizeof(gl_cmd_t));
    if (!gl->cmds)
        return;
    c      = &gl->cmds[gl->n_cmds % CMDS_KEPT];
    c->op  = op;
    c->len = len;
    for (i = 0; i < 6; i++)
        c->p[i] = (i < len) ? p[i] : 0;
    gl->n_cmds++;
}

static int
capture_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h)
{
    FILE       *fp = plat_fopen(path, "wb");
    png_structp png;
    png_infop   info;
    uint32_t    y;

    if (!fp)
        return 0;
    png  = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info = png ? png_create_info_struct(png) : NULL;
    if (!png || !info) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return 0;
    }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return 0;
    }
    png_init_io(png, fp);
    png_set_compression_level(png, 3);
    png_set_IHDR(png, info, w, h, 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    for (y = 0; y < h; y++)
        png_write_row(png, (png_const_bytep) (rgba + (size_t) y * w * 4));
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    return fclose(fp) == 0;
}

static const char *
cmd_name(uint32_t op)
{
    static const char *const names[] = { "OPEN", "CLOSE", "STATE", "DRAW", "CLEAR", "SWAP", "TEXDATA", "PALETTE",
                                         "LFB_READ", "LFB_WRITE", "GAMMA" };

    return ((op >= VGPU98_GL_FIRST) && (op - VGPU98_GL_FIRST < sizeof(names) / sizeof(names[0]))) ? names[op - VGPU98_GL_FIRST]
                                                                                                    : "?";
}

/* A capture's pictures and text, written on a thread of its own so the
   GPU thread goes on. */
typedef struct gl_capture_t {
    vgpu98_gpu_t *gpu;
    char          dir[1024];
    char          why[64];
    uint32_t      serial;
    uint8_t      *pic[3]; /* the front buffer, TMU0's and TMU1's textures (RGBA) */
    uint32_t      w[3], h[3];
    char         *text;
} gl_capture_t;

static atomic_int capture_busy;

static void
capture_thread(void *priv)
{
    gl_capture_t      *c = (gl_capture_t *) priv;
    static const char *names[3] = { "front.png", "tmu0.png", "tmu1.png" };
    char               path[1024];
    FILE              *f;
    uint32_t           i, files = 0;

    for (i = 0; i < 3; i++) {
        if (c->pic[i]) {
            path_append_filename(path, c->dir, names[i]);
            files += capture_png(path, c->pic[i], c->w[i], c->h[i]);
        }
        free(c->pic[i]);
    }
    if (c->text) {
        path_append_filename(path, c->dir, "commands.txt");
        f = plat_fopen(path, "w");
        if (f) {
            fputs(c->text, f);
            files += fclose(f) == 0;
        }
        free(c->text);
    }
    vgpu98_gpu_log(c->gpu, VGPU98_LOG_EVENTS, "VGPU98 Glide: capture %u (%s): %u files in %s\n", c->serial, c->why, files, c->dir);
    free(c);
    atomic_store(&capture_busy, 0);
}

/* A growing text, for the capture's commands. */
static void
text_add(char **t, size_t *n, size_t *cap, const char *fmt, ...)
{
    char    line[256];
    va_list ap;
    int     k;

    va_start(ap, fmt);
    k = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if ((k <= 0) || !*t)
        return;
    if ((size_t) k >= sizeof(line))
        k = sizeof(line) - 1;
    if (*n + (size_t) k + 1 > *cap) {
        size_t nc = *cap * 2 + (size_t) k + 1;
        char  *m  = realloc(*t, nc);

        if (!m)
            return;
        *t   = m;
        *cap = nc;
    }
    memcpy(*t + *n, line, (size_t) k + 1);
    *n += (size_t) k;
}

/* A capture of the session's last frame (at a finding, and once 10 seconds
   into the session; 8 a session at most): its front buffer and each TMU's
   texture as pictures, its registers and the frame's commands as text, in
   a folder of the captures' folder. The pixels are read here; the files go
   on a thread of their own. 0 while the last capture's files are still
   being written (the next swap tries again). */
static int
capture(vgpu98_glide_t *gl, gl_session_t *s, const char *why, int counted)
{
    const char   *base = vgpu98_gpu_capture_dir(gl->gpu);
    char          dir[1024], name[96], path[1024];
    time_t        now = time(NULL);
    struct tm    *tm  = localtime(&now);
    gl_capture_t *c;
    uint16_t     *px;
    uint32_t      i, t;
    size_t        n = 0, cap = 4096;

    if (atomic_load(&capture_busy))
        return 0;
    if (!base || !base[0])
        return 1;
    if (tm)
        snprintf(name, sizeof(name), "%04d-%02d-%02d %02d-%02d-%02d glide capture %u", tm->tm_year + 1900, tm->tm_mon + 1,
                 tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec, gl->capture_serial + 1);
    else
        snprintf(name, sizeof(name), "glide capture %u", gl->capture_serial + 1);
    if (strlen(base) + 1 + strlen(name) + sizeof("commands.txt") >= 260) {
        if (!gl->capture_path_told) {
            gl->capture_path_told = 1;
            gl_log(gl, "no capture: the captures' folder's path is too long (%s)", base);
        }
        return 1;
    }
    snprintf(dir, sizeof(dir), "%s", base);
    if (!plat_dir_check(dir) && plat_dir_create(dir))
        return 1;
    path_append_filename(path, dir, name);
    snprintf(dir, sizeof(dir), "%s", path);
    if (!plat_dir_check(dir) && plat_dir_create(dir))
        return 1;
    c = calloc(1, sizeof(*c));
    if (!c)
        return 1;
    if (counted)
        s->captures++;
    gl->capture_serial++;
    c->gpu    = gl->gpu;
    c->serial = gl->capture_serial;
    snprintf(c->dir, sizeof(c->dir), "%s", dir);
    snprintf(c->why, sizeof(c->why), "%s", why);
    /* The front buffer. */
    px        = malloc((size_t) s->w * s->h * 2);
    c->pic[0] = malloc((size_t) s->w * s->h * 4);
    if (px && c->pic[0] && read_image(gl, s->color[s->front].image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, s->w, s->h, px)) {
        for (i = 0; i < s->w * s->h; i++) {
            const uint32_t v = rgb565_to_888(px[i]);

            c->pic[0][i * 4]     = (uint8_t) (v >> 16);
            c->pic[0][i * 4 + 1] = (uint8_t) (v >> 8);
            c->pic[0][i * 4 + 2] = (uint8_t) v;
            c->pic[0][i * 4 + 3] = 0xff;
        }
        c->w[0] = s->w;
        c->h[0] = s->h;
    } else {
        free(c->pic[0]);
        c->pic[0] = NULL;
    }
    free(px);
    /* Each TMU's texture: its largest level, decoded from its memory. */
    for (t = 0; t < 2; t++) {
        vgpu98_gltex_t g;
        uint32_t       mem, w, h, at;
        uint8_t       *raw;

        if (!s->reg[VGPU98_GS_TMU(t) + VGPU98_GS_TEX_SOURCE] || !tex_desc(s, t, &g, &mem) || !vgpu98_gl_of_parity(&g, g.large))
            continue;
        vgpu98_gl_level_size(g.large, g.aspect, &w, &h);
        at            = s->reg[VGPU98_GS_TMU(t) + VGPU98_GS_TEX_ADDRESS] + vgpu98_gl_level_offset(&g, g.large);
        raw           = malloc(256 * 256 * 2);
        c->pic[1 + t] = malloc((size_t) w * h * 4);
        if (raw && c->pic[1 + t]) {
            tex_read(s, mem, at, vgpu98_gl_level_bytes(&g, g.large), raw);
            if (vgpu98_gl_decode(g.format, raw, w, h, s->palette[mem], (uint32_t *) c->pic[1 + t])) {
                c->w[1 + t] = w;
                c->h[1 + t] = h;
                free(raw);
                continue;
            }
        }
        free(raw);
        free(c->pic[1 + t]);
        c->pic[1 + t] = NULL;
    }
    /* The registers and the frame's commands. */
    c->text = malloc(cap);
    if (c->text) {
        const uint32_t last  = gl->n_cmds;
        uint32_t       first = gl->frame_from;

        c->text[0] = 0;
        text_add(&c->text, &n, &cap, "VGPU98 Glide capture %u (%s): session %x, %ux%u, swap %llu\n\nRegisters:\n", c->serial, why,
                 s->id, s->w, s->h, (unsigned long long) s->swaps);
        for (i = 0; i < VGPU98_GS_COUNT; i++)
            text_add(&c->text, &n, &cap, "%3u: %08X%s", i, s->reg[i], ((i & 7) == 7) ? "\n" : "  ");
        if (last - first > CMDS_KEPT)
            first = last - CMDS_KEPT;
        text_add(&c->text, &n, &cap, "\n\nThe frame's commands (%u):\n", last - first);
        for (i = first; gl->cmds && (i < last); i++) {
            const gl_cmd_t *k = &gl->cmds[i % CMDS_KEPT];

            text_add(&c->text, &n, &cap, "%-9s %5u: %08X %08X %08X %08X %08X %08X\n", cmd_name(k->op), k->len, k->p[0], k->p[1],
                     k->p[2], k->p[3], k->p[4], k->p[5]);
        }
    }
    atomic_store(&capture_busy, 1);
    {
        /* A thread that could not be made is NULL, or -1 from _beginthread:
           the files go on this thread instead. */
        const thread_t *th = thread_create_named(capture_thread, c, "VGPU98 Glide capture");

        if (!th || (th == (thread_t *) (intptr_t) -1))
            capture_thread(c);
    }
    return 1;
}

/* --- Commands ---------------------------------------------------------------------- */

void
vgpu98_glide_packet(vgpu98_glide_t *gl, uint32_t op, const uint32_t *p, uint32_t len, uint32_t generation)
{
    gl_session_t *s;

    if (gl->logging)
        cmd_note(gl, op, p, len);
    if (vgpu98_r3d_broken(gl->r3d)) {
        drop(gl, op, "3D stopped on the host GPU");
        return;
    }
    if (op == VGPU98_GL_OPEN) {
        pkt_open(gl, p, len);
        return;
    }
    if (op == VGPU98_GL_CLOSE) {
        pkt_close(gl, p);
        return;
    }
    s = find(gl, p[0]);
    if (!s) {
        drop(gl, op, "no session %x", p[0]);
        return;
    }
    switch (op) {
        case VGPU98_GL_STATE:
            if (len < VGPU98_GL_LEN_STATE)
                drop(gl, op, "%u dwords", len);
            else
                pkt_state(gl, s, p, len);
            break;
        case VGPU98_GL_CLEAR:
            pkt_clear(gl, s, p, len);
            break;
        case VGPU98_GL_SWAP:
            pkt_swap(gl, s, p, len, generation);
            break;
        case VGPU98_GL_LFB_READ:
            pkt_lfb_read(gl, s, p, len);
            break;
        case VGPU98_GL_LFB_WRITE:
            pkt_lfb_write(gl, s, p, len);
            break;
        case VGPU98_GL_GAMMA:
            pkt_gamma(gl, s, p, len);
            break;
        case VGPU98_GL_DRAW:
            pkt_draw(gl, s, p, len);
            break;
        case VGPU98_GL_TEXDATA:
            pkt_texdata(gl, s, p, len);
            break;
        case VGPU98_GL_PALETTE:
            pkt_palette(gl, s, p, len);
            break;
        default:
            drop(gl, op, "not a Glide command");
            break;
    }
}

void
vgpu98_glide_reset(vgpu98_glide_t *gl, int end)
{
    int i;

    if (!gl)
        return;
    /* The renderer's reset submitted or dropped its recording: no render
       pass of ours is open. */
    gl->pass_open = 0;
    gl->pass_s    = NULL;
    if (!end)
        return;
    if (end == 1) {
        /* A DOS session full screen: its screen, the sessions kept for
           Windows' return in the same mode. */
        if (gl->shown) {
            gl_log(gl, "session %x hidden for a DOS session; shown again at its next swap", gl->shown->id);
            shown_set(gl, NULL);
            vgpu98_disp_glide_end(gl->disp);
        }
        if (gl->front_dirty) {
            gl->front_dirty = NULL;
            vgpu98_gpu_glide_front(gl->gpu, 0);
        }
        return;
    }
    for (i = 0; i < VGPU98_GL_SESSIONS; i++) {
        if (gl->s[i].id)
            gl_log(gl, "session %x ends: %s", gl->s[i].id, "the GPU was reset for a mode change or a new machine");
        session_end(gl, &gl->s[i]);
    }
}

/* --- Start-up ---------------------------------------------------------------- */

static int
pass_make(vgpu98_glide_t *gl, int depth, VkRenderPass *pass)
{
    VkAttachmentDescription ad[2];
    VkAttachmentReference   cr = { 0, VK_IMAGE_LAYOUT_GENERAL };
    VkAttachmentReference   dr = { 1, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription    sd;
    VkRenderPassCreateInfo  ci = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    int                     i;

    memset(ad, 0, sizeof(ad));
    for (i = 0; i < 2; i++) {
        ad[i].samples        = VK_SAMPLE_COUNT_1_BIT;
        ad[i].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        ad[i].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        ad[i].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ad[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ad[i].initialLayout  = VK_IMAGE_LAYOUT_GENERAL;
        ad[i].finalLayout    = VK_IMAGE_LAYOUT_GENERAL;
    }
    ad[0].format = VK_FORMAT_R5G6B5_UNORM_PACK16;
    ad[1].format = VK_FORMAT_D16_UNORM;
    memset(&sd, 0, sizeof(sd));
    sd.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount    = 1;
    sd.pColorAttachments       = &cr;
    sd.pDepthStencilAttachment = depth ? &dr : NULL;
    ci.attachmentCount         = depth ? 2 : 1;
    ci.pAttachments            = ad;
    ci.subpassCount            = 1;
    ci.pSubpasses              = &sd;
    return gl->vk->vkCreateRenderPass(gl->vk->dev, &ci, NULL, pass) == VK_SUCCESS;
}

vgpu98_glide_t *
vgpu98_glide_init(vgpu98_gpu_t *g, vgpu98_r3d_t *r, vgpu98_disp_t *d, const vgpu98_vram_t *vram)
{
    vgpu98_glide_t *gl;
    vgpu98_vk_t    *vk = vgpu98_r3d_vk(r);

    if (!r || !vk)
        return NULL;
    /* Glide's buffers: 16-bit colour that blends, a 16-bit depth buffer. */
    if (!vgpu98_vk_format_has(vk, VK_FORMAT_R5G6B5_UNORM_PACK16,
                              VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                                  VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT) ||
        !vgpu98_vk_format_has(vk, VK_FORMAT_D16_UNORM,
                              VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                  VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) {
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 Glide: the host GPU lacks 16-bit colour or depth images; no Glide\n");
        return NULL;
    }
    gl       = calloc(1, sizeof(vgpu98_glide_t));
    gl->gpu  = g;
    gl->r3d  = r;
    gl->vk   = vk;
    gl->disp = d;
    gl->vram = vram;
    vgpu98_r3d_glide_hook(r, pass_end, gl);
    if (!pass_make(gl, 0, &gl->pass[0]) || !pass_make(gl, 1, &gl->pass[1]) || !draw_objects(gl)) {
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 Glide: setting up the host GPU failed; no Glide\n");
        vgpu98_r3d_glide_submit(r, 1);
        vgpu98_r3d_glide_hook(r, NULL, NULL);
        vgpu98_vk_wait_idle(vk);
        draw_objects_free(gl);
        pool_close(gl);
        if (gl->pass[0])
            vk->vkDestroyRenderPass(vk->dev, gl->pass[0], NULL);
        if (gl->pass[1])
            vk->vkDestroyRenderPass(vk->dev, gl->pass[1], NULL);
        free(gl);
        return NULL;
    }
    return gl;
}

void
vgpu98_glide_close(vgpu98_glide_t *gl)
{
    int i;

    if (!gl)
        return;
    /* A capture's files still being written: a while for them. */
    for (i = 0; atomic_load(&capture_busy) && (i < 100); i++)
        plat_delay_ms(20);
    for (i = 0; i < VGPU98_GL_SESSIONS; i++)
        session_end(gl, &gl->s[i]);
    vgpu98_r3d_glide_submit(gl->r3d, 1);
    vgpu98_r3d_glide_hook(gl->r3d, NULL, NULL);
    vgpu98_vk_wait_idle(gl->vk);
    draw_objects_free(gl);
    pool_close(gl);
    free(gl->cmds);
    for (i = 0; i < 2; i++) {
        if (gl->pass[i])
            gl->vk->vkDestroyRenderPass(gl->vk->dev, gl->pass[i], NULL);
    }
    free(gl);
}

#else /* VGPU98_NO_VULKAN */

vgpu98_glide_t *
vgpu98_glide_init(vgpu98_gpu_t *g, vgpu98_r3d_t *r, vgpu98_disp_t *d, const vgpu98_vram_t *vram)
{
    (void) g;
    (void) r;
    (void) d;
    (void) vram;
    return NULL;
}

void
vgpu98_glide_close(vgpu98_glide_t *gl)
{
    (void) gl;
}

void
vgpu98_glide_packet(vgpu98_glide_t *gl, uint32_t op, const uint32_t *p, uint32_t len, uint32_t generation)
{
    (void) gl;
    (void) op;
    (void) p;
    (void) len;
    (void) generation;
}

void
vgpu98_glide_reset(vgpu98_glide_t *gl, int end)
{
    (void) gl;
    (void) end;
}

void
vgpu98_glide_tick(vgpu98_glide_t *gl)
{
    (void) gl;
}

void
vgpu98_glide_diag(vgpu98_glide_t *gl)
{
    (void) gl;
}

void
vgpu98_glide_screenshot(vgpu98_glide_t *gl)
{
    (void) gl;
}

#endif
