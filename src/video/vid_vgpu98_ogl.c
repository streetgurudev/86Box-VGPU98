/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the OpenGL device, command set 2.6 (PROFILE.OGL).
 *          The OpenGL ICD's share groups, contexts and drawables, their
 *          images on the host GPU, GL 1.1's transform, lighting and
 *          per-fragment operations, textures and pixel rectangles (the
 *          project's spec/opengl.md and spec/gpu-commands.md), and the
 *          extensions of command set 2.6: swap intervals, two texture
 *          units, point sizes by distance, paletted textures.
 *
 *          A device of its own: nothing here touches the 3D renderer's
 *          state or images, nor the Glide device's. It records into the
 *          renderer's command buffers (vid_vgpu98_vk.h), ending its own
 *          render pass whenever the renderer or the Glide device records
 *          or the renderer submits. Unlike Glide it draws into windows on
 *          the desktop: a drawable's front image is written into video
 *          memory through the window's visible rectangles, under the
 *          coherency rule (newer DirectX pixels copied back first, the
 *          pages marked after). A command it cannot take is dropped and
 *          counted, the first ones named in the log; it never stops
 *          the ring. Diagnostics "OpenGL" adds a report every 5 seconds
 *          and findings.
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <86box/thread.h>
#include <86box/vid_vgpu98_card.h>
#include <86box/vid_vgpu98_gpu.h>
#include <86box/vid_vgpu98_oglcmd.h>
#include <86box/vid_vgpu98_dev.h>

#ifndef VGPU98_NO_VULKAN

#    include <png.h>
#    include <86box/86box.h>
#    include <86box/path.h>
#    include <86box/plat.h>
#    include <86box/vid_vgpu98_vk.h>
#    include "vgpu98_ogl_vert.h"
#    include "vgpu98_ogl_frag.h"
#    include "vgpu98_ogl_frag_depth.h"

/* A drawable's images: the front image mirrors the window's client area,
   the back image (double-buffered formats) is drawn into and copied into
   the front image at a swap; the depth image holds depth and stencil. */
typedef struct ogl_image_t {
    VkImage        image;
    VkDeviceMemory mem;
    VkImageView    view;
    VkFormat       format;
    uint32_t       w, h;
} ogl_image_t;

#    define FRONT 0
#    define BACK  1

typedef struct ogl_drawable_t {
    uint32_t      id; /* 0: free */
    uint32_t      w, h;
    uint32_t      format; /* VGPU98_OGL_COLOR_* | depth << VGPU98_OGL_DEPTH_SHIFT | VGPU98_OGL_DOUBLE */
    ogl_image_t   color[2];
    ogl_image_t   depth;
    VkFramebuffer fb[2];
    uint64_t      used; /* the packet that last named it */
    uint32_t      shown_vblank; /* the vertical blank of its last SHOW (a swap interval counts from it) */
    uint32_t      shown_ms;     /* the time of its last SHOW (Diagnostics: the worst frame) */
    /* Diagnostics "OpenGL"'s capture:
       the context that last drew into it, its first SHOW's time, its
       captures (findings and the 10-second one; 86Box screenshots apart) */
    uint32_t      last_ctx;
    uint32_t      first_show_ms;
    uint32_t      captures, shot_captures;
    int           captured_late;
} ogl_drawable_t;

typedef struct ogl_group_t {
    uint32_t id;   /* 0: free */
    uint64_t used; /* the packet that last named it */
} ogl_group_t;

/* The uniform block ogl.vert and ogl.frag read (std140 uvec4s): the
   context's registers, then each unit's texture drawn with (TEX_BLOCK
   uvec4s each, ogl.frag's layout) */
#    define REGS_VEC4S ((VGPU98_OGL_R_COUNT + 3) / 4)
#    define TEX_BLOCK  16
#    define TEX_UNITS  2 /* command set 2.6's texture units */
#    define UBO_BYTES  ((REGS_VEC4S + TEX_UNITS * TEX_BLOCK) * 16)
_Static_assert(REGS_VEC4S == 124, "ogl.vert's and ogl.frag's R_VEC4S: both change with the register count");
_Static_assert(TEX_BLOCK * 4 >= 8 + 4 * VGPU98_OGL_LEVELS, "a texture block holds every level");

typedef struct ogl_context_t {
    uint32_t id; /* 0: free */
    uint32_t group;
    uint32_t drawable; /* bound (BIND), 0 for none */
    uint64_t used;     /* the packet that last named it */
    uint32_t reg[VGPU98_OGL_R_COUNT];
    /* The uniform block's copy in the renderer's uniform buffer, kept while
       the registers and the texture's place do not change and the slot is
       the same */
    int          regs_new;
    uint32_t     tex[TEX_UNITS][TEX_BLOCK * 4];
    uint32_t     ubo_slot, ubo_gen;
    VkBuffer     ubo_buf;
    VkDeviceSize ubo_off;
    /* EXT_shared_texture_palette's palette (PALETTE, texture 0; 2.6): its
       entries, and its copy in the texel arena (made when an index texture
       draws with it, kept until the context ends) */
    uint32_t     spal[VGPU98_OGL_PALETTE_MAX];
    uint32_t     spal_n, spal_base;
    int          spal_resident, spal_dirty;
    VkDeviceSize spal_at;
} ogl_context_t;

/* A share group's texture (TEXIMAGE, TEXPARAM): its levels' texels in the
   host's memory, the truth; its copy in the texel arena,
   made when a draw uses it, dropped least recently used. */
typedef struct ogl_texlevel_t {
    uint32_t  w, h;   /* the border included; 0: not defined */
    uint32_t *texels; /* RGBA8, red in the low byte, the first row (t 0) first; an index level's bytes */
    uint32_t  bpt;    /* its bytes a texel: 4, or 1 for an index level (2.6) */
} ogl_texlevel_t;

typedef struct ogl_tex_t {
    uint32_t          group, id;
    uint32_t          dims, base, border; /* level 0's */
    uint32_t          wraps, filters;     /* TEXPARAM's */
    uint32_t          border_color[4];    /* floats */
    ogl_texlevel_t    level[VGPU98_OGL_LEVELS];
    int               dirty;     /* changed since its arena copy: copied again whole */
    int               resident;  /* it has one */
    /* Each level's rectangle changed since then by texels for a part of it
       (x1 0: none), patched alone */
    uint32_t          cx0[VGPU98_OGL_LEVELS], cy0[VGPU98_OGL_LEVELS], cx1[VGPU98_OGL_LEVELS], cy1[VGPU98_OGL_LEVELS];
    VkDeviceSize      at, bytes; /* its range there */
    uint32_t          first[VGPU98_OGL_LEVELS]; /* each level's first texel there */
    uint64_t          used;      /* the draw that last used it */
    /* Its palette (PALETTE; 2.6), for an index texture: its entries, and its
       place in the arena after the levels (a dword's index) */
    uint32_t          pal[VGPU98_OGL_PALETTE_MAX];
    uint32_t          pal_n, pal_base;
    int               pal_dirty; /* changed since its arena copy */
    uint32_t          pal_first;
    struct ogl_tex_t *next;      /* in its hash bucket */
} ogl_tex_t;

#    define TEX_BUCKETS 1024

/* ogl.vert's and ogl.frag's push constants */
typedef struct ogl_push_t {
    uint32_t control; /* DRAW's control, KIND_CLEAR, or KIND_PIXELS | the image's kind << 4 */
    uint32_t height;  /* the drawable's */
    uint32_t width;
    uint32_t bit;     /* PIXELS of stencil: the bit drawn | the value it takes << 8 */
    float    pos[4];  /* PIXELS: window x, y, z, the fog distance; CLEAR: z in [2] */
    float    zoom[4]; /* PIXELS: zoom x and y, the image's width and height */
    float    color[4];
    float    tex[4];
    float    tex1[4]; /* PIXELS: unit 1's texture coordinates (2.6) */
} ogl_push_t;

#    define KIND_PIXELS 14 /* the shaders' image fragments (PIXELS) */
#    define KIND_CLEAR  15 /* the shaders' clear quad */
#    define MAX_PIPES   256
#    define POOL_SETS   256 /* descriptor sets a slot's pool holds: one while the buffers stay the same */

/* The vertices, indices and images of draws are placed in a chunk of the
   slot's storage buffer, DATA_RANGE spare at its end: the descriptors'
   fixed range then never runs past the buffer. */
#    define DATA_RANGE VGPU98_OGL_PACKET_BYTES
#    define DATA_CHUNK (1u << 20)

/* The texel arena: device memory for the textures drawn with */
#    define ARENA_MAX (128u << 20)
#    define ARENA_MIN (16u << 20)

/* Diagnostics */
#    define DROP_LOGGED  16   /* commands dropped named in the log; the rest counted */
#    define REPORT_MS    5000 /* the OpenGL report's period (Diagnostics "OpenGL") */
#    define PIPE_SLOW_MS 500  /* a pipeline this slow to make named */
#    define WARM_JOIN_MS 10000 /* at close, the warm-up is waited for this long */

/* The OpenGL report's counters, since the last report */
typedef struct ogl_stats_t {
    uint64_t shows, vblank_waits, vblank_ms;
    uint32_t frame_worst_ms; /* between two SHOWs of a drawable */
    uint64_t draws, vertices, draws_tex1, draws_clip, pixels, clears, states, state_words, reads, texreads;
    uint64_t tex_defined, tex_defined_index, tex_rects, tex_bytes_in, palettes;
    /* The texel arena: copies whole and patched, palettes copied, textures
       dropped for room, copied in again unchanged (after such a drop) */
    uint64_t copies, copy_bytes, patches, patch_bytes, pal_copies, evicted, again;
    uint64_t pipes, pass_breaks; /* pipelines made; render passes ended early, for work outside them (copies, read-backs) */
    uint32_t pipe_worst_ms;      /* the slowest pipeline made */
    uint64_t pal_default;        /* draws of index textures with a one-entry palette */
    /* Each unit's texture drawn with, by its wraps as set */
    uint64_t wrap_repeat, wrap_clamp, wrap_edge;
    uint64_t dropped;
    uint32_t dropped_op[16]; /* by opcode, from VGPU98_OGL_FIRST */
} ogl_stats_t;

/* The findings, each said once */
#    define FIND_BLANK   0x01
#    define FIND_THRASH  0x02
#    define FIND_PALETTE 0x04

/* The capture */
#    define CMDS_KEPT     4096  /* packets kept for a capture's frame */
#    define CAPTURES      8     /* a drawable's at most: findings and the 10-second one */
#    define SHOT_CAPTURES 10    /* and at 86Box screenshots, apart */
#    define CAPTURE_MS    10000 /* a drawable's capture this long after its first SHOW */

/* A packet kept for a capture: its opcode, length and first dwords */
typedef struct ogl_cmd_t {
    uint32_t op, len;
    uint32_t p[6];
} ogl_cmd_t;

/* The depth formats the drawables use: D16, and D24S8 where the host has
   it, else D32F with S8, its depth then on the 24-bit grid. */
#    define DEPTH_KINDS 3 /* none, 16, 24+8 */
#    define COLOR_KINDS 2 /* 565, 8888 */

struct vgpu98_ogl_t {
    vgpu98_gpu_t        *gpu;
    vgpu98_r3d_t        *r3d;
    vgpu98_vk_t         *vk;
    vgpu98_disp_t       *disp;
    const vgpu98_vram_t *vram;

    ogl_group_t    groups[VGPU98_OGL_GROUPS];
    ogl_context_t *ctx[VGPU98_OGL_CONTEXTS];
    ogl_drawable_t draw[VGPU98_OGL_DRAWABLES];

    VkFormat     depth24;        /* D24_UNORM_S8_UINT, or D32_SFLOAT_S8_UINT */
    int          depth24_float;  /* the latter: depth computed on the 24-bit grid */
    VkRenderPass pass[COLOR_KINDS][DEPTH_KINDS];

    /* The render pass open in the renderer's command buffer, if any */
    int             pass_open;
    ogl_drawable_t *pass_d;
    int             pass_buf;

    /* Drawing: ogl.vert and ogl.frag (and the latter on the 24-bit grid),
       their layout, a descriptor pool a slot, the pipelines made */
    VkShaderModule        vs, fs, fs_depth;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      layout;
    VkDescriptorPool      dpool[VGPU98_R3D_SLOTS];
    uint32_t              dpool_gen[VGPU98_R3D_SLOTS];
    struct {
        uint64_t   key;
        VkPipeline pipe;
    } pipes[MAX_PIPES];
    uint32_t   n_pipes;
    VkPipeline bound_pipe;
    int        logic_op; /* the host has Vulkan's logicOp */
    int        clamp_exact; /* "OpenGL GL_CLAMP": Exact (else GL_CLAMP sampled as clamp-to-edge) */
    /* The warm-up thread: running,
       asked to stop; a slow pipeline said */
    atomic_int warm_live, warm_stop;
    int        pipe_slow_told;
    /* The descriptor set last written: its slot and the buffers it names
       (the offsets are dynamic) */
    VkDescriptorSet set;
    uint32_t        set_slot, set_gen;
    VkBuffer        set_vbuf, set_ubuf;
    /* The data chunk being filled */
    VkBuffer     chunk_buf;
    VkDeviceSize chunk_off, chunk_used;
    uint8_t     *chunk_map;
    uint32_t     chunk_slot, chunk_gen;

    /* Textures: every group's, by group and ID; the texel arena and its
       free ranges (sorted, apart) */
    ogl_tex_t         *tex[TEX_BUCKETS];
    uint32_t           n_tex;
    uint64_t           tex_bytes; /* the host memory their texels take */
    vgpu98_vk_buffer_t arena;
    struct {
        VkDeviceSize at, n;
    }       *free;
    uint32_t n_free, cap_free;
    uint64_t use;   /* draws that used a texture: its LRU clock */
    uint64_t clock; /* packets taken: the objects' LRU clock */

    uint64_t dropped;
    /* Diagnostics: the drops named so far, the
       packet being taken (a drop's line names it), the report's counters
       and period, the findings said; report periods in a row with SHOWs
       and clears but nothing drawn (a blank screen after three, 15 s) */
    uint32_t    drop_logged;
    uint32_t    op;
    ogl_stats_t st;
    uint32_t    report_ms;
    uint32_t    told;
    uint32_t    blank_run;
    uint64_t    blank_shows;
    double      blank_secs;
    /* The capture: Diagnostics "OpenGL" on (the packets
       noted), the packets kept and the frame's first, a finding's capture
       due at the next SHOW, an 86Box screenshot asked for (the emulation's
       thread), the captures so far */
    int         logging;
    ogl_cmd_t  *cmds;
    uint32_t    n_cmds, frame_from;
    int         capture_due;
    atomic_int  shot_asked;
    uint32_t    capture_serial;
    int         capture_path_told;
};

static const uint32_t one_bits = 0x3f800000u; /* 1.0f */

static float
f32(uint32_t v)
{
    float f;

    memcpy(&f, &v, 4);
    return f;
}

static float
clamp01(float v)
{
    return (v > 0.0f) ? ((v < 1.0f) ? v : 1.0f) : 0.0f; /* not a number: 0 */
}

static const char *
op_name(uint32_t op)
{
    static const char *const names[16] = { "GROUP", "CONTEXT", "DRAWABLE", "BIND",   "STATE", "DRAW",    "CLEAR",   "SWAP",
                                           "TEXIMAGE", "TEXPARAM", "TEXDELETE", "PIXELS", "READ", "TEXREAD", "PALETTE", "5Fh" };

    return ((op >= VGPU98_OGL_FIRST) && (op <= VGPU98_OGL_LAST)) ? names[op - VGPU98_OGL_FIRST] : "?";
}

/* A command the device cannot take: counted, the first ones named, at
   every level but Off. */
static void
drop(vgpu98_ogl_t *o, const char *why, ...)
{
    o->dropped++;
    o->st.dropped++;
    if ((o->op >= VGPU98_OGL_FIRST) && (o->op <= VGPU98_OGL_LAST))
        o->st.dropped_op[o->op - VGPU98_OGL_FIRST]++;
    if (o->drop_logged < DROP_LOGGED) {
        char    text[200];
        va_list ap;

        va_start(ap, why);
        vsnprintf(text, sizeof(text), why, ap);
        va_end(ap);
        o->drop_logged++;
        vgpu98_gpu_log(o->gpu, VGPU98_LOG_EVENTS, "VGPU98 OpenGL: command %02Xh (%s) dropped: %s%s\n", o->op, op_name(o->op), text,
                       (o->drop_logged == DROP_LOGGED) ? " (more are counted, not named)" : "");
    }
}

/* A finding past a real problem's threshold: said once, worded "possible"
   (Diagnostics "OpenGL"). */
static void
finding(vgpu98_ogl_t *o, uint32_t kind, const char *fmt, ...)
{
    char    text[240];
    va_list ap;

    if (o->told & kind)
        return;
    o->told |= kind;
    va_start(ap, fmt);
    vsnprintf(text, sizeof(text), fmt, ap);
    va_end(ap);
    vgpu98_gpu_log(o->gpu, VGPU98_LOG_EVENTS, "VGPU98 OpenGL: possible %s\n", text);
    o->capture_due = 1; /* at the next SHOW */
}

static void cmd_note(vgpu98_ogl_t *o, uint32_t op, const uint32_t *p, uint32_t len);
static void capture_check(vgpu98_ogl_t *o, ogl_drawable_t *d, const uint8_t *px, uint32_t row, uint32_t w, uint32_t h,
                          uint32_t bytes);

/* --- Recording ---------------------------------------------------------------- */

static void
mem_barrier(vgpu98_ogl_t *o, VkCommandBuffer c)
{
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };

    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    o->vk->vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                                NULL, 0, NULL);
}

/* The renderer's hook: the render pass ends before it, or the Glide device,
   records, and before it submits. */
static void
pass_end(void *priv)
{
    vgpu98_ogl_t   *o = (vgpu98_ogl_t *) priv;
    VkCommandBuffer c;

    /* Another records next (its pipeline in the buffer), or a new buffer
       begins: ours is bound again. */
    o->bound_pipe = VK_NULL_HANDLE;
    if (!o->pass_open)
        return;
    c = vgpu98_r3d_glide_rec(o->r3d);
    o->vk->vkCmdEndRenderPass(c);
    mem_barrier(o, c);
    o->pass_open = 0;
    o->pass_d    = NULL;
}

/* Outside any render pass: the command buffer for copies and barriers. */
static VkCommandBuffer
outside(vgpu98_ogl_t *o)
{
    VkCommandBuffer c = vgpu98_r3d_ogl_cmd(o->r3d);

    o->st.pass_breaks += o->pass_open;
    pass_end(o);
    return c;
}

static int
color_kind(uint32_t format)
{
    return ((format & VGPU98_OGL_COLOR_MASK) == VGPU98_OGL_COLOR_565) ? 0 : 1;
}

static int
depth_kind(uint32_t format)
{
    return (int) ((format & VGPU98_OGL_DEPTH_MASK) >> VGPU98_OGL_DEPTH_SHIFT);
}

/* In the drawable's render pass on its front or back image: the command
   buffer. */
static VkCommandBuffer
inside(vgpu98_ogl_t *o, ogl_drawable_t *d, int buf)
{
    VkCommandBuffer       c;
    VkRenderPassBeginInfo bi = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };

    if (o->pass_open && (o->pass_d == d) && (o->pass_buf == buf))
        return vgpu98_r3d_glide_rec(o->r3d);
    c                           = outside(o);
    bi.renderPass               = o->pass[color_kind(d->format)][depth_kind(d->format)];
    bi.framebuffer              = d->fb[buf];
    bi.renderArea.extent.width  = d->w;
    bi.renderArea.extent.height = d->h;
    o->vk->vkCmdBeginRenderPass(c, &bi, VK_SUBPASS_CONTENTS_INLINE);
    o->pass_open = 1;
    o->pass_d    = d;
    o->pass_buf  = buf;
    return c;
}

/* --- Images --------------------------------------------------------------------- */

static void
image_free(vgpu98_ogl_t *o, ogl_image_t *im)
{
    if (im->view)
        vgpu98_r3d_glide_dead(o->r3d, 1, (uint64_t) (uintptr_t) im->view);
    if (im->image)
        vgpu98_r3d_glide_dead(o->r3d, 0, (uint64_t) (uintptr_t) im->image);
    if (im->mem)
        vgpu98_r3d_glide_dead(o->r3d, 2, (uint64_t) (uintptr_t) im->mem);
    memset(im, 0, sizeof(*im));
}

static int
image_make(vgpu98_ogl_t *o, ogl_image_t *im, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage,
           VkImageAspectFlags aspect)
{
    vgpu98_vk_t          *vk = o->vk;
    VkImageCreateInfo     ic = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };

    memset(im, 0, sizeof(*im));
    ic.imageType     = VK_IMAGE_TYPE_2D;
    ic.format        = format;
    ic.extent.width  = w;
    ic.extent.height = h;
    ic.extent.depth  = 1;
    ic.mipLevels     = 1;
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
    if (!vgpu98_r3d_glide_image_memory(o->r3d, im->image, &im->mem, NULL)) {
        vk->vkDestroyImage(vk->dev, im->image, NULL);
        memset(im, 0, sizeof(*im));
        return 0;
    }
    vi.image                       = im->image;
    vi.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    vi.format                      = format;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vk->vkCreateImageView(vk->dev, &vi, NULL, &im->view) != VK_SUCCESS) {
        im->view = VK_NULL_HANDLE;
        image_free(o, im);
        return 0;
    }
    im->format = format;
    im->w      = w;
    im->h      = h;
    /* Into the GENERAL layout, where every image of the device stays. */
    {
        VkCommandBuffer      c  = outside(o);
        VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };

        ib.srcAccessMask               = 0;
        ib.dstAccessMask               = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        ib.oldLayout                   = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout                   = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        ib.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        ib.image                       = im->image;
        ib.subresourceRange.aspectMask = aspect;
        ib.subresourceRange.levelCount = 1;
        ib.subresourceRange.layerCount = 1;
        vk->vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL,
                                 0, NULL, 1, &ib);
    }
    return 1;
}

/* --- Objects --------------------------------------------------------------------- */

static ogl_group_t *
group_find(vgpu98_ogl_t *o, uint32_t id)
{
    int i;

    for (i = 0; id && (i < VGPU98_OGL_GROUPS); i++) {
        if (o->groups[i].id == id)
            return &o->groups[i];
    }
    return NULL;
}

static ogl_context_t *
context_find(vgpu98_ogl_t *o, uint32_t id)
{
    int i;

    for (i = 0; id && (i < VGPU98_OGL_CONTEXTS); i++) {
        if (o->ctx[i] && (o->ctx[i]->id == id))
            return o->ctx[i];
    }
    return NULL;
}

static ogl_drawable_t *
drawable_find(vgpu98_ogl_t *o, uint32_t id)
{
    int i;

    for (i = 0; id && (i < VGPU98_OGL_DRAWABLES); i++) {
        if (o->draw[i].id == id)
            return &o->draw[i];
    }
    return NULL;
}

static void
drawable_images_free(vgpu98_ogl_t *o, ogl_drawable_t *d)
{
    int i;

    if (o->pass_open && (o->pass_d == d))
        outside(o);
    for (i = 0; i < 2; i++) {
        if (d->fb[i])
            vgpu98_r3d_glide_dead(o->r3d, 3, (uint64_t) (uintptr_t) d->fb[i]);
        d->fb[i] = VK_NULL_HANDLE;
        image_free(o, &d->color[i]);
    }
    image_free(o, &d->depth);
}

/* The drawable's images for its size and format, cleared: colour to black
   (alpha 1: an XRGB drawable's alpha reads 1, and an ARGB one's starts
   there), depth to 1, stencil to 0. 0 if the host GPU refused one. */
static int
drawable_images(vgpu98_ogl_t *o, ogl_drawable_t *d)
{
    vgpu98_vk_t            *vk    = o->vk;
    const int               ck    = color_kind(d->format);
    const int               dk    = depth_kind(d->format);
    const VkFormat          cf    = ck ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R5G6B5_UNORM_PACK16;
    const VkFormat          df    = (dk == VGPU98_OGL_DEPTH_16) ? VK_FORMAT_D16_UNORM : o->depth24;
    const VkImageAspectFlags da   = (dk == VGPU98_OGL_DEPTH_16) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                                : (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    const int               n     = (d->format & VGPU98_OGL_DOUBLE) ? 2 : 1;
    int                     i;

    for (i = 0; i < n; i++) {
        if (!image_make(o, &d->color[i], cf, d->w, d->h,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT))
            return 0;
    }
    if (dk && !image_make(o, &d->depth, df, d->w, d->h,
                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                          da))
        return 0;
    for (i = 0; i < n; i++) {
        VkFramebufferCreateInfo fi = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        VkImageView             at[2];

        at[0]              = d->color[i].view;
        at[1]              = d->depth.view;
        fi.renderPass      = o->pass[ck][dk];
        fi.attachmentCount = dk ? 2 : 1;
        fi.pAttachments    = at;
        fi.width           = d->w;
        fi.height          = d->h;
        fi.layers          = 1;
        if (vk->vkCreateFramebuffer(vk->dev, &fi, NULL, &d->fb[i]) != VK_SUCCESS) {
            d->fb[i] = VK_NULL_HANDLE;
            return 0;
        }
    }
    {
        VkCommandBuffer          c = outside(o);
        VkClearColorValue        cv;
        VkClearDepthStencilValue dv;
        VkImageSubresourceRange  rc = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkImageSubresourceRange  rd = { da, 0, 1, 0, 1 };

        memset(&cv, 0, sizeof(cv));
        cv.float32[3] = 1.0f;
        dv.depth      = 1.0f;
        dv.stencil    = 0;
        mem_barrier(o, c);
        for (i = 0; i < n; i++)
            vk->vkCmdClearColorImage(c, d->color[i].image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rc);
        if (dk)
            vk->vkCmdClearDepthStencilImage(c, d->depth.image, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &rd);
        mem_barrier(o, c);
    }
    return 1;
}

static void
drawable_end(vgpu98_ogl_t *o, ogl_drawable_t *d)
{
    int i;

    if (!d->id)
        return;
    drawable_images_free(o, d);
    for (i = 0; i < VGPU98_OGL_CONTEXTS; i++) {
        if (o->ctx[i] && (o->ctx[i]->drawable == d->id))
            o->ctx[i]->drawable = 0;
    }
    memset(d, 0, sizeof(*d));
}

static void range_give(vgpu98_ogl_t *o, VkDeviceSize at, VkDeviceSize n);

static void
context_end(vgpu98_ogl_t *o, int i)
{
    if (o->ctx[i] && o->ctx[i]->spal_resident)
        range_give(o, o->ctx[i]->spal_at, VGPU98_OGL_PALETTE_MAX * 4);
    free(o->ctx[i]);
    o->ctx[i] = NULL;
}

static void tex_group_free(vgpu98_ogl_t *o, uint32_t group);

static void
group_end(vgpu98_ogl_t *o, ogl_group_t *g)
{
    int i;

    if (!g->id)
        return;
    /* Its contexts and textures go with it. */
    for (i = 0; i < VGPU98_OGL_CONTEXTS; i++) {
        if (o->ctx[i] && (o->ctx[i]->group == g->id))
            context_end(o, i);
    }
    tex_group_free(o, g->id);
    memset(g, 0, sizeof(*g));
}

/* A context's registers as GL 1.1 starts them (the ICD sends them whole
   after the open; these stand until then). */
static void
context_defaults(ogl_context_t *c)
{
    uint32_t *r = c->reg;
    int       i;

    /* Its shared palette as it starts: one white entry of RGBA */
    c->spal_n    = 1;
    c->spal_base = VGPU98_OGL_BASE_RGBA;
    c->spal[0]   = 0xffffffffu;

    memset(r, 0, sizeof(c->reg));
    for (i = 0; i < 4; i++) {
        r[VGPU98_OGL_R_MODELVIEW + 5 * i]  = one_bits;
        r[VGPU98_OGL_R_PROJECTION + 5 * i] = one_bits;
        r[VGPU98_OGL_R_TEXTURE + 5 * i]    = one_bits;
        r[VGPU98_OGL_R_UNIT1 + VGPU98_OGL_U_TEXTURE + 5 * i] = one_bits;
    }
    for (i = 0; i < 3; i++)
        r[VGPU98_OGL_R_NORMAL + 5 * i] = one_bits;
    r[VGPU98_OGL_R_ENABLES]    = VGPU98_OGL_EN_DITHER;
    r[VGPU98_OGL_R_SHADE]      = 1;
    r[VGPU98_OGL_R_DEPTH]      = VGPU98_OGL_CMP_LESS | (1u << 8);
    r[VGPU98_OGL_R_DEPTH_RANGE + 1] = one_bits;
    r[VGPU98_OGL_R_STENCIL]    = VGPU98_OGL_CMP_ALWAYS | (0xffu << 16) | (0xffu << 24);
    r[VGPU98_OGL_R_BLEND]      = VGPU98_OGL_BF_ONE | (VGPU98_OGL_BF_ZERO << 8);
    r[VGPU98_OGL_R_LOGIC_OP]   = 3; /* COPY */
    r[VGPU98_OGL_R_COLOR_MASK] = 15;
    r[VGPU98_OGL_R_ALPHA_FUNC] = VGPU98_OGL_CMP_ALWAYS;
    r[VGPU98_OGL_R_POINT_SIZE] = one_bits;
    r[VGPU98_OGL_R_POINT_MAX]  = 0x42800000u; /* 64.0 */
    r[VGPU98_OGL_R_POINT_FADE] = one_bits;
    r[VGPU98_OGL_R_POINT_ATTEN] = one_bits;
    r[VGPU98_OGL_R_LINE_WIDTH] = one_bits;
    r[VGPU98_OGL_R_LINE_STIPPLE] = 0xffffu | (1u << 16);
    r[VGPU98_OGL_R_DRAW_BUFFERS] = 2;
    r[VGPU98_OGL_R_READ_BUFFER]  = 1;
    r[VGPU98_OGL_R_FOG_PER_PIXEL] = 1;
    r[VGPU98_OGL_R_FOG_DENSITY] = one_bits;
    r[VGPU98_OGL_R_FOG_END]     = one_bits;
    for (i = 0; i < 32; i++)
        r[VGPU98_OGL_R_POLYGON_STIPPLE + i] = 0xffffffffu;
}

/* --- Commands: objects -------------------------------------------------------------- */

static void
pkt_group(vgpu98_ogl_t *o, const uint32_t *p, uint32_t len)
{
    ogl_group_t *g;
    int          i;

    if (len < VGPU98_OGL_LEN_GROUP) {
        drop(o, "%u dwords", len);
        return;
    }
    g = group_find(o, p[0]);
    if (p[1] == VGPU98_OGL_CLOSE) {
        if (g)
            group_end(o, g);
        else
            drop(o, "a CLOSE of group %x, which it does not have", p[0]);
        return;
    }
    /* An OPEN of a group that is there: the ICD sending its objects again
       after a GPU restart, which may have lost a batch. */
    if (g && (p[1] == VGPU98_OGL_OPEN))
        return;
    if ((p[1] != VGPU98_OGL_OPEN) || !p[0]) {
        drop(o, "group %x: action %u", p[0], p[1]);
        return;
    }
    for (i = 0; i < VGPU98_OGL_GROUPS; i++) {
        if (!o->groups[i].id) {
            o->groups[i].id   = p[0];
            o->groups[i].used = o->clock;
            return;
        }
    }
    /* Every group in use: a process that ended without the DLL's detach
       (killed, or crashed) holds some. The least recently named goes. */
    g = &o->groups[0];
    for (i = 1; i < VGPU98_OGL_GROUPS; i++) {
        if (o->groups[i].used < g->used)
            g = &o->groups[i];
    }
    group_end(o, g);
    g->id   = p[0];
    g->used = o->clock;
}

static void
pkt_context(vgpu98_ogl_t *o, const uint32_t *p, uint32_t len)
{
    int i;

    if (len < VGPU98_OGL_LEN_CONTEXT) {
        drop(o, "%u dwords", len);
        return;
    }
    if (p[1] == VGPU98_OGL_CLOSE) {
        for (i = 0; i < VGPU98_OGL_CONTEXTS; i++) {
            if (o->ctx[i] && (o->ctx[i]->id == p[0])) {
                context_end(o, i);
                return;
            }
        }
        drop(o, "a CLOSE of context %x, which it does not have", p[0]);
        return;
    }
    if ((p[1] != VGPU98_OGL_OPEN) || !p[0] || !group_find(o, p[2])) {
        drop(o, "context %x: action %u, group %x%s", p[0], p[1], p[2], group_find(o, p[2]) ? "" : " (no such group)");
        return;
    }
    /* Sent again: its registers come again too. */
    if (context_find(o, p[0])) {
        context_find(o, p[0])->group = p[2];
        return;
    }
    for (i = 0; i < VGPU98_OGL_CONTEXTS; i++) {
        if (!o->ctx[i]) {
            o->ctx[i] = calloc(1, sizeof(ogl_context_t));
            if (!o->ctx[i])
                break;
            o->ctx[i]->id       = p[0];
            o->ctx[i]->group    = p[2];
            o->ctx[i]->regs_new = 1;
            o->ctx[i]->used     = o->clock;
            context_defaults(o->ctx[i]);
            return;
        }
    }
    /* Every context in use (or no memory): the least recently named goes,
       and its slot is taken again. */
    {
        int lru = -1;

        for (i = 0; i < VGPU98_OGL_CONTEXTS; i++) {
            if (o->ctx[i] && ((lru < 0) || (o->ctx[i]->used < o->ctx[lru]->used)))
                lru = i;
        }
        if (lru < 0) {
            drop(o, "context %x: no context to take back", p[0]);
            return;
        }
        context_end(o, lru);
        o->ctx[lru] = calloc(1, sizeof(ogl_context_t));
        if (!o->ctx[lru]) {
            drop(o, "context %x: no memory for it", p[0]);
            return;
        }
        o->ctx[lru]->id       = p[0];
        o->ctx[lru]->group    = p[2];
        o->ctx[lru]->regs_new = 1;
        o->ctx[lru]->used     = o->clock;
        context_defaults(o->ctx[lru]);
    }
}

static void
pkt_drawable(vgpu98_ogl_t *o, const uint32_t *p, uint32_t len)
{
    ogl_drawable_t *d;
    uint32_t        w, h, f;
    int             i;

    if (len < VGPU98_OGL_LEN_DRAWABLE) {
        drop(o, "%u dwords", len);
        return;
    }
    d = drawable_find(o, p[0]);
    if (p[1] == VGPU98_OGL_CLOSE) {
        if (d)
            drawable_end(o, d);
        else
            drop(o, "a CLOSE of drawable %x, which it does not have", p[0]);
        return;
    }
    w = p[2] & 0xffff;
    h = p[2] >> 16;
    f = p[3];
    if (!p[0] || !w || !h || (w > VGPU98_OGL_MAX_SIZE) || (h > VGPU98_OGL_MAX_SIZE) ||
        ((f & VGPU98_OGL_COLOR_MASK) < VGPU98_OGL_COLOR_565) || ((f & VGPU98_OGL_COLOR_MASK) > VGPU98_OGL_COLOR_ARGB8888) ||
        (depth_kind(f) >= DEPTH_KINDS) || (f & ~(uint32_t) (VGPU98_OGL_COLOR_MASK | VGPU98_OGL_DEPTH_MASK | VGPU98_OGL_DOUBLE)) ||
        ((p[1] != VGPU98_OGL_OPEN) && ((p[1] != VGPU98_OGL_RESIZE) || !d))) {
        drop(o, "drawable %x: action %u, %ux%u, format %x", p[0], p[1], w, h, f);
        return;
    }
    /* An OPEN of one that is there (sent again after a restart)
       changes nothing if its size and format are the same, and is a RESIZE
       otherwise. */
    if (d && (d->w == w) && (d->h == h) && (d->format == f) && d->fb[0])
        return;
    if (!d) {
        for (i = 0; (i < VGPU98_OGL_DRAWABLES) && !d; i++) {
            if (!o->draw[i].id)
                d = &o->draw[i];
        }
        if (!d) {
            /* Every drawable in use: the least recently named goes. */
            d = &o->draw[0];
            for (i = 1; i < VGPU98_OGL_DRAWABLES; i++) {
                if (o->draw[i].used < d->used)
                    d = &o->draw[i];
            }
            drawable_end(o, d);
        }
        d->used = o->clock;
    } else
        drawable_images_free(o, d);
    d->id     = p[0];
    d->w      = w;
    d->h      = h;
    d->format = f;
    if (!drawable_images(o, d)) {
        /* No room on the host GPU: the drawable stays, without images, and
           what is drawn into it is dropped. */
        drawable_images_free(o, d);
        drop(o, "drawable %x: no room for its %ux%u images on the host GPU (what is drawn into it is dropped)", p[0], w, h);
    }
}

static void
pkt_bind(vgpu98_ogl_t *o, const uint32_t *p, uint32_t len)
{
    ogl_context_t *c = (len >= VGPU98_OGL_LEN_BIND) ? context_find(o, p[0]) : NULL;

    if (!c || (p[1] && !drawable_find(o, p[1]))) {
        drop(o,
             (len < VGPU98_OGL_LEN_BIND) ? "%u dwords"
             : !c                        ? "context %x, which it does not have"
                                         : "drawable %x, which it does not have",
             (len < VGPU98_OGL_LEN_BIND) ? len : !c ? p[0] : p[1]);
        return;
    }
    c->drawable = p[1];
}

static void
pkt_state(vgpu98_ogl_t *o, ogl_context_t *c, const uint32_t *p, uint32_t len)
{
    const uint32_t first = p[1];
    const uint32_t n     = len - 2;

    if ((first >= VGPU98_OGL_R_COUNT) || (n > VGPU98_OGL_R_COUNT - first)) {
        drop(o, "registers %u to %u, past the %u", first, first + n - 1, VGPU98_OGL_R_COUNT);
        return;
    }
    memcpy(&c->reg[first], &p[2], n * sizeof(uint32_t));
    c->regs_new = 1;
    o->st.states++;
    o->st.state_words += n;
}

/* --- Commands: clears ------------------------------------------------------------- */

/* The drawable's buffers the context draws into (front 1, back 2), those it
   has. */
static uint32_t
draw_buffers(const ogl_context_t *c, const ogl_drawable_t *d)
{
    uint32_t b = c->reg[VGPU98_OGL_R_DRAW_BUFFERS] & 3u;

    if (!(d->format & VGPU98_OGL_DOUBLE))
        b &= 1u;
    return b;
}

/* The scissor box (when on) in the image's rows, inside the drawable; a
   zero-sized one for none. GL's window coordinates count up from the
   bottom, the image's rows down from the top. */
static VkRect2D
scissor_rect(const ogl_context_t *c, const ogl_drawable_t *d)
{
    VkRect2D r;
    int64_t  x0 = 0, y0 = 0, x1 = d->w, y1 = d->h;

    if (c->reg[VGPU98_OGL_R_ENABLES] & VGPU98_OGL_EN_SCISSOR) {
        const int64_t sx = (int32_t) c->reg[VGPU98_OGL_R_SCISSOR];
        const int64_t sy = (int32_t) c->reg[VGPU98_OGL_R_SCISSOR + 1];
        const int64_t sw = (int32_t) c->reg[VGPU98_OGL_R_SCISSOR + 2];
        const int64_t sh = (int32_t) c->reg[VGPU98_OGL_R_SCISSOR + 3];

        if (sx > x0)
            x0 = sx;
        if (sy > y0)
            y0 = sy;
        if (sx + sw < x1)
            x1 = sx + sw;
        if (sy + sh < y1)
            y1 = sy + sh;
    }
    if ((x1 <= x0) || (y1 <= y0)) {
        memset(&r, 0, sizeof(r));
        return r;
    }
    r.offset.x      = (int32_t) x0;
    r.offset.y      = (int32_t) (d->h - y1);
    r.extent.width  = (uint32_t) (x1 - x0);
    r.extent.height = (uint32_t) (y1 - y0);
    return r;
}

/* --- Drawing: pipelines ------------------------------------------------------------ */

/* A drawable's colour format: 0 565, 1 XRGB 8888, 2 ARGB 8888; the colour
   components it keeps (the masks' bits: red 1, green 2, blue 4, alpha 8). */
static uint32_t
format_index(uint32_t format)
{
    switch (format & VGPU98_OGL_COLOR_MASK) {
        case VGPU98_OGL_COLOR_565:
            return 0;
        case VGPU98_OGL_COLOR_XRGB8888:
            return 1;
        default:
            return 2;
    }
}

static uint32_t
format_mask(uint32_t format)
{
    return (format_index(format) == 2) ? 15u : 7u;
}

/* What a pipeline is made of (its key's fields):
   - 0-1 the colour format, 2-3 the depth kind (the render pass);
   - 4 blending, 5-8 and 9-12 its factors (VGPU98_OGL_BF_*);
   - 13 the logic op, 14-17 which;
   - 18-21 the colour mask, the drawable's components only;
   - 22 the depth test, 23-25 its function, 26 depth writes;
   - 27 the stencil test, 28-30 its function, 31-39 its operations;
   - 40-41 the faces culled, 42 GL's front face clockwise;
   - 43 depth bias; 44 depth on the 24-bit grid (ogl.frag's variant);
   - 45 the clear quad's. */
#    define K_BLEND      (1ull << 4)
#    define K_LOGIC      (1ull << 13)
#    define K_DEPTH      (1ull << 22)
#    define K_DEPTH_W    (1ull << 26)
#    define K_STENCIL    (1ull << 27)
#    define K_CW         (1ull << 42)
#    define K_BIAS       (1ull << 43)
#    define K_GRID24     (1ull << 44)
#    define K_CLEAR      (1ull << 45)

static uint64_t
key_target(const vgpu98_ogl_t *o, const ogl_drawable_t *d)
{
    const uint32_t dk = (uint32_t) depth_kind(d->format);
    uint64_t       k  = format_index(d->format) | ((uint64_t) dk << 2);

    if ((dk == VGPU98_OGL_DEPTH_24S8) && o->depth24_float)
        k |= K_GRID24;
    return k;
}

/* A DRAW's pipeline key: the context's registers and the draw's kind. */
static uint64_t
draw_key(const vgpu98_ogl_t *o, const ogl_context_t *c, const ogl_drawable_t *d, uint32_t control)
{
    const uint32_t *r    = c->reg;
    const uint32_t  en   = r[VGPU98_OGL_R_ENABLES];
    const uint32_t  kind = control & VGPU98_OGL_DRAW_KIND_MASK;
    const int       dk   = depth_kind(d->format);
    uint64_t        k    = key_target(o, d);

    if (o->logic_op && (en & VGPU98_OGL_EN_LOGIC_OP))
        k |= K_LOGIC | ((uint64_t) (r[VGPU98_OGL_R_LOGIC_OP] & 15u) << 14);
    else if (en & VGPU98_OGL_EN_BLEND)
        k |= K_BLEND | ((uint64_t) (r[VGPU98_OGL_R_BLEND] & 15u) << 5) | ((uint64_t) ((r[VGPU98_OGL_R_BLEND] >> 8) & 15u) << 9);
    k |= (uint64_t) (r[VGPU98_OGL_R_COLOR_MASK] & format_mask(d->format)) << 18;
    /* Without a depth buffer the depth test passes; without a stencil
       buffer the stencil test does (GL 1.1, 4.1.5 and 4.1.6). */
    if (dk && (en & VGPU98_OGL_EN_DEPTH_TEST))
        k |= K_DEPTH | ((uint64_t) (r[VGPU98_OGL_R_DEPTH] & 7u) << 23) | (((r[VGPU98_OGL_R_DEPTH] >> 8) & 1u) ? K_DEPTH_W : 0);
    if ((dk == VGPU98_OGL_DEPTH_24S8) && (en & VGPU98_OGL_EN_STENCIL_TEST)) {
        const uint32_t ops = r[VGPU98_OGL_R_STENCIL_OPS];

        k |= K_STENCIL | ((uint64_t) (r[VGPU98_OGL_R_STENCIL] & 7u) << 28) | ((uint64_t) (ops & 7u) << 31) |
             ((uint64_t) ((ops >> 4) & 7u) << 34) | ((uint64_t) ((ops >> 8) & 7u) << 37);
    }
    if (kind == VGPU98_OGL_TRIANGLES) {
        /* The ICD culled its own triangles (with the facing it lit them by). */
        if ((en & VGPU98_OGL_EN_CULL_FACE) && !(control & VGPU98_OGL_DRAW_CLIP))
            k |= (uint64_t) ((r[VGPU98_OGL_R_FACE] >> 4) & 3u) << 40;
        if (r[VGPU98_OGL_R_FACE] & 1u)
            k |= K_CW;
    }
    if (!(k & K_GRID24)) {
        if (kind == VGPU98_OGL_TRIANGLES)
            k |= (en & VGPU98_OGL_EN_OFFSET_FILL) ? K_BIAS : 0;
        else if (control & VGPU98_OGL_DRAW_POLYGON)
            k |= (en & ((kind == VGPU98_OGL_LINES) ? VGPU98_OGL_EN_OFFSET_LINE : VGPU98_OGL_EN_OFFSET_POINT)) ? K_BIAS : 0;
    }
    return k;
}

static VkBlendFactor
blend_factor(uint32_t f)
{
    static const VkBlendFactor map[11] = {
        VK_BLEND_FACTOR_ZERO,      VK_BLEND_FACTOR_ONE,           VK_BLEND_FACTOR_SRC_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_DST_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
    };

    return (f < 11) ? map[f] : VK_BLEND_FACTOR_ONE;
}

static const VkCompareOp compare_ops[8] = {
    VK_COMPARE_OP_NEVER,   VK_COMPARE_OP_LESS,      VK_COMPARE_OP_EQUAL,            VK_COMPARE_OP_LESS_OR_EQUAL,
    VK_COMPARE_OP_GREATER, VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_ALWAYS,
};

static VkStencilOp
stencil_op(uint32_t op)
{
    static const VkStencilOp map[6] = {
        VK_STENCIL_OP_KEEP, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_REPLACE,
        VK_STENCIL_OP_INCREMENT_AND_CLAMP, VK_STENCIL_OP_DECREMENT_AND_CLAMP, VK_STENCIL_OP_INVERT,
    };

    return (op < 6) ? map[op] : VK_STENCIL_OP_KEEP;
}

/* A key's pipeline, made into the shared pipeline cache: it reads only
   what never changes after the device's start, so the GPU thread
   (pipe_get) and the warm-up thread both call it. */
static VkResult
pipe_build(vgpu98_ogl_t *o, uint64_t key, VkPipeline *made)
{
    vgpu98_vk_t                           *vk = o->vk;
    const int                              dk = (int) ((key >> 2) & 3u);
    VkPipelineShaderStageCreateInfo        st[2];
    VkPipelineVertexInputStateCreateInfo   vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkPipelineViewportStateCreateInfo      vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineMultisampleStateCreateInfo   ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    VkPipelineDepthStencilStateCreateInfo  ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    VkPipelineColorBlendAttachmentState    ba;
    VkPipelineColorBlendStateCreateInfo    cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    static const VkDynamicState            dyn[6] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                                      VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                                      VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo       dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    VkGraphicsPipelineCreateInfo           ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    VkStencilOpState                       so;
    VkPipeline                             pipe = VK_NULL_HANDLE;
    VkResult                               res;
    static const VkCullModeFlags           culls[4] = { VK_CULL_MODE_NONE, VK_CULL_MODE_FRONT_BIT, VK_CULL_MODE_BACK_BIT,
                                                        VK_CULL_MODE_FRONT_AND_BACK };

    memset(st, 0, sizeof(st));
    st[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = o->vs;
    st[0].pName  = "main";
    st[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = (key & K_GRID24) ? o->fs_depth : o->fs;
    st[1].pName  = "main";
    ia.topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; /* points and lines are quads (ogl.vert) */
    vp.viewportCount = 1;
    vp.scissorCount  = 1;
    rs.polygonMode   = VK_POLYGON_MODE_FILL;
    rs.cullMode      = culls[(key >> 40) & 3u];
    /* Vulkan's viewport is GL's upside down, and Vulkan's facing counts its
       area with y down: the two cancel, GL's counter-clockwise is Vulkan's
       (audit 3, item 1). */
    rs.frontFace       = (key & K_CW) ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.depthBiasEnable = (key & K_BIAS) ? VK_TRUE : VK_FALSE;
    rs.lineWidth       = 1.0f;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ds.depthTestEnable      = (key & K_DEPTH) ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable     = ((key & K_DEPTH) && (key & K_DEPTH_W)) ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp       = compare_ops[(key >> 23) & 7u];
    ds.stencilTestEnable    = (key & K_STENCIL) ? VK_TRUE : VK_FALSE;
    memset(&so, 0, sizeof(so));
    so.compareOp     = compare_ops[(key >> 28) & 7u];
    so.failOp        = stencil_op((uint32_t) ((key >> 31) & 7u));
    so.depthFailOp   = stencil_op((uint32_t) ((key >> 34) & 7u));
    so.passOp        = stencil_op((uint32_t) ((key >> 37) & 7u));
    ds.front         = so;
    ds.back          = so;
    memset(&ba, 0, sizeof(ba));
    ba.blendEnable         = (key & K_BLEND) ? VK_TRUE : VK_FALSE;
    ba.srcColorBlendFactor = blend_factor((uint32_t) ((key >> 5) & 15u));
    ba.dstColorBlendFactor = blend_factor((uint32_t) ((key >> 9) & 15u));
    ba.colorBlendOp        = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = ba.srcColorBlendFactor; /* GL 1.1 blends alpha by the same factors */
    ba.dstAlphaBlendFactor = ba.dstColorBlendFactor;
    ba.alphaBlendOp        = VK_BLEND_OP_ADD;
    ba.colorWriteMask      = (VkColorComponentFlags) ((key >> 18) & 15u);
    cb.logicOpEnable       = (key & K_LOGIC) ? VK_TRUE : VK_FALSE;
    cb.logicOp             = (VkLogicOp) ((key >> 14) & 15u); /* GL's order is Vulkan's */
    cb.attachmentCount     = 1;
    cb.pAttachments        = &ba;
    dy.dynamicStateCount   = 6;
    dy.pDynamicStates      = dyn;
    ci.stageCount          = 2;
    ci.pStages             = st;
    ci.pVertexInputState   = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState   = &ms;
    ci.pDepthStencilState  = dk ? &ds : NULL;
    ci.pColorBlendState    = &cb;
    ci.pDynamicState       = &dy;
    ci.layout              = o->layout;
    ci.renderPass          = o->pass[(key & 3u) ? 1 : 0][dk];
    res   = vk->vkCreateGraphicsPipelines(vk->dev, vgpu98_r3d_glide_pcache(o->r3d), 1, &ci, NULL, &pipe);
    *made = (res == VK_SUCCESS) ? pipe : VK_NULL_HANDLE;
    return res;
}

/* The GPU thread's pipeline for a key: made under the watchdog's "making a
   pipeline" step and timed, a slow one named once. */
static VkPipeline
pipe_make(vgpu98_ogl_t *o, uint64_t key)
{
    const uint64_t t0   = vgpu98_diag_us();
    const int      step = vgpu98_gpu_step(o->gpu, VGPU98_STEP_PIPELINE);
    VkPipeline     pipe;
    const VkResult res = pipe_build(o, key, &pipe);
    uint32_t       ms;

    vgpu98_gpu_step(o->gpu, step);
    if (res != VK_SUCCESS) {
        vgpu98_r3d_glide_fail(o->r3d, "making an OpenGL pipeline", (int) res);
        return VK_NULL_HANDLE;
    }
    vgpu98_r3d_glide_new_pipeline(o->r3d);
    ms = (uint32_t) ((vgpu98_diag_us() - t0) / 1000u);
    if (ms > o->st.pipe_worst_ms)
        o->st.pipe_worst_ms = ms;
    if ((ms >= PIPE_SLOW_MS) && !o->pipe_slow_told) {
        o->pipe_slow_told = 1;
        vgpu98_gpu_log(o->gpu, VGPU98_LOG_EVENTS,
                       "VGPU98 OpenGL: making a pipeline took %u ms (not in the pipeline cache: new shaders, or a state not "
                       "made before)%s\n",
                       ms, atomic_load(&o->warm_live) ? "; the warm-up was still running" : "");
    }
    return pipe;
}

static VkPipeline
pipe_get(vgpu98_ogl_t *o, uint64_t key)
{
    VkPipeline pipe;
    uint32_t   i;

    for (i = 0; i < o->n_pipes; i++) {
        if (o->pipes[i].key == key)
            return o->pipes[i].pipe;
    }
    if (o->n_pipes == MAX_PIPES) {
        /* The oldest goes, once the host GPU is done with it. */
        if (!vgpu98_r3d_glide_submit(o->r3d, 1))
            return VK_NULL_HANDLE;
        vgpu98_vk_wait_idle(o->vk);
        o->vk->vkDestroyPipeline(o->vk->dev, o->pipes[0].pipe, NULL);
        memmove(&o->pipes[0], &o->pipes[1], (MAX_PIPES - 1) * sizeof(o->pipes[0]));
        o->n_pipes--;
        o->bound_pipe = VK_NULL_HANDLE;
    }
    pipe = pipe_make(o, key);
    if (!pipe)
        return VK_NULL_HANDLE;
    o->pipes[o->n_pipes].key  = key;
    o->pipes[o->n_pipes].pipe = pipe;
    o->n_pipes++;
    o->st.pipes++;
    return pipe;
}

/* --- Drawing: the data chunk ------------------------------------------------------------ */

static VkDeviceSize
align_up(VkDeviceSize v, VkDeviceSize a)
{
    return (v + a - 1) / a * a;
}

/* Room for a draw's data (at most DATA_RANGE bytes) in the slot's storage
   buffer, inside the chunk being filled (a new chunk begun when it is full
   or the slot changed): the mapped place, its buffer and offset. NULL if
   3D stopped. */
static uint8_t *
data_alloc(vgpu98_ogl_t *o, VkDeviceSize bytes, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off)
{
    uint32_t     gen;
    const uint32_t slot = vgpu98_r3d_glide_slot(o->r3d, &gen);
    VkDeviceSize at     = align_up(o->chunk_used, align);

    if (!o->chunk_map || (o->chunk_slot != slot) || (o->chunk_gen != gen) || (at + bytes > DATA_CHUNK)) {
        o->chunk_map = vgpu98_r3d_glide_alloc(o->r3d, 1, DATA_CHUNK + DATA_RANGE, align, &o->chunk_buf, &o->chunk_off);
        if (!o->chunk_map)
            return NULL;
        o->chunk_slot = vgpu98_r3d_glide_slot(o->r3d, &o->chunk_gen);
        at            = 0;
    }
    o->chunk_used = at + bytes;
    *buf          = o->chunk_buf;
    *off          = o->chunk_off + at;
    return o->chunk_map + at;
}

/* --- Textures and the texel arena --------------------------------------------------------- */

static uint32_t
tex_bucket(uint32_t group, uint32_t id)
{
    return (group * 2654435761u ^ id) % TEX_BUCKETS;
}

static ogl_tex_t *
tex_find(vgpu98_ogl_t *o, uint32_t group, uint32_t id)
{
    ogl_tex_t *t;

    for (t = o->tex[tex_bucket(group, id)]; t; t = t->next) {
        if ((t->group == group) && (t->id == id))
            return t;
    }
    return NULL;
}

/* The texture, made if need be (NULL with the device's 16384 in use) */
static ogl_tex_t *
tex_get(vgpu98_ogl_t *o, uint32_t group, uint32_t id)
{
    ogl_tex_t     *t = tex_find(o, group, id);
    const uint32_t b = tex_bucket(group, id);

    if (t || !id || (o->n_tex >= VGPU98_OGL_TEXTURES))
        return t;
    t = calloc(1, sizeof(ogl_tex_t));
    if (!t)
        return NULL;
    t->group = group;
    t->id    = id;
    /* A palette as a texture starts with: one white entry of RGBA */
    t->pal_n    = 1;
    t->pal_base = VGPU98_OGL_BASE_RGBA;
    t->pal[0]   = 0xffffffffu;
    t->next  = o->tex[b];
    o->tex[b] = t;
    o->n_tex++;
    return t;
}

/* A level's texels freed, the host memory they took counted back */
static void
level_free(vgpu98_ogl_t *o, ogl_texlevel_t *l)
{
    o->tex_bytes -= (uint64_t) l->w * l->h * l->bpt;
    free(l->texels);
    memset(l, 0, sizeof(*l));
}

/* A range of the arena given back: kept sorted, merged with its neighbours. */
static void
range_give(vgpu98_ogl_t *o, VkDeviceSize at, VkDeviceSize n)
{
    uint32_t i;

    for (i = 0; (i < o->n_free) && (o->free[i].at < at); i++)
        ;
    if ((i > 0) && (o->free[i - 1].at + o->free[i - 1].n == at)) {
        o->free[i - 1].n += n;
        if ((i < o->n_free) && (o->free[i - 1].at + o->free[i - 1].n == o->free[i].at)) {
            o->free[i - 1].n += o->free[i].n;
            memmove(&o->free[i], &o->free[i + 1], (o->n_free - i - 1) * sizeof(o->free[0]));
            o->n_free--;
        }
        return;
    }
    if ((i < o->n_free) && (at + n == o->free[i].at)) {
        o->free[i].at = at;
        o->free[i].n += n;
        return;
    }
    if (o->n_free == o->cap_free) {
        const uint32_t cap = o->cap_free ? 2 * o->cap_free : 64;
        void          *nf  = realloc(o->free, cap * sizeof(o->free[0]));

        if (!nf)
            return; /* the range is lost to the arena, nothing worse */
        o->free     = nf;
        o->cap_free = cap;
    }
    memmove(&o->free[i + 1], &o->free[i], (o->n_free - i) * sizeof(o->free[0]));
    o->free[i].at = at;
    o->free[i].n  = n;
    o->n_free++;
}

/* n bytes of the arena, first fit. 0 if no range holds them. */
static int
range_take(vgpu98_ogl_t *o, VkDeviceSize n, VkDeviceSize *at)
{
    uint32_t i;

    for (i = 0; i < o->n_free; i++) {
        if (o->free[i].n >= n) {
            *at = o->free[i].at;
            o->free[i].at += n;
            o->free[i].n -= n;
            if (!o->free[i].n) {
                memmove(&o->free[i], &o->free[i + 1], (o->n_free - i - 1) * sizeof(o->free[0]));
                o->n_free--;
            }
            return 1;
        }
    }
    return 0;
}

/* Its arena copy dropped: draws recorded before keep reading it, as the
   next upload's barrier waits for them (vkCmdPipelineBarrier's first scope
   holds every command submitted before it). */
static void
tex_drop(vgpu98_ogl_t *o, ogl_tex_t *t)
{
    if (t->resident)
        range_give(o, t->at, t->bytes);
    t->resident = 0;
}

static void
tex_free(vgpu98_ogl_t *o, ogl_tex_t *t)
{
    ogl_tex_t **at = &o->tex[tex_bucket(t->group, t->id)];
    int         k;

    while (*at && (*at != t))
        at = &(*at)->next;
    if (*at)
        *at = t->next;
    tex_drop(o, t);
    for (k = 0; k < VGPU98_OGL_LEVELS; k++)
        level_free(o, &t->level[k]);
    free(t);
    o->n_tex--;
}

/* A group's textures, gone with it (group 0: every texture) */
static void
tex_group_free(vgpu98_ogl_t *o, uint32_t group)
{
    uint32_t b;

    for (b = 0; b < TEX_BUCKETS; b++) {
        ogl_tex_t *t = o->tex[b];

        while (t) {
            ogl_tex_t *n = t->next;

            if (!group || (t->group == group))
                tex_free(o, t);
            t = n;
        }
    }
}

/* The levels a draw samples: those defined from 0 on, one after another,
   each of level 0's kind (indices or colours) */
static uint32_t
tex_levels(const ogl_tex_t *t)
{
    uint32_t n = 0, q = 0, size;

    /* At most level 0's chain down to 1 (q + 1 levels): a level defined
       past it is never sampled. */
    if (t->level[0].w) {
        size = (t->level[0].w > t->level[0].h) ? t->level[0].w : t->level[0].h;
        size = (size > 2 * t->border) ? size - 2 * t->border : 1;
        while (size > 1) {
            size >>= 1;
            q++;
        }
    }
    while ((n <= q) && (n < VGPU98_OGL_LEVELS) && t->level[n].w && t->level[n].texels &&
           (t->level[n].bpt == t->level[0].bpt))
        n++;
    return n;
}

/* A level's place in the arena: whole dwords, so that each level starts on
   one (an index level's bytes, 2.6) */
static VkDeviceSize
level_bytes(const ogl_texlevel_t *l)
{
    return align_up((VkDeviceSize) l->w * l->h * l->bpt, 4);
}

/* A palette's entries into the arena at dst (2.6). 0 when 3D stopped. */
static int
pal_upload(vgpu98_ogl_t *o, const uint32_t *e, uint32_t n, VkDeviceSize dst)
{
    VkDeviceSize off;
    VkBuffer     sbuf;
    uint8_t     *map = vgpu98_r3d_glide_alloc(o->r3d, 0, (VkDeviceSize) n * 4, 4, &sbuf, &off);

    if (!map)
        return 0;
    memcpy(map, e, (size_t) n * 4);
    o->st.pal_copies++;
    {
        VkCommandBuffer c = outside(o);
        VkBufferCopy    bc;

        bc.srcOffset = off;
        bc.dstOffset = dst;
        bc.size      = (VkDeviceSize) n * 4;
        mem_barrier(o, c);
        o->vk->vkCmdCopyBuffer(c, sbuf, o->arena.buffer, 1, &bc);
        mem_barrier(o, c);
    }
    return 1;
}

/* The least recently used texture with an arena copy dropped for room: not
   `keep`, nor one the same draw uses (`used` the draw's clock). 0 if there
   is none. */
static int
tex_evict(vgpu98_ogl_t *o, const ogl_tex_t *keep)
{
    ogl_tex_t *lru = NULL;
    uint32_t   b;

    for (b = 0; b < TEX_BUCKETS; b++) {
        ogl_tex_t *x;

        for (x = o->tex[b]; x; x = x->next) {
            if (x->resident && (x != keep) && (x->used != o->use) && (!lru || (x->used < lru->used)))
                lru = x;
        }
    }
    if (!lru)
        return 0;
    tex_drop(o, lru);
    o->st.evicted++;
    return 1;
}

/* The changed rectangles of a resident texture's levels into its copy:
   each changed row's span a copy region. 0 when 3D stopped. */
static int
tex_patch(vgpu98_ogl_t *o, ogl_tex_t *t, uint32_t n)
{
    VkDeviceSize  bytes = 0, so = 0, off;
    VkBuffer      sbuf;
    VkBufferCopy *bc;
    uint8_t      *map;
    uint32_t      k, rows = 0, r = 0;

    for (k = 0; k < n; k++) {
        if (t->cx1[k]) {
            bytes += (VkDeviceSize) (t->cx1[k] - t->cx0[k]) * (t->cy1[k] - t->cy0[k]) * t->level[k].bpt;
            rows += t->cy1[k] - t->cy0[k];
        }
    }
    if (!rows) {
        memset(t->cx1, 0, sizeof(t->cx1));
        return 1;
    }
    map = vgpu98_r3d_glide_alloc(o->r3d, 0, bytes, 4, &sbuf, &off);
    bc  = map ? malloc(rows * sizeof(VkBufferCopy)) : NULL;
    if (!bc)
        return 0;
    o->st.patches++;
    o->st.patch_bytes += bytes;
    for (k = 0; k < n; k++) {
        const ogl_texlevel_t *l = &t->level[k];
        const uint32_t        span = (t->cx1[k] - t->cx0[k]) * l->bpt;
        uint32_t              y;

        if (!t->cx1[k])
            continue;
        for (y = t->cy0[k]; y < t->cy1[k]; y++, r++) {
            const VkDeviceSize at = ((VkDeviceSize) y * l->w + t->cx0[k]) * l->bpt;

            memcpy(map + so, (const uint8_t *) l->texels + at, span);
            bc[r].srcOffset = off + so;
            bc[r].dstOffset = (VkDeviceSize) t->first[k] * 4 + at;
            bc[r].size      = span;
            so += span;
        }
    }
    {
        VkCommandBuffer c = outside(o);

        mem_barrier(o, c);
        o->vk->vkCmdCopyBuffer(c, sbuf, o->arena.buffer, rows, bc);
        mem_barrier(o, c);
    }
    free(bc);
    memset(t->cx1, 0, sizeof(t->cx1));
    return 1;
}

/* The texture's copy in the arena, made or brought up to date (the least
   recently used dropped for room, never one this draw uses: `used` the
   draw's clock). An index texture's palette follows its levels, with room
   for the largest (2.6). 0 when it cannot have one. */
static int
tex_resident(vgpu98_ogl_t *o, ogl_tex_t *t)
{
    const uint32_t n     = tex_levels(t);
    const int      index = t->base == VGPU98_OGL_BASE_INDEX;
    VkDeviceSize   bytes = 0, at, off;
    VkBuffer       sbuf;
    uint8_t       *map;
    uint32_t       k;

    if (!n || !o->arena.buffer)
        return 0;
    for (k = 0; k < n; k++)
        bytes += level_bytes(&t->level[k]);
    bytes = align_up(bytes + (index ? VGPU98_OGL_PALETTE_MAX * 4 : 0), 256);
    if (t->resident && !t->dirty && (t->bytes == bytes)) {
        if (!tex_patch(o, t, n) || (index && t->pal_dirty && !pal_upload(o, t->pal, t->pal_n, (VkDeviceSize) t->pal_first * 4))) {
            tex_drop(o, t);
            return 0;
        }
        t->pal_dirty = 0;
        return 1;
    }
    if (t->resident && (t->bytes != bytes))
        tex_drop(o, t);
    if (!t->resident) {
        while (!range_take(o, bytes, &at)) {
            if (!tex_evict(o, t))
                return 0; /* larger than the arena (with this draw's others) */
        }
        t->at       = at;
        t->bytes    = bytes;
        t->resident = 1;
    }
    /* The levels (and the palette) through the slot's staging buffer */
    map = vgpu98_r3d_glide_alloc(o->r3d, 0, bytes, 4, &sbuf, &off);
    if (!map) {
        tex_drop(o, t);
        return 0;
    }
    at = 0;
    for (k = 0; k < n; k++) {
        memcpy(map + at, t->level[k].texels, (size_t) t->level[k].w * t->level[k].h * t->level[k].bpt);
        t->first[k] = (uint32_t) ((t->at + at) / 4);
        at += level_bytes(&t->level[k]);
    }
    if (index) {
        memcpy(map + at, t->pal, (size_t) t->pal_n * 4);
        t->pal_first = (uint32_t) ((t->at + at) / 4);
        at += (VkDeviceSize) t->pal_n * 4;
    }
    /* Copied whole; unchanged since its last copy, it was dropped for room */
    o->st.copies++;
    o->st.copy_bytes += at;
    o->st.again += !t->dirty;
    {
        VkCommandBuffer c = outside(o);
        VkBufferCopy    bc;

        bc.srcOffset = off;
        bc.dstOffset = t->at;
        bc.size      = at;
        mem_barrier(o, c);
        o->vk->vkCmdCopyBuffer(c, sbuf, o->arena.buffer, 1, &bc);
        mem_barrier(o, c);
    }
    t->dirty     = 0;
    t->pal_dirty = 0;
    memset(t->cx1, 0, sizeof(t->cx1));
    return 1;
}

/* The context's shared palette in the arena, made or brought up to date
   (2.6; the least recently used texture dropped for room, never one this
   draw uses). 0 when it cannot have one. */
static int
spal_prepare(vgpu98_ogl_t *o, ogl_context_t *c)
{
    if (!o->arena.buffer)
        return 0;
    if (!c->spal_resident) {
        while (!range_take(o, VGPU98_OGL_PALETTE_MAX * 4, &c->spal_at)) {
            if (!tex_evict(o, NULL))
                return 0;
        }
        c->spal_resident = 1;
        c->spal_dirty    = 1;
    }
    if (c->spal_dirty) {
        if (!pal_upload(o, c->spal, c->spal_n, c->spal_at))
            return 0;
        c->spal_dirty = 0;
    }
    return 1;
}

/* The uniform block's texture part for a unit's texture drawn with
   (ogl.frag's layout; all zero for none), its copy made first. An index
   texture's block names its palette: INDEXED in [0], the palette's base
   format in place of the texture's, its width less one in [2]'s second
   byte, its first entry's dword in [3] (2.6). */
#    define TEX_INDEXED (1u << 24)

static void
tex_block(vgpu98_ogl_t *o, ogl_context_t *c, int unit, uint32_t *b)
{
    const uint32_t id = c->reg[unit ? VGPU98_OGL_R_UNIT1 + VGPU98_OGL_U_TEXTURE_ID : VGPU98_OGL_R_TEXTURE_ID];
    ogl_tex_t     *t;
    uint32_t       n, k, base;

    memset(b, 0, TEX_BLOCK * 16);
    t = id ? tex_find(o, c->group, id) : NULL;
    if (!t || !t->base)
        return;
    t->used = o->use; /* this draw's: kept while the other unit's is made */
    if (!tex_resident(o, t))
        return;
    n    = tex_levels(t);
    base = t->base;
    if (base == VGPU98_OGL_BASE_INDEX) {
        /* Its own palette, or the context's shared one when that is on */
        if (c->reg[VGPU98_OGL_R_ENABLES] & VGPU98_OGL_EN_SHARED_PALETTE) {
            if (!spal_prepare(o, c))
                return;
            base = c->spal_base;
            b[2] = (c->spal_n - 1) << 8;
            b[3] = (uint32_t) (c->spal_at / 4);
        } else {
            base = t->pal_base;
            b[2] = (t->pal_n - 1) << 8;
            b[3] = t->pal_first;
        }
        b[0] = TEX_INDEXED;
        /* A one-entry palette: every texel one colour (the starting palette:
           white), a possible missing palette */
        o->st.pal_default += ((b[2] >> 8) & 0xffu) == 0;
    }
    /* The wraps as sampled: GL_CLAMP as clamp-to-edge, as period drivers
       did, unless the setting is Exact */
    {
        uint32_t ws = t->wraps & 15u, wt = (t->wraps >> 4) & 15u;

        /* The report's wraps, as set */
        if ((ws == VGPU98_OGL_WRAP_CLAMP) || (wt == VGPU98_OGL_WRAP_CLAMP))
            o->st.wrap_clamp++;
        else if ((ws == VGPU98_OGL_WRAP_CLAMP_EDGE) || (wt == VGPU98_OGL_WRAP_CLAMP_EDGE))
            o->st.wrap_edge++;
        else
            o->st.wrap_repeat++;
        if (!o->clamp_exact) {
            ws = (ws == VGPU98_OGL_WRAP_CLAMP) ? VGPU98_OGL_WRAP_CLAMP_EDGE : ws;
            wt = (wt == VGPU98_OGL_WRAP_CLAMP) ? VGPU98_OGL_WRAP_CLAMP_EDGE : wt;
        }
        b[0] |= (base & 15u) | ((t->dims & 15u) << 4) | (ws << 8) | (wt << 12) | ((t->filters & 15u) << 16) |
                (((t->filters >> 4) & 15u) << 20);
    }
    b[1] = n;
    b[2] |= t->border;
    memcpy(&b[4], t->border_color, 16);
    for (k = 0; k < n; k++) {
        b[8 + 4 * k]  = t->first[k];
        b[9 + 4 * k]  = t->level[k].w;
        b[10 + 4 * k] = t->level[k].h;
    }
}

/* Both units' texture parts brought up to date, for one draw: a change
   makes the uniform block go again. */
static void
tex_prepare(vgpu98_ogl_t *o, ogl_context_t *c)
{
    uint32_t b[TEX_BLOCK * 4];
    int      u;

    /* Both units' textures this draw's first, so that making one resident
       (or the shared palette) never drops the other's (the final audit) */
    o->use++;
    for (u = 0; u < TEX_UNITS; u++) {
        const uint32_t id = c->reg[u ? VGPU98_OGL_R_UNIT1 + VGPU98_OGL_U_TEXTURE_ID : VGPU98_OGL_R_TEXTURE_ID];
        ogl_tex_t     *t  = id ? tex_find(o, c->group, id) : NULL;

        if (t)
            t->used = o->use;
    }
    for (u = 0; u < TEX_UNITS; u++) {
        tex_block(o, c, u, b);
        if (memcmp(b, c->tex[u], sizeof(b))) {
            memcpy(c->tex[u], b, sizeof(b));
            c->regs_new = 1;
        }
    }
}


/* --- Drawing: buffers ----------------------------------------------------------------- */

/* A descriptor set from the slot's pool, the pool reset when the slot was
   begun again (its last use done); written for the buffers. */
static VkDescriptorSet
set_make(vgpu98_ogl_t *o, uint32_t slot, uint32_t gen, VkBuffer vbuf, VkBuffer ubuf)
{
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorBufferInfo      bi[4];
    VkWriteDescriptorSet        w[4];
    VkDescriptorSet             set;
    int                         k;

    if (slot >= VGPU98_R3D_SLOTS)
        return VK_NULL_HANDLE;
    if (o->dpool_gen[slot] != gen) {
        o->vk->vkResetDescriptorPool(o->vk->dev, o->dpool[slot], 0);
        o->dpool_gen[slot] = gen;
        if (o->set_slot == slot)
            o->set = VK_NULL_HANDLE;
    }
    ai.descriptorPool     = o->dpool[slot];
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &o->dsl;
    if (o->vk->vkAllocateDescriptorSets(o->vk->dev, &ai, &set) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    /* Fixed ranges: the uniform block's; a draw's data, at most DATA_RANGE
       (data_alloc keeps that much spare at a chunk's end); the arena whole */
    bi[0].buffer = ubuf;
    bi[0].offset = 0;
    bi[0].range  = UBO_BYTES;
    bi[1].buffer = vbuf;
    bi[1].offset = 0;
    bi[1].range  = DATA_RANGE;
    bi[2]        = bi[1];
    bi[3].buffer = o->arena.buffer;
    bi[3].offset = 0;
    bi[3].range  = VK_WHOLE_SIZE;
    memset(w, 0, sizeof(w));
    for (k = 0; k < 4; k++) {
        w[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet          = set;
        w[k].dstBinding      = (uint32_t) k;
        w[k].descriptorCount = 1;
        w[k].descriptorType  = (k == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                        : (k < 3) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[k].pBufferInfo     = &bi[k];
    }
    o->vk->vkUpdateDescriptorSets(o->vk->dev, 4, w, 0, NULL);
    return set;
}

/* A draw's buffers: its two data parts (vertices and indices, or an image)
   in the data chunk, the context's uniform block (the copy made before,
   while unchanged in this slot), and the descriptor set naming them; offs,
   the set's dynamic offsets. 0 if 3D stopped. */
static int
buffers(vgpu98_ogl_t *o, ogl_context_t *c, const void *d1, VkDeviceSize n1, const void *d2, VkDeviceSize n2,
        VkDescriptorSet *set, uint32_t *offs)
{
    const VkPhysicalDeviceLimits *lim    = &o->vk->props.limits;
    const VkDeviceSize            salign = (lim->minStorageBufferOffsetAlignment > 4) ? lim->minStorageBufferOffsetAlignment : 4;
    const VkDeviceSize            ualign = (lim->minUniformBufferOffsetAlignment > 16) ? lim->minUniformBufferOffsetAlignment : 16;
    int                           tries;

    for (tries = 0; tries < 4; tries++) {
        uint32_t     slot0, gen0, slot1, gen1;
        VkBuffer     b1, b2, ubuf;
        VkDeviceSize o1, o2, uoff;
        uint8_t     *m1, *m2, *umap = NULL;

        slot0 = vgpu98_r3d_glide_slot(o->r3d, &gen0);
        m1    = data_alloc(o, n1 ? n1 : 4, salign, &b1, &o1);
        m2    = m1 ? data_alloc(o, n2 ? n2 : 4, salign, &b2, &o2) : NULL;
        if (!m1 || !m2)
            return 0;
        if (c->regs_new || !c->ubo_buf || (c->ubo_slot != slot0) || (c->ubo_gen != gen0)) {
            umap = vgpu98_r3d_glide_alloc(o->r3d, 2, UBO_BYTES, ualign, &ubuf, &uoff);
            if (!umap)
                return 0;
        } else {
            ubuf = c->ubo_buf;
            uoff = c->ubo_off;
        }
        slot1 = vgpu98_r3d_glide_slot(o->r3d, &gen1);
        if ((slot0 != slot1) || (gen0 != gen1) || (b1 != b2))
            continue; /* the slot was submitted between, or a chunk began in another buffer: again */
        if (umap) {
            memset(umap, 0, UBO_BYTES);
            memcpy(umap, c->reg, sizeof(c->reg));
            memcpy(umap + REGS_VEC4S * 16, c->tex, sizeof(c->tex));
            c->regs_new = 0;
            c->ubo_slot = slot1;
            c->ubo_gen  = gen1;
            c->ubo_buf  = ubuf;
            c->ubo_off  = uoff;
        }
        if (n1)
            memcpy(m1, d1, (size_t) n1);
        if (n2)
            memcpy(m2, d2, (size_t) n2);
        if (!o->set || (o->set_slot != slot1) || (o->set_gen != gen1) || (o->set_vbuf != b1) || (o->set_ubuf != ubuf)) {
            VkDescriptorSet s = set_make(o, slot1, gen1, b1, ubuf);

            if (!s) {
                /* The slot's pool is full: the next slot. */
                if (!vgpu98_r3d_glide_submit(o->r3d, 0))
                    return 0;
                continue;
            }
            o->set      = s;
            o->set_slot = slot1;
            o->set_gen  = gen1;
            o->set_vbuf = b1;
            o->set_ubuf = ubuf;
        }
        *set    = o->set;
        offs[0] = (uint32_t) uoff;
        offs[1] = (uint32_t) o1;
        offs[2] = (uint32_t) o2;
        return 1;
    }
    return 0;
}

/* GL's viewport as Vulkan's, upside down (GL's rows count up from the
   bottom), inside the host's limits; 0 when it is empty. */
static int
viewport_gl(const vgpu98_ogl_t *o, const ogl_context_t *c, const ogl_drawable_t *d, VkViewport *vp)
{
    const VkPhysicalDeviceLimits *lim = &o->vk->props.limits;
    double                        x   = (int32_t) c->reg[VGPU98_OGL_R_VIEWPORT];
    double                        y   = (int32_t) c->reg[VGPU98_OGL_R_VIEWPORT + 1];
    double                        w   = (int32_t) c->reg[VGPU98_OGL_R_VIEWPORT + 2];
    double                        h   = (int32_t) c->reg[VGPU98_OGL_R_VIEWPORT + 3];
    const double                  lo  = lim->viewportBoundsRange[0], hi = lim->viewportBoundsRange[1];
    double                        fy;

    if ((w <= 0.0) || (h <= 0.0))
        return 0;
    if (w > lim->maxViewportDimensions[0])
        w = lim->maxViewportDimensions[0];
    if (h > lim->maxViewportDimensions[1])
        h = lim->maxViewportDimensions[1];
    fy = (double) d->h - y; /* GL's bottom edge in the image's rows */
    if (x < lo)
        x = lo;
    if (x + w > hi)
        x = hi - w;
    if (fy > hi)
        fy = hi;
    if (fy - h < lo)
        fy = lo + h;
    vp->x        = (float) x;
    vp->y        = (float) fy;
    vp->width    = (float) w;
    vp->height   = -(float) h;
    vp->minDepth = clamp01(f32(c->reg[VGPU98_OGL_R_DEPTH_RANGE]));
    vp->maxDepth = clamp01(f32(c->reg[VGPU98_OGL_R_DEPTH_RANGE + 1]));
    return 1;
}

/* The whole drawable as Vulkan's viewport, upside down, depth 0-1 (points,
   lines and images are placed in window coordinates by ogl.vert, which
   applies the depth range: GL clips them only at the window). */
static void
viewport_window(const ogl_drawable_t *d, VkViewport *vp)
{
    vp->x        = 0.0f;
    vp->y        = (float) d->h;
    vp->width    = (float) d->w;
    vp->height   = -(float) d->h;
    vp->minDepth = 0.0f;
    vp->maxDepth = 1.0f;
}

/* In the drawable's pass on buffer b: the pipeline, the set, the dynamic
   state and the push constants for a draw. */
static VkCommandBuffer
draw_begin(vgpu98_ogl_t *o, const ogl_context_t *c, ogl_drawable_t *d, int b, VkPipeline pipe, VkDescriptorSet set,
           const uint32_t *offs, const VkViewport *vp, const VkRect2D *sc, const ogl_push_t *pc, uint32_t stencil_write,
           uint32_t stencil_ref)
{
    vgpu98_vk_t    *vk = o->vk;
    VkCommandBuffer cb = inside(o, d, b);

    if (o->bound_pipe != pipe) {
        vk->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        o->bound_pipe = pipe;
    }
    vk->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, o->layout, 0, 1, &set, 3, offs);
    vk->vkCmdSetViewport(cb, 0, 1, vp);
    vk->vkCmdSetScissor(cb, 0, 1, sc);
    vk->vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, (c->reg[VGPU98_OGL_R_STENCIL] >> 16) & 0xffu);
    vk->vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_write);
    vk->vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref);
    vk->vkCmdSetDepthBias(cb, f32(c->reg[VGPU98_OGL_R_POLYGON_OFFSET + 1]), 0.0f, f32(c->reg[VGPU98_OGL_R_POLYGON_OFFSET]));
    vk->vkCmdPushConstants(cb, o->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(*pc), pc);
    return cb;
}

/* The buffers a draw goes into (front 1, back 2): GL's draw buffers, or
   with none of them (GL_NONE) one buffer through a colour mask of 0, so
   depth and stencil are still updated (GL 1.1, 4.2.1). */
static uint32_t
targets(const ogl_context_t *c, const ogl_drawable_t *d, uint64_t *key)
{
    const uint32_t b = draw_buffers(c, d);

    if (b)
        return b;
    *key &= ~(15ull << 18);
    return (d->format & VGPU98_OGL_DOUBLE) ? 2u : 1u;
}

/* --- Commands: DRAW, PIXELS ---------------------------------------------------------------- */

static void
pkt_draw(vgpu98_ogl_t *o, ogl_context_t *c, const uint32_t *p, uint32_t len)
{
    const uint32_t  ctl  = (len >= VGPU98_OGL_LEN_DRAW) ? p[1] : 0;
    const uint32_t  kind = ctl & VGPU98_OGL_DRAW_KIND_MASK;
    const uint32_t  nv   = (len >= VGPU98_OGL_LEN_DRAW) ? p[2] : 0;
    const uint32_t  ni   = (len >= VGPU98_OGL_LEN_DRAW) ? p[3] : 0;
    const uint32_t  per  = (kind == VGPU98_OGL_POINTS) ? 1u : (kind == VGPU98_OGL_LINES) ? 2u : 3u;
    const uint32_t  ok   = VGPU98_OGL_DRAW_KIND_MASK | VGPU98_OGL_DRAW_CLIP | VGPU98_OGL_DRAW_STIPPLE_ON | VGPU98_OGL_DRAW_POLYGON |
                        VGPU98_OGL_DRAW_TEX1;
    const uint32_t  vd   = (ctl & VGPU98_OGL_DRAW_TEX1) ? VGPU98_OGL_VERTEX1_DWORDS : VGPU98_OGL_VERTEX_DWORDS;
    const uint16_t *idx;
    ogl_drawable_t *d;
    VkDescriptorSet set;
    VkPipeline      pipe;
    VkViewport      vp;
    VkRect2D        sc;
    ogl_push_t      pc;
    uint64_t        key;
    uint32_t        offs[3], bufs, i;
    int             b;

    if ((len < VGPU98_OGL_LEN_DRAW) || (kind > VGPU98_OGL_TRIANGLES) || (ctl & ~ok) || !nv || (nv > 65536) || !ni ||
        (ni % per) || ((uint64_t) len != VGPU98_OGL_LEN_DRAW + (uint64_t) nv * vd + (ni + 1) / 2) ||
        ((uint64_t) nv * vd * 4 > DATA_RANGE) || ((uint64_t) ni * 2 > DATA_RANGE)) {
        drop(o, "kind %u, control %x, %u vertices, %u indices in %u dwords", kind, ctl, nv, ni, len);
        return;
    }
    /* The indices: 16 bits each, the first of a dword in its low half */
    idx = (const uint16_t *) &p[VGPU98_OGL_LEN_DRAW + nv * vd];
    for (i = 0; i < ni; i++) {
        if (idx[i] >= nv) {
            drop(o, "an index %u past its %u vertices", idx[i], nv);
            return;
        }
    }
    o->st.draws++;
    o->st.vertices += nv;
    o->st.draws_tex1 += (ctl & VGPU98_OGL_DRAW_TEX1) != 0;
    o->st.draws_clip += (ctl & VGPU98_OGL_DRAW_CLIP) != 0;
    d = c->drawable ? drawable_find(o, c->drawable) : NULL;
    if (!d || !d->fb[0])
        return; /* nothing to draw into */
    d->last_ctx = c->id; /* its textures, for a capture */
    sc = scissor_rect(c, d);
    if (!sc.extent.width)
        return;
    /* Triangles in GL's viewport; points and lines in the window (their
       quads in window coordinates: ogl.vert) */
    if (kind == VGPU98_OGL_TRIANGLES) {
        if (!viewport_gl(o, c, d, &vp))
            return;
    } else
        viewport_window(d, &vp);
    key  = draw_key(o, c, d, ctl);
    bufs = targets(c, d, &key);
    pipe = pipe_get(o, key);
    if (!pipe)
        return;
    tex_prepare(o, c);
    if (!buffers(o, c, &p[VGPU98_OGL_LEN_DRAW], (VkDeviceSize) nv * vd * 4, idx, (VkDeviceSize) ni * 2, &set, offs))
        return;
    memset(&pc, 0, sizeof(pc));
    pc.control = ctl;
    pc.height  = d->h;
    pc.width   = d->w;
    for (b = 0; b < 2; b++) {
        if (bufs & (1u << b)) {
            VkCommandBuffer cb = draw_begin(o, c, d, b, pipe, set, offs, &vp, &sc, &pc, c->reg[VGPU98_OGL_R_STENCIL] >> 24,
                                            (c->reg[VGPU98_OGL_R_STENCIL] >> 8) & 0xffu);

            o->vk->vkCmdDraw(cb, (kind == VGPU98_OGL_TRIANGLES) ? ni : 6 * (ni / per), 1, 0, 0);
        }
    }
}

/* An image's fragments: a quad over its zoomed rectangle;
   ogl.frag finds each fragment's pixel. Stencil images write their indices
   one bit at a time: for each bit the write mask lets through, the
   fragments whose bit is 1 replace it with 1, then those whose bit is 0
   with 0 (no colour, no tests: GL 1.1, 4.3.1). */
static void
pkt_pixels(vgpu98_ogl_t *o, ogl_context_t *c, const uint32_t *p, uint32_t len)
{
    const uint32_t  kind = (len >= VGPU98_OGL_LEN_PIXELS) ? (p[1] & ~(uint32_t) VGPU98_OGL_PIXELS_TEX1) : 0;
    const uint32_t  head = VGPU98_OGL_LEN_PIXELS + (((len >= VGPU98_OGL_LEN_PIXELS) && (p[1] & VGPU98_OGL_PIXELS_TEX1)) ? 4 : 0);
    const uint32_t  w    = (len >= VGPU98_OGL_LEN_PIXELS) ? (p[2] & 0xffff) : 0;
    const uint32_t  h    = (len >= VGPU98_OGL_LEN_PIXELS) ? (p[2] >> 16) : 0;
    const uint32_t  rw   = (kind >= VGPU98_OGL_PIXELS_STENCIL) ? (w + 3) / 4 : w;
    ogl_drawable_t *d;
    VkDescriptorSet set;
    VkViewport      vp;
    VkRect2D        sc;
    ogl_push_t      pc;
    uint32_t        offs[3], bufs;
    int             k, b;

    if ((len < VGPU98_OGL_LEN_PIXELS) || (kind > VGPU98_OGL_PIXELS_BITMAP) || !w || !h ||
        ((uint64_t) len != head + (uint64_t) rw * h) || ((uint64_t) rw * h * 4 > DATA_RANGE)) {
        drop(o, "kind %x, %ux%u in %u dwords", (len >= VGPU98_OGL_LEN_PIXELS) ? p[1] : 0, w, h, len);
        return;
    }
    o->st.pixels++;
    d = c->drawable ? drawable_find(o, c->drawable) : NULL;
    if (!d || !d->fb[0])
        return;
    sc = scissor_rect(c, d);
    if (!sc.extent.width)
        return;
    if ((kind == VGPU98_OGL_PIXELS_STENCIL) && ((depth_kind(d->format) != VGPU98_OGL_DEPTH_24S8) || !(c->reg[VGPU98_OGL_R_STENCIL] >> 24)))
        return; /* no stencil buffer, or no bit written */
    viewport_window(d, &vp);
    memset(&pc, 0, sizeof(pc));
    pc.control = KIND_PIXELS | (kind << 4);
    pc.height  = d->h;
    pc.width   = d->w;
    pc.pos[0]  = f32(p[3]);
    pc.pos[1]  = f32(p[4]);
    pc.pos[2]  = clamp01(f32(p[5]));
    pc.pos[3]  = f32(p[16]);
    pc.zoom[0] = f32(p[6]);
    pc.zoom[1] = f32(p[7]);
    pc.zoom[2] = (float) w;
    pc.zoom[3] = (float) h;
    for (k = 0; k < 4; k++) {
        pc.color[k] = clamp01(f32(p[8 + k]));
        pc.tex[k]   = f32(p[12 + k]);
        pc.tex1[k]  = (head > VGPU98_OGL_LEN_PIXELS) ? f32(p[17 + k]) : ((k == 3) ? 1.0f : 0.0f);
    }
    if (kind == VGPU98_OGL_PIXELS_STENCIL) {
        const uint32_t swm = c->reg[VGPU98_OGL_R_STENCIL] >> 24;
        const uint64_t key = key_target(o, d) | K_CLEAR | K_STENCIL | ((uint64_t) VGPU98_OGL_CMP_ALWAYS << 28) |
                             ((uint64_t) VGPU98_OGL_SOP_REPLACE << 31) | ((uint64_t) VGPU98_OGL_SOP_REPLACE << 34) |
                             ((uint64_t) VGPU98_OGL_SOP_REPLACE << 37);
        const VkPipeline pipe = pipe_get(o, key);
        int              bit, v;

        if (!pipe)
            return;
        tex_prepare(o, c);
        if (!buffers(o, c, &p[head], (VkDeviceSize) rw * h * 4, NULL, 0, &set, offs))
            return;
        b = (d->format & VGPU98_OGL_DOUBLE) ? BACK : FRONT;
        for (bit = 0; bit < 8; bit++) {
            if (!(swm & (1u << bit)))
                continue;
            for (v = 1; v >= 0; v--) {
                pc.bit = (uint32_t) bit | ((uint32_t) v << 8);
                o->vk->vkCmdDraw(draw_begin(o, c, d, b, pipe, set, offs, &vp, &sc, &pc, 1u << bit, v ? 0xffu : 0u), 6, 1, 0, 0);
            }
        }
        return;
    }
    {
        /* Through the per-fragment operations as a point's fragments are;
           a depth image's depths put by ogl.frag on the 24-bit grid */
        uint64_t   key = draw_key(o, c, d, VGPU98_OGL_POINTS) | ((kind == VGPU98_OGL_PIXELS_DEPTH) ? K_GRID24 : 0);
        VkPipeline pipe;

        bufs = targets(c, d, &key);
        pipe = pipe_get(o, key);
        if (!pipe)
            return;
        tex_prepare(o, c);
        if (!buffers(o, c, &p[head], (VkDeviceSize) rw * h * 4, NULL, 0, &set, offs))
            return;
        for (b = 0; b < 2; b++) {
            if (bufs & (1u << b))
                o->vk->vkCmdDraw(draw_begin(o, c, d, b, pipe, set, offs, &vp, &sc, &pc, c->reg[VGPU98_OGL_R_STENCIL] >> 24,
                                            (c->reg[VGPU98_OGL_R_STENCIL] >> 8) & 0xffu),
                                 6, 1, 0, 0);
        }
    }
}

/* --- A clear through partial masks ----------------------------------------------------- */

/* GL's clear honours the colour and stencil write masks, which Vulkan's
   clears do not: the shaders' clear quad over the scissor box, through a
   pipeline with those masks. */
static void
clear_quad(vgpu98_ogl_t *o, ogl_context_t *c, ogl_drawable_t *d, uint32_t colors, uint32_t cmask, int depth, int stencil,
           const uint32_t *p, const VkRect2D *rect)
{
    VkDescriptorSet set;
    VkViewport      vp;
    ogl_push_t      pc;
    VkPipeline      pipes[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t        offs[3];
    int             b, first = 1;

    /* Depth and stencil with the first buffer drawn (or a buffer of their
       own when no colour is cleared). */
    if (!colors && (depth || stencil)) {
        colors = (d->format & VGPU98_OGL_DOUBLE) ? 2u : 1u;
        cmask  = 0;
    }
    /* The pipelines first: making one may submit the slot. */
    for (b = 0; b < 2; b++) {
        uint64_t key;

        if (!(colors & (1u << b)))
            continue;
        key = key_target(o, d) | K_CLEAR | ((uint64_t) cmask << 18);
        if (first && depth)
            key |= K_DEPTH | K_DEPTH_W | ((uint64_t) VGPU98_OGL_CMP_ALWAYS << 23);
        if (first && stencil)
            key |= K_STENCIL | ((uint64_t) VGPU98_OGL_CMP_ALWAYS << 28) | ((uint64_t) VGPU98_OGL_SOP_REPLACE << 31) |
                   ((uint64_t) VGPU98_OGL_SOP_REPLACE << 34) | ((uint64_t) VGPU98_OGL_SOP_REPLACE << 37);
        first    = 0;
        pipes[b] = pipe_get(o, key);
        if (!pipes[b])
            return;
    }
    if (!buffers(o, c, NULL, 0, NULL, 0, &set, offs))
        return;
    vp.x        = 0.0f;
    vp.y        = 0.0f;
    vp.width    = (float) d->w;
    vp.height   = (float) d->h;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    memset(&pc, 0, sizeof(pc));
    pc.control  = KIND_CLEAR;
    pc.height   = d->h;
    pc.width    = d->w;
    pc.color[0] = clamp01(f32(p[2]));
    pc.color[1] = clamp01(f32(p[3]));
    pc.color[2] = clamp01(f32(p[4]));
    pc.color[3] = ((d->format & VGPU98_OGL_COLOR_MASK) == VGPU98_OGL_COLOR_ARGB8888) ? clamp01(f32(p[5])) : 1.0f;
    pc.pos[2]   = clamp01(f32(p[6]));
    for (b = 0; b < 2; b++) {
        if (pipes[b])
            o->vk->vkCmdDraw(draw_begin(o, c, d, b, pipes[b], set, offs, &vp, rect, &pc, c->reg[VGPU98_OGL_R_STENCIL] >> 24,
                                        p[7] & 0xffu),
                             6, 1, 0, 0);
    }
}

/* --- Commands: textures --------------------------------------------------------------------- */

/* The host memory all textures' texels may take */
#    define TEX_BYTES_MAX (512ull << 20)

/* A row of texels from TEXIMAGE's data into a level: dwords as they come,
   or an index level's bytes, four a dword (the first in the low byte), the
   row padded to a whole dword (2.6) */
static void
row_in(ogl_texlevel_t *l, uint32_t x, uint32_t y, const uint32_t *src, uint32_t w)
{
    if (l->bpt == 4) {
        memcpy(l->texels + (size_t) y * l->w + x, src, (size_t) w * 4);
        return;
    }
    {
        uint8_t *dst = (uint8_t *) l->texels + (size_t) y * l->w + x;
        uint32_t i;

        for (i = 0; i < w; i++)
            dst[i] = (uint8_t) (src[i >> 2] >> ((i & 3u) * 8u));
    }
}

static void
pkt_teximage(vgpu98_ogl_t *o, const uint32_t *p, uint32_t len)
{
    const uint32_t ctl    = (len >= VGPU98_OGL_LEN_TEXIMAGE) ? p[2] : 0;
    const uint32_t dims   = ctl & VGPU98_OGL_TEX_DIMS_MASK;
    const uint32_t level  = (ctl >> VGPU98_OGL_TEX_LEVEL_SHIFT) & 15u;
    const uint32_t sw     = (len >= VGPU98_OGL_LEN_TEXIMAGE) ? (p[3] & 0xffff) : 0;
    const uint32_t sh     = (len >= VGPU98_OGL_LEN_TEXIMAGE) ? (p[3] >> 16) : 0;
    const uint32_t px     = (len >= VGPU98_OGL_LEN_TEXIMAGE) ? (p[4] & 0xffff) : 0;
    const uint32_t py     = (len >= VGPU98_OGL_LEN_TEXIMAGE) ? (p[4] >> 16) : 0;
    const uint32_t base   = (len >= VGPU98_OGL_LEN_TEXIMAGE) ? p[5] : 0;
    const uint32_t bpt    = (base == VGPU98_OGL_BASE_INDEX) ? 1 : 4;
    /* A row's dwords in the data: an index level's bytes four a dword */
    const uint32_t rw     = (bpt == 1) ? (sw + 3) / 4 : sw;
    const uint32_t ok     = VGPU98_OGL_TEX_DIMS_MASK | (15u << VGPU98_OGL_TEX_LEVEL_SHIFT) | VGPU98_OGL_TEX_BORDER |
                            VGPU98_OGL_TEX_DEFINE | VGPU98_OGL_TEX_TEXELS;
    const uint64_t n      = (uint64_t) sw * sh;
    ogl_tex_t     *t;
    ogl_texlevel_t *l;
    uint32_t       y;

    if ((len < VGPU98_OGL_LEN_TEXIMAGE) || !group_find(o, p[0]) || !p[1] || (ctl & ~ok) || ((dims != 1) && (dims != 2)) ||
        (level >= VGPU98_OGL_LEVELS) || (base < VGPU98_OGL_BASE_ALPHA) || (base > VGPU98_OGL_BASE_INDEX) || !sw || !sh ||
        (sw > VGPU98_OGL_MAX_SIZE + 2) || (sh > VGPU98_OGL_MAX_SIZE + 2) || ((dims == 1) && (ctl & VGPU98_OGL_TEX_DEFINE) && (sh != 1))) {
        drop(o,
             (len < VGPU98_OGL_LEN_TEXIMAGE) ? "%u dwords"
             : !group_find(o, p[0])          ? "group %x, which it does not have"
                                             : "texture %x: control %x, %ux%u, base format %u",
             (len < VGPU98_OGL_LEN_TEXIMAGE) ? len : !group_find(o, p[0]) ? p[0] : p[1], ctl, sw, sh, base);
        return;
    }
    if (ctl & VGPU98_OGL_TEX_DEFINE) {
        uint32_t *texels;

        if ((uint64_t) len != VGPU98_OGL_LEN_TEXIMAGE + ((ctl & VGPU98_OGL_TEX_TEXELS) ? (uint64_t) rw * sh : 0)) {
            drop(o, "texture %x level %u: %u dwords for %ux%u", p[1], level, len, sw, sh);
            return;
        }
        t = tex_get(o, p[0], p[1]);
        if (!t || (o->tex_bytes - (uint64_t) t->level[level].w * t->level[level].h * t->level[level].bpt + n * bpt > TEX_BYTES_MAX)) {
            drop(o, "texture %x: the device's 16384 textures, or their 512 MB, all in use", p[1]);
            return;
        }
        /* Whole dwords, so that an index level's bytes may be read as them */
        texels = calloc((size_t) (n * bpt + 3) / 4, 4);
        if (!texels) {
            drop(o, "texture %x level %u: no memory for %ux%u", p[1], level, sw, sh);
            return;
        }
        l = &t->level[level];
        level_free(o, l);
        l->texels = texels;
        l->w      = sw;
        l->h      = sh;
        l->bpt    = bpt;
        if (ctl & VGPU98_OGL_TEX_TEXELS) {
            for (y = 0; y < sh; y++)
                row_in(l, 0, y, &p[VGPU98_OGL_LEN_TEXIMAGE + y * rw], sw);
        }
        o->tex_bytes += n * bpt;
        if (level == 0) {
            t->dims   = dims;
            t->base   = base;
            t->border = (ctl & VGPU98_OGL_TEX_BORDER) ? 1 : 0;
        }
        t->dirty = 1;
        o->st.tex_defined++;
        o->st.tex_defined_index += bpt == 1;
        o->st.tex_bytes_in += (uint64_t) len * 4;
        return;
    }
    /* Texels for a rectangle of a level defined before, of the same kind
       (indices or colours) */
    t = tex_find(o, p[0], p[1]);
    l = t ? &t->level[level] : NULL;
    if (!l || !l->texels || (l->bpt != bpt) || (px + sw > l->w) || (py + sh > l->h) ||
        ((uint64_t) len != VGPU98_OGL_LEN_TEXIMAGE + (uint64_t) rw * sh)) {
        drop(o, "texture %x level %u: %ux%u at %u,%u %s", p[1], level, sw, sh, px, py,
             (!l || !l->texels)                         ? "of a level not defined"
             : (l->bpt != bpt)                          ? "of the other kind (indices or colours)"
             : ((px + sw > l->w) || (py + sh > l->h)) ? "outside the level"
                                                        : "in a packet of the wrong length");
        return;
    }
    o->st.tex_rects++;
    o->st.tex_bytes_in += (uint64_t) len * 4;
    for (y = 0; y < sh; y++)
        row_in(l, px, py + y, &p[VGPU98_OGL_LEN_TEXIMAGE + y * rw], sw);
    /* The arena's copy patched where the texels changed, unless it is to be
       copied whole anyway */
    if (t->resident && !t->dirty) {
        if (!t->cx1[level]) {
            t->cx0[level] = px;
            t->cy0[level] = py;
            t->cx1[level] = px + sw;
            t->cy1[level] = py + sh;
        } else {
            if (px < t->cx0[level])
                t->cx0[level] = px;
            if (py < t->cy0[level])
                t->cy0[level] = py;
            if (px + sw > t->cx1[level])
                t->cx1[level] = px + sw;
            if (py + sh > t->cy1[level])
                t->cy1[level] = py + sh;
        }
    } else
        t->dirty = 1;
}

static void
pkt_texparam(vgpu98_ogl_t *o, const uint32_t *p, uint32_t len)
{
    ogl_tex_t *t = ((len >= VGPU98_OGL_LEN_TEXPARAM) && group_find(o, p[0])) ? tex_get(o, p[0], p[1]) : NULL;

    if (!t || ((p[2] & 15u) > VGPU98_OGL_WRAP_CLAMP_EDGE) || (((p[2] >> 4) & 15u) > VGPU98_OGL_WRAP_CLAMP_EDGE) ||
        ((p[3] & 15u) > VGPU98_OGL_FILTER_L_MIP_L) || (((p[3] >> 4) & 15u) > VGPU98_OGL_FILTER_LINEAR)) {
        drop(o, !t ? "texture %x: no group, or no texture to make" : "texture %x: wraps %x, filters %x", (len >= 2) ? p[1] : 0,
             (len >= 3) ? p[2] : 0, (len >= 4) ? p[3] : 0);
        return;
    }
    t->wraps   = p[2] & 0xffu;
    t->filters = p[3] & 0xffu;
    memcpy(t->border_color, &p[4], 16);
}

static void
pkt_texdelete(vgpu98_ogl_t *o, const uint32_t *p, uint32_t len)
{
    ogl_tex_t *t = (len >= VGPU98_OGL_LEN_TEXDELETE) ? tex_find(o, p[0], p[1]) : NULL;

    if (t)
        tex_free(o, t);
}

/* --- Commands: read-backs ------------------------------------------------------------------- */

/* Rows to video memory: each `bytes` long, `stride` apart from `offset`,
   inside video memory. 0 if they are not. */
static int
vram_rows_ok(const vgpu98_ogl_t *o, uint32_t offset, uint32_t stride, uint32_t rows, uint32_t bytes)
{
    return rows && bytes && (stride >= bytes) &&
           ((uint64_t) offset + (uint64_t) (rows - 1) * stride + bytes <= o->vram->size);
}

/* The rows stay off the frame shown: a staging surface found before a
   mode change may lie inside the new desktop (audit 4's note). */
static int
off_screen(vgpu98_ogl_t *o, uint32_t offset, uint32_t stride, uint32_t rows, uint32_t bytes)
{
    const uint64_t a0 = offset, a1 = (uint64_t) offset + (uint64_t) (rows - 1) * stride + bytes;
    int            on;
    uint32_t       start, pitch, height, bpp;

    vgpu98_gpu_visible(o->gpu, &on, &start, &pitch, &height, &bpp);
    return !on || (a1 <= start) || (a0 >= (uint64_t) start + (uint64_t) pitch * height);
}

/* READ: the rectangle of a buffer into video memory, the
   rows from the bottom, colour as the drawable keeps it, depth as 32-bit
   integers, stencil as bytes. The part outside the drawable is left as
   video memory has it (GL leaves those pixels undefined). */
static void
pkt_read(vgpu98_ogl_t *o, ogl_context_t *c, const uint32_t *p, uint32_t len, uint32_t generation)
{
    ogl_drawable_t     *d    = c->drawable ? drawable_find(o, c->drawable) : NULL;
    const uint32_t      what = (len >= VGPU98_OGL_LEN_READ) ? p[1] : 0;
    const int32_t       x    = (len >= VGPU98_OGL_LEN_READ) ? (int16_t) (p[2] & 0xffff) : 0;
    const int32_t       y    = (len >= VGPU98_OGL_LEN_READ) ? (int16_t) (p[2] >> 16) : 0;
    const uint32_t      w    = (len >= VGPU98_OGL_LEN_READ) ? (p[3] & 0xffff) : 0;
    const uint32_t      h    = (len >= VGPU98_OGL_LEN_READ) ? (p[3] >> 16) : 0;
    const uint32_t      offset = (len >= VGPU98_OGL_LEN_READ) ? p[4] : 0;
    const uint32_t      stride = (len >= VGPU98_OGL_LEN_READ) ? p[5] : 0;
    const int           dk   = d ? depth_kind(d->format) : 0;
    uint32_t            out_bpp, src_bpp;
    const ogl_image_t  *im;
    VkImageAspectFlags  aspect;
    int32_t             x0, y0, x1, y1, j;
    uint32_t            cw, ch;
    VkBuffer            buf;
    VkDeviceSize        off;
    uint8_t            *map;
    VkBufferImageCopy   bc;
    VkCommandBuffer     cb;

    if ((len < VGPU98_OGL_LEN_READ) || (what > VGPU98_OGL_READ_STENCIL) || !w || !h || !d) {
        drop(o, "what %u, %ux%u%s", what, w, h, d ? "" : " (no drawable bound)");
        return;
    }
    if (!d->fb[0])
        return;
    switch (what) {
        case VGPU98_OGL_READ_DEPTH:
            if (!dk) {
                drop(o, "depth read from a drawable without depth");
                return;
            }
            im      = &d->depth;
            aspect  = VK_IMAGE_ASPECT_DEPTH_BIT;
            out_bpp = 4;
            src_bpp = (dk == VGPU98_OGL_DEPTH_16) ? 2 : 4;
            break;
        case VGPU98_OGL_READ_STENCIL:
            if (dk != VGPU98_OGL_DEPTH_24S8) {
                drop(o, "stencil read from a drawable without stencil");
                return;
            }
            im      = &d->depth;
            aspect  = VK_IMAGE_ASPECT_STENCIL_BIT;
            out_bpp = 1;
            src_bpp = 1;
            break;
        default:
            im      = &d->color[((what == VGPU98_OGL_READ_BACK) && (d->format & VGPU98_OGL_DOUBLE)) ? BACK : FRONT];
            aspect  = VK_IMAGE_ASPECT_COLOR_BIT;
            out_bpp = color_kind(d->format) ? 4 : 2;
            src_bpp = out_bpp;
            break;
    }
    if (!vram_rows_ok(o, offset, stride, h, w * out_bpp) || !off_screen(o, offset, stride, h, w * out_bpp)) {
        drop(o, "%u rows at %x (stride %u): outside video memory, or on the screen", h, offset, stride);
        return;
    }
    o->st.reads++;
    /* The part inside the drawable */
    x0 = (x > 0) ? x : 0;
    y0 = (y > 0) ? y : 0;
    x1 = ((int64_t) x + w < (int64_t) d->w) ? x + (int32_t) w : (int32_t) d->w;
    y1 = ((int64_t) y + h < (int64_t) d->h) ? y + (int32_t) h : (int32_t) d->h;
    if ((x1 <= x0) || (y1 <= y0))
        return;
    cw  = (uint32_t) (x1 - x0);
    ch  = (uint32_t) (y1 - y0);
    map = vgpu98_r3d_glide_alloc(o->r3d, 0, (VkDeviceSize) cw * ch * src_bpp, 4, &buf, &off);
    if (!map)
        return;
    memset(&bc, 0, sizeof(bc));
    bc.bufferOffset                = off;
    bc.imageSubresource.aspectMask = aspect;
    bc.imageSubresource.layerCount = 1;
    bc.imageOffset.x               = x0;
    bc.imageOffset.y               = (int32_t) d->h - y1; /* GL's top row of it, in the image's rows */
    bc.imageExtent.width           = cw;
    bc.imageExtent.height          = ch;
    bc.imageExtent.depth           = 1;
    cb                             = outside(o);
    mem_barrier(o, cb);
    o->vk->vkCmdCopyImageToBuffer(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &bc);
    mem_barrier(o, cb);
    if (!vgpu98_r3d_glide_submit(o->r3d, 1) || vgpu98_gpu_reset_since(o->gpu, generation))
        return; /* a mode change meanwhile: video memory laid out anew */
    for (j = y0; j < y1; j++) {
        const uint8_t *src = map + (size_t) ((int32_t) d->h - 1 - j - ((int32_t) d->h - y1)) * cw * src_bpp;
        const uint32_t at  = offset + (uint32_t) (j - y) * stride + (uint32_t) (x0 - x) * out_bpp;
        uint8_t       *dst = o->vram->data + at;
        uint32_t       i;

        vgpu98_r3d_sync(o->r3d, at, cw * out_bpp, VGPU98_CB_2D);
        if ((what == VGPU98_OGL_READ_DEPTH) && (dk == VGPU98_OGL_DEPTH_16)) {
            for (i = 0; i < cw; i++) {
                const uint32_t v = (uint32_t) src[2 * i] | ((uint32_t) src[2 * i + 1] << 8);

                memcpy(dst + 4 * i, &v, 4);
            }
        } else if (what == VGPU98_OGL_READ_DEPTH) {
            for (i = 0; i < cw; i++) {
                uint32_t v;

                memcpy(&v, src + 4 * i, 4);
                if (o->depth24_float) {
                    float f;

                    memcpy(&f, &v, 4);
                    v = (uint32_t) floor((double) clamp01(f) * 16777215.0 + 0.5);
                } else
                    v &= 0xffffffu;
                memcpy(dst + 4 * i, &v, 4);
            }
        } else
            memcpy(dst, src, (size_t) cw * out_bpp);
        vgpu98_gpu_wrote(o->gpu, at, cw * out_bpp);
    }
}

/* TEXREAD: rows of a level's texels into video memory (from the host's
   copy, the truth): RGBA8, or an index level's bytes (2.6). */
static void
pkt_texread(vgpu98_ogl_t *o, const uint32_t *p, uint32_t len, uint32_t generation)
{
    ogl_tex_t            *t     = (len >= VGPU98_OGL_LEN_TEXREAD) ? tex_find(o, p[0], p[1]) : NULL;
    const ogl_texlevel_t *l     = (t && (p[2] < VGPU98_OGL_LEVELS)) ? &t->level[p[2]] : NULL;
    const uint32_t        first = (len >= VGPU98_OGL_LEN_TEXREAD) ? (p[5] & 0xffff) : 0;
    const uint32_t        rows  = (len >= VGPU98_OGL_LEN_TEXREAD) ? (p[5] >> 16) : 0;
    const uint32_t        rb    = l ? l->w * l->bpt : 0; /* a row's bytes */
    uint32_t              r;

    if (!l || !l->texels || (first + rows > l->h) || !vram_rows_ok(o, p[3], p[4], rows, rb) ||
        !off_screen(o, p[3], p[4], rows, rb)) {
        drop(o,
             (!l || !l->texels) ? "texture %x level %u: not defined"
                                : "texture %x level %u: rows %u to %u at %x outside the level or video memory",
             (len >= 2) ? p[1] : 0, (len >= 3) ? p[2] : 0, first, first + rows - 1, (len >= 4) ? p[3] : 0);
        return;
    }
    o->st.texreads++;
    if (vgpu98_gpu_reset_since(o->gpu, generation))
        return;
    for (r = 0; r < rows; r++) {
        const uint32_t at = p[3] + r * p[4];

        vgpu98_r3d_sync(o->r3d, at, rb, VGPU98_CB_2D);
        memcpy(o->vram->data + at, (const uint8_t *) l->texels + (size_t) (first + r) * rb, rb);
        vgpu98_gpu_wrote(o->gpu, at, rb);
    }
}

/* PALETTE (2.6): a texture's palette, or the context's shared one (texture
   0). Its arena copy is brought up to date by the next draw that uses it. */
static void
pkt_palette(vgpu98_ogl_t *o, ogl_context_t *c, const uint32_t *p, uint32_t len)
{
    const uint32_t base  = (len >= VGPU98_OGL_LEN_PALETTE) ? p[2] : 0;
    const uint32_t width = (len >= VGPU98_OGL_LEN_PALETTE) ? p[3] : 0;
    ogl_tex_t     *t     = NULL;

    if ((len < VGPU98_OGL_LEN_PALETTE) || (base < VGPU98_OGL_BASE_ALPHA) || (base > VGPU98_OGL_BASE_RGBA) || !width ||
        (width > VGPU98_OGL_PALETTE_MAX) || (width & (width - 1)) || (len != VGPU98_OGL_LEN_PALETTE + width) ||
        (p[1] && !(t = tex_get(o, c->group, p[1])))) {
        drop(o, "texture %x: base format %u, width %u in %u dwords", (len >= 2) ? p[1] : 0, base, width, len);
        return;
    }
    o->st.palettes++;
    if (t) {
        memcpy(t->pal, &p[VGPU98_OGL_LEN_PALETTE], (size_t) width * 4);
        t->pal_n     = width;
        t->pal_base  = base;
        t->pal_dirty = 1;
        return;
    }
    memcpy(c->spal, &p[VGPU98_OGL_LEN_PALETTE], (size_t) width * 4);
    c->spal_n     = width;
    c->spal_base  = base;
    c->spal_dirty = 1;
}

static void
pkt_clear(vgpu98_ogl_t *o, ogl_context_t *c, const uint32_t *p, uint32_t len)
{
    ogl_drawable_t   *d = c->drawable ? drawable_find(o, c->drawable) : NULL;
    const uint32_t    mask = (len >= VGPU98_OGL_LEN_CLEAR) ? p[1] : 0;
    const int         dk   = d ? depth_kind(d->format) : 0;
    const uint32_t    cmask = c->reg[VGPU98_OGL_R_COLOR_MASK] & 15u;
    VkClearAttachment att[2];
    VkClearRect       cr;
    uint32_t          bufs, n_ds = 0;
    int               depth_done = 0;
    int               b;

    if ((len < VGPU98_OGL_LEN_CLEAR) || (mask & ~7u)) {
        drop(o, "buffers %x in %u dwords", mask, len);
        return;
    }
    o->st.clears++;
    if (!d || !d->fb[0])
        return; /* nothing to clear: no drawable, or none on the host GPU */
    cr.rect           = scissor_rect(c, d);
    cr.baseArrayLayer = 0;
    cr.layerCount     = 1;
    if (!cr.rect.extent.width)
        return;
    /* A colour mask that keeps some of the drawable's components, or a
       stencil write mask that keeps some bits: the clear quad. */
    {
        const uint32_t fmask = format_mask(d->format);
        const uint32_t cm    = (mask & VGPU98_OGL_CLEAR_COLOR) ? (cmask & fmask) : 0;
        const uint32_t swm   = c->reg[VGPU98_OGL_R_STENCIL] >> 24;
        const int      sten  = (mask & VGPU98_OGL_CLEAR_STENCIL) && (dk == VGPU98_OGL_DEPTH_24S8) && swm;

        if ((cm && (cm != fmask)) || (sten && (swm != 0xffu))) {
            clear_quad(o, c, d, cm ? draw_buffers(c, d) : 0, cm,
                       dk && (mask & VGPU98_OGL_CLEAR_DEPTH) && ((c->reg[VGPU98_OGL_R_DEPTH] >> 8) & 1u), sten, p, &cr.rect);
            return;
        }
    }
    memset(att, 0, sizeof(att));
    /* Depth and stencil, as their write masks allow (GL clears through
       them); the stencil index to the buffer's bits. */
    if (dk) {
        if ((mask & VGPU98_OGL_CLEAR_DEPTH) && ((c->reg[VGPU98_OGL_R_DEPTH] >> 8) & 1u)) {
            float z = clamp01(f32(p[6]));

            if ((dk == VGPU98_OGL_DEPTH_24S8) && o->depth24_float)
                z = (float) (floor((double) z * 16777215.0 + 0.5) / 16777215.0);
            att[1].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
            att[1].clearValue.depthStencil.depth = z;
        }
        if ((mask & VGPU98_OGL_CLEAR_STENCIL) && (dk == VGPU98_OGL_DEPTH_24S8) && (c->reg[VGPU98_OGL_R_STENCIL] >> 24)) {
            att[1].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            att[1].clearValue.depthStencil.stencil = p[7] & 0xffu;
        }
        if (att[1].aspectMask)
            n_ds = 1;
    }
    bufs = (mask & VGPU98_OGL_CLEAR_COLOR) && cmask ? draw_buffers(c, d) : 0;
    att[0].aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
    att[0].colorAttachment = 0;
    att[0].clearValue.color.float32[0] = clamp01(f32(p[2]));
    att[0].clearValue.color.float32[1] = clamp01(f32(p[3]));
    att[0].clearValue.color.float32[2] = clamp01(f32(p[4]));
    att[0].clearValue.color.float32[3] =
        ((d->format & VGPU98_OGL_COLOR_MASK) == VGPU98_OGL_COLOR_ARGB8888) ? clamp01(f32(p[5])) : 1.0f;
    for (b = 0; b < 2; b++) {
        if (bufs & (1u << b)) {
            VkCommandBuffer cb = inside(o, d, b);

            if (!depth_done && n_ds) {
                o->vk->vkCmdClearAttachments(cb, 2, att, 1, &cr);
                depth_done = 1;
            } else
                o->vk->vkCmdClearAttachments(cb, 1, att, 1, &cr);
        }
    }
    if (!depth_done && n_ds)
        o->vk->vkCmdClearAttachments(inside(o, d, BACK - !(d->format & VGPU98_OGL_DOUBLE)), 1, &att[1], 1, &cr);
}

/* --- Commands: swaps --------------------------------------------------------------- */

typedef struct ogl_rect_t {
    int32_t x0, y0, x1, y1; /* on the screen, exclusive */
} ogl_rect_t;

/* The SWAP's rectangles, each inside the drawable (at the client area's
   place ox, oy) and inside video memory as the screen lays it out. 0 if one
   is not. */
static int
swap_rects(vgpu98_ogl_t *o, const ogl_drawable_t *d, const uint32_t *p, uint32_t n, int32_t ox, int32_t oy,
           uint32_t offset, uint32_t pitch, uint32_t bytes, ogl_rect_t *out)
{
    uint32_t i;

    for (i = 0; i < n; i++) {
        ogl_rect_t *r = &out[i];
        uint64_t    end;

        r->x0 = (int16_t) (p[2 * i] & 0xffff);
        r->y0 = (int16_t) (p[2 * i] >> 16);
        r->x1 = (int16_t) (p[2 * i + 1] & 0xffff);
        r->y1 = (int16_t) (p[2 * i + 1] >> 16);
        if ((r->x0 < 0) || (r->y0 < 0) || (r->x1 <= r->x0) || (r->y1 <= r->y0) || (r->x0 < ox) || (r->y0 < oy) ||
            (r->x1 > ox + (int32_t) d->w) || (r->y1 > oy + (int32_t) d->h) || ((uint64_t) r->x1 * bytes > pitch))
            return 0;
        end = (uint64_t) offset + (uint64_t) (r->y1 - 1) * pitch + (uint64_t) r->x1 * bytes;
        if (end > o->vram->size)
            return 0;
    }
    return 1;
}

/* LOAD: the front image takes the window's visible pixels from video memory
   (newer DirectX pixels there copied back first). */
static void
swap_load(vgpu98_ogl_t *o, ogl_drawable_t *d, const ogl_rect_t *rects, uint32_t n, int32_t ox, int32_t oy, uint32_t offset,
          uint32_t pitch, uint32_t bytes)
{
    uint32_t i;

    for (i = 0; i < n; i++) {
        const ogl_rect_t *r    = &rects[i];
        const uint32_t    w    = (uint32_t) (r->x1 - r->x0), h = (uint32_t) (r->y1 - r->y0);
        const uint32_t    row  = (w * bytes + 3u) & ~3u;
        const uint32_t    at   = offset + (uint32_t) r->y0 * pitch + (uint32_t) r->x0 * bytes;
        VkBuffer          buf;
        VkDeviceSize      off;
        uint8_t          *map;
        VkBufferImageCopy bc;
        VkCommandBuffer   c;
        uint32_t          y;

        vgpu98_r3d_sync(o->r3d, at, (h - 1) * pitch + w * bytes, VGPU98_CB_2D);
        map = vgpu98_r3d_glide_alloc(o->r3d, 0, (VkDeviceSize) row * h, 4, &buf, &off);
        if (!map)
            return; /* 3D stopped */
        for (y = 0; y < h; y++)
            memcpy(map + (size_t) y * row, o->vram->data + at + (size_t) y * pitch, (size_t) w * bytes);
        memset(&bc, 0, sizeof(bc));
        bc.bufferOffset                = off;
        bc.bufferRowLength             = row / bytes;
        bc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bc.imageSubresource.layerCount = 1;
        bc.imageOffset.x               = r->x0 - ox;
        bc.imageOffset.y               = r->y0 - oy;
        bc.imageExtent.width           = w;
        bc.imageExtent.height          = h;
        bc.imageExtent.depth           = 1;
        c                              = outside(o);
        mem_barrier(o, c);
        o->vk->vkCmdCopyBufferToImage(c, buf, d->color[FRONT].image, VK_IMAGE_LAYOUT_GENERAL, 1, &bc);
        mem_barrier(o, c);
    }
}

/* SHOW: the front image's pixels under the rectangles go into video memory.
   They are read back to the host's memory (the simple path), then
   written row by row; a reset meanwhile (a mode change: video memory laid out
   anew) writes nothing. With a swap interval (WGL_EXT_swap_control) the
   image is read back first, then the interval's vertical blanks waited for,
   so the write lands at the start of the blank. The read-back is copied out
   of the staging buffer at once: the wait and the coherency rule's copies
   back may begin other slots, and reuse it. The writes go under the display's layer
   lock, so the host's display composes all of the old frame or all of the
   new. */
/* A SHOW's wait for its swap interval's vertical blanks, timed for the
   report */
static void
vblank_wait(vgpu98_ogl_t *o, ogl_drawable_t *d, uint32_t interval, uint32_t generation)
{
    const uint32_t t0 = plat_get_ticks();

    vgpu98_gpu_wait_vblanks_since(o->gpu, interval, &d->shown_vblank, generation);
    if (interval) {
        o->st.vblank_waits++;
        o->st.vblank_ms += plat_get_ticks() - t0;
    }
}

static void
swap_show(vgpu98_ogl_t *o, ogl_drawable_t *d, const ogl_rect_t *rects, uint32_t n, int32_t ox, int32_t oy, uint32_t offset,
          uint32_t pitch, uint32_t bytes, uint32_t interval, uint32_t generation)
{
    int32_t           bx0 = INT32_MAX, by0 = INT32_MAX, bx1 = INT32_MIN, by1 = INT32_MIN;
    uint32_t          w, h, row, i;
    VkBuffer          buf;
    VkDeviceSize      off;
    uint8_t          *map, *kept = NULL;
    VkBufferImageCopy bc;
    VkCommandBuffer   c;

    /* The report's frames: between two SHOWs of the drawable */
    {
        const uint32_t now = plat_get_ticks();

        if (d->shown_ms && ((uint32_t) (now - d->shown_ms) > o->st.frame_worst_ms))
            o->st.frame_worst_ms = now - d->shown_ms;
        d->shown_ms = now ? now : 1;
        o->st.shows++;
    }
    for (i = 0; i < n; i++) {
        if (rects[i].x0 < bx0)
            bx0 = rects[i].x0;
        if (rects[i].y0 < by0)
            by0 = rects[i].y0;
        if (rects[i].x1 > bx1)
            bx1 = rects[i].x1;
        if (rects[i].y1 > by1)
            by1 = rects[i].y1;
    }
    if (!n) {
        /* Nothing of the window shown: the swap still keeps its pace. */
        vblank_wait(o, d, interval, generation);
        return;
    }
    w   = (uint32_t) (bx1 - bx0);
    h   = (uint32_t) (by1 - by0);
    row = (w * bytes + 3u) & ~3u;
    map = vgpu98_r3d_glide_alloc(o->r3d, 0, (VkDeviceSize) row * h, 4, &buf, &off);
    if (!map)
        return;
    memset(&bc, 0, sizeof(bc));
    bc.bufferOffset                = off;
    bc.bufferRowLength             = row / bytes;
    bc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bc.imageSubresource.layerCount = 1;
    bc.imageOffset.x               = bx0 - ox;
    bc.imageOffset.y               = by0 - oy;
    bc.imageExtent.width           = w;
    bc.imageExtent.height          = h;
    bc.imageExtent.depth           = 1;
    c                              = outside(o);
    mem_barrier(o, c);
    o->vk->vkCmdCopyImageToBuffer(c, d->color[FRONT].image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &bc);
    mem_barrier(o, c);
    if (!vgpu98_r3d_glide_submit(o->r3d, 1) || vgpu98_gpu_reset_since(o->gpu, generation))
        return;
    kept = malloc((size_t) row * h);
    if (!kept)
        return;
    memcpy(kept, map, (size_t) row * h);
    map = kept;
    vblank_wait(o, d, interval, generation);
    if (vgpu98_gpu_reset_since(o->gpu, generation)) {
        free(kept);
        return; /* a mode change during the wait: video memory laid out anew */
    }
    /* Newer DirectX pixels under the rectangles copied back first (the
       coherency rule), then the writes in one go, then the pages marked. */
    for (i = 0; i < n; i++) {
        const ogl_rect_t *r = &rects[i];

        vgpu98_r3d_sync(o->r3d, offset + (uint32_t) r->y0 * pitch + (uint32_t) r->x0 * bytes,
                        (uint32_t) (r->y1 - r->y0 - 1) * pitch + (uint32_t) (r->x1 - r->x0) * bytes, VGPU98_CB_2D);
    }
    /* Under the display's layer lock when there is a display (its own
       setup may have failed while the host GPU works) */
    if (o->disp) {
        const int step = vgpu98_gpu_step(o->gpu, VGPU98_STEP_LAYER);

        vgpu98_disp_layer_lock(o->disp);
        vgpu98_gpu_step(o->gpu, step);
    }
    for (i = 0; i < n; i++) {
        const ogl_rect_t *r  = &rects[i];
        const uint32_t    rw = (uint32_t) (r->x1 - r->x0), rh = (uint32_t) (r->y1 - r->y0);
        const uint32_t    at = offset + (uint32_t) r->y0 * pitch + (uint32_t) r->x0 * bytes;
        uint32_t          y;

        for (y = 0; y < rh; y++)
            memcpy(o->vram->data + at + (size_t) y * pitch,
                   map + (size_t) (r->y0 - by0 + (int32_t) y) * row + (size_t) (r->x0 - bx0) * bytes, (size_t) rw * bytes);
    }
    if (o->disp)
        vgpu98_disp_layer_unlock(o->disp);
    for (i = 0; i < n; i++) {
        const ogl_rect_t *r = &rects[i];

        vgpu98_gpu_wrote(o->gpu, offset + (uint32_t) r->y0 * pitch + (uint32_t) r->x0 * bytes,
                         (uint32_t) (r->y1 - r->y0 - 1) * pitch + (uint32_t) (r->x1 - r->x0) * bytes);
    }
    /* Diagnostics "OpenGL": a capture of what this SHOW wrote, when one is
       due */
    if (o->logging)
        capture_check(o, d, kept, row, w, h, bytes);
    free(kept);
}

#    define SWAP_RECTS_MAX 1024

static void
pkt_swap(vgpu98_ogl_t *o, const uint32_t *p, uint32_t len, uint32_t generation)
{
    ogl_drawable_t *d      = (len >= VGPU98_OGL_LEN_SWAP) ? drawable_find(o, p[0]) : NULL;
    const uint32_t  flags  = (len >= VGPU98_OGL_LEN_SWAP) ? p[1] : 0;
    const uint32_t  offset = (len >= VGPU98_OGL_LEN_SWAP) ? p[2] : 0;
    const uint32_t  pitch  = (len >= VGPU98_OGL_LEN_SWAP) ? p[3] : 0;
    const uint32_t  bpp    = (len >= VGPU98_OGL_LEN_SWAP) ? p[4] : 0;
    const int32_t   ox     = (len >= VGPU98_OGL_LEN_SWAP) ? (int16_t) (p[5] & 0xffff) : 0;
    const int32_t   oy     = (len >= VGPU98_OGL_LEN_SWAP) ? (int16_t) (p[5] >> 16) : 0;
    uint32_t        n      = (len >= VGPU98_OGL_LEN_SWAP) ? p[6] : 0;
    const uint32_t  interval = (flags & VGPU98_OGL_SWAP_INTERVAL_MASK) >> VGPU98_OGL_SWAP_INTERVAL_SHIFT;
    ogl_rect_t     *rects;
    uint32_t        bytes;

    if (!d || (flags & ~(7u | VGPU98_OGL_SWAP_INTERVAL_MASK | VGPU98_OGL_SWAP_OVERRIDE)) ||
        (interval > VGPU98_OGL_SWAP_INTERVAL_MAX) ||
        (n > SWAP_RECTS_MAX) || (n > (len - VGPU98_OGL_LEN_SWAP) / 2) || ((bpp != 16) && (bpp != 32))) {
        drop(o, !d ? "drawable %x, which it does not have" : "drawable %x: flags %x, %u rectangles, %u bits a pixel",
             (len >= 1) ? p[0] : 0, flags, n, bpp);
        return;
    }
    if (!d->fb[0])
        return; /* no images on the host GPU */
    bytes = bpp / 8;
    /* The window on a screen of another depth (a mode change since): no
       transfer, as video memory is laid out for another format. */
    if ((flags & (VGPU98_OGL_SWAP_LOAD | VGPU98_OGL_SWAP_SHOW)) && (bytes != (color_kind(d->format) ? 4u : 2u)))
        n = 0;
    /* A swap recorded before a mode change: the screen it names is not the
       one shown now (audit 1, item 6), and its rectangles are not written. */
    if ((flags & (VGPU98_OGL_SWAP_LOAD | VGPU98_OGL_SWAP_SHOW)) && n) {
        int      on;
        uint32_t start, vpitch, vheight, vbpp, i;

        vgpu98_gpu_visible(o->gpu, &on, &start, &vpitch, &vheight, &vbpp);
        if (!on || (start != offset) || (vpitch != pitch) || (vbpp != bpp))
            n = 0;
        for (i = 0; i < n; i++) {
            if ((p[VGPU98_OGL_LEN_SWAP + 2 * i + 1] >> 16) > vheight)
                n = 0;
        }
    }
    rects = n ? malloc(n * sizeof(ogl_rect_t)) : NULL;
    if (n && (!rects || !swap_rects(o, d, &p[VGPU98_OGL_LEN_SWAP], n, ox, oy, offset, pitch, bytes, rects))) {
        free(rects);
        drop(o, "drawable %x: a rectangle outside the screen or video memory", p[0]);
        return;
    }
    if (flags & VGPU98_OGL_SWAP_LOAD)
        swap_load(o, d, rects, n, ox, oy, offset, pitch, bytes);
    if ((flags & VGPU98_OGL_SWAP_COPY) && (d->format & VGPU98_OGL_DOUBLE)) {
        VkCommandBuffer c = outside(o);
        VkImageCopy     ic;

        memset(&ic, 0, sizeof(ic));
        ic.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ic.srcSubresource.layerCount = 1;
        ic.dstSubresource            = ic.srcSubresource;
        ic.extent.width              = d->w;
        ic.extent.height             = d->h;
        ic.extent.depth              = 1;
        mem_barrier(o, c);
        o->vk->vkCmdCopyImage(c, d->color[BACK].image, VK_IMAGE_LAYOUT_GENERAL, d->color[FRONT].image,
                              VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
        mem_barrier(o, c);
    }
    /* The V-sync overrides on the interval the ICD asked for, on the SHOWs
       it marks as a full-screen program's own: never a
       window on the desktop's or a front-buffer show's. */
    if (flags & VGPU98_OGL_SWAP_SHOW)
        swap_show(o, d, rects, n, ox, oy, offset, pitch, bytes,
                  (flags & VGPU98_OGL_SWAP_OVERRIDE) ? vgpu98_gpu_vsync_interval(o->gpu, VGPU98_VSYNC_OGL_SHIFT, interval)
                                                     : interval,
                  generation);
    free(rects);
}

/* --- Commands ---------------------------------------------------------------------- */

void
vgpu98_ogl_packet(vgpu98_ogl_t *o, uint32_t op, const uint32_t *p, uint32_t len, uint32_t generation)
{
    ogl_context_t *c;

    o->op = op;
    if (o->logging)
        cmd_note(o, op, p, len);
    if (vgpu98_r3d_broken(o->r3d)) {
        drop(o, "3D stopped on the host GPU");
        return;
    }
    /* The objects a packet names were used now (their LRU clock: objects of
       a process that ended without its detach are taken back when a kind
       is full). */
    o->clock++;
    if (len >= 1) {
        ogl_group_t    *g;
        ogl_drawable_t *d;

        switch (op) {
            case VGPU98_OGL_GROUP:
            case VGPU98_OGL_TEXIMAGE:
            case VGPU98_OGL_TEXPARAM:
            case VGPU98_OGL_TEXDELETE:
            case VGPU98_OGL_TEXREAD:
                if ((g = group_find(o, p[0])) != NULL)
                    g->used = o->clock;
                break;
            case VGPU98_OGL_DRAWABLE:
            case VGPU98_OGL_SWAP:
                if ((d = drawable_find(o, p[0])) != NULL)
                    d->used = o->clock;
                break;
            default:
                if ((c = context_find(o, p[0])) != NULL) {
                    c->used = o->clock;
                    if ((g = group_find(o, c->group)) != NULL)
                        g->used = o->clock;
                    if ((d = drawable_find(o, c->drawable)) != NULL)
                        d->used = o->clock;
                }
                break;
        }
    }
    switch (op) {
        case VGPU98_OGL_GROUP:
            pkt_group(o, p, len);
            return;
        case VGPU98_OGL_CONTEXT:
            pkt_context(o, p, len);
            return;
        case VGPU98_OGL_DRAWABLE:
            pkt_drawable(o, p, len);
            return;
        case VGPU98_OGL_BIND:
            pkt_bind(o, p, len);
            return;
        case VGPU98_OGL_SWAP:
            pkt_swap(o, p, len, generation);
            return;
        case VGPU98_OGL_TEXIMAGE:
            pkt_teximage(o, p, len);
            return;
        case VGPU98_OGL_TEXPARAM:
            pkt_texparam(o, p, len);
            return;
        case VGPU98_OGL_TEXDELETE:
            pkt_texdelete(o, p, len);
            return;
        case VGPU98_OGL_TEXREAD:
            pkt_texread(o, p, len, generation);
            return;
        default:
            break;
    }
    c = context_find(o, p[0]);
    switch (op) {
        case VGPU98_OGL_STATE:
            if (!c || (len < VGPU98_OGL_LEN_STATE))
                drop(o, !c ? "context %x, which it does not have" : "%u dwords", !c ? p[0] : len);
            else
                pkt_state(o, c, p, len);
            break;
        case VGPU98_OGL_CLEAR:
            if (!c)
                drop(o, "context %x, which it does not have", p[0]);
            else
                pkt_clear(o, c, p, len);
            break;
        case VGPU98_OGL_DRAW:
            if (!c)
                drop(o, "context %x, which it does not have", p[0]);
            else
                pkt_draw(o, c, p, len);
            break;
        case VGPU98_OGL_PIXELS:
            if (!c)
                drop(o, "context %x, which it does not have", p[0]);
            else
                pkt_pixels(o, c, p, len);
            break;
        case VGPU98_OGL_READ:
            if (!c)
                drop(o, "context %x, which it does not have", p[0]);
            else
                pkt_read(o, c, p, len, generation);
            break;
        case VGPU98_OGL_PALETTE:
            if (!c)
                drop(o, "context %x, which it does not have", p[0]);
            else
                pkt_palette(o, c, p, len);
            break;
        default:
            drop(o, "not an OpenGL command");
            break;
    }
}

void
vgpu98_ogl_reset(vgpu98_ogl_t *o, int end)
{
    int i;

    if (!o)
        return;
    /* The renderer's reset submitted or dropped its recording: no render
       pass of ours is open, and what the slots held is not counted on. */
    o->pass_open  = 0;
    o->pass_d     = NULL;
    o->bound_pipe = VK_NULL_HANDLE;
    o->set        = VK_NULL_HANDLE;
    o->chunk_map  = NULL;
    /* An upload into the arena may have been dropped with the recording
       (a shared palette's too). */
    for (i = 0; i < VGPU98_OGL_CONTEXTS; i++) {
        if (o->ctx[i]) {
            o->ctx[i]->regs_new   = 1;
            o->ctx[i]->spal_dirty = 1;
        }
    }
    {
        uint32_t b;

        for (b = 0; b < TEX_BUCKETS; b++) {
            ogl_tex_t *t;

            for (t = o->tex[b]; t; t = t->next)
                t->dirty = 1;
        }
    }
    if (!end)
        return;
    for (i = 0; i < VGPU98_OGL_DRAWABLES; i++)
        drawable_end(o, &o->draw[i]);
    for (i = 0; i < VGPU98_OGL_GROUPS; i++)
        group_end(o, &o->groups[i]);
    for (i = 0; i < VGPU98_OGL_CONTEXTS; i++) {
        if (o->ctx[i])
            context_end(o, i);
    }
}

/* --- Diagnostics "OpenGL" -------------------------- */

/* Every 5 seconds while OpenGL works: what was drawn and how, the textures
   and the arena, the commands dropped; the findings of the period. */
void
vgpu98_ogl_diag(vgpu98_ogl_t *o)
{
    const uint32_t now = plat_get_ticks();
    ogl_stats_t   *st;

    if (!o)
        return;
    st         = &o->st;
    o->logging = vgpu98_gpu_logging(o->gpu, VGPU98_LOG_OPENGL);
    if (!o->report_ms) {
        memset(st, 0, sizeof(*st));
        o->report_ms = now;
        return;
    }
    if ((uint32_t) (now - o->report_ms) < REPORT_MS)
        return;
    if (st->shows || st->draws || st->pixels || st->clears || st->tex_defined || st->tex_rects || st->dropped) {
        const double secs = (now - o->report_ms) / 1000.0;
        char         ops[160];
        size_t       at = 0;
        VkDeviceSize spare = 0;
        uint32_t     i;

        ops[0] = 0;
        for (i = 0; i < 16; i++) {
            if (st->dropped_op[i] && (at + 24 < sizeof(ops)))
                at += (size_t) snprintf(ops + at, sizeof(ops) - at, "%s%s %u", at ? ", " : " (", op_name(VGPU98_OGL_FIRST + i),
                                        st->dropped_op[i]);
        }
        if (at)
            snprintf(ops + at, sizeof(ops) - at, ")");
        for (i = 0; i < o->n_free; i++)
            spare += o->free[i].n;
        /* Two lines: the drawing; then state, read-backs and drops */
        vgpu98_gpu_log(o->gpu, VGPU98_LOG_OPENGL,
                       "VGPU98 OpenGL: %.1f s: %llu SHOWs (worst frame %u ms), %llu waits for vertical blanks (%llu ms); %llu DRAWs "
                       "of %llu vertices (%llu with unit 1, %llu by the ICD's path), %llu PIXELS, %llu CLEARs\n",
                       secs, (unsigned long long) st->shows, st->frame_worst_ms, (unsigned long long) st->vblank_waits,
                       (unsigned long long) st->vblank_ms, (unsigned long long) st->draws, (unsigned long long) st->vertices,
                       (unsigned long long) st->draws_tex1, (unsigned long long) st->draws_clip, (unsigned long long) st->pixels,
                       (unsigned long long) st->clears);
        vgpu98_gpu_log(o->gpu, VGPU98_LOG_OPENGL,
                       "VGPU98 OpenGL:   %llu STATEs (%llu registers); %llu READs, %llu TEXREADs (each a stall); %llu commands "
                       "dropped%s\n",
                       (unsigned long long) st->states, (unsigned long long) st->state_words, (unsigned long long) st->reads,
                       (unsigned long long) st->texreads, (unsigned long long) st->dropped, ops);
        /* The textures in two lines: what came in and what went to the host
           GPU; then what is kept */
        if (st->tex_defined || st->tex_rects || st->copies || st->patches || st->palettes || st->pal_copies || st->evicted) {
            vgpu98_gpu_log(o->gpu, VGPU98_LOG_OPENGL,
                           "VGPU98 OpenGL:   textures: %llu levels defined (%llu of indices), %llu rectangles, %llu KB received, "
                           "%llu palettes; to the host GPU %llu whole (%llu KB), %llu patched (%llu KB), %llu palettes\n",
                           (unsigned long long) st->tex_defined, (unsigned long long) st->tex_defined_index,
                           (unsigned long long) st->tex_rects, (unsigned long long) (st->tex_bytes_in >> 10),
                           (unsigned long long) st->palettes, (unsigned long long) st->copies,
                           (unsigned long long) (st->copy_bytes >> 10), (unsigned long long) st->patches,
                           (unsigned long long) (st->patch_bytes >> 10), (unsigned long long) st->pal_copies);
            vgpu98_gpu_log(o->gpu, VGPU98_LOG_OPENGL,
                           "VGPU98 OpenGL:   textures kept: %u (%llu KB); %llu dropped for room, %llu copied in again unchanged; "
                           "the arena %llu of %llu MB in use\n",
                           o->n_tex, (unsigned long long) (o->tex_bytes >> 10), (unsigned long long) st->evicted,
                           (unsigned long long) st->again, (unsigned long long) ((o->arena.size - spare) >> 20),
                           (unsigned long long) (o->arena.size >> 20));
        }
        /* One a texture unit a draw, so beside the DRAWs it can be nearly
           twice as many (item 4) */
        if (st->wrap_repeat || st->wrap_clamp || st->wrap_edge)
            vgpu98_gpu_log(o->gpu, VGPU98_LOG_OPENGL,
                           "VGPU98 OpenGL:   textures drawn (one a unit a draw): %llu repeating, %llu with GL_CLAMP (sampled %s), "
                           "%llu with clamp-to-edge asked for\n",
                           (unsigned long long) st->wrap_repeat, (unsigned long long) st->wrap_clamp,
                           o->clamp_exact ? "exactly" : "as clamp-to-edge", (unsigned long long) st->wrap_edge);
        /* The slowest pipeline named only when one was made (item 2) */
        if (st->pipes)
            vgpu98_gpu_log(o->gpu, VGPU98_LOG_OPENGL,
                           "VGPU98 OpenGL:   %llu pipelines made (%u kept, the slowest %u ms), %llu render passes ended early, for "
                           "work outside them\n",
                           (unsigned long long) st->pipes, o->n_pipes, st->pipe_worst_ms, (unsigned long long) st->pass_breaks);
        else if (st->pass_breaks)
            vgpu98_gpu_log(o->gpu, VGPU98_LOG_OPENGL, "VGPU98 OpenGL:   %llu render passes ended early, for work outside them\n",
                           (unsigned long long) st->pass_breaks);
        /* Findings past real problems' thresholds. A blank screen: SHOWs
           with nothing drawn; with clears alone, three periods in a row
           (15 s), so a loading screen's few seconds are not a finding. */
        if ((st->shows >= 10) && !st->draws && !st->pixels && !st->clears) {
            finding(o, FIND_BLANK, "blank screen: %llu SHOWs in %.0f s with nothing drawn", (unsigned long long) st->shows, secs);
            o->blank_run = 0;
        } else if ((st->shows >= 10) && !st->draws && !st->pixels) {
            o->blank_shows += st->shows;
            o->blank_secs += secs;
            if (++o->blank_run >= 3) {
                finding(o, FIND_BLANK, "blank screen: %llu SHOWs in %.0f s with nothing drawn but clears",
                        (unsigned long long) o->blank_shows, o->blank_secs);
                o->blank_run = 0;
            }
        } else
            o->blank_run = 0;
        if (!o->blank_run) {
            o->blank_shows = 0;
            o->blank_secs  = 0.0;
        }
        if (st->again >= 100)
            finding(o, FIND_THRASH,
                    "texture memory thrashing: %llu textures copied to the host GPU again in %.0f s, unchanged, after being "
                    "dropped for room (the arena %llu MB)",
                    (unsigned long long) st->again, secs, (unsigned long long) (o->arena.size >> 20));
        if (st->pal_default >= 100)
            finding(o, FIND_PALETTE,
                    "missing palette: %llu draws in %.0f s with an index texture whose palette has one entry, drawn in one "
                    "colour (white unless set)",
                    (unsigned long long) st->pal_default, secs);
    } else {
        /* A period without OpenGL's work ends a run of blank ones. */
        o->blank_run   = 0;
        o->blank_shows = 0;
        o->blank_secs  = 0.0;
    }
    memset(st, 0, sizeof(*st));
    o->report_ms = now;
}

/* An 86Box screenshot at Diagnostics "OpenGL" (the emulation's thread): a
   capture of the frame shown at the next SHOW. */
void
vgpu98_ogl_screenshot(vgpu98_ogl_t *o)
{
    if (o)
        atomic_store(&o->shot_asked, 1);
}

/* A packet kept for the capture's frame. */
static void
cmd_note(vgpu98_ogl_t *o, uint32_t op, const uint32_t *p, uint32_t len)
{
    ogl_cmd_t *k;
    uint32_t   i;

    if (!o->cmds)
        o->cmds = calloc(CMDS_KEPT, sizeof(ogl_cmd_t));
    if (!o->cmds)
        return;
    k      = &o->cmds[o->n_cmds % CMDS_KEPT];
    k->op  = op;
    k->len = len;
    for (i = 0; i < 6; i++)
        k->p[i] = (i < len) ? p[i] : 0;
    o->n_cmds++;
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

/* A capture's pictures and text, written on a thread of its own so the
   GPU thread goes on. */
typedef struct ogl_capture_t {
    vgpu98_gpu_t *gpu;
    char          dir[1024];
    char          why[64];
    uint32_t      serial;
    uint8_t      *pic[3]; /* what the SHOW wrote, unit 0's and unit 1's textures (RGBA) */
    uint32_t      w[3], h[3];
    char         *text;
} ogl_capture_t;

static atomic_int capture_busy;

static void
capture_thread(void *priv)
{
    ogl_capture_t     *k = (ogl_capture_t *) priv;
    static const char *names[3] = { "front.png", "unit0.png", "unit1.png" };
    char               path[1024];
    FILE              *f;
    uint32_t           i, files = 0;

    for (i = 0; i < 3; i++) {
        if (k->pic[i]) {
            path_append_filename(path, k->dir, names[i]);
            files += capture_png(path, k->pic[i], k->w[i], k->h[i]);
        }
        free(k->pic[i]);
    }
    if (k->text) {
        path_append_filename(path, k->dir, "commands.txt");
        f = plat_fopen(path, "w");
        if (f) {
            fputs(k->text, f);
            files += fclose(f) == 0;
        }
        free(k->text);
    }
    vgpu98_gpu_log(k->gpu, VGPU98_LOG_EVENTS, "VGPU98 OpenGL: capture %u (%s): %u files in %s\n", k->serial, k->why, files, k->dir);
    free(k);
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

/* A unit's texture drawn with, its level 0 as RGBA from the host's copy
   (an index level through the palette it draws with): NULL for none. */
static uint8_t *
capture_texture(const vgpu98_ogl_t *o, const ogl_context_t *c, int unit, uint32_t *w, uint32_t *h)
{
    const uint32_t        id = c->reg[unit ? VGPU98_OGL_R_UNIT1 + VGPU98_OGL_U_TEXTURE_ID : VGPU98_OGL_R_TEXTURE_ID];
    ogl_tex_t            *t  = id ? tex_find((vgpu98_ogl_t *) o, c->group, id) : NULL;
    const ogl_texlevel_t *l  = t ? &t->level[0] : NULL;
    uint8_t              *rgba;
    uint32_t              i, n;

    if (!l || !l->texels || !l->w || !l->h)
        return NULL;
    n    = l->w * l->h;
    rgba = malloc((size_t) n * 4);
    if (!rgba)
        return NULL;
    if (l->bpt == 4)
        memcpy(rgba, l->texels, (size_t) n * 4); /* RGBA8, red in the low byte: the bytes in order */
    else {
        const int       shared = (c->reg[VGPU98_OGL_R_ENABLES] & VGPU98_OGL_EN_SHARED_PALETTE) != 0;
        const uint32_t *pal    = shared ? c->spal : t->pal;
        const uint32_t  mask   = (shared ? c->spal_n : t->pal_n) - 1;

        for (i = 0; i < n; i++)
            memcpy(rgba + (size_t) i * 4, &pal[((const uint8_t *) l->texels)[i] & mask], 4);
    }
    *w = l->w;
    *h = l->h;
    return rgba;
}

/* A capture of what a SHOW wrote (at a finding, 10 seconds after the
   drawable's first SHOW, and at an 86Box screenshot): its
   pixels, the textures its last context drew with, the registers and the
   frame's packets, in a folder of the captures' folder. The pixels are
   taken here; the files go on a thread of their own. 0 while the last
   capture's files are still being written (the next SHOW tries again). */
static int
capture(vgpu98_ogl_t *o, ogl_drawable_t *d, const uint8_t *px, uint32_t row, uint32_t w, uint32_t h, uint32_t bytes,
        const char *why)
{
    const char          *base = vgpu98_gpu_capture_dir(o->gpu);
    const ogl_context_t *c    = d->last_ctx ? context_find(o, d->last_ctx) : NULL;
    char                 dir[1024], name[96], path[1024];
    time_t               now = time(NULL);
    struct tm           *tm  = localtime(&now);
    ogl_capture_t       *k;
    uint32_t             i, x, y;
    size_t               n = 0, cap = 8192;

    if (atomic_load(&capture_busy))
        return 0;
    if (!base || !base[0])
        return 1;
    if (tm)
        snprintf(name, sizeof(name), "%04d-%02d-%02d %02d-%02d-%02d opengl capture %u", tm->tm_year + 1900, tm->tm_mon + 1,
                 tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec, o->capture_serial + 1);
    else
        snprintf(name, sizeof(name), "opengl capture %u", o->capture_serial + 1);
    if (strlen(base) + 1 + strlen(name) + sizeof("commands.txt") >= 260) {
        if (!o->capture_path_told) {
            o->capture_path_told = 1;
            vgpu98_gpu_log(o->gpu, VGPU98_LOG_EVENTS, "VGPU98 OpenGL: no capture: the captures' folder's path is too long (%s)\n",
                           base);
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
    k = calloc(1, sizeof(*k));
    if (!k)
        return 1;
    o->capture_serial++;
    k->gpu    = o->gpu;
    k->serial = o->capture_serial;
    snprintf(k->dir, sizeof(k->dir), "%s", dir);
    snprintf(k->why, sizeof(k->why), "%s", why);
    /* What the SHOW wrote: 565 or BGRA 8888, as RGBA */
    k->pic[0] = malloc((size_t) w * h * 4);
    if (k->pic[0]) {
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                const uint8_t *s   = px + (size_t) y * row + (size_t) x * bytes;
                uint8_t       *out = k->pic[0] + ((size_t) y * w + x) * 4;

                if (bytes == 2) {
                    const uint32_t v = (uint32_t) s[0] | ((uint32_t) s[1] << 8);

                    out[0] = (uint8_t) (((v >> 11) & 31u) * 255u / 31u);
                    out[1] = (uint8_t) (((v >> 5) & 63u) * 255u / 63u);
                    out[2] = (uint8_t) ((v & 31u) * 255u / 31u);
                } else {
                    out[0] = s[2];
                    out[1] = s[1];
                    out[2] = s[0];
                }
                out[3] = 0xff;
            }
        }
        k->w[0] = w;
        k->h[0] = h;
    }
    /* Each unit's texture, from the host's copy */
    for (i = 0; c && (i < 2); i++)
        k->pic[1 + i] = capture_texture(o, c, (int) i, &k->w[1 + i], &k->h[1 + i]);
    /* The registers and the frame's packets */
    k->text = malloc(cap);
    if (k->text) {
        const uint32_t last  = o->n_cmds;
        uint32_t       first = o->frame_from;

        k->text[0] = 0;
        text_add(&k->text, &n, &cap, "VGPU98 OpenGL capture %u (%s): drawable %x, %ux%u (format %x), shown %ux%u; context %x\n",
                 k->serial, why, d->id, d->w, d->h, d->format, w, h, c ? c->id : 0);
        if (c) {
            text_add(&k->text, &n, &cap, "\nRegisters:\n");
            for (i = 0; i < VGPU98_OGL_R_COUNT; i++)
                text_add(&k->text, &n, &cap, "%3u: %08X%s", i, c->reg[i], ((i & 7) == 7) ? "\n" : "  ");
        }
        if (last - first > CMDS_KEPT)
            first = last - CMDS_KEPT;
        text_add(&k->text, &n, &cap, "\n\nThe frame's packets (%u):\n", last - first);
        for (i = first; o->cmds && (i < last); i++) {
            const ogl_cmd_t *m = &o->cmds[i % CMDS_KEPT];

            text_add(&k->text, &n, &cap, "%-9s %5u: %08X %08X %08X %08X %08X %08X\n", op_name(m->op), m->len, m->p[0], m->p[1],
                     m->p[2], m->p[3], m->p[4], m->p[5]);
        }
    }
    atomic_store(&capture_busy, 1);
    {
        /* A thread that could not be made is NULL, or -1 from _beginthread:
           the files go on this thread instead. */
        const thread_t *th = thread_create_named(capture_thread, k, "VGPU98 OpenGL capture");

        if (!th || (th == (thread_t *) (intptr_t) -1))
            capture_thread(k);
    }
    return 1;
}

/* At each SHOW with Diagnostics "OpenGL": a capture if one is due; the
   next frame's packets start here. */
static void
capture_check(vgpu98_ogl_t *o, ogl_drawable_t *d, const uint8_t *px, uint32_t row, uint32_t w, uint32_t h, uint32_t bytes)
{
    const uint32_t now = plat_get_ticks();

    if (!d->first_show_ms)
        d->first_show_ms = now ? now : 1;
    if (atomic_load(&o->shot_asked) && (d->shot_captures < SHOT_CAPTURES)) {
        if (capture(o, d, px, row, w, h, bytes, "an 86Box screenshot")) {
            atomic_store(&o->shot_asked, 0);
            d->shot_captures++;
        }
    } else if (o->capture_due && (d->captures < CAPTURES)) {
        if (capture(o, d, px, row, w, h, bytes, "a finding")) {
            o->capture_due = 0;
            d->captures++;
        }
    } else if (!d->captured_late && ((uint32_t) (now - d->first_show_ms) >= CAPTURE_MS) && (d->captures < CAPTURES)) {
        if (capture(o, d, px, row, w, h, bytes, "10 seconds in")) {
            d->captured_late = 1;
            d->captures++;
        }
    }
    o->frame_from = o->n_cmds;
}

/* --- Start-up ---------------------------------------------------------------- */

/* The warm-up: a fixed set of
   pipelines made into the shared pipeline cache as 86Box starts, and let
   go, so the first draw finds the shaders' first pipeline made (run 001's
   first draw took 1.2 s): each colour format with each depth kind, plainly
   and alpha-blended. */
static void
warm_thread(void *priv)
{
    vgpu98_ogl_t  *o  = (vgpu98_ogl_t *) priv;
    const uint64_t t0 = vgpu98_diag_us();
    uint32_t       made = 0, failed = 0, f, dk, b;

    for (f = 0; (f < 3) && !atomic_load(&o->warm_stop); f++) {
        for (dk = 0; (dk < DEPTH_KINDS) && !atomic_load(&o->warm_stop); dk++) {
            for (b = 0; (b < 2) && !atomic_load(&o->warm_stop); b++) {
                uint64_t   k = f | ((uint64_t) dk << 2) | ((uint64_t) ((f == 2) ? 15u : 7u) << 18);
                VkPipeline p;

                if ((dk == VGPU98_OGL_DEPTH_24S8) && o->depth24_float)
                    k |= K_GRID24;
                if (b)
                    k |= K_BLEND | ((uint64_t) VGPU98_OGL_BF_SRC_ALPHA << 5) | ((uint64_t) VGPU98_OGL_BF_ONE_MINUS_SRC_ALPHA << 9);
                if (pipe_build(o, k, &p) == VK_SUCCESS) {
                    o->vk->vkDestroyPipeline(o->vk->dev, p, NULL);
                    made++;
                } else
                    failed++;
            }
        }
    }
    /* Stopped by the close only if it ended before the last of the 18 */
    vgpu98_gpu_log(o->gpu, VGPU98_LOG_EVENTS, "VGPU98 OpenGL: %u pipelines readied in the background in %u ms%s%s\n", made,
                   (uint32_t) ((vgpu98_diag_us() - t0) / 1000u), failed ? " (some could not be made)" : "",
                   (made + failed < 3u * DEPTH_KINDS * 2u) ? ", stopped by the close" : "");
    atomic_store(&o->warm_live, 0);
}

static int
pass_make(vgpu98_ogl_t *o, VkFormat color, VkFormat depth, VkRenderPass *pass)
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
        ad[i].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
        ad[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
        ad[i].initialLayout  = VK_IMAGE_LAYOUT_GENERAL;
        ad[i].finalLayout    = VK_IMAGE_LAYOUT_GENERAL;
    }
    ad[0].format = color;
    ad[1].format = depth;
    memset(&sd, 0, sizeof(sd));
    sd.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount    = 1;
    sd.pColorAttachments       = &cr;
    sd.pDepthStencilAttachment = (depth != VK_FORMAT_UNDEFINED) ? &dr : NULL;
    ci.attachmentCount         = (depth != VK_FORMAT_UNDEFINED) ? 2 : 1;
    ci.pAttachments            = ad;
    ci.subpassCount            = 1;
    ci.pSubpasses              = &sd;
    return o->vk->vkCreateRenderPass(o->vk->dev, &ci, NULL, pass) == VK_SUCCESS;
}

/* ogl.vert's and ogl.frag's modules, their layout, each slot's descriptor
   pool. 0 if the host GPU refused one. */
static int
draw_objects(vgpu98_ogl_t *o)
{
    vgpu98_vk_t                    *vk = o->vk;
    VkShaderModuleCreateInfo        smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkDescriptorSetLayoutBinding    b[4];
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPushConstantRange             pcr;
    VkPipelineLayoutCreateInfo      pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkDescriptorPoolSize            ps[3];
    VkDescriptorPoolCreateInfo      dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    int                             i;

    smi.codeSize = sizeof(vgpu98_ogl_vert);
    smi.pCode    = vgpu98_ogl_vert;
    if (vk->vkCreateShaderModule(vk->dev, &smi, NULL, &o->vs) != VK_SUCCESS)
        return 0;
    smi.codeSize = sizeof(vgpu98_ogl_frag);
    smi.pCode    = vgpu98_ogl_frag;
    if (vk->vkCreateShaderModule(vk->dev, &smi, NULL, &o->fs) != VK_SUCCESS)
        return 0;
    smi.codeSize = sizeof(vgpu98_ogl_frag_depth);
    smi.pCode    = vgpu98_ogl_frag_depth;
    if (vk->vkCreateShaderModule(vk->dev, &smi, NULL, &o->fs_depth) != VK_SUCCESS)
        return 0;
    /* The uniform block, the vertices (or an image), the indices: their
       offsets dynamic, so one set serves while the slot's buffers stay the
       same; the texel arena. */
    memset(b, 0, sizeof(b));
    for (i = 0; i < 4; i++) {
        b[i].binding         = (uint32_t) i;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    b[2].stageFlags     = VK_SHADER_STAGE_VERTEX_BIT;
    b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[3].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
    dli.bindingCount = 4;
    dli.pBindings    = b;
    if (vk->vkCreateDescriptorSetLayout(vk->dev, &dli, NULL, &o->dsl) != VK_SUCCESS)
        return 0;
    pcr.stageFlags             = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset                 = 0;
    pcr.size                   = sizeof(ogl_push_t);
    pli.setLayoutCount         = 1;
    pli.pSetLayouts            = &o->dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges    = &pcr;
    if (vk->vkCreatePipelineLayout(vk->dev, &pli, NULL, &o->layout) != VK_SUCCESS)
        return 0;
    ps[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    ps[0].descriptorCount = POOL_SETS;
    ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    ps[1].descriptorCount = POOL_SETS * 2;
    ps[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[2].descriptorCount = POOL_SETS;
    dpi.maxSets           = POOL_SETS;
    dpi.poolSizeCount     = 3;
    dpi.pPoolSizes        = ps;
    for (i = 0; i < VGPU98_R3D_SLOTS; i++) {
        if (vk->vkCreateDescriptorPool(vk->dev, &dpi, NULL, &o->dpool[i]) != VK_SUCCESS)
            return 0;
        o->dpool_gen[i] = 0xffffffffu;
    }
    /* The texel arena: as large as the host gives, from ARENA_MAX down */
    {
        VkDeviceSize n;

        for (n = ARENA_MAX; n >= ARENA_MIN; n /= 2) {
            if (vgpu98_vk_buffer_create_local(vk, &o->arena, n,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT))
                break;
            memset(&o->arena, 0, sizeof(o->arena));
        }
        if (!o->arena.buffer)
            return 0;
        o->arena.size = n;
        range_give(o, 0, n);
    }
    return 1;
}

static void
objects_free(vgpu98_ogl_t *o)
{
    vgpu98_vk_t *vk = o->vk;
    uint32_t     i;
    int          c, d;

    for (i = 0; i < o->n_pipes; i++)
        vk->vkDestroyPipeline(vk->dev, o->pipes[i].pipe, NULL);
    o->n_pipes = 0;
    tex_group_free(o, 0);
    if (o->arena.buffer)
        vgpu98_vk_buffer_destroy(vk, &o->arena);
    memset(&o->arena, 0, sizeof(o->arena));
    free(o->free);
    o->free   = NULL;
    o->n_free = o->cap_free = 0;
    for (i = 0; i < VGPU98_R3D_SLOTS; i++) {
        if (o->dpool[i])
            vk->vkDestroyDescriptorPool(vk->dev, o->dpool[i], NULL);
        o->dpool[i] = VK_NULL_HANDLE;
    }
    if (o->layout)
        vk->vkDestroyPipelineLayout(vk->dev, o->layout, NULL);
    if (o->dsl)
        vk->vkDestroyDescriptorSetLayout(vk->dev, o->dsl, NULL);
    if (o->fs_depth)
        vk->vkDestroyShaderModule(vk->dev, o->fs_depth, NULL);
    if (o->fs)
        vk->vkDestroyShaderModule(vk->dev, o->fs, NULL);
    if (o->vs)
        vk->vkDestroyShaderModule(vk->dev, o->vs, NULL);
    o->layout   = VK_NULL_HANDLE;
    o->dsl      = VK_NULL_HANDLE;
    o->fs_depth = VK_NULL_HANDLE;
    o->fs       = VK_NULL_HANDLE;
    o->vs       = VK_NULL_HANDLE;
    for (c = 0; c < COLOR_KINDS; c++) {
        for (d = 0; d < DEPTH_KINDS; d++) {
            if (o->pass[c][d])
                o->vk->vkDestroyRenderPass(o->vk->dev, o->pass[c][d], NULL);
            o->pass[c][d] = VK_NULL_HANDLE;
        }
    }
}

vgpu98_ogl_t *
vgpu98_ogl_init(vgpu98_gpu_t *g, vgpu98_r3d_t *r, vgpu98_disp_t *d, const vgpu98_vram_t *vram)
{
    static const VkFormatFeatureFlags color_needs = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                                                    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    static const VkFormatFeatureFlags depth_needs = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                                    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    vgpu98_ogl_t *o;
    vgpu98_vk_t  *vk = vgpu98_r3d_vk(r);
    VkFormat      depth24;
    int           ok, c;

    if (!r || !vk)
        return NULL;
    /* The drawables' formats: 565 and 8888 colour that blends,
       D16, and D24S8 or D32F-S8. */
    if (vgpu98_vk_format_has(vk, VK_FORMAT_D24_UNORM_S8_UINT, depth_needs))
        depth24 = VK_FORMAT_D24_UNORM_S8_UINT;
    else if (vgpu98_vk_format_has(vk, VK_FORMAT_D32_SFLOAT_S8_UINT, depth_needs))
        depth24 = VK_FORMAT_D32_SFLOAT_S8_UINT;
    else
        depth24 = VK_FORMAT_UNDEFINED;
    if (!vgpu98_vk_format_has(vk, VK_FORMAT_R5G6B5_UNORM_PACK16, color_needs) ||
        !vgpu98_vk_format_has(vk, VK_FORMAT_B8G8R8A8_UNORM, color_needs) ||
        !vgpu98_vk_format_has(vk, VK_FORMAT_D16_UNORM, depth_needs) || (depth24 == VK_FORMAT_UNDEFINED))
        return NULL; /* the start-up line then names no command set 2.5 */
    o = calloc(1, sizeof(vgpu98_ogl_t));
    if (!o)
        return NULL;
    o->gpu           = g;
    o->r3d           = r;
    o->vk            = vk;
    o->disp          = d;
    o->vram          = vram;
    o->depth24       = depth24;
    o->depth24_float = depth24 == VK_FORMAT_D32_SFLOAT_S8_UINT;
    o->logic_op      = vk->logic_op;
    o->clamp_exact   = (vgpu98_gpu_options(g) & VGPU98_OPT_OGL_CLAMP_EXACT) != 0;
    vgpu98_r3d_ogl_hook(r, pass_end, o);
    ok = 1;
    for (c = 0; ok && (c < COLOR_KINDS); c++) {
        const VkFormat cf = c ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R5G6B5_UNORM_PACK16;

        ok = pass_make(o, cf, VK_FORMAT_UNDEFINED, &o->pass[c][0]) && pass_make(o, cf, VK_FORMAT_D16_UNORM, &o->pass[c][1]) &&
             pass_make(o, cf, depth24, &o->pass[c][2]);
    }
    ok = ok && draw_objects(o);
    if (!ok) {
        vgpu98_r3d_ogl_hook(r, NULL, NULL);
        objects_free(o);
        free(o);
        return NULL;
    }
    /* The warm-up, with the pipeline cache On */
    if (vgpu98_r3d_glide_pcache(r) && !(vgpu98_gpu_options(g) & (VGPU98_OPT_PCACHE_EMPTY | VGPU98_OPT_PCACHE_OFF))) {
        const thread_t *th;

        atomic_store(&o->warm_live, 1);
        th = thread_create_named(warm_thread, o, "VGPU98 OpenGL warm-up");
        if (!th || (th == (thread_t *) (intptr_t) -1))
            atomic_store(&o->warm_live, 0); /* no thread: the pipelines are made as draws need them */
    }
    return o;
}

int
vgpu98_ogl_close(vgpu98_ogl_t *o)
{
    int i;

    if (!o)
        return 1;
    /* The warm-up stopped and waited for: it uses the shaders and passes
       freed below. One still making a pipeline after the wait leaves them
       (and the device) as they are, as the renderer's own warm-up does. */
    {
        uint32_t waited = 0;

        atomic_store(&o->warm_stop, 1);
        while (atomic_load(&o->warm_live) && (waited < WARM_JOIN_MS)) {
            plat_delay_ms(10);
            waited += 10;
        }
    }
    for (i = 0; i < VGPU98_OGL_DRAWABLES; i++)
        drawable_end(o, &o->draw[i]);
    for (i = 0; i < VGPU98_OGL_CONTEXTS; i++) {
        if (o->ctx[i])
            context_end(o, i);
    }
    vgpu98_r3d_glide_submit(o->r3d, 1);
    vgpu98_r3d_ogl_hook(o->r3d, NULL, NULL);
    vgpu98_vk_wait_idle(o->vk);
    if (atomic_load(&o->warm_live)) {
        vgpu98_gpu_log(o->gpu, VGPU98_LOG_EVENTS,
                       "VGPU98 OpenGL: the warm-up did not end within %u s of the close; its objects left as they are\n",
                       WARM_JOIN_MS / 1000);
        return 0;
    }
    objects_free(o);
    free(o->cmds);
    free(o);
    return 1;
}

#else /* VGPU98_NO_VULKAN */

vgpu98_ogl_t *
vgpu98_ogl_init(vgpu98_gpu_t *g, vgpu98_r3d_t *r, vgpu98_disp_t *d, const vgpu98_vram_t *vram)
{
    (void) g;
    (void) r;
    (void) d;
    (void) vram;
    return NULL;
}

int
vgpu98_ogl_close(vgpu98_ogl_t *o)
{
    (void) o;
    return 1;
}

void
vgpu98_ogl_packet(vgpu98_ogl_t *o, uint32_t op, const uint32_t *p, uint32_t len, uint32_t generation)
{
    (void) o;
    (void) op;
    (void) p;
    (void) len;
    (void) generation;
}

void
vgpu98_ogl_reset(vgpu98_ogl_t *o, int end)
{
    (void) o;
    (void) end;
}

void
vgpu98_ogl_diag(vgpu98_ogl_t *o)
{
    (void) o;
}

void
vgpu98_ogl_screenshot(vgpu98_ogl_t *o)
{
    (void) o;
}

#endif
