/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the interfaces between the device (vid_vgpu98.c),
 *          its command processor (vid_vgpu98_gpu.c), its 3D renderer on
 *          the host's GPU (vid_vgpu98_r3d.c, vid_vgpu98_vk.c) and its
 *          display on the host's GPU (vid_vgpu98_disp.c).
 *
 *          The command processor runs on its own thread. The device
 *          calls into it from the emulation thread (registers, video
 *          memory writes, vertical blank); the renderer runs on the
 *          command processor's thread only. The display has a thread of
 *          its own, fed by the device at every vertical blank.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#ifndef VIDEO_VGPU98_DEV_H
#define VIDEO_VGPU98_DEV_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <86box/vid_vgpu98_dispdiag.h>

/* Host diagnostics (the card's "Diagnostics" setting, spec/registers.md):
   what goes to 86Box's log. Every line starts with the local date and time. */
#define VGPU98_LOG_DEBUG_REGISTER 0x01 /* lines the guest writes to the debug registers */
#define VGPU98_LOG_EVENTS         0x02 /* rejected packets, resets and their causes, holds, the host GPU's start-up and failures */
#define VGPU98_LOG_SUMMARY        0x04 /* a line of counters every few seconds while the GPU works */
#define VGPU98_LOG_CAPTURE        0x08 /* the recorder, written out as a snapshot around a failure */
#define VGPU98_LOG_COMMANDS       0x10 /* every packet and coherency event as it happens (slow) */
#define VGPU98_LOG_AUDIT          0x20 /* the DirectX driver audits its calls' vertex ranges (BAR0 DEBUG reads VGPU98_DEBUG_AUDIT) */
#define VGPU98_LOG_AUDIT8         0x40 /* and its DirectX 8 draws (BAR0 DEBUG reads VGPU98_DEBUG_AUDIT8) */
#define VGPU98_LOG_MINIMAL        0x80 /* "Minimal": the DirectX driver keeps to its event lines (BAR0 DEBUG reads VGPU98_DEBUG_EVENTS, not LISTENING) */
#define VGPU98_LOG_TEXTURES       0x100 /* the texture audit (BAR0 DEBUG reads VGPU98_DEBUG_TEXTURES) */
#define VGPU98_LOG_GLIDE          0x200 /* "Glide": the Glide DLLs' full lines, the Glide device's report and captures (BAR0 DEBUG reads VGPU98_DEBUG_GLIDE) */
#define VGPU98_LOG_OPENGL         0x400 /* "OpenGL": the ICD's full lines, the OpenGL device's report, findings and captures (BAR0 DEBUG reads VGPU98_DEBUG_OPENGL) */
#define VGPU98_LOG_DISPLAY        0x800 /* "Display": the screen's transitions, 86Box's screen sizes and the frames after */

/* The texture audit's findings (vgpu98_gpu_find): each kind's first asks
   for a capture at once, the rest at most one per 30 s. */
#define VGPU98_FIND_STALE      0 /* a texture's bytes in video memory changed, its pages' versions not */
#define VGPU98_FIND_SET        1 /* a stage reads a coordinate set the vertices lack, or too few of its components */
#define VGPU98_FIND_COORD      2 /* a coordinate not finite, or beyond +-65536 */
#define VGPU98_FIND_RHW        3 /* a pre-transformed vertex's RHW 0 or less or not finite, or a triangle's beyond 100000:1 */
#define VGPU98_FIND_STRETCH    4 /* a triangle (pre-transformed, or through a texture matrix) magnified both ways, a texel over more than 64 x 64 pixels */
#define VGPU98_FIND_PALETTE    5 /* an 8-bit texture drawn through a palette of zeros */
#define VGPU98_FIND_SHOWN      6 /* frames drawn mostly into the surface the last FLIP showed */
#define VGPU98_FIND_RACE       7 /* a draw's vertices changed between the DirectX driver's call and the GPU's read */
#define VGPU98_FIND_ARENA      8 /* the arena's copy of a resident buffer differs from what the DirectX driver drew */
#define VGPU98_FIND_MATRIX     9 /* a texture matrix not finite */
#define VGPU98_FINDS           10

/* Why the 3D path dropped a draw without an error (vgpu98_stats_t.drops). */
#define VGPU98_DROP_STOPPED  0 /* 3D stopped after a host GPU failure, until a reset */
#define VGPU98_DROP_IMAGES   1 /* its target, Z buffer or a texture could not be made ready */
#define VGPU98_DROP_HOST     2 /* a host GPU object: a command buffer, the render pass, a sampler */
#define VGPU98_DROP_SCISSOR  3 /* scissored out entirely */
#define VGPU98_DROP_PIPELINE 4 /* no pipeline */
#define VGPU98_DROP_MEMORY   5 /* no memory for its vertices and uniforms, or no descriptor set */
#define VGPU98_DROPS         6

/* The local date and time, "2026-10-10 22:41:07", for a log line. */
void     vgpu98_log_stamp(char *buf, size_t size);
/* A clock in microseconds, for the summary's time figures. */
uint64_t vgpu98_diag_us(void);

/* Why GPU-newer pixels were copied back to video memory (the summary's
   copies back by cause). */
#define VGPU98_CB_TRAP    0 /* the guest's CPU touched a page */
#define VGPU98_CB_SYNC    1 /* a SYNC packet (a Lock) */
#define VGPU98_CB_2D      2 /* FILL, BLT or UPLOAD read or wrote the bytes */
#define VGPU98_CB_FRAME   3 /* the frame shown, without the display on the host GPU or past its layer */
#define VGPU98_CB_IMAGE   4 /* another image of the same bytes was used */
#define VGPU98_CB_DISCARD 5 /* a DISCARD kept what lay outside its range */
#define VGPU98_CB_EVICT   6 /* the image was evicted */
#define VGPU98_CB_RESET   7 /* a RESET */

/* Counters for the summary and the capture's slices: totals since power-on,
   kept by the command processor's thread (the renderer adds its own). */
typedef struct vgpu98_stats_t {
    uint32_t packets;
    uint32_t flips;
    uint32_t fences;
    uint32_t syncs;
    uint32_t fills;
    uint32_t blits;
    uint32_t draws;
    uint32_t triangles;
    uint32_t lines;
    uint32_t points;
    uint32_t clears;
    uint32_t new_images;
    uint32_t uploads;
    uint32_t copies_back;
    uint32_t copies_back_by[8]; /* by cause: VGPU98_CB_* */
    uint32_t upload_rects;      /* UPLOAD rectangles written into host images as well */
    uint32_t views;             /* attachment views of levels and faces made */
    uint32_t merges;            /* images of a level or face taken over by a texture's image */
    uint32_t upload_packets;    /* UPLOAD packets */
    uint64_t upload_packet_bytes;
    uint32_t evictions;
    uint32_t stale_evictions; /* of which clean images unused for 10 seconds */
    uint32_t discards;
    uint32_t new_pipelines;
    uint32_t resets;
    uint32_t errors;
    uint32_t submits;       /* command buffers sent to the host GPU */
    uint32_t flips_waited;  /* flips that waited for vertical blank */
    uint32_t ring_low_room; /* doorbells with less than an eighth of the ring free */
    uint32_t publishes;     /* images' GPU-newer pixels copied into the display's layer */
    uint64_t publish_bytes;
    uint64_t upload_bytes;
    uint64_t copy_back_bytes;
    uint64_t ring_bytes;   /* packets taken from the ring, headers included */
    uint64_t vertex_bytes; /* of which INLINE packets (vertices and indices) */
    /* The vertex pipeline (summary vertices) */
    uint32_t draws_pre;        /* draws of pre-transformed vertices */
    uint32_t draws_fixed;      /* draws through the fixed-function transform */
    uint32_t shape_wrap;       /* draws needing whole primitives: WRAPn */
    uint32_t shape_points;     /* point fill */
    uint32_t shape_wire;       /* wireframe */
    uint32_t shape_lines;      /* lines */
    uint32_t shape_sprites;    /* points drawn as squares (sized or sprites; command set 2.1) */
    uint32_t draws_streams[4]; /* draws by the streams their declaration reads: 1, 2, 3-4, 5 or more */
    uint32_t draws_palette;    /* indexed vertex blending (2.1) */
    uint32_t draws_tween;      /* tweening (2.1) */
    uint32_t draws_psize;      /* with a per-vertex point size (2.1) */
    uint32_t draws_palette_noidx; /* indexed vertex blending without BLENDINDICES: indices 0-3 (2.1) */
    uint64_t vertices;         /* vertex numbers the draws' windows spanned */
    uint64_t indices;
    uint64_t vertex_copy_bytes; /* stream bytes copied for the host GPU */
    /* Resident buffers (summary buffers): stream and index buffer reads the
       arena served, buffers read from guest RAM instead, buffers that went
       in and their bytes, those evicted, the bytes of vertices the draws did
       not copy again, draws straight from the arena */
    uint32_t res_hits;
    uint32_t res_misses;
    uint32_t res_inserts;
    uint64_t res_insert_bytes;
    uint32_t res_evictions;
    uint64_t res_saved_bytes;
    uint32_t draws_direct;
    uint64_t pipeline_us; /* time making pipelines (the cache's effect shows here) */
    uint32_t sets_reused; /* draws that bound an earlier draw's textures' descriptor set */
    uint32_t res_stale;   /* resident buffers the vertex audit found changed in guest RAM, their version not */
    /* The streams (summary streams): streams draws read from GPU addresses
       and inline; by stride (up to 16, 32, 64 bytes, more); declaration
       elements by the GPU's type (1-8); indexed draws by index size (16,
       32 bits) and of those with inline indices; draws whose streams at GPU
       addresses were all, some or none resident */
    uint32_t streams_gpuvm;
    uint32_t streams_inline;
    uint32_t stream_strides[4];
    uint32_t elements[9];
    uint32_t draws_idx[2];
    uint32_t draws_idx_inline[2];
    uint32_t draws_res[3];
    /* Main memory read through the page table (summary buffers), by
       VGPU98_GPUVM_* */
    uint64_t gpuvm_bytes[4]; /* by VGPU98_GPUVM_* (VGPU98_GPUVM_KINDS) */
    /* The fixed-function transform (summary transform), of draws through it */
    uint32_t draws_lit;     /* with LIGHTING */
    uint32_t lights_used;   /* lights enabled in them, summed */
    uint32_t draws_blend;   /* with vertex blending */
    uint32_t draws_planes;  /* with user clip planes */
    uint32_t draws_texgen;  /* with generated texture coordinates */
    uint32_t draws_texxform; /* with texture transforms */
    /* Microseconds of the command processor's thread spent: */
    uint64_t idle_us;      /* waiting for the doorbell */
    uint64_t flip_us;      /* waiting for a flip's vertical blank */
    uint64_t upload_us;    /* uploading host images */
    uint64_t copy_back_us; /* copying back to video memory */
    uint64_t gpu_wait_us;  /* waiting for the host GPU to free a command buffer */
    /* The vertex audit (summary audit): draws whose copy lacked vertices of
       their window (they read as 0), and those vertices; draws dropped
       without an error, by VGPU98_DROP_* */
    uint32_t draws_past;
    uint64_t verts_past;
    uint32_t drops[VGPU98_DROPS];
    /* The depth (summary depth): draws W-buffered, of those into 32-bit float
       images; depth images moved to a float format; W ranges with an
       infinite far plane taken */
    uint32_t draws_w;
    uint32_t draws_w_float;
    uint32_t depth_moves;
    uint32_t wrange_infinite;
    /* Textures (summary textures): stage uses (an active stage of a draw
       with a texture bound), by coordinate source (vertex sets, camera
       normal, camera position, reflection), transformed and of those
       projected; anisotropic, trilinear, without mipmaps, with a mip bias;
       bump stages; the mip check's textures and the levels it named */
    uint32_t tex_uses;
    uint32_t tex_src[4];
    uint32_t tex_xform;
    uint32_t tex_projected;
    uint32_t tex_aniso;
    uint32_t tex_trilinear;
    uint32_t tex_nomip;
    uint32_t tex_lodbias;
    uint32_t tex_dot3;
    uint32_t tex_bumpenv;
    /* Command set 2.1's pixel side: stages sampling a volume, with
       MULTIPLYADD or LERP, writing TEMP; volume uploads; draws blending other
       than by adding, with channels masked */
    uint32_t tex_volume;
    uint32_t tex_madd_lerp;
    uint32_t tex_temp;
    uint32_t vol_uploads;
    uint32_t draws_blendop;
    uint32_t draws_masked;
    uint32_t mip_textures;
    uint32_t mip_named;
    uint32_t mip_again; /* textures measured again, their content changed */
    uint64_t mip_us; /* microseconds the mip check took */
} vgpu98_stats_t;

/* What the recorder notes besides packets (vgpu98_gpu_note's arguments). */
#define VGPU98_NOTE_NEW_IMAGE 1 /* id, format | levels << 8, width << 16 | height, lo, hi */
#define VGPU98_NOTE_UPLOAD    2 /* id, lo, hi, bytes */
#define VGPU98_NOTE_COPY_BACK 3 /* id, x0 << 16 | y0, x1 << 16 | y1, start, len */
#define VGPU98_NOTE_EVICT     4 /* id, lo, hi */
#define VGPU98_NOTE_DISCARD   5 /* id, offset, len */
#define VGPU98_NOTE_PIPELINE  6 /* number, shape | program << 4 | blend << 8 | src << 16 | dst << 24, test | write << 8 | func << 16 | stencil << 24 */
#define VGPU98_NOTE_FAILURE   7 /* VkResult */

#define VGPU98_PAGE_SHIFT 12

typedef struct vgpu98_gpu_t vgpu98_gpu_t;
typedef struct vgpu98_r3d_t vgpu98_r3d_t;
typedef struct vgpu98_glide_t vgpu98_glide_t;
typedef struct vgpu98_ogl_t   vgpu98_ogl_t;

/* What the command processor needs from the device. Called on the command
   processor's thread. */
typedef struct vgpu98_host_t {
    void *priv;
    /* The display engine's frame: whether it is on, START, PITCH, the
       visible height in lines of video memory, and bits per pixel. */
    void (*display)(void *priv, int *enabled, uint32_t *start, uint32_t *pitch, uint32_t *height, uint32_t *bpp);
    uint32_t options; /* the card's settings the GPU follows: VGPU98_OPT_* */
    /* The host GPU's pipeline cache between sessions: its file ("" for
       none), opened as 86Box opens files. */
    char  pipeline_cache[1024];
    FILE *(*open)(const char *path, const char *mode);
    /* The texture audit's dumps go in folders here ("" for none): the
       machine's folder's vgpu98-textures. */
    char  capture_dir[1024];
} vgpu98_host_t;

/* vgpu98_host_t.options */
#define VGPU98_OPT_DEPTH16_FULL 0x01 /* "16-bit depth precision": Full (D16 in a 32-bit float image, without the grid) */
#define VGPU98_OPT_NO_FIXED8    0x02 /* "DirectX 8 interface": Off (no PROFILE.FIXED8: a command set 2.0 GPU) */
#define VGPU98_OPT_NO_PROGRAMS  0x04 /* "DirectX 8 shaders": Off (no PROFILE.PROGRAMS: a command set 2.1 GPU) */
#define VGPU98_OPT_PCACHE_EMPTY 0x08 /* "Pipeline cache": Start empty (nothing read or warmed; saved, merged) */
#define VGPU98_OPT_PCACHE_OFF   0x10 /* "Pipeline cache": Off (neither read nor saved) */
#define VGPU98_OPT_NO_DOWNLOAD  0x20 /* "Locks through main memory": Off (no PROFILE.DOWNLOAD: a command set 2.2 GPU's Locks) */
#define VGPU98_OPT_OGL_CLAMP_EXACT 0x40 /* "OpenGL GL_CLAMP": Exact (OpenGL 1.1's rule; else sampled as clamp-to-edge, as period drivers did) */
#define VGPU98_OPT_SOFT_CURSOR  0x80 /* "Cursor": Software (no PROFILE.CURSOR: the DIB engine draws the cursor into video memory) */
#define VGPU98_OPT_NO_GAMMA     0x100 /* "Gamma": Off (no PROFILE.GAMMA: the card has no gamma table) */
#define VGPU98_OPT_NO_VSYNC     0x200 /* "V-sync overrides": Off (no PROFILE.VSYNC: programs decide) */

/* --- The command processor (vid_vgpu98_gpu.c) --- */

vgpu98_gpu_t *vgpu98_gpu_init(uint8_t *vram, uint32_t vram_size, const vgpu98_host_t *host, int log_mask);
/* 0 when a thread that reads video memory or the device (the GPU thread,
   the watchdog, the display's) had not ended in time: it is left, and so
   must they be. The renderer's own threads left keep only what they use. */
int           vgpu98_gpu_close(vgpu98_gpu_t *g);
/* A machine reset: the registers' power-on state. */
void          vgpu98_gpu_reset(vgpu98_gpu_t *g);
/* GPU_RESET (the VGA BIOS at POST): the same, logged as such. */
void          vgpu98_gpu_power_on(vgpu98_gpu_t *g);
/* BAR0 registers 10h and up (the identification registers are the device's). */
uint32_t      vgpu98_gpu_read(vgpu98_gpu_t *g, uint32_t reg);
void          vgpu98_gpu_write(vgpu98_gpu_t *g, uint32_t reg, uint32_t val);
uint32_t      vgpu98_gpu_profile(const vgpu98_gpu_t *g);
/* The card's settings it was made with (vgpu98_host_t.options). */
uint32_t      vgpu98_gpu_options(const vgpu98_gpu_t *g);
/* A validator refusing an item of a list packet (a state pair, a
   declaration's element) names it for the rejection line: the dword of the
   payload where the item starts (command processor's thread). */
void          vgpu98_gpu_fail_item(vgpu98_gpu_t *g, uint32_t dword);
/* Reads `len` bytes at GPU address `address` through the page table into
   `dst` (NULL: only checks they can be read), counted as `kind`. Returns 0
   or the ERROR_CODE of the first page that cannot (command processor's
   thread). */
#define VGPU98_GPUVM_VERTICES 0
#define VGPU98_GPUVM_INDICES  1
#define VGPU98_GPUVM_UPLOAD   2
#define VGPU98_GPUVM_DOWNLOAD 3 /* DOWNLOAD's destinations, written (command set 2.3) */
#define VGPU98_GPUVM_KINDS    4
int           vgpu98_gpu_gpuvm_read(vgpu98_gpu_t *g, uint32_t address, uint32_t len, uint8_t *dst, int kind);
/* The same, written: len bytes from src into main memory at a GPU address
   (DOWNLOAD); every page is checked before any byte is written. */
int           vgpu98_gpu_gpuvm_write(vgpu98_gpu_t *g, uint32_t address, uint32_t len, const uint8_t *src, int kind);
/* The start of vertical blank. Returns 1 with *start set when a flip takes
   effect now: the device loads START with it. */
int           vgpu98_gpu_vblank(vgpu98_gpu_t *g, uint32_t *start);
/* Video memory pages the command processor wrote since the last call
   (first page, page count), for the display's redraw; 0 if none. */
int           vgpu98_gpu_take_dirty(vgpu98_gpu_t *g, uint32_t *first, uint32_t *count);
/* The per-page CPU write counters: the device bumps one for every guest
   write to video memory (emulation thread only). Element -1 counts every
   write, so a reader can tell cheaply that nothing changed. */
uint32_t     *vgpu98_gpu_cpu_pages(vgpu98_gpu_t *g);
/* The GPU-newer page map: per 4 KB page of video memory, how many host
   images hold pixels newer than video memory there. Non-zero: a guest
   access to the page must call vgpu98_gpu_touch first. */
volatile uint16_t *vgpu98_gpu_newer(vgpu98_gpu_t *g);
/* A guest access (emulation thread) to a page the GPU holds newer pixels
   for: they are copied back, and video memory is current when it returns. */
void          vgpu98_gpu_touch(vgpu98_gpu_t *g, uint32_t addr);
/* The display on the host's GPU stopped (emulation thread): what it showed
   from the GPU goes back to video memory, for 86Box's own display. */
void          vgpu98_gpu_sync_visible_request(vgpu98_gpu_t *g);

static inline void
vgpu98_gpu_cpu_write(uint32_t *cpu_pages, uint32_t addr)
{
    cpu_pages[addr >> VGPU98_PAGE_SHIFT]++;
    cpu_pages[-1]++;
}

/* --- Shared with the renderer (command processor thread) --- */

/* The command processor's view of video memory, for the renderer. */
typedef struct vgpu98_vram_t {
    uint8_t           *data;
    uint32_t           size;
    uint32_t          *cpu_pages; /* written by the emulation thread; [-1] counts all writes */
    uint32_t          *gpu_pages; /* written by the command processor's thread; [-1] likewise */
    volatile uint16_t *newer;     /* the GPU-newer page map, kept by the renderer (command processor's thread) */
    int                fixed8;    /* PROFILE.FIXED8: command set 2.1's formats are surfaces too */
    int                programs;  /* PROFILE.PROGRAMS: command set 2.2's formats too */
} vgpu98_vram_t;

/* Changes whenever any page's version changes. */
static inline uint32_t
vgpu98_vram_epoch(const vgpu98_vram_t *v)
{
    return *(volatile uint32_t *) &v->cpu_pages[-1] + v->gpu_pages[-1];
}

/* A page's version: changes whenever the guest's CPU or the GPU writes it. */
static inline uint32_t
vgpu98_page_version(const vgpu98_vram_t *v, uint32_t page)
{
    return *(volatile uint32_t *) &v->cpu_pages[page] + v->gpu_pages[page];
}

/* The GPU wrote [offset, offset + len): every host image of those bytes is
   stale, and the display redraws them. */
void vgpu98_gpu_wrote(vgpu98_gpu_t *g, uint32_t offset, uint32_t len);
/* The command processor's reset count, and whether it has been reset since
   a value of it: long operations check before writing video memory. */
uint32_t vgpu98_gpu_generation(vgpu98_gpu_t *g);
int      vgpu98_gpu_reset_since(vgpu98_gpu_t *g, uint32_t generation);
/* A line to 86Box's log, with the date and time, when one of `sources` is
   on. Any thread. */
void vgpu98_gpu_log(vgpu98_gpu_t *g, int sources, const char *fmt, ...);
/* Is one of `sources` on? */
int  vgpu98_gpu_logging(const vgpu98_gpu_t *g, int sources);
/* The counters (command processor thread). */
vgpu98_stats_t *vgpu98_gpu_stats(vgpu98_gpu_t *g);
/* Notes an event for the recorder and, in the developer option, the live
   stream (command processor thread). Nothing is formatted otherwise. */
void vgpu98_gpu_note(vgpu98_gpu_t *g, int kind, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e);
/* Something failed: with captures on, a snapshot of the seconds around now.
   Any thread; the lock must not be held. */
void vgpu98_gpu_trigger(vgpu98_gpu_t *g, const char *why);
/* The DirectX driver's last Direct3D context went (DEBUG_IDLE): the watchdog
   takes drawing as ended. */
void vgpu98_gpu_idle(vgpu98_gpu_t *g);
/* What the command processor's thread waits on inside a packet or a batch's
   end, for the watchdog: its line for a phase lasting too long names the
   step. Command processor thread; returns the step before, which the caller
   puts back when its wait ends (so the innermost is named). */
#define VGPU98_STEP_NONE      0
#define VGPU98_STEP_SUBMIT    1 /* handing work to the host GPU: the queue's lock, vkQueueSubmit */
#define VGPU98_STEP_HOST_GPU  2 /* waiting for the host GPU to finish a submission */
#define VGPU98_STEP_COPY_BACK 3 /* copying GPU-newer pixels back to video memory */
#define VGPU98_STEP_LAYER     4 /* waiting for the display's layer (its lock) */
#define VGPU98_STEP_PIPELINE  5 /* making a pipeline */
#define VGPU98_STEP_PCACHE    6 /* saving the pipeline cache */
#define VGPU98_STEP_LOCK      7 /* waiting for the command processor's lock */
#define VGPU98_STEP_IDLE      8 /* waiting for the host GPU to go idle */
int  vgpu98_gpu_step(vgpu98_gpu_t *g, int step);
/* A wait for the host GPU (VGPU98_STEP_HOST_GPU) has gone `seconds` without
   its work done (vgpu98_vk_wait_fence's `second`, with the command
   processor as `priv`): noted for the watchdog. Nonzero when 86Box closes:
   the wait is given up. */
int  vgpu98_gpu_waited(void *g, uint32_t seconds);
/* A screenshot was taken: a capture, with captures on (a limit of its own,
   and a line when it is not taken). */
void vgpu98_gpu_screenshot(vgpu98_gpu_t *g);
/* The texture audit found something of a kind (VGPU98_FIND_*) about host
   image `image` (0 for none), its level or face `sub` (index plus one, 0
   for none): with captures on, a snapshot, the first of each kind at once,
   the rest at most one per 30 s and 12 a session; while one records, a
   named finding (one the log's lines name) goes in its text. `why` is at
   most VGPU98_FIND_WHY bytes. Command processor thread. */
#define VGPU98_FIND_WHY 704
void vgpu98_gpu_find(vgpu98_gpu_t *g, int kind, uint32_t image, uint32_t sub, const char *why, int named);
/* Would an unnamed finding of the kind now start or join a capture? When
   it would not for the limits, it is counted as not captured (the next
   finding capture's text says how many). Command processor thread. */
int  vgpu98_gpu_find_wants(vgpu98_gpu_t *g, int kind);

/* Surfaces as packets describe them (spec/gpu-commands.md, SURF). */
typedef struct vgpu98_surf_t {
    uint32_t offset;
    uint32_t width;
    uint32_t height;
    uint32_t pitch;
    uint32_t format; /* VGPU98_FMT_* */
    uint32_t bytes;  /* from offset to the end of its last row */
} vgpu98_surf_t;

/* Bytes per pixel of a format; 0 for the DXT formats and unknown codes. */
int      vgpu98_fmt_bpp(uint32_t format);
/* Is the format one of command set 2.1's (A8, X4R4G4B4, A4L4, Q8W8V8U8),
   which a packet takes only with PROFILE.FIXED8? */
int      vgpu98_fmt_fixed8(uint32_t format);
int      vgpu98_fmt_programs(uint32_t format);
int      vgpu98_fmt_is_dxt(uint32_t format);
int      vgpu98_fmt_is_depth(uint32_t format);
/* Reads a SURF descriptor. Returns 0, or ERROR_CODE 3 for a surface outside
   video memory or with an unknown format. */
int      vgpu98_surf_parse(const vgpu98_vram_t *v, const uint32_t *p, vgpu98_surf_t *s);

/* --- The vertex input (vid_vgpu98_vtx.c): command set 2.0's declaration,
       streams, index buffer and inline buffer, and a draw's vertices read
       from them (spec/gpu-commands.md, Vertex input), with command set 2.1's
       element types and uses. Command processor thread. --- */

/* The declaration resolved for the vertex shader: one slot per element use
   the pipeline reads. */
#define VGPU98_SLOT_POSITION 0
#define VGPU98_SLOT_WEIGHT   1
#define VGPU98_SLOT_NORMAL   2
#define VGPU98_SLOT_DIFFUSE  3
#define VGPU98_SLOT_SPECULAR 4
#define VGPU98_SLOT_TEX0     5
#define VGPU98_SLOT_PSIZE    13 /* 2.1: the point size */
#define VGPU98_SLOT_INDICES  14 /* 2.1: BLENDINDICES */
#define VGPU98_SLOT_POSITION2 15 /* 2.1: tweening's second position and normal */
#define VGPU98_SLOT_NORMAL2  16
#define VGPU98_SLOTS         17
/* A slot's type flag for the vertex shader: the element is read from the
   renderer's arena of resident buffers, not the draw's copy. */
#define VGPU98_SLOT_RES      0x100

/* A stream's bytes as one draw copies them (vgpu98_vin_layout). */
typedef struct vgpu98_vin_copy_t {
    uint64_t lo;    /* the first byte copied, from the stream's address */
    uint32_t words; /* how many words */
    uint32_t at;    /* where they go in the draw's copy, in words */
} vgpu98_vin_copy_t;

typedef struct vgpu98_vin_t {
    int      have_decl;
    int      program;      /* 0: pre-transformed (POSITIONT), 1: fixed function (POSITION), 2: INPUT elements for a vertex program (2.2; the slots are the input registers) */
    uint32_t weights;      /* BLENDWEIGHT components, 0 without */
    uint32_t streams_used; /* a bit per stream the declaration reads */
    uint32_t n_elements;
    uint32_t decl[2 * VGPU98_DECL_MAX]; /* as DECLARE sent it, for the diagnostics */
    struct {
        int      on;
        uint32_t stream;
        uint32_t type;   /* VGPU98_DT_* */
        uint32_t offset; /* bytes into the stream's vertex */
    } slot[VGPU98_SLOTS];
    struct {
        int      set;
        uint32_t source; /* VGPU98_SRC_* */
        uint32_t address;
        uint32_t size;
        uint32_t stride;
        uint32_t version;
        uint32_t extent; /* bytes the declaration reads of a vertex: its elements' largest end */
    } stream[VGPU98_MAX_STREAMS];
    struct {
        int      set;
        uint32_t isize; /* 2 or 4 */
        uint32_t source;
        uint32_t address;
        uint32_t size;
        uint32_t version;
    } index;
    uint8_t      *inl; /* the inline buffer, VGPU98_INLINE_BYTES */
    vgpu98_gpu_t *gpu; /* reads main memory through the page table */

    /* The draw being read */
    vgpu98_vin_copy_t copy[VGPU98_MAX_STREAMS];
    uint32_t          inl_lo;    /* the inline buffer's bytes the draw copies, once for every stream in it: from here */
    uint32_t          inl_words; /* this many words, at the start of the draw's copy */
    uint8_t          *gv;     /* the draw's streams at GPU addresses, read from guest RAM: laid out as in the draw's copy, after the inline buffer's bytes */
    uint32_t          gv_cap;
    uint32_t          gv_words; /* the words of them, which follow the inline buffer's in the draw's copy */
    uint8_t          *gidx; /* an index buffer at a GPU address: the draw's indices, read from guest RAM */
    uint32_t          gidx_cap;
    uint32_t         *local; /* an indexed draw's indices as window numbers */
    uint32_t          local_cap;
    uint32_t          past;        /* the window's vertices the draw's copy lacks in the stream that lacks most (they read as 0) */
    uint32_t          past_stream; /* that stream */
    /* Buffers at GPU addresses the renderer holds whole in its arena of
       resident buffers (set before vgpu98_vin_window and vgpu98_vin_layout
       for each draw): read there, never copied or read from guest RAM. */
    uint32_t          res_mask;                      /* bit s: stream s is */
    uint32_t          res_word[VGPU98_MAX_STREAMS];  /* the word of the arena its address is at */
    const uint8_t    *res_map;                       /* the arena's bytes */
    const uint8_t    *index_res;                     /* the index buffer's bytes there, or NULL */
} vgpu98_vin_t;

/* A draw as the vertex input reads it. Its vertices are numbered within a
   window of vertex numbers, [first, first + nverts): the vertex shader's
   gl_VertexIndex, and the numbers in an indexed draw's index list and in
   the primitive lists, are window numbers, and a window number from nverts
   up reads every element as 0 (a vertex outside the vertex numbers the
   card has, spec/gpu-commands.md). */
typedef struct vgpu98_vin_draw_t {
    uint32_t        prim; /* VGPU98_PRIM_* */
    uint32_t        count;
    uint32_t        start; /* the DRAW's: its first vertex, or its first index */
    int32_t         base;  /* and an indexed draw's base */
    int             indexed;
    const uint32_t *edges; /* edge flags (4 bits per triangle), NULL without */
    uint32_t        ntri;  /* triangles the topology makes */
    int64_t         first;
    uint32_t        nverts;
    const uint32_t *local; /* indexed: the `count` indices as window numbers */
    int             outside; /* it uses a vertex outside the window (it reads as 0, so draws at the origin) */
} vgpu98_vin_draw_t;

int      vgpu98_vin_init(vgpu98_vin_t *v, vgpu98_gpu_t *gpu);
void     vgpu98_vin_close(vgpu98_vin_t *v);
/* RESET: the declaration, streams and index buffer go (the inline buffer's
   bytes stay; the driver may not rely on them). */
void     vgpu98_vin_reset(vgpu98_vin_t *v);
/* DECLARE, STREAM, INDICES, INLINE. Returns 0 or an ERROR_CODE. */
int      vgpu98_vin_packet(vgpu98_vin_t *v, uint32_t op, const uint32_t *p, uint32_t len, uint32_t profile);
/* Reads a DRAW packet: validates it against the state, finds the window of
   vertices it reads and, for an indexed draw, its window numbers. Returns 0
   or an ERROR_CODE. vgpu98_vin_check is its first half (no index is read:
   an indexed draw's window is not known yet), vgpu98_vin_window the second. */
int      vgpu98_vin_draw(vgpu98_vin_t *v, const uint32_t *p, uint32_t len, vgpu98_vin_draw_t *d);
int      vgpu98_vin_check(vgpu98_vin_t *v, const uint32_t *p, uint32_t len, vgpu98_vin_draw_t *d);
int      vgpu98_vin_window(vgpu98_vin_t *v, vgpu98_vin_draw_t *d);
/* Which bytes of each stream the draw's window needs, at most `max_words`
   in all (the vertices past them read 0); those at GPU addresses are read
   from guest RAM now. Returns 0 or an ERROR_CODE, with *words the words the
   draw's copy of its streams takes, and `past` the window's vertices the
   copy lacks (past a stream's end, or past those limits). */
int      vgpu98_vin_layout(vgpu98_vin_t *v, const vgpu98_vin_draw_t *d, uint32_t max_words, uint32_t *words);
/* Copies those bytes into `dst` and resolves the slots for the vertex
   shader: first word (of window vertex 0), stride in words, type (0
   absent), end word (the first past the stream's copy). */
void     vgpu98_vin_copy(const vgpu98_vin_t *v, uint32_t *dst, uint32_t slots[VGPU98_SLOTS][4]);
/* The bounds of the window's positions as the vertex shader reads them
   (after vgpu98_vin_layout): x and y least, then greatest, each clamped to
   +-2^22 (NaN to the low end), the origin included when a vertex reads as
   0 there. 0 when the draw uses no vertex. */
int      vgpu98_vin_bounds(const vgpu98_vin_t *v, const vgpu98_vin_draw_t *d, float *b);
/* How many primitives the draw makes (lines or triangles; points for a
   point list). */
uint32_t vgpu98_vin_nprims(const vgpu98_vin_draw_t *d);
/* The draw's lines or triangles in assembled order, four ints each, in
   window numbers: a triangle's three vertices and its edge flags, or a
   line's two and whether it may draw its last pixel (with `lastpixel`,
   every line of a list and the last of a strip). */
void     vgpu98_vin_prims(const vgpu98_vin_draw_t *d, int lastpixel, int32_t *out);
/* For the texture audit: corner k of the draw's
   primitive i in vgpu98_vin_prims's order, as its place among the draw's
   vertices or indices (element), and in window numbers (corner); and an
   element of window vertex i as the vertex shader reads it (after
   vgpu98_vin_layout), as up to four floats (those it lacks 0, a fourth 1),
   returning its components, 0 when the draw's copy lacks it (it reads as
   0) or the declaration has no element in that slot. */
uint32_t vgpu98_vin_element(const vgpu98_vin_draw_t *d, uint32_t i, uint32_t k);
int32_t  vgpu98_vin_corner(const vgpu98_vin_draw_t *d, uint32_t i, uint32_t k);
int      vgpu98_vin_fetch(const vgpu98_vin_t *v, uint32_t slot, int32_t i, float *out);
/* Window vertex i of stream s as the draw's copy holds it: up to `max` of
   its bytes into buf, their number returned (0 outside the copy). */
uint32_t vgpu98_vin_raw(const vgpu98_vin_t *v, uint32_t s, int32_t i, uint32_t from, uint8_t *buf, uint32_t max);

/* --- The 3D renderer (vid_vgpu98_r3d.c) --- */

/* Starts the host GPU. NULL if it cannot (no Vulkan): the profile then lacks
   RENDER3D, and nothing else changes. */
vgpu98_r3d_t *vgpu98_r3d_init(vgpu98_gpu_t *g, const vgpu98_vram_t *vram);
/* 0 when a pipeline cache save or warm-up thread had not ended: what it uses
   (the host GPU's device, the renderer) is left, not freed, and a line says
   so. */
int           vgpu98_r3d_close(vgpu98_r3d_t *r);
/* RESET: copies back what the images hold newer (until deadline_us, on
   vgpu98_diag_us's clock), then drops every host image and the 3D state;
   with forget (a new session: power on, a machine reset, the driver's
   start), the mip check's records too. */
void          vgpu98_r3d_reset(vgpu98_r3d_t *r, uint64_t deadline_us, int forget);
/* Copies back to video memory every GPU-newer byte overlapping the range. */
void          vgpu98_r3d_sync(vgpu98_r3d_t *r, uint32_t offset, uint32_t len, int cause);
/* UPLOAD wrote a rectangle of a surface into video memory (as the GPU's own
   write, already counted): the host images of the surface that were current
   take the rectangle too, so no upload of them follows. */
void          vgpu98_r3d_uploaded(vgpu98_r3d_t *r, const vgpu98_surf_t *s, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                                  uint32_t start, uint32_t len);
/* FILL, BLT or UPLOAD wrote a rectangle of a surface into video memory: a
   float depth image under those bytes takes them again at its next use,
   even where they equal what its shadow holds (a W-buffer's form in video
   memory can repeat: with an infinite range every W reads 0). Called
   before vgpu98_r3d_uploaded. */
void          vgpu98_r3d_written(vgpu98_r3d_t *r, const vgpu98_surf_t *s, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1);
/* DISCARD: drops the host images overlapping the range, first copying back
   their GPU-newer data outside it. */
void          vgpu98_r3d_discard(vgpu98_r3d_t *r, uint32_t offset, uint32_t len);
/* --- The display on the host's GPU (vid_vgpu98_disp.c) --- */

typedef struct vgpu98_disp_t vgpu98_disp_t;

struct bitmap_t;

/* A frame to show: the display engine's state at the start of vertical
   blank, and where the frame goes in 86Box's frame buffer. */
typedef struct vgpu98_frame_t {
    uint32_t start;  /* video memory offset of the first row */
    uint32_t pitch;
    uint32_t width;
    uint32_t height;
    uint32_t bpp;    /* 8, 15, 16 or 32 */
    uint32_t dscan;  /* 1, or 2 for double scan */
    uint32_t blank;  /* the screen is blanked */
    uint32_t palette[256]; /* 8 bpp: the 32-bit colour of each byte value, through the DAC mask */
    int      x, y;   /* the frame's first pixel in the monitor's frame buffer */
    /* The hardware cursor as taken at this vertical blank:
       shown or not, its pattern's top-left
       corner in the mode's pixels, and a number that changes with its
       pattern. */
    int      cur_on;
    int32_t  cur_x, cur_y;
    uint32_t cur_serial;
    /* The gamma table as taken at this vertical blank:
       applied or not (15, 16 and 32 bpp only), and a number that changes
       with the table. */
    int      gamma_on;
    uint32_t gamma_serial;
} vgpu98_frame_t;

/* The hardware cursor's pattern at its largest: the AND mask, then the
   image at 4 bytes a pixel (spec/gpu-registers.md, The hardware cursor). */
#define VGPU98_CURSOR_BYTES (VGPU98_CURSOR_XOR + VGPU98_CURSOR_SIZE * VGPU98_CURSOR_SIZE * 4)

typedef struct vgpu98_disp_stats_t {
    uint64_t frames;       /* composed on the host GPU */
    uint64_t upload_bytes; /* video memory uploaded */
    uint64_t read_bytes;   /* rows read back into 86Box's frame buffer */
    uint64_t busy_us;      /* the display thread's work */
} vgpu98_disp_stats_t;

/* A rectangle of GPU-newer pixels in the display's layer: an image's dirty
   rectangle, at its place in video memory. */
typedef struct vgpu98_layer_rect_t {
    uint32_t id;     /* the image's */
    uint32_t offset; /* video memory offset of the image's first pixel */
    uint32_t pitch;
    uint32_t bytes;  /* per pixel: 2 or 4 */
    uint32_t x0, y0, x1, y1;
} vgpu98_layer_rect_t;

/* On the 3D renderer's device; NULL without one. */
vgpu98_disp_t *vgpu98_disp_init(vgpu98_gpu_t *g, vgpu98_r3d_t *r, const vgpu98_vram_t *vram);
/* 0 when its thread had not ended in time: what it uses is left, not freed. */
int            vgpu98_disp_close(vgpu98_disp_t *d);
/* 0 once the host GPU failed: the device draws the frames itself again. */
int            vgpu98_disp_ok(const vgpu98_disp_t *d);
/* Emulation thread: start showing frames (everything uploaded and written
   anew), and stop (work in progress is dropped). */
void           vgpu98_disp_start(vgpu98_disp_t *d);
void           vgpu98_disp_stop(vgpu98_disp_t *d);
/* Emulation thread, at every vertical blank while started: the rows of the
   frame composed since go into the monitor's frame buffer `tb` at `now`'s
   place (returns 1, with the frame they are of in `taken` unless NULL; 0
   when none is ready), then the next frame is posted. A frame composed for
   another mode or place than `now`'s (width, height, depth, pitch, double
   scan, x, y) is held back (returns -1, `taken` filled): the new mode's
   first frame is the caller's. */
int            vgpu98_disp_take(vgpu98_disp_t *d, struct bitmap_t *tb, const vgpu98_frame_t *now, vgpu98_frame_t *taken);
/* `cursor` is the hardware cursor's pattern (VGPU98_CURSOR_BYTES), read
   when the frame's cur_serial is new; `gamma` the gamma table
   (VGPU98_GAMMA_ENTRIES words), read when its gamma_serial is new. */
void           vgpu98_disp_frame(vgpu98_disp_t *d, const vgpu98_frame_t *f, const uint8_t *cursor, const uint32_t *gamma);
/* The counters since power-on, for the summary. */
void           vgpu98_disp_stats(const vgpu98_disp_t *d, vgpu98_disp_stats_t *s);
/* The display thread's step, for the watchdog (any thread, no lock): what it
   is in (VGPU98_DSTEP_*) since `since` (plat_get_ticks), and while it waits
   for the host GPU, the seconds without its work done and when that was
   last checked (0 before the first second). 0 without a display. */
#define VGPU98_DSTEP_FRAME    0 /* waiting for a frame to compose */
#define VGPU98_DSTEP_LAYER    1 /* waiting for its layer's lock */
#define VGPU98_DSTEP_COMPOSE  2 /* uploading video memory and recording the frame */
#define VGPU98_DSTEP_SUBMIT   3 /* handing the frame to the host GPU */
#define VGPU98_DSTEP_HOST_GPU 4 /* waiting for the host GPU to finish it */
int            vgpu98_disp_watch(const vgpu98_disp_t *d, uint32_t *since, uint32_t *seconds, uint32_t *checked);
/* 86Box closes: a wait for the host GPU in progress is given up (any
   thread, before the close). */
void           vgpu98_disp_closing(vgpu98_disp_t *d);
/* The command processor's display, NULL without one. */
vgpu98_disp_t *vgpu98_gpu_disp(vgpu98_gpu_t *g);
/* Showing frames on the host GPU now (any thread; a hint for the renderer). */
int            vgpu98_disp_showing(const vgpu98_disp_t *d);
/* The layer (command processor's thread): a buffer the size of video
   memory into which the renderer copies GPU-newer pixels at their video
   memory offsets, and the rectangles that hold them. The lock is held
   around the copy's submission, so the display never composes from a
   layer being written. */
void           vgpu98_disp_layer_lock(vgpu98_disp_t *d);
void           vgpu98_disp_layer_unlock(vgpu98_disp_t *d);
uint64_t       vgpu98_disp_layer_buffer(vgpu98_disp_t *d); /* a VkBuffer */
/* Under the layer lock: sets an image's rectangle (replacing the last), or
   returns 0 when the layer holds as many as it can. */
int            vgpu98_disp_layer_set(vgpu98_disp_t *d, const vgpu98_layer_rect_t *rect);
/* An image's pixels are no longer newer than video memory (copied back or
   dropped): its rectangle goes. Takes the layer lock. */
void           vgpu98_disp_layer_drop(vgpu98_disp_t *d, uint32_t id);
/* Diagnostics "Display" (any thread but the display's): the layer's
   rectangles of `bytes` a pixel overlapping [start, start + len) of video
   memory, a frame that wraps past its end taken as not wrapping. Takes the
   layer lock. */
uint32_t       vgpu98_disp_layer_over(vgpu98_disp_t *d, uint32_t start, uint32_t len, uint32_t bytes);
/* The Glide screen (the command processor's thread):
   a buffer of 2048x2048 16-bit pixels, a VkBuffer, into which
   the Glide device copies a session's front buffer at each swap, rows of
   `width` pixels, its submission done under the layer lock; then, still
   under it, vgpu98_disp_glide_frame says what the buffer holds and gives
   the session's gamma table (256 words: red in bits 7:0, green 15:8, blue
   23:16). From then until vgpu98_disp_glide_end (which takes the layer
   lock), the display composes from it instead of video memory. The
   buffer is 0 without the Glide screen on the host GPU. */
uint64_t       vgpu98_disp_glide_buffer(vgpu98_disp_t *d);
void           vgpu98_disp_glide_frame(vgpu98_disp_t *d, uint32_t width, uint32_t height, const uint32_t *gamma);
void           vgpu98_disp_glide_end(vgpu98_disp_t *d);
int            vgpu98_disp_glide_on(const vgpu98_disp_t *d);

/* Executes a 3D packet whose payload has been read. Returns 0 or an
   ERROR_CODE. */
int           vgpu98_r3d_packet(vgpu98_r3d_t *r, uint32_t op, const uint32_t *payload, uint32_t len);
/* The host images held, their bytes, and the budget, for the summary. */
void          vgpu98_r3d_usage(const vgpu98_r3d_t *r, uint32_t *images, uint64_t *bytes, uint64_t *budget);
/* The pipeline cache's file `which` (0 or 1), opened (NULL without one);
   its name. */
FILE         *vgpu98_gpu_cache_open(const vgpu98_gpu_t *g, int which, const char *mode);
void          vgpu98_gpu_cache_name(const vgpu98_gpu_t *g, int which, char *buf, size_t size);
/* The folder the texture audit's dumps go in ("" for none). */
const char   *vgpu98_gpu_capture_dir(const vgpu98_gpu_t *g);
/* A capture begins at a texture level: its textures read back and written
   as pictures on a thread of their own,
   the finding's host image `image` (0 for none) first, at its level or face
   `sub` (its index plus one; 0 for none). Returns the length of the lines
   naming them written to buf, for the capture's text. */
size_t        vgpu98_r3d_dump(vgpu98_r3d_t *r, uint32_t capture, uint32_t image, uint32_t sub, char *buf, size_t size);
/* The volume textures held (2.1) and their bytes. */
void          vgpu98_r3d_vol_usage(const vgpu98_r3d_t *r, uint32_t *volumes, uint64_t *bytes);
/* The W range now (WRANGE), for the summary. */
void          vgpu98_r3d_wrange(const vgpu98_r3d_t *r, float *wnear, float *wfar);
/* A capture records after its trigger: until the census ends, the first draw
   of each kind is described in full. The end returns the text (malloc'd;
   NULL without any) for the snapshot. */
void          vgpu98_r3d_census_start(vgpu98_r3d_t *r);
char         *vgpu98_r3d_census_end(vgpu98_r3d_t *r);
/* The texture audit's summary lines since the last:
   stages drawn by format and conversion, churn, the checks and findings;
   its figures start again. */
void          vgpu98_r3d_tex_summary(vgpu98_r3d_t *r);
/* A FLIP to `start` ended a frame: the texture audit's display check judges
   it. */
void          vgpu98_r3d_frame(vgpu98_r3d_t *r, uint32_t start);
/* A NOP's payload with the texture audit: the DirectX driver's audit notes
   (VGPU98_NOTE_*); others are ignored. */
void          vgpu98_r3d_note(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len, uint64_t at, uint64_t known);
/* The 3D state as text lines (each starting with two spaces), for a
   snapshot. Returns the length written. */
size_t        vgpu98_r3d_describe(const vgpu98_r3d_t *r, char *buf, size_t size);
/* The end of a batch, or a FENCE: work recorded so far goes to the host
   GPU. */
void          vgpu98_r3d_flush(vgpu98_r3d_t *r);
/* GPU-newer pixels over [lo, hi), the frame on the screen (bpp bits per
   pixel), go to the display's layer straight from the host GPU; those it
   cannot show as they are, and all of them while the display is not on
   the host GPU, go back to video memory. */
void          vgpu98_r3d_publish(vgpu98_r3d_t *r, vgpu98_disp_t *d, uint32_t lo, uint32_t hi, uint32_t bpp);
/* The display that takes the published pixels (set once both exist). */
void          vgpu98_r3d_set_disp(vgpu98_r3d_t *r, vgpu98_disp_t *d);
/* 86Box closes: the pipeline cache's warm-up stops after the pipelines in
   hand (any thread, before the close). */
void          vgpu98_r3d_closing(vgpu98_r3d_t *r);

/* --- The Glide device (vid_vgpu98_glide.c) --- */

/* Command set 2.4 on the renderer's Vulkan device, NULL without one (the
   profile then lacks GLIDE). Its sessions, images and pipelines are its own;
   it shares the renderer's command buffers and the display's screen. */
vgpu98_glide_t *vgpu98_glide_init(vgpu98_gpu_t *g, vgpu98_r3d_t *r, vgpu98_disp_t *d, const vgpu98_vram_t *vram);
void            vgpu98_glide_close(vgpu98_glide_t *gl);
/* A Glide command whose payload has been read (the command processor's
   thread): taken, or dropped and counted, never an error. */
void            vgpu98_glide_packet(vgpu98_glide_t *gl, uint32_t op, const uint32_t *p, uint32_t len, uint32_t generation);
/* RESET: `end` 2 (a mode change, a new machine) every session ends and the
   desktop comes back; 1 (a full-screen DOS session) the screen given back
   and the sessions kept, shown again at their next swap (Windows comes back
   in the same mode); 0 (the GPU's other resets) the sessions left as they
   are. */
void            vgpu98_glide_reset(vgpu98_glide_t *gl, int end);
/* At a batch's end and while the command processor is idle (its thread,
   outside its lock): a front buffer drawn into since it was last shown goes
   to the screen, once a vertical blank at most. */
void            vgpu98_glide_tick(vgpu98_glide_t *gl);
/* Diagnostics "Glide" (the command processor's thread, outside its lock):
   the Glide report every 5 seconds while Glide works, and its findings. */
void            vgpu98_glide_diag(vgpu98_glide_t *gl);
/* An 86Box screenshot at Diagnostics "Glide" (the emulation's thread): a
   capture of the frame shown at the next swap. */
void            vgpu98_glide_screenshot(vgpu98_glide_t *gl);

/* --- The OpenGL device (vid_vgpu98_ogl.c) --- */

/* Command set 2.5 on the renderer's Vulkan device, NULL without one (the
   profile then lacks OGL). Its groups, contexts, drawables, textures and
   pipelines are its own; it shares the renderer's command buffers, and
   writes its frames into video memory under the coherency rule. */
vgpu98_ogl_t *vgpu98_ogl_init(vgpu98_gpu_t *g, vgpu98_r3d_t *r, vgpu98_disp_t *d, const vgpu98_vram_t *vram);
/* 0 when its warm-up was still making a pipeline after the close's wait:
   its objects are left, and the GPU keeps the renderer and its record (as
   for the renderer's own threads). */
int           vgpu98_ogl_close(vgpu98_ogl_t *o);
/* An OpenGL command whose payload has been read (the command processor's
   thread): taken, or dropped and counted, never an error. */
void          vgpu98_ogl_packet(vgpu98_ogl_t *o, uint32_t op, const uint32_t *p, uint32_t len, uint32_t generation);
/* RESET: `end` 1 (a new machine) every object ends; 0 (every other reset:
   a mode change, a DOS session, a restart) they are kept, as GL 1.1 has no
   lost context. */
void          vgpu98_ogl_reset(vgpu98_ogl_t *o, int end);
/* Diagnostics "OpenGL" (the GPU thread's periodic work): the device's report
   every 5 seconds while OpenGL works, and its findings. */
void          vgpu98_ogl_diag(vgpu98_ogl_t *o);
/* An 86Box screenshot at Diagnostics "OpenGL" (the emulation's thread): a
   capture of the frame shown at the next SHOW. */
void          vgpu98_ogl_screenshot(vgpu98_ogl_t *o);

/* The command processor, for the Glide device: a swap's wait for `interval`
   vertical blanks since the last swap (one at least from now; none for 0),
   given up at a reset (its generation); the visible frame (the host's
   view); the vertical blanks so far. The OpenGL device's swaps wait the
   same way from a mark of their own (`since`: the vertical blank of the
   drawable's last swap, set to the one the wait ends at). */
void     vgpu98_gpu_wait_vblanks(vgpu98_gpu_t *g, uint32_t interval, uint32_t generation);
void     vgpu98_gpu_wait_vblanks_since(vgpu98_gpu_t *g, uint32_t interval, uint32_t *since, uint32_t generation);
void     vgpu98_gpu_visible(vgpu98_gpu_t *g, int *enabled, uint32_t *start, uint32_t *pitch, uint32_t *height, uint32_t *bpp);
uint32_t vgpu98_gpu_vblanks(vgpu98_gpu_t *g);
/* While `on`, every vertical blank wakes the command processor, for
   vgpu98_glide_tick. */
void     vgpu98_gpu_glide_front(vgpu98_gpu_t *g, int on);
/* A Glide session has the screen (the Glide device, at its swaps and its
   sessions' ends), and whether one has (the emulation thread, at vertical
   blank): the hardware cursor is not shown over it, as a Voodoo3's Glide
   turns its cursor off. */
void     vgpu98_gpu_glide_screen(vgpu98_gpu_t *g, int on);
int      vgpu98_gpu_glide_screen_on(vgpu98_gpu_t *g);
/* The V-sync overrides: VSYNC_CONTROL as written (the
   emulation thread; a machine reset and POST clear it), as read, and the
   vertical blanks a flip or swap waits after it, for the field at `shift`
   and the blanks the program asked for (the command processor's thread,
   which logs the first wait it changes for each API after each write). */
void     vgpu98_gpu_set_vsync(vgpu98_gpu_t *g, uint32_t control);
uint32_t vgpu98_gpu_vsync(vgpu98_gpu_t *g);
uint32_t vgpu98_gpu_vsync_interval(vgpu98_gpu_t *g, int shift, uint32_t asked);
/* An 86Box screenshot, for the Glide device (vgpu98_glide_screenshot). */
void     vgpu98_gpu_glide_screenshot(vgpu98_gpu_t *g);
/* And for the OpenGL device (vgpu98_ogl_screenshot, Diagnostics "OpenGL"). */
void     vgpu98_gpu_ogl_screenshot(vgpu98_gpu_t *g);

/* --- Diagnostics "Display" (vid_vgpu98_dd.c) --- */

/* It watches the screen at its transitions: for 30 vertical syncs after
   each, what 86Box did with the screen, logged, and the frames, captured.
   Everything but vgpu98_dd_screenshot is called on the emulation thread. */
typedef struct vgpu98_dd_t vgpu98_dd_t;

/* Why the host GPU does not compose the frames. */
#define VGPU98_DD_HOST_ON       0 /* it does */
#define VGPU98_DD_HOST_ENGINE   1 /* the display engine is off: 86Box draws the VGA's modes */
#define VGPU98_DD_HOST_DPMS     2 /* the monitor sleeps */
#define VGPU98_DD_HOST_OVERRIDE 3 /* a pass-through card has the screen */
#define VGPU98_DD_HOST_SIZE     4 /* the frame does not fit 86Box's frame buffer */
#define VGPU98_DD_HOST_NONE     5 /* no display on the host GPU, or it failed */

/* The screen as the card shows it at a vertical sync: what a transition
   changes. */
typedef struct vgpu98_dd_state_t {
    int      engine;                         /* the display engine's mode, else the VGA's */
    uint32_t width, height, bpp, pitch, dscan, refresh; /* the engine's */
    int      vga_width, vga_lines, vga_text; /* the VGA's picture as 86Box times it */
    int      blank;                          /* the screen blanked (SR01 bit 5) */
    int      dpms;                           /* the monitor asleep */
    int      override;                       /* a pass-through card has the screen */
    int      glide;                          /* a Glide session's screen */
    int      host;                           /* VGPU98_DD_HOST_* */
} vgpu98_dd_state_t;

/* A vertical sync, just after 86Box's blit of the frame. */
typedef struct vgpu98_dd_frame_t {
    vgpu98_dd_state_t state;
    int               shown;              /* 86Box blitted a frame (svga_doblit, not a pass-through card's) */
    int               lines;              /* the lines it counted (svga->vdisp) */
    int               rows;               /* the mode's (svga->dispend) */
    int               screen_w, screen_h; /* 86Box's screen */
    const struct bitmap_t *tb;            /* the monitor's frame buffer, and the picture's place in it */
    int               x, y;
    const uint8_t    *vram;               /* video memory, and the engine's frame in it */
    uint32_t          vram_mask;
    uint32_t          start;
    const uint32_t   *palette;            /* 256: each byte value's colour, through the DAC mask */
} vgpu98_dd_frame_t;

/* `capture_base` the captures' folder ("" for none). NULL without memory. */
vgpu98_dd_t *vgpu98_dd_init(vgpu98_disp_t *disp, const char *capture_base);
/* The session's totals; a capture being made is dropped. */
void         vgpu98_dd_close(vgpu98_dd_t *dd);
/* A port write changed 86Box's vertical timings or the picture's width:
   the scan as it stood with the timings now, and the frame total before. */
void         vgpu98_dd_timing(vgpu98_dd_t *dd, const vgpu98_dd_scan_t *scan, int old_vtotal);
/* The card ended a frame its own change stranded:
   the scan before and after. Comes before the port write's
   vgpu98_dd_timing. */
void         vgpu98_dd_ended(vgpu98_dd_t *dd, const vgpu98_dd_scan_t *before, const vgpu98_dd_scan_t *after);
/* A write of the DAC's colours. */
void         vgpu98_dd_palette(vgpu98_dd_t *dd);
/* The vertical blank's host display: a restart, the frame blacked out,
   and what the take returned (`taken` the frame its rows are of). */
void         vgpu98_dd_vblank(vgpu98_dd_t *dd, int restarted, int blanked, int took, const vgpu98_frame_t *taken);
void         vgpu98_dd_vsync(vgpu98_dd_t *dd, const vgpu98_dd_frame_t *f);
/* An 86Box screenshot (any thread): a capture from the next vertical sync. */
void         vgpu98_dd_screenshot(vgpu98_dd_t *dd);

#endif /* VIDEO_VGPU98_DEV_H */
