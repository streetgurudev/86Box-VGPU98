/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the GPU's command processor. BAR0 interface
 *          2.0 (the command ring, CONTROL with its hold bits, STATUS,
 *          errors, fences, vertical blank) and the packets of command
 *          set 1.0 that need no 3D: NOP, FENCE, SYNC, FLIP, DISCARD,
 *          and FILL and BLT, which run on the host CPU directly in
 *          video memory. 3D packets go to the renderer on the host's
 *          GPU. The project's spec/gpu-registers.md, gpu-commands.md
 *          and gpu-coherency.md define all of it.
 *
 *          The command processor runs on its own thread. The guest's
 *          register accesses and the display's vertical blank come
 *          from the emulation thread; one mutex keeps the registers
 *          consistent between the two, and is never held while a
 *          packet executes.
 *
 *          The card's diagnostics live here too: counters, a short
 *          summary, and a recorder written out as a snapshot of the
 *          seconds around a failure.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/plat.h>
#include <86box/thread.h>
#include <86box/mem.h>
#include <86box/vid_vgpu98_card.h>
#include <86box/vid_vgpu98_gpu.h>
#include <86box/vid_vgpu98_dev.h>
#include <86box/vid_vgpu98_glcmd.h>
#include <86box/vid_vgpu98_oglcmd.h>

#define DEBUG_LINE 160

/* Diagnostics. */
#define REC_COUNT         8192  /* records in the recorder; a power of two */
#define SLICE_MS          100
#define SLICE_COUNT       64    /* 6.4 s of slices */
#define SUMMARY_MS        5000
/* summary frames: frame times kept for its median and 95th percentile, and
   a slow frame's */
#define FRAMES_KEPT       4096
#define FRAME_SLOW_US     50000
#define CAPTURE_BEFORE_MS 3000
#define CAPTURE_AFTER_MS  2000
#define CAPTURE_PRE_MAX   4096  /* records written out from before the trigger */
#define CAPTURE_POST_MAX  1024  /* and from after it */
#define CAPTURE_GAP_MS    10000 /* between snapshots */
#define CAPTURE_MAX       5     /* per session, of failures */
#define CAPTURE_SHOTS_MAX 10    /* per session, of screenshots (vgpu98_gpu_screenshot) */
#define CAPTURE_FINDS_MAX 12    /* per session, of the texture audit's findings (vgpu98_gpu_find) */
#define CAPTURE_FIND_GAP_MS 30000 /* between findings' snapshots, but for each kind's first */
#define CAPTURE_HEAD      24576 /* bytes of a snapshot's first lines (with the texture audit's list of pictures) */
#define OUT_BLOCK         4096  /* bytes gathered before they go to the log */

#define REC_PACKET 0x80 /* a packet; the VGPU98_NOTE_* kinds are below */
#define REC_RESET  0x81 /* cause, reset count */

/* What a RESET was for, as the device can tell from the write. */
enum {
    CAUSE_MACHINE,
    CAUSE_HOLD_DRIVER,
    CAUSE_HOLD_VDD,
    CAUSE_START,
    CAUSE_AFTER_ERROR,
    CAUSE_NO_PROGRESS,
    CAUSE_POWER_ON
};

static const char *const reset_causes[] = {
    "machine reset",
    "the display driver holds the GPU (a mode change)",
    "the mini-VDD holds the GPU (a VGA mode: a DOS session or message mode)",
    "the DirectX driver or a Glide DLL starts the GPU",
    "the DirectX driver or a Glide DLL restarts the GPU after its error",
    "the DirectX driver or a Glide DLL restarts a GPU that made no progress for a second",
    "the VGA BIOS puts the GPU in its power-on state (POST)",
};

typedef struct rec_t {
    uint32_t ms;
    uint8_t  kind;
    uint8_t  op;
    uint16_t pad;
    uint32_t v[8]; /* a packet: ring offset, length, its first six dwords */
} rec_t;

/* A kind of rejection: its error, its packet and the packet's first argument
   (a list's failing item). The first few of a kind are logged in full (and
   again after a quiet spell), the rest counted for the summary. */
#define ERR_SIGS      8
#define ERR_LOG_FIRST 3     /* rejections of a kind logged in full since it was last quiet */
#define ERR_QUIET_MS  10000 /* a kind not seen for this long is logged in full again */
#define NO_ITEM       0xffffffffu
#define NO_PACKET     0xffffffffu

typedef struct err_sig_t {
    int      used;
    uint32_t code;
    uint32_t op;
    uint32_t key;
    uint32_t last_ms;
    uint32_t logged;   /* in full, since it was last quiet */
    uint32_t count;    /* since the summary */
    uint32_t unlogged; /* of those */
    char     what[160]; /* the packet, as its first rejection named it */
} err_sig_t;

/* The counters' totals at the start of a slice. */
typedef struct slice_t {
    uint32_t       ms;
    vgpu98_stats_t total;
} slice_t;

/* The registers when a trigger fired. */
typedef struct trig_regs_t {
    uint32_t status;
    uint32_t error_code;
    uint32_t error_offset;
    uint32_t fence;
    uint32_t ring_base;
    uint32_t ring_size;
    uint32_t ring_head;
    uint32_t ring_tail;
    int      hold_driver;
    int      hold_vdd;
    uint32_t generation;
} trig_regs_t;

struct vgpu98_gpu_t {
    vgpu98_host_t host;
    vgpu98_vram_t  vram;
    vgpu98_r3d_t  *r3d;
    vgpu98_disp_t *disp;
    vgpu98_glide_t *glide; /* the Glide device (command set 2.4), NULL without 3D */
    vgpu98_ogl_t   *ogl;   /* the OpenGL device (command set 2.5), NULL without 3D */
    uint32_t       profile;
    int           log_mask;

    thread_t *thread;
    event_t  *wake;      /* doorbell, CONTROL, reset, quit */
    /* The watchdog (with the log on): what the GPU
       thread is in (DOG_*), the packet, since when, and the last vertical
       blank, read by a thread of its own every second. */
    thread_t            *dog;
    event_t             *dog_wake;
    event_t             *dog_exited;
    atomic_int           dog_quit;
    atomic_int           dog_phase;
    atomic_uint          dog_op;
    atomic_uint          dog_since;  /* plat_get_ticks (milliseconds) */
    atomic_uint          dog_vblank;
    atomic_uint          dog_packet; /* the last packet's end, and the last DRAW's */
    atomic_uint          dog_draw;
    atomic_uint          dog_reset;  /* the last RESET's (a mode change, a VGA mode, a restart), or the DirectX driver's last context's end */
    /* What the GPU thread waits on inside its phase (VGPU98_STEP_*), since
       when, and in a wait for the host GPU the seconds without its work
       done and when that was last checked. */
    atomic_int           dog_step;
    atomic_uint          dog_step_since;
    atomic_uint          dog_wait_s;
    atomic_uint          dog_wait_at;
    atomic_int           closing;    /* 86Box closes: waits for the host GPU are given up */
    event_t  *flip_wake; /* vertical blank, reset, quit */
    event_t  *reset_ack; /* the thread has dropped what it held */
    event_t  *exited;
    event_t  *touch_done; /* a guest access's copy back is done */
    mutex_t  *lock;

    /* Registers (under lock). */
    uint32_t ring_base; /* RING_BASE as written: an offset in video memory, or SYSMEM and a physical address */
    uint32_t ring_size;
    uint32_t ring_shown_base; /* the ring the log last named */
    uint32_t ring_shown_size;
    uint32_t status_base;     /* STATUS_BASE: the status block in guest RAM, 0 none */
    /* The page table (interface 3.0), written under the lock and read by the
       command processor without it: each a dword, and an entry is checked
       when it is read. */
    volatile uint32_t gpuvm_base;    /* GPUVM_BASE: the table in guest RAM, 0 none */
    volatile uint32_t gpuvm_entries; /* GPUVM_ENTRIES */
    uint32_t ring_head;
    uint32_t ring_tail;
    /* The ring's bytes run so far, and when a packet began, those the
       driver had written (the texture audit's notes: a note past them was
       written after that packet ran). */
    uint64_t ring_done;
    uint64_t ring_known;
    int      run;
    int      hold_driver;
    int      hold_vdd;
    int      error;
    uint32_t error_offset;
    uint32_t error_code;
    uint32_t fence;
    uint32_t vblank_count;
    uint32_t vblank_waiters; /* the Glide device waits for vertical blanks: each one wakes it */
    uint32_t glide_vblank;   /* the vertical blank of the Glide device's last swap */
    int      glide_front;    /* a Glide front buffer waits to be shown: each vertical blank wakes the thread */
    atomic_int glide_screen; /* a Glide session has the screen (no lock: the hardware cursor's) */
    /* The V-sync overrides: VSYNC_CONTROL as written (the
       emulation thread), and a count of its writes; the APIs whose first
       changed wait since the last write is in the log (the GPU thread). */
    atomic_uint vsync_control;
    atomic_uint vsync_writes;
    uint32_t    vsync_seen;
    uint32_t    vsync_told;

    /* The thread's state (under lock). */
    int      executing;  /* a packet or the end of a batch is in progress */
    int      batch_open; /* packets were taken since the last end of batch */
    int      reset_req;
    int      reset_forget; /* a RESET since the 3D side's last that starts a new session (power on, a machine reset, the driver's start) */
    int      reset_glide;  /* the strongest RESET for the Glide device since its last: 2 ends its sessions, 1 hides them */
    int      reset_ogl;    /* and for the OpenGL device: 1 ends its objects (a new machine), 0 keeps them */
    uint32_t generation; /* changes with every RESET */
    int      quit;
    uint32_t touch_serial; /* guest accesses to GPU-newer pages, asked for */
    uint32_t touch_served; /* and copied back */
    uint32_t touch_page;
    int      sync_visible_req; /* the display left the host GPU: the visible frame goes back to video memory */
    uint32_t touch_count;  /* for the summary */
    uint64_t touch_us;

    /* Flips (under lock). */
    int      flip_pending; /* STATUS.FLIP_PENDING */
    int      flip_queued;  /* a FLIP waits for vertical blank */
    int      flip_done;
    uint32_t flip_start;
    uint32_t flip_interval;
    int      imm_pending; /* an IMMEDIATE flip's start, loaded at the next blank */
    uint32_t imm_start;
    uint32_t last_flip_vblank;

    /* Pages the thread wrote, for the display (under lock). */
    uint32_t dirty_first;
    uint32_t dirty_end; /* 0: none */

    uint32_t *payload; /* the packet being executed */
    uint8_t  *tmp;     /* BLT's copy of its source */
    size_t    tmp_size;

    /* Diagnostics. The thread owns all of it but what is marked. */
    vgpu98_stats_t stats;
    vgpu98_stats_t summary_from; /* the totals at the last summary */
    uint32_t       summary_ms;
    /* Frames (summary frames): the host's time between FLIPs since the last
       summary (the first FRAMES_KEPT for the median and percentile), the
       worst, the totals at the last FLIP, and what the thread did in the
       slow frames (over FRAME_SLOW_US) */
    uint64_t       frame_last;
    uint32_t       frame_us[FRAMES_KEPT];
    uint32_t       n_frames;
    uint32_t       frame_worst;
    vgpu98_stats_t at_flip;
    struct {
        uint32_t frames;
        uint32_t pipelines;
        uint64_t us, idle, flip, upload, copy, wait, pipe;
    } slow;
    rec_t         *rec;   /* NULL unless captures are on */
    uint32_t       rec_n; /* records written, ever */
    slice_t        slice[SLICE_COUNT];
    uint32_t       n_slices; /* ever */
    char          *out;      /* lines gathered for the log */
    size_t         out_len;
    time_t         stamp_time;
    char           stamp[24];
    uint32_t       error_count;  /* under lock */
    uint32_t       fail_item;    /* the thread's: the payload dword where the packet being executed failed, or NO_ITEM */
    uint32_t       fetch_hdr;    /* the thread's: the header of the packet last fetched */
    err_sig_t      err_sig[ERR_SIGS]; /* under lock: the kinds of rejection seen lately */
    int            err_quiet;    /* under lock: the last rejection went unlogged, and so does the reset after it */
    uint32_t       ring_peak;    /* under lock: the most bytes in the ring at a doorbell since the summary */
    vgpu98_disp_stats_t disp_from; /* the display's counters at the summary's start */
    uint32_t       touch_from;   /* touch_count and touch_us at the summary's start */
    uint64_t       touch_us_from;
    uint32_t       ring_low;     /* under lock: doorbells with under an eighth of the ring free */
    int            reset_cause;  /* under lock: the last RESET's CAUSE_* */
    uint64_t       reset_us;     /* under lock: when it was asked for (vgpu98_diag_us) */
    int            trig_pending; /* under lock: a failure's trigger, with the two below */
    char           trig_why[160];
    trig_regs_t    trig_regs;
    int            shot_pending; /* under lock: a screenshot's, with the one below (a failure's goes first) */
    trig_regs_t    shot_regs;
    int            find_pending; /* under lock: a finding's (vgpu98_gpu_find), with the three below (a screenshot's goes first) */
    uint32_t       find_image;   /* the host image it names, 0 for none (the capture's textures) ... */
    uint32_t       find_sub;     /* ... and its level or face, index plus one, 0 for none */
    char           find_why[VGPU98_FIND_WHY];
    trig_regs_t    find_regs;
    int            cap_active; /* a capture is recording its last seconds */
    uint32_t       cap_ms;     /* when it was triggered */
    uint32_t       cap_rec;    /* rec_n then */
    uint32_t       cap_post;   /* records since */
    char          *cap_head;   /* its first lines, written at the trigger */
    uint32_t       cap_count;  /* snapshots written for failures ... */
    uint32_t       cap_shots;  /* ... for screenshots ... */
    uint32_t       cap_finds;  /* ... and for findings */
    int            cap_kind;   /* the one recording's: CAP_* */
    uint32_t       cap_end_ms;  /* the end of the last snapshot for a failure or a screenshot ... */
    uint32_t       fail_end_ms; /* ... of the last for a failure (a screenshot's or finding's never holds one back) ... */
    uint32_t       find_end_ms; /* ... and of the last for a finding (which holds back findings alone) */
    uint32_t       cap_skipped; /* triggers not captured since the last snapshot */
    uint32_t       find_seen;    /* the thread's: the finding kinds a snapshot has held, a bit each */
    uint32_t       find_skipped; /* the thread's: findings not captured since the last finding's snapshot */
    int            find_full_told; /* the thread's: the findings' limit was named */
};

/* What a capture is for. */
#define CAP_FAIL 0
#define CAP_SHOT 1
#define CAP_FIND 2

static uint32_t
cap_number(const vgpu98_gpu_t *g)
{
    return g->cap_count + g->cap_shots + g->cap_finds + 1;
}

static int    gpu_can_run(const vgpu98_gpu_t *g);
static void   gpu_thread(void *priv);

/* What the GPU thread is in, for the watchdog. */
#define DOG_IDLE    0
#define DOG_PACKET  1
#define DOG_BATCH   2
#define DOG_CAPTURE 3
#define DOG_RESET   4
#define DOG_TOUCH   5
#define DOG_SHOW    6
#define DOG_LOG     7
#define DOG_LOCK    8 /* waiting for the GPU's lock */
/* Nothing sent for this long after 3D drawing, with a device open: named
   (wall clock; at this length the emulation's pace no longer matters). */
#define DOG_QUIET_MS 300000

static void
dog_set(vgpu98_gpu_t *g, int phase, uint32_t op)
{
    if (!g->dog)
        return;
    if (phase != DOG_IDLE) {
        atomic_store_explicit(&g->dog_op, op, memory_order_relaxed);
        atomic_store_explicit(&g->dog_since, plat_get_ticks(), memory_order_relaxed);
    }
    atomic_store_explicit(&g->dog_phase, phase, memory_order_release);
}

int
vgpu98_gpu_step(vgpu98_gpu_t *g, int step)
{
    const int before = atomic_load_explicit(&g->dog_step, memory_order_relaxed);

    if (!g->dog || (step == before))
        return before;
    atomic_store_explicit(&g->dog_wait_s, 0, memory_order_relaxed);
    atomic_store_explicit(&g->dog_step_since, plat_get_ticks(), memory_order_relaxed);
    atomic_store_explicit(&g->dog_step, step, memory_order_release);
    return before;
}

int
vgpu98_gpu_waited(void *priv, uint32_t seconds)
{
    vgpu98_gpu_t *g = (vgpu98_gpu_t *) priv;

    if (g->dog) {
        atomic_store_explicit(&g->dog_wait_at, plat_get_ticks(), memory_order_relaxed);
        atomic_store_explicit(&g->dog_wait_s, seconds, memory_order_release);
    }
    return atomic_load(&g->closing);
}
static size_t surf_row_bytes(const vgpu98_surf_t *s);

/* --- Logging -------------------------------------------------------------- */

static void
format_stamp(time_t t, char *buf, size_t size)
{
    const struct tm *tm;
#ifdef _WIN32
    tm = localtime(&t); /* per thread in the Windows C runtime */
#else
    struct tm tmb;

    tm = localtime_r(&t, &tmb);
#endif
    if (!tm || !strftime(buf, size, "%Y-%m-%d %H:%M:%S", tm))
        snprintf(buf, size, "%lld", (long long) t);
}

void
vgpu98_log_stamp(char *buf, size_t size)
{
    format_stamp(time(NULL), buf, size);
}

uint64_t
vgpu98_diag_us(void)
{
    struct timespec ts;

    if (!timespec_get(&ts, TIME_UTC))
        return 0;
    return (uint64_t) ts.tv_sec * 1000000u + (uint64_t) (ts.tv_nsec / 1000);
}

static uint64_t
us_since(uint64_t t0)
{
    const uint64_t t = vgpu98_diag_us();

    return (t > t0) ? t - t0 : 0;
}

/* The thread's timestamp, formatted once a second. */
static const char *
gpu_stamp(vgpu98_gpu_t *g)
{
    time_t t = time(NULL);

    if (t != g->stamp_time) {
        g->stamp_time = t;
        format_stamp(t, g->stamp, sizeof(g->stamp));
    }
    return g->stamp;
}

void
vgpu98_gpu_log(vgpu98_gpu_t *g, int sources, const char *fmt, ...)
{
    char    line[1024];
    char    stamp[24];
    va_list ap;

    if (!(g->log_mask & sources))
        return;
    vgpu98_log_stamp(stamp, sizeof(stamp));
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    pclog("%s %s", stamp, line);
}

int
vgpu98_gpu_logging(const vgpu98_gpu_t *g, int sources)
{
    return (g->log_mask & sources) != 0;
}

vgpu98_stats_t *
vgpu98_gpu_stats(vgpu98_gpu_t *g)
{
    return &g->stats;
}

/* Lines gathered into blocks (the thread only): 86Box's logger flushes the
   file on every call. */
static void
out_flush(vgpu98_gpu_t *g)
{
    if (g->out_len) {
        pclog("%s", g->out);
        g->out_len = 0;
        g->out[0]  = 0;
    }
}

static void
out_line(vgpu98_gpu_t *g, const char *fmt, ...)
{
    char    line[512];
    int     n;
    va_list ap;

    va_start(ap, fmt);
    n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n <= 0)
        return;
    if (n >= (int) sizeof(line))
        n = sizeof(line) - 1;
    if (g->out_len + n >= OUT_BLOCK)
        out_flush(g);
    memcpy(g->out + g->out_len, line, n + 1);
    g->out_len += n;
}

/* Text of several lines, each ending in a newline. */
static void
out_text(vgpu98_gpu_t *g, const char *s)
{
    while (*s) {
        const char *e = strchr(s, '\n');
        int         n = e ? (int) (e - s) : (int) strlen(s);

        out_line(g, "%.*s\n", (n > 500) ? 500 : n, s);
        s += n + (e ? 1 : 0);
    }
}

static const char *
op_name(uint32_t op)
{
    switch (op) {
        case VGPU98_OP_NOP:
            return "NOP";
        case VGPU98_OP_FENCE:
            return "FENCE";
        case VGPU98_OP_SYNC:
            return "SYNC";
        case VGPU98_OP_FLIP:
            return "FLIP";
        case VGPU98_OP_DISCARD:
            return "DISCARD";
        case VGPU98_OP_FILL:
            return "FILL";
        case VGPU98_OP_UPLOAD:
            return "UPLOAD";
        case VGPU98_OP_DOWNLOAD:
            return "DOWNLOAD";
        case VGPU98_OP_BLT:
            return "BLT";
        case VGPU98_OP_TARGET:
            return "TARGET";
        case VGPU98_OP_VIEWPORT:
            return "VIEWPORT";
        case VGPU98_OP_WRANGE:
            return "WRANGE";
        case VGPU98_OP_STATE:
            return "STATE";
        case VGPU98_OP_TEXTURE:
            return "TEXTURE";
        case VGPU98_OP_PALETTE:
            return "PALETTE";
        case VGPU98_OP_DECLARE:
            return "DECLARE";
        case VGPU98_OP_STREAM:
            return "STREAM";
        case VGPU98_OP_INDICES:
            return "INDICES";
        case VGPU98_OP_INLINE:
            return "INLINE";
        case VGPU98_OP_DRAW:
            return "DRAW";
        case VGPU98_OP_ZRANGE:
            return "ZRANGE";
        case VGPU98_OP_TRANSFORM:
            return "TRANSFORM";
        case VGPU98_OP_LIGHT:
            return "LIGHT";
        case VGPU98_OP_LIGHTENABLE:
            return "LIGHTENABLE";
        case VGPU98_OP_MATERIAL:
            return "MATERIAL";
        case VGPU98_OP_CLIPPLANE:
            return "CLIPPLANE";
        case VGPU98_OP_PROGRAM:
            return "PROGRAM";
        case VGPU98_OP_BIND:
            return "BIND";
        case VGPU98_OP_CONSTANTS:
            return "CONSTANTS";
        case VGPU98_OP_CLEAR:
            return "CLEAR";
        case VGPU98_GL_OPEN:
            return "GL_OPEN";
        case VGPU98_GL_CLOSE:
            return "GL_CLOSE";
        case VGPU98_GL_STATE:
            return "GL_STATE";
        case VGPU98_GL_DRAW:
            return "GL_DRAW";
        case VGPU98_GL_CLEAR:
            return "GL_CLEAR";
        case VGPU98_GL_SWAP:
            return "GL_SWAP";
        case VGPU98_GL_TEXDATA:
            return "GL_TEXDATA";
        case VGPU98_GL_PALETTE:
            return "GL_PALETTE";
        case VGPU98_GL_LFB_READ:
            return "GL_LFB_READ";
        case VGPU98_GL_LFB_WRITE:
            return "GL_LFB_WRITE";
        case VGPU98_GL_GAMMA:
            return "GL_GAMMA";
        case VGPU98_OGL_GROUP:
            return "OGL_GROUP";
        case VGPU98_OGL_CONTEXT:
            return "OGL_CONTEXT";
        case VGPU98_OGL_DRAWABLE:
            return "OGL_DRAWABLE";
        case VGPU98_OGL_BIND:
            return "OGL_BIND";
        case VGPU98_OGL_STATE:
            return "OGL_STATE";
        case VGPU98_OGL_DRAW:
            return "OGL_DRAW";
        case VGPU98_OGL_CLEAR:
            return "OGL_CLEAR";
        case VGPU98_OGL_SWAP:
            return "OGL_SWAP";
        case VGPU98_OGL_TEXIMAGE:
            return "OGL_TEXIMAGE";
        case VGPU98_OGL_TEXPARAM:
            return "OGL_TEXPARAM";
        case VGPU98_OGL_TEXDELETE:
            return "OGL_TEXDELETE";
        case VGPU98_OGL_PIXELS:
            return "OGL_PIXELS";
        case VGPU98_OGL_READ:
            return "OGL_READ";
        case VGPU98_OGL_TEXREAD:
            return "OGL_TEXREAD";
        default:
            return ((op >= VGPU98_GL_FIRST) && (op <= VGPU98_GL_LAST))     ? "GL_?"
                   : ((op >= VGPU98_OGL_FIRST) && (op <= VGPU98_OGL_LAST)) ? "OGL_?"
                                                                           : "?";
    }
}

/* --- Formats and surfaces -------------------------------------------------- */

int
vgpu98_fmt_bpp(uint32_t format)
{
    switch (format) {
        case VGPU98_FMT_P8:
        case VGPU98_FMT_L8:
        case VGPU98_FMT_A8:   /* 2.1 */
        case VGPU98_FMT_A4L4: /* 2.1 */
            return 1;
        case VGPU98_FMT_R5G6B5:
        case VGPU98_FMT_X1R5G5B5:
        case VGPU98_FMT_A1R5G5B5:
        case VGPU98_FMT_A4R4G4B4:
        case VGPU98_FMT_X4R4G4B4: /* 2.1 */
        case VGPU98_FMT_A8L8:
        case VGPU98_FMT_V8U8:
        case VGPU98_FMT_L6V5U5:
        case VGPU98_FMT_D16:
            return 2;
        case VGPU98_FMT_X8R8G8B8:
        case VGPU98_FMT_A8R8G8B8:
        case VGPU98_FMT_X8L8V8U8:
        case VGPU98_FMT_Q8W8V8U8:    /* 2.1 */
        case VGPU98_FMT_V16U16:      /* 2.2 */
        case VGPU98_FMT_G16R16:      /* 2.2 */
        case VGPU98_FMT_A2W10V10U10: /* 2.2 */
        case VGPU98_FMT_D24X8:
        case VGPU98_FMT_D24S8:
            return 4;
        default:
            return 0;
    }
}

int
vgpu98_fmt_fixed8(uint32_t format)
{
    return (format == VGPU98_FMT_A8) || (format == VGPU98_FMT_X4R4G4B4) || (format == VGPU98_FMT_A4L4) ||
           (format == VGPU98_FMT_Q8W8V8U8);
}

/* Command set 2.2's formats: surfaces only with PROFILE.PROGRAMS (a GPU
   without it is exactly command set 2.1). */
int
vgpu98_fmt_programs(uint32_t format)
{
    return (format == VGPU98_FMT_V16U16) || (format == VGPU98_FMT_G16R16) || (format == VGPU98_FMT_A2W10V10U10);
}

int
vgpu98_fmt_is_dxt(uint32_t format)
{
    return (format >= VGPU98_FMT_DXT1) && (format <= VGPU98_FMT_DXT5);
}

int
vgpu98_fmt_is_depth(uint32_t format)
{
    return (format >= VGPU98_FMT_D16) && (format <= VGPU98_FMT_D24S8);
}

/* Bytes of one row: pixels, or for DXT a row of 4x4 blocks. */
static size_t
surf_row_bytes(const vgpu98_surf_t *s)
{
    if (vgpu98_fmt_is_dxt(s->format))
        return (size_t) ((s->width + 3) / 4) * ((s->format == VGPU98_FMT_DXT1) ? 8 : 16);
    return (size_t) s->width * vgpu98_fmt_bpp(s->format);
}

int
vgpu98_surf_parse(const vgpu98_vram_t *v, const uint32_t *p, vgpu98_surf_t *s)
{
    uint32_t rows;
    uint64_t end;
    size_t   row;

    s->offset = p[0];
    s->width  = p[1] & 0xffff;
    s->height = p[1] >> 16;
    s->pitch  = p[2] & 0xffff;
    s->format = (p[2] >> 16) & 0xff;
    s->bytes  = 0;
    if ((p[2] >> 24) || !s->width || !s->height)
        return VGPU98_ERR_ARGUMENT;
    if (!vgpu98_fmt_bpp(s->format) && !vgpu98_fmt_is_dxt(s->format))
        return VGPU98_ERR_ARGUMENT;

    row  = surf_row_bytes(s);
    rows = vgpu98_fmt_is_dxt(s->format) ? (s->height + 3) / 4 : s->height;
    if (s->pitch < row)
        return VGPU98_ERR_ARGUMENT;
    end = (uint64_t) s->offset + (uint64_t) (rows - 1) * s->pitch + row;
    if (end > v->size)
        return VGPU98_ERR_ARGUMENT;
    s->bytes = (uint32_t) (end - s->offset);
    return 0;
}

typedef struct rect_t {
    uint32_t left, top, right, bottom;
} rect_t;

static int
rect_parse(const uint32_t *p, const vgpu98_surf_t *s, rect_t *r)
{
    r->left   = p[0] & 0xffff;
    r->top    = p[0] >> 16;
    r->right  = p[1] & 0xffff;
    r->bottom = p[1] >> 16;
    if ((r->right > s->width) || (r->bottom > s->height) || (r->left > r->right) || (r->top > r->bottom))
        return VGPU98_ERR_ARGUMENT;
    return 0;
}

static int
rect_empty(const rect_t *r)
{
    return (r->left == r->right) || (r->top == r->bottom);
}

/* The bytes a rectangle of a (non-DXT) surface spans: [*start, *start + *len). */
static void
rect_span(const vgpu98_surf_t *s, const rect_t *r, uint32_t *start, uint32_t *len)
{
    const uint32_t bpp = (uint32_t) vgpu98_fmt_bpp(s->format);

    *start = s->offset + r->top * s->pitch + r->left * bpp;
    *len   = (r->bottom - r->top - 1) * s->pitch + (r->right - r->left) * bpp;
}

/* --- Video memory written by the GPU ------------------------------------- */

void
vgpu98_gpu_wrote(vgpu98_gpu_t *g, uint32_t offset, uint32_t len)
{
    uint32_t first;
    uint32_t end;
    uint32_t p;
    int      step;

    if (!len)
        return;
    first = offset >> VGPU98_PAGE_SHIFT;
    end   = ((offset + len - 1) >> VGPU98_PAGE_SHIFT) + 1;
    /* The bytes before their pages' versions, as for the CPU's writes. */
    atomic_thread_fence(memory_order_release);
    for (p = first; p < end; p++)
        g->vram.gpu_pages[p]++;
    g->vram.gpu_pages[-1]++;

    step = vgpu98_gpu_step(g, VGPU98_STEP_LOCK);
    thread_wait_mutex(g->lock);
    vgpu98_gpu_step(g, step);
    if (!g->dirty_end) {
        g->dirty_first = first;
        g->dirty_end   = end;
    } else {
        if (first < g->dirty_first)
            g->dirty_first = first;
        if (end > g->dirty_end)
            g->dirty_end = end;
    }
    thread_release_mutex(g->lock);
}

int
vgpu98_gpu_take_dirty(vgpu98_gpu_t *g, uint32_t *first, uint32_t *count)
{
    int ret = 0;

    thread_wait_mutex(g->lock);
    if (g->dirty_end) {
        *first         = g->dirty_first;
        *count         = g->dirty_end - g->dirty_first;
        g->dirty_end   = 0;
        g->dirty_first = 0;
        ret            = 1;
    }
    thread_release_mutex(g->lock);
    return ret;
}

uint32_t *
vgpu98_gpu_cpu_pages(vgpu98_gpu_t *g)
{
    return g->vram.cpu_pages;
}

uint32_t
vgpu98_gpu_generation(vgpu98_gpu_t *g)
{
    return *(volatile uint32_t *) &g->generation;
}

int
vgpu98_gpu_reset_since(vgpu98_gpu_t *g, uint32_t generation)
{
    return *(volatile uint32_t *) &g->generation != generation;
}

static void status_write_locked(const vgpu98_gpu_t *g);

/* Brings [offset, offset + len) up to date in video memory. */
static void
gpu_sync(vgpu98_gpu_t *g, uint32_t offset, uint32_t len, int cause)
{
    if (g->r3d && len)
        vgpu98_r3d_sync(g->r3d, offset, len, cause);
}

/* A frame's GPU-newer pixels, [start, start + len), to the display's layer
   (`copy_back`: to video memory). Past the end of video memory the display
   wraps to its start, and so does this. */
static void
gpu_show_frame(vgpu98_gpu_t *g, uint32_t start, uint64_t len, uint32_t bpp, int copy_back)
{
    uint64_t first;

    if (len > g->vram.size)
        len = g->vram.size;
    first = len;
    if (first > g->vram.size - start)
        first = g->vram.size - start;
    if (copy_back) {
        gpu_sync(g, start, (uint32_t) first, VGPU98_CB_FRAME);
        gpu_sync(g, 0, (uint32_t) (len - first), VGPU98_CB_FRAME);
    } else {
        vgpu98_r3d_publish(g->r3d, g->disp, start, start + (uint32_t) first, bpp);
        vgpu98_r3d_publish(g->r3d, g->disp, 0, (uint32_t) (len - first), bpp);
    }
}

/* The frame the display engine scans out is never stale: at the end of
   every batch and at every FENCE, its GPU-newer pixels go to the display's
   layer straight from the host GPU (or, without the display on the host
   GPU, back to video memory). `copy_back`: to video memory in any case. */
static void
gpu_show_visible(vgpu98_gpu_t *g, int copy_back)
{
    int      enabled;
    uint32_t start;
    uint32_t pitch;
    uint32_t height;
    uint32_t bpp;
    uint64_t len;

    if (!g->r3d)
        return;
    g->host.display(g->host.priv, &enabled, &start, &pitch, &height, &bpp);
    if (!enabled || (start >= g->vram.size))
        return;
    len = (uint64_t) pitch * height;
    vgpu98_r3d_flush(g->r3d);
    gpu_show_frame(g, start, len, bpp, copy_back);
}

/* A guest access to a page the GPU holds newer pixels for (under lock; let
   go meanwhile): the images over the page are copied back. */
static void
touch_service_locked(vgpu98_gpu_t *g)
{
    while (g->touch_served != g->touch_serial) {
        const uint32_t serial = g->touch_serial;
        const uint32_t page   = g->touch_page;

        thread_release_mutex(g->lock);
        gpu_sync(g, page << VGPU98_PAGE_SHIFT, 1u << VGPU98_PAGE_SHIFT, VGPU98_CB_TRAP);
        thread_wait_mutex(g->lock);
        g->touch_served = serial;
        thread_set_event(g->touch_done);
    }
}

/* --- 2D on the host CPU ---------------------------------------------------- */

static uint32_t
px_get(const uint8_t *p, int bpp)
{
    switch (bpp) {
        case 1:
            return p[0];
        case 2:
            return p[0] | (p[1] << 8);
        default:
            return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t) p[3] << 24);
    }
}

static void
px_put(uint8_t *p, int bpp, uint32_t v)
{
    p[0] = (uint8_t) v;
    if (bpp >= 2)
        p[1] = (uint8_t) (v >> 8);
    if (bpp == 4) {
        p[2] = (uint8_t) (v >> 16);
        p[3] = (uint8_t) (v >> 24);
    }
}

/* The bits a colour key compares: the format's colour bits, not alpha or
   unused bits (spec/gpu-commands.md, BLT). */
static uint32_t
key_mask(uint32_t format)
{
    switch (format) {
        case VGPU98_FMT_X1R5G5B5:
        case VGPU98_FMT_A1R5G5B5:
            return 0x7fff;
        case VGPU98_FMT_A4R4G4B4:
        case VGPU98_FMT_X4R4G4B4: /* 2.1 */
            return 0x0fff;
        case VGPU98_FMT_A8L8:
            return 0x00ff;
        case VGPU98_FMT_A4L4: /* 2.1 */
            return 0x0f;
        case VGPU98_FMT_A8: /* 2.1: no colour bits, so every pixel matches */
            return 0;
        case VGPU98_FMT_X8R8G8B8:
        case VGPU98_FMT_A8R8G8B8:
        case VGPU98_FMT_X8L8V8U8:
            return 0x00ffffff;
        case VGPU98_FMT_A2W10V10U10: /* 2.2: alpha ignored, as every format's */
            return 0x3fffffff;
        case VGPU98_FMT_D24X8:
            return 0xffffff00;
        default:
            return (vgpu98_fmt_bpp(format) == 4) ? 0xffffffff : ((vgpu98_fmt_bpp(format) == 2) ? 0xffff : 0xff);
    }
}

static int
blit_fill(vgpu98_gpu_t *g, const uint32_t *p)
{
    vgpu98_surf_t s;
    rect_t        r;
    uint32_t      start;
    uint32_t      len;
    uint32_t      x;
    uint32_t      y;
    int           bpp;
    int           err;

    if ((err = vgpu98_surf_parse(&g->vram, p, &s)) || (err = rect_parse(p + 3, &s, &r)))
        return err;
    if (vgpu98_fmt_is_dxt(s.format))
        return VGPU98_ERR_ARGUMENT;
    if ((vgpu98_fmt_fixed8(s.format) && !g->vram.fixed8) || (vgpu98_fmt_programs(s.format) && !g->vram.programs))
        return VGPU98_ERR_PROFILE;
    if (rect_empty(&r))
        return 0;
    bpp = vgpu98_fmt_bpp(s.format);
    rect_span(&s, &r, &start, &len);
    gpu_sync(g, start, len, VGPU98_CB_2D);
    for (y = r.top; y < r.bottom; y++) {
        uint8_t *row = g->vram.data + s.offset + y * s.pitch + r.left * bpp;

        if (bpp == 1)
            memset(row, (uint8_t) p[5], r.right - r.left);
        else {
            for (x = r.left; x < r.right; x++, row += bpp)
                px_put(row, bpp, p[5]);
        }
    }
    vgpu98_gpu_wrote(g, start, len);
    if (g->r3d)
        vgpu98_r3d_written(g->r3d, &s, r.left, r.top, r.right, r.bottom);
    return 0;
}

/* A DXT rectangle of whole blocks: its left and top on the grid of 4, its
   right and bottom too or at the surface's edge. */
static int
dxt_aligned(const vgpu98_surf_t *s, const rect_t *r)
{
    return !(r->left & 3) && !(r->top & 3) && (!(r->right & 3) || (r->right == s->width)) &&
           (!(r->bottom & 3) || (r->bottom == s->height));
}

/* BLT of DXT surfaces: whole blocks copied as they are (a texture load
   between textures in video memory). The same size, no flags. */
static int
blit_copy_blocks(vgpu98_gpu_t *g, const vgpu98_surf_t *ss, const rect_t *sr, const vgpu98_surf_t *ds, const rect_t *dr,
                 uint32_t flags)
{
    const uint32_t block = (ss->format == VGPU98_FMT_DXT1) ? 8 : 16;
    uint32_t       rows;
    uint32_t       row_bytes;
    uint32_t       s0;
    uint32_t       d0;
    uint32_t       dlen;
    uint32_t       y;
    size_t         need;

    if (flags || ((sr->right - sr->left) != (dr->right - dr->left)) || ((sr->bottom - sr->top) != (dr->bottom - dr->top)) ||
        !dxt_aligned(ss, sr) || !dxt_aligned(ds, dr))
        return VGPU98_ERR_ARGUMENT;
    if (rect_empty(sr))
        return 0;
    rows      = (sr->bottom - sr->top + 3) / 4;
    row_bytes = ((sr->right - sr->left + 3) / 4) * block;
    s0        = ss->offset + (sr->top / 4) * ss->pitch + (sr->left / 4) * block;
    d0        = ds->offset + (dr->top / 4) * ds->pitch + (dr->left / 4) * block;
    dlen      = (rows - 1) * ds->pitch + row_bytes;
    gpu_sync(g, s0, (rows - 1) * ss->pitch + row_bytes, VGPU98_CB_2D);
    gpu_sync(g, d0, dlen, VGPU98_CB_2D);
    /* The whole source read first: overlapping copies work. */
    need = (size_t) rows * row_bytes;
    if (need > g->tmp_size) {
        free(g->tmp);
        g->tmp      = malloc(need);
        g->tmp_size = g->tmp ? need : 0;
        if (!g->tmp)
            return 0; /* out of host memory: the copy is lost, the GPU goes on */
    }
    for (y = 0; y < rows; y++)
        memcpy(g->tmp + (size_t) y * row_bytes, g->vram.data + s0 + (size_t) y * ss->pitch, row_bytes);
    for (y = 0; y < rows; y++)
        memcpy(g->vram.data + d0 + (size_t) y * ds->pitch, g->tmp + (size_t) y * row_bytes, row_bytes);
    vgpu98_gpu_wrote(g, d0, dlen);
    if (g->r3d)
        vgpu98_r3d_written(g->r3d, ds, dr->left, dr->top, dr->right, dr->bottom);
    return 0;
}

static int
blit_copy(vgpu98_gpu_t *g, const uint32_t *p)
{
    vgpu98_surf_t  ss;
    vgpu98_surf_t  ds;
    rect_t         sr;
    rect_t         dr;
    uint32_t       flags = p[10];
    uint32_t       mask;
    uint32_t       skey;
    uint32_t       dkey;
    uint32_t       sw;
    uint32_t       sh;
    uint32_t       dw;
    uint32_t       dh;
    uint32_t       start;
    uint32_t       len;
    uint32_t       x;
    uint32_t       y;
    size_t         need;
    int            bpp;
    int            err;

    if ((err = vgpu98_surf_parse(&g->vram, p, &ss)) || (err = rect_parse(p + 3, &ss, &sr)) ||
        (err = vgpu98_surf_parse(&g->vram, p + 5, &ds)) || (err = rect_parse(p + 8, &ds, &dr)))
        return err;
    if ((ss.format != ds.format) || (flags & ~0xfu))
        return VGPU98_ERR_ARGUMENT;
    if ((vgpu98_fmt_fixed8(ss.format) && !g->vram.fixed8) || (vgpu98_fmt_programs(ss.format) && !g->vram.programs))
        return VGPU98_ERR_PROFILE;
    if (vgpu98_fmt_is_dxt(ss.format))
        return blit_copy_blocks(g, &ss, &sr, &ds, &dr, flags);
    if (rect_empty(&sr) || rect_empty(&dr))
        return 0;
    bpp  = vgpu98_fmt_bpp(ss.format);
    mask = key_mask(ss.format);
    skey = p[11] & mask;
    dkey = p[12] & mask;
    sw   = sr.right - sr.left;
    sh   = sr.bottom - sr.top;
    dw   = dr.right - dr.left;
    dh   = dr.bottom - dr.top;

    /* Both up to date in video memory; then the whole source is read before
       anything is written, so overlapping copies work. */
    rect_span(&ss, &sr, &start, &len);
    gpu_sync(g, start, len, VGPU98_CB_2D);
    rect_span(&ds, &dr, &start, &len);
    gpu_sync(g, start, len, VGPU98_CB_2D);

    need = (size_t) sw * sh * bpp;
    if (need > g->tmp_size) {
        free(g->tmp);
        g->tmp      = malloc(need);
        g->tmp_size = g->tmp ? need : 0;
        if (!g->tmp)
            return 0; /* out of host memory: the copy is lost, the GPU goes on */
    }
    for (y = 0; y < sh; y++)
        memcpy(g->tmp + (size_t) y * sw * bpp, g->vram.data + ss.offset + (sr.top + y) * ss.pitch + sr.left * bpp,
               (size_t) sw * bpp);

    for (y = 0; y < dh; y++) {
        /* Point sampling at the destination pixel's centre. */
        uint32_t sy  = (uint32_t) (((uint64_t) (2 * y + 1) * sh) / (2 * (uint64_t) dh));
        uint8_t *row = g->vram.data + ds.offset + (dr.top + y) * ds.pitch + dr.left * bpp;

        if (flags & VGPU98_BLT_MIRROR_Y)
            sy = sh - 1 - sy;
        for (x = 0; x < dw; x++, row += bpp) {
            uint32_t sx = (uint32_t) (((uint64_t) (2 * x + 1) * sw) / (2 * (uint64_t) dw));
            uint32_t v;

            if (flags & VGPU98_BLT_MIRROR_X)
                sx = sw - 1 - sx;
            v = px_get(g->tmp + ((size_t) sy * sw + sx) * bpp, bpp);
            if ((flags & VGPU98_BLT_SRCKEY) && ((v & mask) == skey))
                continue;
            if ((flags & VGPU98_BLT_DSTKEY) && ((px_get(row, bpp) & mask) != dkey))
                continue;
            px_put(row, bpp, v);
        }
    }
    vgpu98_gpu_wrote(g, start, len);
    if (g->r3d)
        vgpu98_r3d_written(g->r3d, &ds, dr.left, dr.top, dr.right, dr.bottom);
    return 0;
}

/* --- Main memory through the page table (interface 3.0) ---------------------- */

int
vgpu98_gpu_gpuvm_read(vgpu98_gpu_t *g, uint32_t address, uint32_t len, uint8_t *dst, int kind)
{
    const uint64_t ram_bytes = ram ? (uint64_t) mem_size * 1024 : 0;
    const uint32_t base      = g->gpuvm_base;
    const uint32_t entries   = g->gpuvm_entries;
    uint64_t       at        = address;
    const uint64_t end       = (uint64_t) address + len;

    while (at < end) {
        const uint32_t n     = (uint32_t) (at >> VGPU98_PAGE_SHIFT);
        const uint32_t off   = (uint32_t) (at & 0xfff);
        const uint32_t chunk = (end - at < 0x1000 - off) ? (uint32_t) (end - at) : 0x1000 - off;
        uint64_t       entry_at;
        uint32_t       e;
        uint64_t       page;

        if (!base || (n >= entries))
            return VGPU98_ERR_UNMAPPED;
        /* The entry itself in guest RAM (GPUVM_ENTRIES may have grown since
           GPUVM_BASE was checked), then the page it names. */
        entry_at = (uint64_t) base + 4 * (uint64_t) n;
        if (entry_at + 4 > ram_bytes)
            return VGPU98_ERR_BAD_PAGE;
        e = ram[entry_at] | (ram[entry_at + 1] << 8) | (ram[entry_at + 2] << 16) | ((uint32_t) ram[entry_at + 3] << 24);
        if (!(e & VGPU98_GPUVM_VALID))
            return VGPU98_ERR_UNMAPPED;
        /* Any page of guest RAM but the adapter area (A0000h-FFFFFh), whose
           RAM is not what the processor sees there. */
        page = e & ~0xfffu;
        if (((page >= 0xa0000) && (page < 0x100000)) || (page + 0x1000 > ram_bytes))
            return VGPU98_ERR_BAD_PAGE;
        if (dst) {
            memcpy(dst, ram + page + off, chunk);
            dst += chunk;
        }
        at += chunk;
    }
    if (dst && (kind >= 0) && (kind < VGPU98_GPUVM_KINDS))
        g->stats.gpuvm_bytes[kind] += len;
    return 0;
}

/* Main memory written through the page table (DOWNLOAD): the pages are
   checked as a read checks them (vgpu98_gpu_gpuvm_read without a
   destination), all of them before any byte goes, and each entry again as
   its page is written (the guest may change the table meanwhile: nothing
   goes outside its RAM whatever it does). The destination is the DirectX
   driver's pool, data the guest's CPU never runs, so the recompiler's page
   records (the emulation thread's) are left alone. */
int
vgpu98_gpu_gpuvm_write(vgpu98_gpu_t *g, uint32_t address, uint32_t len, const uint8_t *src, int kind)
{
    uint64_t       at  = address;
    const uint64_t end = (uint64_t) address + len;
    int            err;

    if ((err = vgpu98_gpu_gpuvm_read(g, address, len, NULL, -1)))
        return err;
    while (at < end) {
        const uint64_t ram_bytes = ram ? (uint64_t) mem_size * 1024 : 0;
        const uint32_t base      = g->gpuvm_base;
        const uint32_t n         = (uint32_t) (at >> VGPU98_PAGE_SHIFT);
        const uint32_t off       = (uint32_t) (at & 0xfff);
        const uint32_t chunk     = (end - at < 0x1000 - off) ? (uint32_t) (end - at) : 0x1000 - off;
        const uint64_t e_at      = (uint64_t) base + 4 * (uint64_t) n;
        uint32_t       e;
        uint64_t       page;

        if (!base || (n >= g->gpuvm_entries) || (e_at + 4 > ram_bytes))
            return VGPU98_ERR_UNMAPPED;
        e = ram[e_at] | (ram[e_at + 1] << 8) | (ram[e_at + 2] << 16) | ((uint32_t) ram[e_at + 3] << 24);
        page = e & ~0xfffu;
        if (!(e & VGPU98_GPUVM_VALID) || ((page >= 0xa0000) && (page < 0x100000)) || (page + 0x1000 > ram_bytes))
            return VGPU98_ERR_BAD_PAGE;
        memcpy(ram + page + off, src, chunk);
        src += chunk;
        at += chunk;
    }
    if ((kind >= 0) && (kind < VGPU98_GPUVM_KINDS))
        g->stats.gpuvm_bytes[kind] += len;
    return 0;
}

/* UPLOAD: bytes from outside video memory into a rectangle of a surface,
   exactly as they are (spec/gpu-commands.md). Video memory takes them as the
   GPU's own write, after GPU-newer pixels over the same bytes are copied
   back; the host images of the surface that were current take them too. */
static int
blit_upload(vgpu98_gpu_t *g, const uint32_t *p, uint32_t len)
{
    const uint32_t source  = p[5];
    const uint32_t pitch   = p[6];
    const uint32_t address = p[7];
    const int      gpuvm   = (source & VGPU98_UPLOAD_GPUVM) != 0;
    vgpu98_surf_t  s;
    rect_t         r;
    const uint8_t *data;
    uint32_t       rows;
    uint32_t       row_bytes;
    uint32_t       first_row;
    uint32_t       left_bytes;
    uint32_t       start;
    uint32_t       span;
    uint32_t       y;
    uint64_t       need;
    uint32_t       src_pitch = pitch;
    uint32_t       lo;
    uint32_t       hi;
    int            err;

    if ((err = vgpu98_surf_parse(&g->vram, p, &s)) || (err = rect_parse(p + 3, &s, &r)))
        return err;
    if (source & ~VGPU98_UPLOAD_GPUVM)
        return VGPU98_ERR_ARGUMENT;
    if ((gpuvm && !(g->profile & VGPU98_PROFILE_GPUVM)) || (vgpu98_fmt_fixed8(s.format) && !g->vram.fixed8) ||
        (vgpu98_fmt_programs(s.format) && !g->vram.programs))
        return VGPU98_ERR_PROFILE;
    if (rect_empty(&r))
        return 0;
    if (vgpu98_fmt_is_dxt(s.format)) {
        const uint32_t block = (s.format == VGPU98_FMT_DXT1) ? 8 : 16;

        if (!dxt_aligned(&s, &r))
            return VGPU98_ERR_ARGUMENT;
        rows       = (r.bottom - r.top + 3) / 4;
        row_bytes  = ((r.right - r.left + 3) / 4) * block;
        first_row  = r.top / 4;
        left_bytes = (r.left / 4) * block;
    } else {
        const uint32_t bpp = (uint32_t) vgpu98_fmt_bpp(s.format);

        rows       = r.bottom - r.top;
        row_bytes  = (r.right - r.left) * bpp;
        first_row  = r.top;
        left_bytes = r.left * bpp;
    }
    if (pitch < row_bytes)
        return VGPU98_ERR_ARGUMENT;
    need = (uint64_t) (rows - 1) * pitch + row_bytes;
    if (gpuvm) {
        /* The rows read from guest RAM before anything is written: a row
           that cannot be read stops the packet with nothing written. */
        if ((uint64_t) address + need > 0x100000000ull)
            return VGPU98_ERR_UNMAPPED;
        if ((size_t) rows * row_bytes > g->tmp_size) {
            free(g->tmp);
            g->tmp      = malloc((size_t) rows * row_bytes);
            g->tmp_size = g->tmp ? (size_t) rows * row_bytes : 0;
            if (!g->tmp)
                return 0; /* out of host memory: the load is lost, the GPU goes on */
        }
        for (y = 0; y < rows; y++) {
            if ((err = vgpu98_gpu_gpuvm_read(g, address + y * pitch, row_bytes, g->tmp + (size_t) y * row_bytes,
                                             VGPU98_GPUVM_UPLOAD)))
                return err;
        }
        data      = g->tmp;
        src_pitch = row_bytes;
    } else {
        if ((uint64_t) (len - VGPU98_LEN_UPLOAD) * 4 < need)
            return VGPU98_ERR_LENGTH;
        data = (const uint8_t *) (p + VGPU98_LEN_UPLOAD);
    }
    start = s.offset + first_row * s.pitch + left_bytes;
    span  = (rows - 1) * s.pitch + row_bytes;
    gpu_sync(g, start, span, VGPU98_CB_2D);
    g->stats.upload_packets++;
    g->stats.upload_packet_bytes += (uint64_t) rows * row_bytes;
    /* From main memory (the copy a Lock went through):
       only the rows from the first to the last
       that differ from video memory go, so a Lock that wrote nothing moves no
       version and reloads no image (a depth buffer keeps its host precision). */
    lo = 0;
    hi = rows;
    if (gpuvm) {
        while ((lo < hi) && !memcmp(g->vram.data + start + (size_t) lo * s.pitch, data + (size_t) lo * src_pitch, row_bytes))
            lo++;
        while ((hi > lo) &&
               !memcmp(g->vram.data + start + (size_t) (hi - 1) * s.pitch, data + (size_t) (hi - 1) * src_pitch, row_bytes))
            hi--;
        if (lo == hi)
            return 0;
    }
    for (y = lo; y < hi; y++) {
        uint8_t *row = g->vram.data + start + (size_t) y * s.pitch;

        memcpy(row, data + (size_t) y * src_pitch, row_bytes);
    }
    {
        const uint32_t unit   = vgpu98_fmt_is_dxt(s.format) ? 4 : 1; /* pixel rows a row holds */
        const uint32_t top    = r.top + lo * unit;
        const uint32_t bottom = (r.top + hi * unit < r.bottom) ? r.top + hi * unit : r.bottom;
        const uint32_t wstart = start + lo * s.pitch;
        const uint32_t wspan  = (hi - lo - 1) * s.pitch + row_bytes;

        vgpu98_gpu_wrote(g, wstart, wspan);
        if (g->r3d) {
            vgpu98_r3d_written(g->r3d, &s, r.left, top, r.right, bottom);
            vgpu98_r3d_uploaded(g->r3d, &s, r.left, top, r.right, bottom, wstart, wspan);
        }
    }
    return 0;
}

/* DOWNLOAD (command set 2.3): a rectangle of a surface into main memory at
   a GPU address, `pitch` apart, exactly as video memory holds it, its
   GPU-newer pixels copied back first as SYNC does (spec/gpu-commands.md).
   Every row's pages are checked before any byte is written. */
static int
blit_download(vgpu98_gpu_t *g, const uint32_t *p)
{
    const uint32_t pitch   = p[5];
    const uint32_t address = p[6];
    vgpu98_surf_t  s;
    rect_t         r;
    uint32_t       rows;
    uint32_t       row_bytes;
    uint32_t       first_row;
    uint32_t       left_bytes;
    uint32_t       start;
    uint32_t       span;
    uint32_t       y;
    int            err;

    if ((err = vgpu98_surf_parse(&g->vram, p, &s)) || (err = rect_parse(p + 3, &s, &r)))
        return err;
    if ((vgpu98_fmt_fixed8(s.format) && !g->vram.fixed8) || (vgpu98_fmt_programs(s.format) && !g->vram.programs))
        return VGPU98_ERR_PROFILE;
    if (rect_empty(&r))
        return 0;
    if (vgpu98_fmt_is_dxt(s.format)) {
        const uint32_t block = (s.format == VGPU98_FMT_DXT1) ? 8 : 16;

        if (!dxt_aligned(&s, &r))
            return VGPU98_ERR_ARGUMENT;
        rows       = (r.bottom - r.top + 3) / 4;
        row_bytes  = ((r.right - r.left + 3) / 4) * block;
        first_row  = r.top / 4;
        left_bytes = (r.left / 4) * block;
    } else {
        const uint32_t bpp = (uint32_t) vgpu98_fmt_bpp(s.format);

        rows       = r.bottom - r.top;
        row_bytes  = (r.right - r.left) * bpp;
        first_row  = r.top;
        left_bytes = r.left * bpp;
    }
    if (pitch < row_bytes)
        return VGPU98_ERR_ARGUMENT;
    if ((uint64_t) address + (uint64_t) (rows - 1) * pitch + row_bytes > 0x100000000ull)
        return VGPU98_ERR_UNMAPPED;
    for (y = 0; y < rows; y++) {
        if ((err = vgpu98_gpu_gpuvm_read(g, address + y * pitch, row_bytes, NULL, -1)))
            return err;
    }
    start = s.offset + first_row * s.pitch + left_bytes;
    span  = (rows - 1) * s.pitch + row_bytes;
    if (g->r3d)
        vgpu98_r3d_flush(g->r3d);
    gpu_sync(g, start, span, VGPU98_CB_SYNC);
    for (y = 0; y < rows; y++) {
        if ((err = vgpu98_gpu_gpuvm_write(g, address + y * pitch, row_bytes, g->vram.data + start + (size_t) y * s.pitch,
                                          VGPU98_GPUVM_DOWNLOAD)))
            return err;
    }
    return 0;
}

/* --- Packets ---------------------------------------------------------------- */

/* Minimum payload and the PROFILE bit each opcode needs. */
static int
op_info(uint32_t op, uint32_t *min_len, uint32_t *profile)
{
    *profile = VGPU98_PROFILE_COMMANDS;
    switch (op) {
        case VGPU98_OP_NOP:
            *min_len = 0;
            return 1;
        case VGPU98_OP_FENCE:
            *min_len = VGPU98_LEN_FENCE;
            return 1;
        case VGPU98_OP_SYNC:
        case VGPU98_OP_FLIP:
        case VGPU98_OP_DISCARD:
            *min_len = 2;
            return 1;
        case VGPU98_OP_FILL:
            *min_len = VGPU98_LEN_FILL;
            *profile = VGPU98_PROFILE_BLIT;
            return 1;
        case VGPU98_OP_UPLOAD:
            *min_len = VGPU98_LEN_UPLOAD;
            *profile = VGPU98_PROFILE_BLIT;
            return 1;
        case VGPU98_OP_DOWNLOAD:
            *min_len = VGPU98_LEN_DOWNLOAD;
            *profile = VGPU98_PROFILE_BLIT | VGPU98_PROFILE_GPUVM | VGPU98_PROFILE_DOWNLOAD;
            return 1;
        case VGPU98_OP_BLT:
            *min_len = VGPU98_LEN_BLT;
            *profile = VGPU98_PROFILE_BLIT;
            return 1;
        case VGPU98_OP_TARGET:
            *min_len = VGPU98_LEN_TARGET;
            break;
        case VGPU98_OP_VIEWPORT:
            *min_len = VGPU98_LEN_VIEWPORT;
            break;
        case VGPU98_OP_WRANGE:
            *min_len = VGPU98_LEN_WRANGE;
            break;
        case VGPU98_OP_STATE:
            *min_len = 0;
            break;
        case VGPU98_OP_TEXTURE:
            *min_len = VGPU98_LEN_TEXTURE;
            break;
        case VGPU98_OP_PALETTE:
            *min_len = VGPU98_LEN_PALETTE;
            break;
        case VGPU98_OP_CLEAR:
            *min_len = VGPU98_LEN_CLEAR;
            break;
        case VGPU98_OP_DECLARE:
            *min_len = VGPU98_LEN_DECLARE;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_STREAM:
            *min_len = VGPU98_LEN_STREAM;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_INDICES:
            *min_len = VGPU98_LEN_INDICES;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_INLINE:
            *min_len = VGPU98_LEN_INLINE;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_DRAW:
            *min_len = VGPU98_LEN_DRAW;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_ZRANGE:
            *min_len = VGPU98_LEN_ZRANGE;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_TRANSFORM:
            *min_len = VGPU98_LEN_TRANSFORM;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_LIGHT:
            *min_len = VGPU98_LEN_LIGHT;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_LIGHTENABLE:
            *min_len = VGPU98_LEN_LIGHTENABLE;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_MATERIAL:
            *min_len = VGPU98_LEN_MATERIAL;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_CLIPPLANE:
            *min_len = VGPU98_LEN_CLIPPLANE;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
            return 1;
        case VGPU98_OP_PROGRAM:
            *min_len = VGPU98_LEN_PROGRAM;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX | VGPU98_PROFILE_PROGRAMS;
            return 1;
        case VGPU98_OP_BIND:
            *min_len = VGPU98_LEN_BIND;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX | VGPU98_PROFILE_PROGRAMS;
            return 1;
        case VGPU98_OP_CONSTANTS:
            *min_len = VGPU98_LEN_CONSTANTS;
            *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX | VGPU98_PROFILE_PROGRAMS;
            return 1;
        default:
            /* Command set 2.4's Glide commands: their session at least; the
               Glide device takes or drops the rest, never stopping. */
            if ((op >= VGPU98_GL_FIRST) && (op <= VGPU98_GL_LAST)) {
                *min_len = 1;
                *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_GLIDE;
                return 1;
            }
            /* Command set 2.5's OpenGL commands likewise. */
            if ((op >= VGPU98_OGL_FIRST) && (op <= VGPU98_OGL_LAST)) {
                *min_len = 1;
                *profile = VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_OGL;
                return 1;
            }
            return 0;
    }
    *profile = VGPU98_PROFILE_RENDER3D;
    return 1;
}

/* The ring's first byte: in video memory, or in guest RAM when RING_BASE has
   SYSMEM (interface 2.1). The machine's RAM block is taken afresh each time:
   86Box replaces it only at a hard reset, which closes the card first. */
static const uint8_t *
ring_data(const vgpu98_gpu_t *g)
{
    if (g->ring_base & VGPU98_RING_SYSMEM)
        return ram + (g->ring_base & ~0xfffu);
    return g->vram.data + g->ring_base;
}

/* Reads the packet at RING_HEAD (under lock). Returns 0 with the opcode and
   payload, or an ERROR_CODE. RING_TAIL was written under the same lock after
   the guest wrote the packet, so everything up to it is there to read. */
static int
gpu_fetch(vgpu98_gpu_t *g, uint32_t *op, uint32_t *len)
{
    const uint8_t *ring = ring_data(g);
    uint32_t       at   = g->ring_head;
    uint32_t       avail;
    uint32_t       hdr;
    uint32_t       min_len;
    uint32_t       need_profile;

    hdr  = ring[at] | (ring[at + 1] << 8) | (ring[at + 2] << 16) | ((uint32_t) ring[at + 3] << 24);
    *op  = VGPU98_PKT_OP(hdr);
    *len = VGPU98_PKT_LEN(hdr);
    /* For the rejection line, should the header be refused. */
    g->fetch_hdr = hdr;
    if ((hdr >> 24) || !op_info(*op, &min_len, &need_profile))
        return VGPU98_ERR_OPCODE;
    if ((uint64_t) at + 4 + (uint64_t) *len * 4 > g->ring_size)
        return VGPU98_ERR_RING_END;
    avail = (g->ring_tail > at) ? (g->ring_tail - at) : (g->ring_size - at);
    if ((*len < min_len) || (4 + *len * 4 > avail))
        return VGPU98_ERR_LENGTH;
    if ((g->profile & need_profile) != need_profile)
        return VGPU98_ERR_PROFILE;
    memcpy(g->payload, ring + at + 4, (size_t) *len * 4);
    return 0;
}

static int
range_ok(const vgpu98_gpu_t *g, uint32_t offset, uint32_t len)
{
    return (offset <= g->vram.size) && (len <= g->vram.size - offset);
}

/* A FLIP ends a frame (summary frames): its time since the last one, and
   for a slow one, what the thread did meanwhile. The first FLIP after a
   reset (a mode change, a DOS session) starts the clock again. */
static void
frame_mark(vgpu98_gpu_t *g)
{
    const uint64_t        now = vgpu98_diag_us();
    const vgpu98_stats_t *s   = &g->stats;
    const vgpu98_stats_t *a   = &g->at_flip;

    if (g->frame_last && (s->resets == a->resets)) {
        /* The wall clock may be stepped back. */
        const uint64_t us = (now > g->frame_last) ? now - g->frame_last : 0;
        const uint32_t u  = (us > 0xffffffffu) ? 0xffffffffu : (uint32_t) us;

        if (g->n_frames < FRAMES_KEPT)
            g->frame_us[g->n_frames] = u;
        g->n_frames++;
        if (u > g->frame_worst)
            g->frame_worst = u;
        if (us > FRAME_SLOW_US) {
            g->slow.frames++;
            g->slow.us += us;
            g->slow.idle += s->idle_us - a->idle_us;
            g->slow.flip += s->flip_us - a->flip_us;
            g->slow.upload += s->upload_us - a->upload_us;
            g->slow.copy += s->copy_back_us - a->copy_back_us;
            g->slow.wait += s->gpu_wait_us - a->gpu_wait_us;
            g->slow.pipe += s->pipeline_us - a->pipeline_us;
            g->slow.pipelines += s->new_pipelines - a->new_pipelines;
        }
    }
    g->frame_last = now;
    g->at_flip    = *s;
}

static int
exec_flip(vgpu98_gpu_t *g, const uint32_t *p, uint32_t generation)
{
    uint32_t start    = p[0];
    uint32_t flags    = p[1];
    uint32_t interval = (flags & VGPU98_FLIP_INTERVAL_MASK) >> VGPU98_FLIP_INTERVAL_SHIFT;
    int      enabled;
    uint32_t cur;
    uint32_t pitch;
    uint32_t height;
    uint32_t bpp;
    uint64_t len;
    uint64_t flip_t0 = 0;
    int      step;

    if ((start & 3) || (start >= g->vram.size) || (interval > 4) || (flags & ~0xfu))
        return VGPU98_ERR_ARGUMENT;
    if (interval < 1)
        interval = 1;
    /* The V-sync overrides: an immediate flip asks for no
       wait, and no wait is an immediate flip. */
    interval = vgpu98_gpu_vsync_interval(g, VGPU98_VSYNC_D3D_SHIFT, (flags & VGPU98_FLIP_IMMEDIATE) ? 0 : interval);
    if (interval)
        flags &= ~VGPU98_FLIP_IMMEDIATE;
    else
        flags |= VGPU98_FLIP_IMMEDIATE;
    if (g->log_mask & VGPU98_LOG_SUMMARY)
        frame_mark(g);
    /* The texture audit's display check: the frame drawn so far ended, and
       the one this FLIP shows is at start. */
    if (g->r3d && (g->log_mask & VGPU98_LOG_TEXTURES))
        vgpu98_r3d_frame(g->r3d, start);

    /* The new frame's GPU-newer pixels go to the display's layer before it
       is shown (to video memory without the display on the host GPU). */
    g->host.display(g->host.priv, &enabled, &cur, &pitch, &height, &bpp);
    len = (uint64_t) pitch * height;
    if (g->r3d) {
        vgpu98_r3d_flush(g->r3d);
        gpu_show_frame(g, start, len, bpp, 0);
    }

    step = vgpu98_gpu_step(g, VGPU98_STEP_LOCK);
    thread_wait_mutex(g->lock);
    vgpu98_gpu_step(g, step);
    if (g->generation != generation) {
        thread_release_mutex(g->lock);
        return 0;
    }
    if (flags & VGPU98_FLIP_IMMEDIATE) {
        g->imm_start        = start;
        g->imm_pending      = 1;
        g->flip_queued      = 0;
        g->last_flip_vblank = g->vblank_count;
        thread_release_mutex(g->lock);
        return 0;
    }
    g->flip_pending  = 1;
    g->flip_queued   = 1;
    g->flip_done     = 0;
    g->flip_start    = start;
    g->flip_interval = interval;
    status_write_locked(g);
    if (g->log_mask) {
        g->stats.flips_waited++;
        flip_t0 = vgpu98_diag_us();
    }
    while (!g->flip_done && !g->quit && (g->generation == generation)) {
        /* The vertical blank comes from the emulation thread, which may be
           waiting for a guest access's copy back meanwhile. */
        touch_service_locked(g);
        thread_release_mutex(g->lock);
        thread_wait_event(g->flip_wake, 100);
        /* Manual-reset events off Windows; the state is checked again below. */
        thread_reset_event(g->flip_wake);
        thread_wait_mutex(g->lock);
    }
    if (g->generation == generation) {
        g->flip_pending = 0;
        status_write_locked(g);
    }
    if (flip_t0)
        g->stats.flip_us += us_since(flip_t0);
    thread_release_mutex(g->lock);
    return 0;
}

static void
count_packet(vgpu98_stats_t *s, uint32_t op, uint32_t len)
{
    s->packets++;
    s->ring_bytes += 4 + (uint64_t) len * 4;
    if (op == VGPU98_OP_INLINE)
        s->vertex_bytes += 4 + (uint64_t) len * 4;
    switch (op) {
        case VGPU98_OP_FENCE:
            s->fences++;
            break;
        case VGPU98_OP_SYNC:
            s->syncs++;
            break;
        case VGPU98_OP_FLIP:
            s->flips++;
            break;
        case VGPU98_OP_FILL:
            s->fills++;
            break;
        case VGPU98_OP_BLT:
            s->blits++;
            break;
        default:
            break;
    }
}

static int
gpu_execute(vgpu98_gpu_t *g, uint32_t op, uint32_t len, uint32_t generation)
{
    const uint32_t *p = g->payload;

    switch (op) {
        case VGPU98_OP_NOP:
            /* With the texture audit, the DirectX driver's audit notes;
               every other payload is skipped. */
            if (len && g->r3d && (g->log_mask & VGPU98_LOG_TEXTURES))
                vgpu98_r3d_note(g->r3d, p, len, g->ring_done, g->ring_known);
            return 0;
        case VGPU98_OP_FENCE: {
            int step;

            /* Everything before it is complete, the visible frame shown
               (this ends the batch the driver wrote). */
            gpu_show_visible(g, 0);
            step = vgpu98_gpu_step(g, VGPU98_STEP_LOCK);
            thread_wait_mutex(g->lock);
            vgpu98_gpu_step(g, step);
            if (g->generation == generation) {
                g->fence = p[0];
                status_write_locked(g);
            }
            thread_release_mutex(g->lock);
            return 0;
        }
        case VGPU98_OP_SYNC:
            if (!range_ok(g, p[0], p[1]))
                return VGPU98_ERR_ARGUMENT;
            if (g->r3d)
                vgpu98_r3d_flush(g->r3d);
            gpu_sync(g, p[0], p[1], VGPU98_CB_SYNC);
            return 0;
        case VGPU98_OP_FLIP:
            return exec_flip(g, p, generation);
        case VGPU98_OP_DISCARD:
            if (!range_ok(g, p[0], p[1]))
                return VGPU98_ERR_ARGUMENT;
            if (g->r3d && p[1])
                vgpu98_r3d_discard(g->r3d, p[0], p[1]);
            return 0;
        case VGPU98_OP_FILL:
            return blit_fill(g, p);
        case VGPU98_OP_UPLOAD:
            return blit_upload(g, p, len);
        case VGPU98_OP_DOWNLOAD:
            return blit_download(g, p);
        case VGPU98_OP_BLT:
            return blit_copy(g, p);
        default:
            if ((op >= VGPU98_GL_FIRST) && (op <= VGPU98_GL_LAST)) {
                if (g->glide)
                    vgpu98_glide_packet(g->glide, op, p, len, generation);
                return 0;
            }
            if ((op >= VGPU98_OGL_FIRST) && (op <= VGPU98_OGL_LAST)) {
                if (g->ogl)
                    vgpu98_ogl_packet(g->ogl, op, p, len, generation);
                return 0;
            }
            return g->r3d ? vgpu98_r3d_packet(g->r3d, op, p, len) : VGPU98_ERR_PROFILE;
    }
}

/* --- The Glide device's use of the command processor ---------------------- */

void
vgpu98_gpu_wait_vblanks_since(vgpu98_gpu_t *g, uint32_t interval, uint32_t *since, uint32_t generation)
{
    uint32_t start;
    int      step;

    step = vgpu98_gpu_step(g, VGPU98_STEP_LOCK);
    thread_wait_mutex(g->lock);
    vgpu98_gpu_step(g, step);
    start = g->vblank_count;
    if (interval) {
        /* The interval's vertical blanks since the last swap, and one at
           least from now, as FLIP waits. */
        g->vblank_waiters++;
        while (((g->vblank_count == start) || ((uint32_t) (g->vblank_count - *since) < interval)) && !g->quit &&
               (g->generation == generation)) {
            touch_service_locked(g);
            thread_release_mutex(g->lock);
            thread_wait_event(g->flip_wake, 100);
            thread_reset_event(g->flip_wake);
            thread_wait_mutex(g->lock);
        }
        g->vblank_waiters--;
    }
    *since = g->vblank_count;
    thread_release_mutex(g->lock);
}

void
vgpu98_gpu_wait_vblanks(vgpu98_gpu_t *g, uint32_t interval, uint32_t generation)
{
    vgpu98_gpu_wait_vblanks_since(g, interval, &g->glide_vblank, generation);
}

void
vgpu98_gpu_visible(vgpu98_gpu_t *g, int *enabled, uint32_t *start, uint32_t *pitch, uint32_t *height, uint32_t *bpp)
{
    g->host.display(g->host.priv, enabled, start, pitch, height, bpp);
}

uint32_t
vgpu98_gpu_vblanks(vgpu98_gpu_t *g)
{
    uint32_t v;

    thread_wait_mutex(g->lock);
    v = g->vblank_count;
    thread_release_mutex(g->lock);
    return v;
}

void
vgpu98_gpu_glide_front(vgpu98_gpu_t *g, int on)
{
    thread_wait_mutex(g->lock);
    g->glide_front = on;
    thread_release_mutex(g->lock);
}

/* --- The V-sync overrides --- */

void
vgpu98_gpu_set_vsync(vgpu98_gpu_t *g, uint32_t control)
{
    /* A change starts the per-API notes again; the same value written again
       (OK with nothing changed) does not. */
    if (atomic_exchange(&g->vsync_control, control) != control)
        atomic_fetch_add(&g->vsync_writes, 1);
}

uint32_t
vgpu98_gpu_vsync(vgpu98_gpu_t *g)
{
    return g ? atomic_load(&g->vsync_control) : 0;
}

uint32_t
vgpu98_gpu_vsync_interval(vgpu98_gpu_t *g, int shift, uint32_t asked)
{
    static const char *const apis[3]  = { "Direct3D's flip", "OpenGL's swap", "Glide's swap" };
    static const char *const modes[4] = { "", "always on", "always off", "" };
    const uint32_t           control  = atomic_load(&g->vsync_control);
    const uint32_t           writes   = atomic_load(&g->vsync_writes);
    const uint32_t           got      = vgpu98_vsync_interval(control, shift, asked);
    const uint32_t           api      = (uint32_t) shift / 2;

    if (writes != g->vsync_seen) {
        g->vsync_seen = writes;
        g->vsync_told = 0;
    }
    /* The first wait an override changed for each API since the register's
       last write. */
    if ((got != asked) && !(g->vsync_told & (1u << api))) {
        g->vsync_told |= 1u << api;
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS,
                       "VGPU98 GPU: V-sync: %s waits %u vertical blank%s (%s; the program asked for %u)\n", apis[api],
                       got, (got == 1) ? "" : "s", modes[(control >> shift) & 3], asked);
    }
    return got;
}

void
vgpu98_gpu_glide_screen(vgpu98_gpu_t *g, int on)
{
    atomic_store_explicit(&g->glide_screen, on, memory_order_relaxed);
}

int
vgpu98_gpu_glide_screen_on(vgpu98_gpu_t *g)
{
    return g && atomic_load_explicit(&g->glide_screen, memory_order_relaxed);
}

void
vgpu98_gpu_glide_screenshot(vgpu98_gpu_t *g)
{
    if (g && g->glide)
        vgpu98_glide_screenshot(g->glide);
}

void
vgpu98_gpu_ogl_screenshot(vgpu98_gpu_t *g)
{
    if (g && g->ogl)
        vgpu98_ogl_screenshot(g->ogl);
}

/* --- Diagnostics: the recorder, the summary, captures ---------------------- */

/* A record as a line (no time, no prefix). */
static void
rec_format(const rec_t *r, char *buf, size_t size)
{
    const uint32_t *v = r->v;
    int             n = 0;
    uint32_t        i;

    switch (r->kind) {
        case REC_PACKET:
            n = snprintf(buf, size, "%05X %-8s %4u", v[0], op_name(r->op), v[1]);
            for (i = 0; (i < v[1]) && (i < 6) && (n > 0) && ((size_t) n < size); i++)
                n += snprintf(buf + n, size - n, " %08X", v[2 + i]);
            if ((v[1] > 6) && (n > 0) && ((size_t) n < size))
                snprintf(buf + n, size - n, " ...");
            break;
        case VGPU98_NOTE_NEW_IMAGE:
            snprintf(buf, size, "new host image %u: format %02X, %u levels, %ux%u, video memory %06X-%06X", v[0],
                     v[1] & 0xff, v[1] >> 8, v[2] >> 16, v[2] & 0xffff, v[3], v[4]);
            break;
        case VGPU98_NOTE_UPLOAD:
            snprintf(buf, size, "upload image %u (%06X-%06X, %u KB)", v[0], v[1], v[2], (v[3] + 1023) / 1024);
            break;
        case VGPU98_NOTE_COPY_BACK:
            snprintf(buf, size, "copy back image %u, %u,%u-%u,%u (%06X+%X)", v[0], v[1] >> 16, v[1] & 0xffff, v[2] >> 16,
                     v[2] & 0xffff, v[3], v[4]);
            break;
        case VGPU98_NOTE_EVICT:
            snprintf(buf, size, "evict image %u (%06X-%06X)", v[0], v[1], v[2]);
            break;
        case VGPU98_NOTE_DISCARD:
            snprintf(buf, size, "discard image %u (%06X+%X)", v[0], v[1], v[2]);
            break;
        case VGPU98_NOTE_PIPELINE:
            snprintf(buf, size, "new pipeline %u: shape %u, program %u, blend %u %u/%u, depth test %u write %u func %u, stencil %u", v[0],
                     v[1] & 0xf, (v[1] >> 4) & 0xf, (v[1] >> 8) & 0xff, (v[1] >> 16) & 0xff, v[1] >> 24, v[2] & 0xff, (v[2] >> 8) & 0xff,
                     (v[2] >> 16) & 0xff, v[2] >> 24);
            break;
        case VGPU98_NOTE_FAILURE:
            snprintf(buf, size, "host GPU failure (VkResult %d)", (int) v[0]);
            break;
        case REC_RESET:
            snprintf(buf, size, "reset %u: %s", v[1], (v[0] < (sizeof(reset_causes) / sizeof(reset_causes[0]))) ? reset_causes[v[0]] : "?");
            break;
        default:
            snprintf(buf, size, "record %02X", r->kind);
            break;
    }
}

static void
rec_put(vgpu98_gpu_t *g, const rec_t *r)
{
    if (g->rec && !(g->cap_active && (g->cap_post >= CAPTURE_POST_MAX))) {
        g->rec[g->rec_n & (REC_COUNT - 1)] = *r;
        g->rec_n++;
        if (g->cap_active)
            g->cap_post++;
    }
    if (g->log_mask & VGPU98_LOG_COMMANDS) {
        char line[256];

        rec_format(r, line, sizeof(line));
        out_line(g, "%s VGPU98 %s %s\n", gpu_stamp(g), (r->kind == REC_PACKET) ? "cmd:" : "3D:", line);
    }
}

static void
note_packet(vgpu98_gpu_t *g, uint32_t at, uint32_t op, uint32_t len)
{
    rec_t    r;
    uint32_t i;

    memset(&r, 0, sizeof(r));
    r.ms   = plat_get_ticks();
    r.kind = REC_PACKET;
    r.op   = (uint8_t) op;
    r.v[0] = at;
    r.v[1] = len;
    for (i = 0; (i < len) && (i < 6); i++)
        r.v[2 + i] = g->payload[i];
    rec_put(g, &r);
}

void
vgpu98_gpu_note(vgpu98_gpu_t *g, int kind, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e)
{
    rec_t r;

    if (!(g->log_mask & (VGPU98_LOG_CAPTURE | VGPU98_LOG_COMMANDS)))
        return;
    memset(&r, 0, sizeof(r));
    r.ms   = plat_get_ticks();
    r.kind = (uint8_t) kind;
    r.v[0] = a;
    r.v[1] = b;
    r.v[2] = c;
    r.v[3] = d;
    r.v[4] = e;
    rec_put(g, &r);
}

static void
fmt_bytes(char *buf, size_t size, uint64_t bytes)
{
    if (bytes >= (1u << 20))
        snprintf(buf, size, "%.1f MB", (double) bytes / (1u << 20));
    else
        snprintf(buf, size, "%u KB", (unsigned) ((bytes + 1023) / 1024));
}

/* A slice of a capture: the counters' change from `a` to `b`. */
static void
slice_delta(char *buf, size_t size, const vgpu98_stats_t *a, const vgpu98_stats_t *b)
{
    char up[24];
    char back[24];

    fmt_bytes(up, sizeof(up), b->upload_bytes - a->upload_bytes);
    fmt_bytes(back, sizeof(back), b->copy_back_bytes - a->copy_back_bytes);
    snprintf(buf, size,
             "%u packets, %u flips, %u draws, %u triangles, %u uploads (%s), %u copies back (%s), %u evictions, %u errors; "
             "%u ms at flips, %u ms copying back, %u KB of ring",
             b->packets - a->packets, b->flips - a->flips, b->draws - a->draws, b->triangles - a->triangles,
             b->uploads - a->uploads, up, b->copies_back - a->copies_back, back, b->evictions - a->evictions,
             b->errors - a->errors, (unsigned) ((b->flip_us - a->flip_us) / 1000),
             (unsigned) ((b->copy_back_us - a->copy_back_us) / 1000), (unsigned) ((b->ring_bytes - a->ring_bytes) / 1024));
}

static double
pct(uint64_t part, double whole)
{
    return (whole > 0.0) ? (100.0 * (double) part / whole) : 0.0;
}

/* A percentage as the summary prints it: one decimal under 10. */
static const char *
pct_text(char *buf, size_t size, double v)
{
    snprintf(buf, size, (v < 10.0) ? "%.1f%%" : "%.0f%%", v);
    return buf;
}

/* Frame times in order, for the median and percentile. */
static int
cmp_u32(const void *x, const void *y)
{
    const uint32_t a = *(const uint32_t *) x;
    const uint32_t b = *(const uint32_t *) y;

    return (a > b) - (a < b);
}

/* The summary: the work, the host images, where the thread's time went, the
   ring. */
static void
summary_write(vgpu98_gpu_t *g, uint32_t now)
{
    const vgpu98_stats_t *a       = &g->summary_from;
    const vgpu98_stats_t *b       = &g->stats;
    const double          seconds = (now - g->summary_ms) / 1000.0;
    const double          us      = (now - g->summary_ms) * 1000.0;
    const uint32_t        flips   = b->flips - a->flips;
    const uint32_t        waited  = b->flips_waited - a->flips_waited;
    const uint64_t        ring    = b->ring_bytes - a->ring_bytes;
    char                  up[24];
    char                  back[24];
    char                  held[24];
    char                  budget[24];
    char                  total[24];
    char                  rate[24];
    char                  frame[24];
    uint32_t              images = 0;
    uint64_t              bytes  = 0;
    uint64_t              limit  = 0;
    uint32_t              peak;
    uint32_t              size;
    int                   sysmem;
    double                idle, flip, upload, copy, wait, busy, other;
    char                  p[7][16];

    vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                   "VGPU98 summary, %.1f s: %u flips (%.1f fps), %u fences, %u syncs, %u fills, %u blits, "
                   "%u draws (%u triangles, %u lines, %u points), %u clears, %u resets, %u errors\n",
                   seconds, flips, (seconds > 0.0) ? flips / seconds : 0.0, b->fences - a->fences, b->syncs - a->syncs,
                   b->fills - a->fills, b->blits - a->blits, b->draws - a->draws, b->triangles - a->triangles,
                   b->lines - a->lines, b->points - a->points, b->clears - a->clears, b->resets - a->resets,
                   b->errors - a->errors);

    /* The kinds of rejection since the last summary, the first four named. */
    {
        char     list[720];
        int      n        = 0;
        int      shown    = 0;
        uint32_t total    = 0;
        uint32_t unlogged = 0;
        int      i;

        list[0] = 0;
        thread_wait_mutex(g->lock);
        for (i = 0; i < ERR_SIGS; i++) {
            err_sig_t *e = &g->err_sig[i];

            if (!e->used || !e->count)
                continue;
            total += e->count;
            unlogged += e->unlogged;
            if ((shown < 4) && (n >= 0) && ((size_t) n < sizeof(list)))
                n += snprintf(list + n, sizeof(list) - n, "%s%s, error %u: %u", shown ? "; " : "", e->what, e->code, e->count);
            shown++;
            e->count    = 0;
            e->unlogged = 0;
        }
        thread_release_mutex(g->lock);
        if (total)
            vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY, "VGPU98 summary errors: %u packets rejected (%u not logged): %s%s\n", total,
                           unlogged, list, (shown > 4) ? "; ..." : "");
    }

    if (b->draws != a->draws) {
        fmt_bytes(up, sizeof(up), b->vertex_copy_bytes - a->vertex_copy_bytes);
        fmt_bytes(back, sizeof(back), b->vertex_bytes - a->vertex_bytes);
        vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                       "VGPU98 summary vertices: %u draws pre-transformed, %u through the transform; whole primitives "
                       "for WRAPn %u, point fill %u, wireframe %u, lines %u; %llu vertices in the draws' windows, "
                       "%llu indices, %s copied for the host GPU, %s of INLINE packets\n",
                       b->draws_pre - a->draws_pre, b->draws_fixed - a->draws_fixed, b->shape_wrap - a->shape_wrap,
                       b->shape_points - a->shape_points, b->shape_wire - a->shape_wire, b->shape_lines - a->shape_lines,
                       (unsigned long long) (b->vertices - a->vertices), (unsigned long long) (b->indices - a->indices), up,
                       back);
        /* Command set 2.1's (DirectX 8's) part of them. */
        {
            uint32_t sd[4];
            uint32_t any = 0;
            int      i;

            for (i = 0; i < 4; i++)
                sd[i] = b->draws_streams[i] - a->draws_streams[i];
            any = sd[1] | sd[2] | sd[3] | (b->draws_palette - a->draws_palette) | (b->draws_tween - a->draws_tween) |
                  (b->draws_psize - a->draws_psize) | (b->shape_sprites - a->shape_sprites);
            if (any)
                vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                               "VGPU98 summary vertices 2.1: draws reading 1 stream %u, 2 %u, 3-4 %u, 5 or more %u; "
                               "indexed vertex blending %u (without indices, so 0-3, %u), tweening %u; points as squares %u, "
                               "with a per-vertex size %u\n",
                               sd[0], sd[1], sd[2], sd[3], b->draws_palette - a->draws_palette,
                               b->draws_palette_noidx - a->draws_palette_noidx, b->draws_tween - a->draws_tween,
                               b->shape_sprites - a->shape_sprites, b->draws_psize - a->draws_psize);
        }
        /* The streams: where draws read them, their strides and elements,
           the indices, residency. */
        if ((b->streams_gpuvm != a->streams_gpuvm) || (b->streams_inline != a->streams_inline)) {
            uint32_t e[9];
            uint32_t st[4];
            int      i;

            for (i = 0; i < 9; i++)
                e[i] = b->elements[i] - a->elements[i];
            for (i = 0; i < 4; i++)
                st[i] = b->stream_strides[i] - a->stream_strides[i];
            vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                           "VGPU98 summary streams: %u read at GPU addresses, %u inline; strides up to 16 bytes %u, 32 %u, 64 %u, "
                           "more %u; elements FLOAT1 %u, FLOAT2 %u, FLOAT3 %u, FLOAT4 %u, D3DCOLOR %u, UBYTE4 %u, SHORT2 %u, "
                           "SHORT4 %u; indexed draws 16-bit %u (inline %u), 32-bit %u (inline %u); draws with their streams "
                           "resident %u, some %u, none %u\n",
                           b->streams_gpuvm - a->streams_gpuvm, b->streams_inline - a->streams_inline, st[0], st[1], st[2],
                           st[3], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], b->draws_idx[0] - a->draws_idx[0],
                           b->draws_idx_inline[0] - a->draws_idx_inline[0], b->draws_idx[1] - a->draws_idx[1],
                           b->draws_idx_inline[1] - a->draws_idx_inline[1], b->draws_res[0] - a->draws_res[0],
                           b->draws_res[1] - a->draws_res[1], b->draws_res[2] - a->draws_res[2]);
        }
    }
    {
        uint32_t drops[VGPU98_DROPS];
        uint32_t any = b->draws_past - a->draws_past;
        int      i;

        for (i = 0; i < VGPU98_DROPS; i++)
            any |= drops[i] = b->drops[i] - a->drops[i];
        if (any)
            vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                           "VGPU98 summary vertex audit: %u draws lacked %llu vertices of their windows (read as 0); "
                           "draws dropped: 3D stopped %u, images %u, host objects %u, scissored out %u, no pipeline %u, "
                           "memory %u\n",
                           b->draws_past - a->draws_past, (unsigned long long) (b->verts_past - a->verts_past),
                           drops[VGPU98_DROP_STOPPED], drops[VGPU98_DROP_IMAGES], drops[VGPU98_DROP_HOST],
                           drops[VGPU98_DROP_SCISSOR], drops[VGPU98_DROP_PIPELINE], drops[VGPU98_DROP_MEMORY]);
    }
    if ((b->draws_w != a->draws_w) || (b->depth_moves != a->depth_moves) || (b->wrange_infinite != a->wrange_infinite)) {
        float wnear = 1.0f;
        float wfar  = 1.0f;

        if (g->r3d)
            vgpu98_r3d_wrange(g->r3d, &wnear, &wfar);
        vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                       "VGPU98 summary depth: %u draws W-buffered (%u into 32-bit float images), %u depth images moved to "
                       "32-bit float, %u W ranges with an infinite far plane; W range now %g to %g\n",
                       b->draws_w - a->draws_w, b->draws_w_float - a->draws_w_float, b->depth_moves - a->depth_moves,
                       b->wrange_infinite - a->wrange_infinite, wnear, wfar);
    }
    {
        const uint64_t v = b->gpuvm_bytes[VGPU98_GPUVM_VERTICES] - a->gpuvm_bytes[VGPU98_GPUVM_VERTICES];
        const uint64_t i = b->gpuvm_bytes[VGPU98_GPUVM_INDICES] - a->gpuvm_bytes[VGPU98_GPUVM_INDICES];
        const uint64_t u = b->gpuvm_bytes[VGPU98_GPUVM_UPLOAD] - a->gpuvm_bytes[VGPU98_GPUVM_UPLOAD];
        const uint64_t w = b->gpuvm_bytes[VGPU98_GPUVM_DOWNLOAD] - a->gpuvm_bytes[VGPU98_GPUVM_DOWNLOAD];

        if (v || i || u || w) {
            char vs[24], is[24], us[24], ws[24];

            fmt_bytes(vs, sizeof(vs), v);
            fmt_bytes(is, sizeof(is), i);
            fmt_bytes(us, sizeof(us), u);
            fmt_bytes(ws, sizeof(ws), w);
            vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                           "VGPU98 summary buffers: read from main memory through the page table: vertices %s, indices %s, "
                           "UPLOAD sources %s; written there: DOWNLOAD %s; the table spans %u KB\n",
                           vs, is, us, ws, g->gpuvm_entries * 4);
        }
        if ((b->res_hits != a->res_hits) || (b->res_misses != a->res_misses) || (b->sets_reused != a->sets_reused) ||
            (b->res_stale != a->res_stale)) {
            char ins[24], sav[24];

            fmt_bytes(ins, sizeof(ins), b->res_insert_bytes - a->res_insert_bytes);
            fmt_bytes(sav, sizeof(sav), b->res_saved_bytes - a->res_saved_bytes);
            vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                           "VGPU98 summary buffers resident: %u reads from the arena, %u from guest RAM; %u buffers went in "
                           "(%s), %u evicted; %s of vertices not copied again; %u draws straight from the arena; "
                           "%u draws reused a descriptor set; %u found stale\n",
                           b->res_hits - a->res_hits, b->res_misses - a->res_misses, b->res_inserts - a->res_inserts, ins,
                           b->res_evictions - a->res_evictions, sav, b->draws_direct - a->draws_direct,
                           b->sets_reused - a->sets_reused, b->res_stale - a->res_stale);
        }
    }
    if (b->tex_uses != a->tex_uses) {
        char mip[160];

        mip[0] = 0;
        if (g->log_mask & VGPU98_LOG_CAPTURE)
            snprintf(mip, sizeof(mip), "; mip check: %u textures, %u measured again, %u levels named (%.1f ms)",
                     b->mip_textures - a->mip_textures, b->mip_again - a->mip_again, b->mip_named - a->mip_named,
                     (double) (b->mip_us - a->mip_us) / 1000.0);
        vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                       "VGPU98 summary textures: %u stage uses; coordinates from vertices %u, camera normal %u, camera position %u, "
                       "reflection %u; transformed %u, projected %u; asking for anisotropy %u, trilinear %u, without mipmaps %u, "
                       "with a mip bias %u; bump stages: DOTPRODUCT3 %u, BUMPENVMAP %u%s\n",
                       b->tex_uses - a->tex_uses, b->tex_src[0] - a->tex_src[0], b->tex_src[1] - a->tex_src[1],
                       b->tex_src[2] - a->tex_src[2], b->tex_src[3] - a->tex_src[3], b->tex_xform - a->tex_xform,
                       b->tex_projected - a->tex_projected, b->tex_aniso - a->tex_aniso, b->tex_trilinear - a->tex_trilinear,
                       b->tex_nomip - a->tex_nomip, b->tex_lodbias - a->tex_lodbias, b->tex_dot3 - a->tex_dot3,
                       b->tex_bumpenv - a->tex_bumpenv, mip);
    }
    /* Command set 2.1's (DirectX 8's) pixel side. */
    if ((b->tex_volume != a->tex_volume) || (b->tex_madd_lerp != a->tex_madd_lerp) || (b->tex_temp != a->tex_temp) ||
        (b->draws_blendop != a->draws_blendop) || (b->draws_masked != a->draws_masked) || (b->vol_uploads != a->vol_uploads))
    {
        uint32_t nv = 0;
        uint64_t vb = 0;
        char     vbs[32];

        if (g->r3d)
            vgpu98_r3d_vol_usage(g->r3d, &nv, &vb);
        fmt_bytes(vbs, sizeof(vbs), vb);
        vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                       "VGPU98 summary pixels 2.1: stages sampling a volume %u, with MULTIPLYADD or LERP %u, writing TEMP %u; "
                       "volume uploads %u, volumes held %u (%s); draws blending other than by adding %u, with channels masked %u\n",
                       b->tex_volume - a->tex_volume, b->tex_madd_lerp - a->tex_madd_lerp, b->tex_temp - a->tex_temp,
                       b->vol_uploads - a->vol_uploads, nv, vbs, b->draws_blendop - a->draws_blendop,
                       b->draws_masked - a->draws_masked);
    }
    if (b->draws_fixed != a->draws_fixed) {
        const uint32_t lit = b->draws_lit - a->draws_lit;

        vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                       "VGPU98 summary transform: of %u draws through it, %u lit (%.1f lights each), %u blended, "
                       "%u with clip planes, %u with generated coordinates, %u with texture transforms\n",
                       b->draws_fixed - a->draws_fixed, lit, lit ? (double) (b->lights_used - a->lights_used) / lit : 0.0,
                       b->draws_blend - a->draws_blend, b->draws_planes - a->draws_planes, b->draws_texgen - a->draws_texgen,
                       b->draws_texxform - a->draws_texxform);
    }

    if (g->r3d)
        vgpu98_r3d_usage(g->r3d, &images, &bytes, &limit);
    fmt_bytes(up, sizeof(up), b->upload_bytes - a->upload_bytes);
    fmt_bytes(back, sizeof(back), b->copy_back_bytes - a->copy_back_bytes);
    fmt_bytes(held, sizeof(held), bytes);
    fmt_bytes(budget, sizeof(budget), limit);
    vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                   "VGPU98 summary images: %u uploads (%s), %u copies back (%s), %u evictions (%u stale), %u discards, "
                   "%u new images, %u new pipelines (%.1f ms), %u submits; host images %u (%s of %s)\n",
                   b->uploads - a->uploads, up, b->copies_back - a->copies_back, back, b->evictions - a->evictions,
                   b->stale_evictions - a->stale_evictions, b->discards - a->discards, b->new_images - a->new_images, b->new_pipelines - a->new_pipelines,
                   (double) (b->pipeline_us - a->pipeline_us) / 1000.0, b->submits - a->submits, images, held, budget);
    if ((b->copies_back != a->copies_back) || (b->upload_packets != a->upload_packets) || (b->views != a->views) ||
        (b->merges != a->merges)) {
        const uint32_t *x = a->copies_back_by;
        const uint32_t *y = b->copies_back_by;

        fmt_bytes(up, sizeof(up), b->upload_packet_bytes - a->upload_packet_bytes);
        vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                       "VGPU98 summary images by cause: copies back for guest accesses %u, Locks %u, 2D packets %u, frames %u, "
                       "other images %u, DISCARDs %u, evictions %u, resets %u; %u UPLOAD packets (%s), written into %u "
                       "host images as well; %u levels and faces drawn into first, %u images taken over by textures\n",
                       y[VGPU98_CB_TRAP] - x[VGPU98_CB_TRAP], y[VGPU98_CB_SYNC] - x[VGPU98_CB_SYNC],
                       y[VGPU98_CB_2D] - x[VGPU98_CB_2D], y[VGPU98_CB_FRAME] - x[VGPU98_CB_FRAME],
                       y[VGPU98_CB_IMAGE] - x[VGPU98_CB_IMAGE], y[VGPU98_CB_DISCARD] - x[VGPU98_CB_DISCARD],
                       y[VGPU98_CB_EVICT] - x[VGPU98_CB_EVICT], y[VGPU98_CB_RESET] - x[VGPU98_CB_RESET],
                       b->upload_packets - a->upload_packets, up, b->upload_rects - a->upload_rects, b->views - a->views,
                       b->merges - a->merges);
    }

    idle   = pct(b->idle_us - a->idle_us, us);
    flip   = pct(b->flip_us - a->flip_us, us);
    upload = pct(b->upload_us - a->upload_us, us);
    copy   = pct(b->copy_back_us - a->copy_back_us, us);
    wait   = pct(b->gpu_wait_us - a->gpu_wait_us, us);
    busy   = 100.0 - idle - flip;
    if (busy < 0.0)
        busy = 0.0;
    other = busy - upload - copy - wait;
    if (other < 0.0)
        other = 0.0;
    vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                   "VGPU98 summary time: busy %s (uploads %s, copies back %s, host GPU waits %s, other %s), "
                   "at flips %s, idle %s; %u of %u flips waited, %.1f ms on average\n",
                   pct_text(p[0], sizeof(p[0]), busy), pct_text(p[1], sizeof(p[1]), upload),
                   pct_text(p[2], sizeof(p[2]), copy), pct_text(p[3], sizeof(p[3]), wait),
                   pct_text(p[4], sizeof(p[4]), other), pct_text(p[5], sizeof(p[5]), flip),
                   pct_text(p[6], sizeof(p[6]), idle), waited, flips,
                   waited ? (double) (b->flip_us - a->flip_us) / waited / 1000.0 : 0.0);
    if (g->n_frames) {
        /* Of the summary's first FRAMES_KEPT frames (sorted in place: they
           go after it); the 95th percentile by nearest rank. */
        const uint32_t kept = (g->n_frames < FRAMES_KEPT) ? g->n_frames : FRAMES_KEPT;
        const double   sus  = (double) g->slow.us;
        double         med;
        double         p95;
        double         rest;
        char           q[7][16];

        qsort(g->frame_us, kept, sizeof(uint32_t), cmp_u32);
        med = g->frame_us[kept / 2] / 1000.0;
        p95 = g->frame_us[(kept * 95 + 99) / 100 - 1] / 1000.0;
        rest = sus - (double) (g->slow.idle + g->slow.flip + g->slow.upload + g->slow.copy + g->slow.wait + g->slow.pipe);
        if (rest < 0.0)
            rest = 0.0;
        vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                       "VGPU98 summary frames: %u frames (between FLIPs), median %.1f ms, 95th percentile %.1f ms, worst %.1f ms; "
                       "%u over %u ms (%.0f ms in all), of their time the GPU thread waited for the guest %s, at flips %s, "
                       "uploaded %s, copied back %s, waited for the host GPU %s, made %u pipelines in %s, other %s\n",
                       g->n_frames, med, p95, g->frame_worst / 1000.0, g->slow.frames, FRAME_SLOW_US / 1000, sus / 1000.0,
                       pct_text(q[0], sizeof(q[0]), pct(g->slow.idle, sus)), pct_text(q[1], sizeof(q[1]), pct(g->slow.flip, sus)),
                       pct_text(q[2], sizeof(q[2]), pct(g->slow.upload, sus)), pct_text(q[3], sizeof(q[3]), pct(g->slow.copy, sus)),
                       pct_text(q[4], sizeof(q[4]), pct(g->slow.wait, sus)), g->slow.pipelines,
                       pct_text(q[5], sizeof(q[5]), pct(g->slow.pipe, sus)), pct_text(q[6], sizeof(q[6]), pct(rest, sus)));
    }

    thread_wait_mutex(g->lock);
    peak         = g->ring_peak;
    size         = g->ring_size;
    sysmem       = !!(g->ring_base & VGPU98_RING_SYSMEM);
    g->ring_peak = 0;
    thread_release_mutex(g->lock);
    fmt_bytes(total, sizeof(total), ring);
    fmt_bytes(rate, sizeof(rate), (seconds > 0.0) ? (uint64_t) (ring / seconds) : 0);
    fmt_bytes(frame, sizeof(frame), flips ? ring / flips : 0);
    vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                   "VGPU98 summary ring: %u KB in %s memory, %s (%s/s, %s a frame, %.0f%% INLINE data), at most %.0f%% full, "
                   "%u doorbells with under 1/8 free\n",
                   size >> 10, sysmem ? "main" : "video", total, rate, frame,
                   pct(b->vertex_bytes - a->vertex_bytes, (double) ring), pct(peak, (double) size),
                   b->ring_low_room - a->ring_low_room);

    if (g->disp) {
        vgpu98_disp_stats_t ds;

        uint32_t touches;
        uint64_t touch_us;

        vgpu98_disp_stats(g->disp, &ds);
        thread_wait_mutex(g->lock);
        touches  = g->touch_count - g->touch_from;
        touch_us = g->touch_us - g->touch_us_from;
        g->touch_from    = g->touch_count;
        g->touch_us_from = g->touch_us;
        thread_release_mutex(g->lock);
        fmt_bytes(up, sizeof(up), ds.upload_bytes - g->disp_from.upload_bytes);
        fmt_bytes(back, sizeof(back), ds.read_bytes - g->disp_from.read_bytes);
        fmt_bytes(frame, sizeof(frame), b->publish_bytes - a->publish_bytes);
        vgpu98_gpu_log(g, VGPU98_LOG_SUMMARY,
                       "VGPU98 summary display: %u frames composed on the host GPU, %s of video memory uploaded, "
                       "%s read back for 86Box's window, %u publishes of GPU pixels (%s); %u guest accesses to "
                       "GPU-newer pages copied back (%.1f ms waited); display thread busy %s\n",
                       (unsigned) (ds.frames - g->disp_from.frames), up, back, b->publishes - a->publishes, frame,
                       touches, touch_us / 1000.0, pct_text(p[0], sizeof(p[0]), pct(ds.busy_us - g->disp_from.busy_us, us)));
    }
    /* The texture audit's lines. */
    if (g->r3d && (g->log_mask & VGPU98_LOG_TEXTURES))
        vgpu98_r3d_tex_summary(g->r3d);
}

static void
status_text(char *buf, size_t size, uint32_t st)
{
    snprintf(buf, size, "%s%s%s%s%s", (st & VGPU98_STATUS_BUSY) ? " BUSY" : "", (st & VGPU98_STATUS_ERROR) ? " ERROR" : "",
             (st & VGPU98_STATUS_RUNNING) ? " RUNNING" : "", (st & VGPU98_STATUS_FLIP_PENDING) ? " FLIP_PENDING" : "",
             (st & VGPU98_STATUS_HELD) ? " HELD" : "");
}

/* The registers a trigger noted, as a snapshot's line. Returns its length. */
static size_t
regs_text(char *buf, size_t size, const trig_regs_t *t)
{
    char st[64];
    int  n;

    status_text(st, sizeof(st), t->status);
    n = snprintf(buf, size,
                 "  Registers: STATUS %02X (%s ), ERROR_CODE %u, ERROR_OFFSET %05X, FENCE %u, ring %08X+%X, "
                 "HEAD %05X, TAIL %05X, holds: display driver %d, mini-VDD %d; resets so far %u\n",
                 t->status, st, t->error_code, t->error_offset, t->fence, t->ring_base, t->ring_size, t->ring_head,
                 t->ring_tail, t->hold_driver, t->hold_vdd, t->generation);
    return (n < 0) ? 0 : ((size_t) n >= size) ? size - 1 : (size_t) n;
}

/* What came while a capture records, in its text: "Then, 120 ms after the
   screenshot, a failure: ...", with a failure's registers and 3D state. */
static void
cap_head_then(vgpu98_gpu_t *g, uint32_t now, const char *what, const char *why, const trig_regs_t *t)
{
    const size_t size = CAPTURE_HEAD;
    size_t       n    = strlen(g->cap_head);
    int          k;

    if (n + 512 >= size)
        return;
    k = snprintf(g->cap_head + n, size - n, "  Then, %d ms after the %s, %s: %s\n", (int32_t) (now - g->cap_ms),
                 (g->cap_kind == CAP_SHOT) ? "screenshot" : (g->cap_kind == CAP_FIND) ? "finding" : "failure", what, why);
    n += (k < 0) ? 0 : ((size_t) k >= size - n) ? size - n - 1 : (size_t) k;
    if (t) {
        n += regs_text(g->cap_head + n, size - n, t);
        if (g->r3d && (n + 1 < size))
            n += vgpu98_r3d_describe(g->r3d, g->cap_head + n, size - n);
    }
}

/* A trigger reached the thread: the state now, then 2 more seconds. A
   screenshot's has a limit of its own, and says when it is not taken (the
   tester expects one); vgpu98_gpu_find decides which findings ask for one.
   A failure while a screenshot's or a finding's capture records makes the
   snapshot the failure's, with its reason, registers and 3D state; a
   finding while any records goes in its text. `image` is the host image a
   finding names (0 for none), `sub` its level or face (index plus one). */
static void
capture_start(vgpu98_gpu_t *g, const char *why, const trig_regs_t *t, int kind, uint32_t image, uint32_t sub)
{
    const uint32_t now  = plat_get_ticks();
    const size_t   size = CAPTURE_HEAD;
    char           stamp[24];
    char           finds[80];
    size_t         n;

    if (g->cap_active) {
        if (kind == CAP_SHOT)
            vgpu98_gpu_log(g, VGPU98_LOG_CAPTURE, "VGPU98 capture: a screenshot while capture %u records; it covers it\n",
                           cap_number(g));
        else if (kind == CAP_FIND)
            cap_head_then(g, now, "a finding", why, NULL);
        else if ((g->cap_kind != CAP_FAIL) && (g->cap_count < CAPTURE_MAX)) {
            const char *was = (g->cap_kind == CAP_SHOT) ? "screenshot" : "finding";

            cap_head_then(g, now, "a failure", why, t);
            g->cap_kind = CAP_FAIL;
            vgpu98_gpu_log(g, VGPU98_LOG_CAPTURE, "VGPU98 capture %u: %s, while a %s's capture records: the snapshot is the failure's\n",
                           cap_number(g), why, was);
        }
        return; /* the capture recording now covers it */
    }
    if (kind == CAP_SHOT) {
        if (g->cap_shots >= CAPTURE_SHOTS_MAX) {
            vgpu98_gpu_log(g, VGPU98_LOG_CAPTURE,
                           "VGPU98 capture: a screenshot not captured: %u screenshot snapshots written, the limit for a session\n",
                           CAPTURE_SHOTS_MAX);
            return;
        }
        if ((g->cap_count || g->cap_shots) && ((uint32_t) (now - g->cap_end_ms) < CAPTURE_GAP_MS)) {
            vgpu98_gpu_log(g, VGPU98_LOG_CAPTURE,
                           "VGPU98 capture: a screenshot not captured: %u s after the last snapshot (one per %u s)\n",
                           (uint32_t) (now - g->cap_end_ms) / 1000, CAPTURE_GAP_MS / 1000);
            return;
        }
    } else if (kind == CAP_FIND) {
        if (g->cap_finds >= CAPTURE_FINDS_MAX)
            return; /* vgpu98_gpu_find named the limit */
    } else if (g->cap_count >= CAPTURE_MAX) {
        if (!g->cap_skipped++)
            vgpu98_gpu_log(g, VGPU98_LOG_CAPTURE, "VGPU98 capture: %u snapshots written, the limit for a session; no more until 86Box restarts\n",
                           CAPTURE_MAX);
        return;
    }
    if ((kind == CAP_FAIL) && g->cap_count && ((uint32_t) (now - g->fail_end_ms) < CAPTURE_GAP_MS)) {
        g->cap_skipped++;
        return;
    }
    g->cap_head = malloc(size);
    if (!g->cap_head)
        return;
    finds[0] = 0;
    if (kind == CAP_FIND)
        snprintf(finds, sizeof(finds), "; findings since the last finding's snapshot not captured: %u", g->find_skipped);
    vgpu98_log_stamp(stamp, sizeof(stamp));
    n = (size_t) snprintf(g->cap_head, size,
                          "%s VGPU98 capture %u BEGIN: %s\n"
                          "  From %u s before the trigger to %u s after it; times are relative to the trigger. "
                          "Triggers since the last snapshot not captured: %u%s\n",
                          stamp, cap_number(g), why, CAPTURE_BEFORE_MS / 1000, CAPTURE_AFTER_MS / 1000, g->cap_skipped, finds);
    if (n >= size)
        n = size - 1;
    n += regs_text(g->cap_head + n, size - n, t);
    if (g->r3d)
        n += vgpu98_r3d_describe(g->r3d, g->cap_head + n, size - n);
    else if (n + 40 < size)
        n += (size_t) snprintf(g->cap_head + n, size - n, "  3D: not available\n");
    /* At a texture level, its textures as pictures (written on a thread of
       their own; their comparison lines follow the capture). */
    if (g->r3d && (g->log_mask & VGPU98_LOG_TEXTURES) && (n + 1 < size))
        n += vgpu98_r3d_dump(g->r3d, cap_number(g), image, sub, g->cap_head + n, size - n);
    g->cap_active = 1;
    g->cap_kind   = kind;
    g->cap_ms     = now;
    g->cap_rec    = g->rec_n;
    g->cap_post   = 0;
    if (g->r3d)
        vgpu98_r3d_census_start(g->r3d);
    vgpu98_gpu_log(g, VGPU98_LOG_CAPTURE, "VGPU98 capture %u: %s; recording %u s more\n", cap_number(g), why,
                   CAPTURE_AFTER_MS / 1000);
}

static void
capture_write(vgpu98_gpu_t *g)
{
    const uint32_t now    = plat_get_ticks();
    const uint32_t oldest = (g->rec_n > REC_COUNT) ? g->rec_n - REC_COUNT : 0;
    uint32_t       first  = (g->cap_rec > CAPTURE_PRE_MAX) ? g->cap_rec - CAPTURE_PRE_MAX : 0;
    uint32_t       k;
    uint32_t       i;
    char           line[512];
    char           stamp[24];

    out_text(g, g->cap_head);

    /* The slices around the trigger. */
    out_line(g, "  Activity per slice of about %u ms:\n", SLICE_MS);
    k = (g->n_slices > SLICE_COUNT) ? g->n_slices - SLICE_COUNT : 0;
    for (; k < g->n_slices; k++) {
        const slice_t        *s   = &g->slice[k % SLICE_COUNT];
        const vgpu98_stats_t *end = (k + 1 < g->n_slices) ? &g->slice[(k + 1) % SLICE_COUNT].total : &g->stats;
        const int32_t         at  = (int32_t) (s->ms - g->cap_ms);

        if ((at < -CAPTURE_BEFORE_MS - SLICE_MS) || (at > CAPTURE_AFTER_MS))
            continue;
        if (end->packets == s->total.packets) {
            out_line(g, "  %+6d ms  idle\n", at);
            continue;
        }
        slice_delta(line, sizeof(line), &s->total, end);
        out_line(g, "  %+6d ms  %s\n", at, line);
    }

    /* The records. */
    if (first < oldest)
        first = oldest;
    out_line(g, "  Records before the trigger (at most %u):\n", CAPTURE_PRE_MAX);
    for (i = first; i != g->cap_rec; i++) {
        const rec_t *r = &g->rec[i & (REC_COUNT - 1)];

        if ((int32_t) (g->cap_ms - r->ms) > CAPTURE_BEFORE_MS)
            continue;
        rec_format(r, line, sizeof(line));
        out_line(g, "  %+6d ms  %s\n", (int32_t) (r->ms - g->cap_ms), line);
    }
    out_line(g, "  Records after the trigger (at most %u)%s:\n", CAPTURE_POST_MAX,
             (g->cap_post >= CAPTURE_POST_MAX) ? ", the rest not recorded" : "");
    for (i = g->cap_rec; i != g->rec_n; i++) {
        const rec_t *r = &g->rec[i & (REC_COUNT - 1)];

        rec_format(r, line, sizeof(line));
        out_line(g, "  %+6d ms  %s\n", (int32_t) (r->ms - g->cap_ms), line);
    }
    /* The census of the draws after the trigger. */
    if (g->r3d) {
        char *census = vgpu98_r3d_census_end(g->r3d);

        if (census) {
            out_text(g, census);
            free(census);
        }
    }
    vgpu98_log_stamp(stamp, sizeof(stamp));
    out_line(g, "%s VGPU98 capture %u END\n", stamp, cap_number(g));
    out_flush(g);

    free(g->cap_head);
    g->cap_head    = NULL;
    g->cap_active  = 0;
    /* A finding's snapshot holds back neither a screenshot (cap_end_ms) nor
       the failures' count of triggers not captured. */
    if (g->cap_kind == CAP_FIND) {
        g->cap_finds++;
        g->find_end_ms  = now;
        g->find_skipped = 0;
        return;
    }
    if (g->cap_kind == CAP_SHOT)
        g->cap_shots++;
    else {
        g->cap_count++;
        g->fail_end_ms = now;
    }
    g->cap_end_ms  = now;
    g->cap_skipped = 0;
}

/* The thread's periodic work, with the lock released: slices, the summary,
   a capture that has recorded long enough, and the gathered lines. */
static void
diag_tick(vgpu98_gpu_t *g)
{
    const uint32_t now = plat_get_ticks();

    if (g->rec && (!g->n_slices || ((uint32_t) (now - g->slice[(g->n_slices - 1) % SLICE_COUNT].ms) >= SLICE_MS))) {
        slice_t *s = &g->slice[g->n_slices % SLICE_COUNT];

        s->ms    = now;
        s->total = g->stats;
        g->n_slices++;
    }
    if ((g->log_mask & VGPU98_LOG_SUMMARY) && ((uint32_t) (now - g->summary_ms) >= SUMMARY_MS)) {
        vgpu98_disp_stats_t ds;

        /* While the GPU worked, or the display composed frames. */
        vgpu98_disp_stats(g->disp, &ds);
        if ((g->stats.packets != g->summary_from.packets) || (ds.frames != g->disp_from.frames)) {
            dog_set(g, DOG_LOG, 0);
            summary_write(g, now);
            dog_set(g, DOG_IDLE, 0);
        }
        g->summary_from = g->stats;
        g->disp_from    = ds;
        g->summary_ms   = now;
        g->n_frames     = 0;
        g->frame_worst  = 0;
        memset(&g->slow, 0, sizeof(g->slow));
    }
    if (g->cap_active && ((uint32_t) (now - g->cap_ms) >= CAPTURE_AFTER_MS)) {
        dog_set(g, DOG_CAPTURE, 0);
        capture_write(g);
        dog_set(g, DOG_IDLE, 0);
    }
    /* Diagnostics "Glide": the Glide device's report; "OpenGL": the OpenGL
       device's. */
    if (g->glide && (g->log_mask & VGPU98_LOG_GLIDE)) {
        dog_set(g, DOG_LOG, 0);
        vgpu98_glide_diag(g->glide);
        dog_set(g, DOG_IDLE, 0);
    }
    if (g->ogl && (g->log_mask & VGPU98_LOG_OPENGL)) {
        dog_set(g, DOG_LOG, 0);
        vgpu98_ogl_diag(g->ogl);
        dog_set(g, DOG_IDLE, 0);
    }
}

/* --- The thread -------------------------------------------------------------- */

static int
gpu_can_run(const vgpu98_gpu_t *g)
{
    return g->run && !g->error && !g->hold_driver && !g->hold_vdd;
}

static const char *
error_name(uint32_t code)
{
    static const char *const names[] = { "?",
                                         "unknown opcode",
                                         "bad length",
                                         "bad argument",
                                         "packet past the ring's end",
                                         "not in the profile",
                                         "bad RING_TAIL",
                                         "a GPU address the page table does not map",
                                         "a page table entry outside guest RAM" };

    return (code < (sizeof(names) / sizeof(names[0]))) ? names[code] : "?";
}

/* STATUS (under lock). */
static uint32_t
status_locked(const vgpu98_gpu_t *g)
{
    uint32_t v = 0;

    if (gpu_can_run(g) && ((g->ring_head != g->ring_tail) || g->executing || g->batch_open || g->flip_queued))
        v |= VGPU98_STATUS_BUSY;
    if (g->error)
        v |= VGPU98_STATUS_ERROR;
    if (g->run && !g->error)
        v |= VGPU98_STATUS_RUNNING;
    if (g->flip_pending)
        v |= VGPU98_STATUS_FLIP_PENDING;
    if (g->hold_driver || g->hold_vdd)
        v |= VGPU98_STATUS_HELD;
    return v;
}

/* The status block in guest RAM (interface 2.2, under lock): FENCE,
   RING_HEAD, STATUS and VBLANK_COUNT, written whenever one of them changes,
   so the drivers poll memory instead of registers. A dword at a time; each
   field is current on its own. */
static void
status_write_locked(const vgpu98_gpu_t *g)
{
    volatile uint32_t *p;

    if (!g->status_base)
        return;
    /* Whatever the GPU wrote before (copies back into video memory) is
       visible before the values that report it. */
    atomic_thread_fence(memory_order_release);
    p    = (volatile uint32_t *) (ram + g->status_base);
    p[0] = g->fence;
    p[1] = g->ring_head;
    p[2] = status_locked(g);
    p[3] = g->vblank_count;
}

/* The registers as a trigger notes them (under lock). */
static void
trig_regs_locked(const vgpu98_gpu_t *g, trig_regs_t *t)
{
    t->status       = status_locked(g);
    t->error_code   = g->error ? g->error_code : 0;
    t->error_offset = g->error ? g->error_offset : 0;
    t->fence        = g->fence;
    t->ring_base    = g->ring_base;
    t->ring_size    = g->ring_size;
    t->ring_head    = g->ring_head;
    t->ring_tail    = g->ring_tail;
    t->hold_driver  = g->hold_driver;
    t->hold_vdd     = g->hold_vdd;
    t->generation   = g->generation;
}

/* A failure's trigger (under lock): the thread starts the capture. The
   first wins until the thread has taken it. A screenshot's waits apart. */
static void
trigger_locked(vgpu98_gpu_t *g, const char *fmt, ...)
{
    va_list ap;

    if (!(g->log_mask & VGPU98_LOG_CAPTURE) || g->trig_pending)
        return;
    va_start(ap, fmt);
    vsnprintf(g->trig_why, sizeof(g->trig_why), fmt, ap);
    va_end(ap);
    trig_regs_locked(g, &g->trig_regs);
    g->trig_pending = 1;
    thread_set_event(g->wake);
}

void
vgpu98_gpu_idle(vgpu98_gpu_t *g)
{
    atomic_store_explicit(&g->dog_reset, plat_get_ticks(), memory_order_relaxed);
}

void
vgpu98_gpu_trigger(vgpu98_gpu_t *g, const char *why)
{
    thread_wait_mutex(g->lock);
    trigger_locked(g, "%s", why);
    thread_release_mutex(g->lock);
}

void
vgpu98_gpu_screenshot(vgpu98_gpu_t *g)
{
    thread_wait_mutex(g->lock);
    if (g->log_mask & VGPU98_LOG_CAPTURE) {
        if (g->shot_pending)
            vgpu98_gpu_log(g, VGPU98_LOG_CAPTURE, "VGPU98 capture: a screenshot not captured: another is waiting\n");
        else {
            trig_regs_locked(g, &g->shot_regs);
            g->shot_pending = 1;
            thread_set_event(g->wake);
        }
    }
    thread_release_mutex(g->lock);
}

/* vgpu98_gpu_find's decision, ahead of it, for an unnamed finding (so the
   renderer formats nothing it would drop). find_pending is the GPU
   thread's to set and clear. */
int
vgpu98_gpu_find_wants(vgpu98_gpu_t *g, int kind)
{
    const int first = !(g->find_seen & (1u << kind));

    if (!(g->log_mask & VGPU98_LOG_CAPTURE))
        return 0;
    if (g->cap_active)
        return first;
    if (g->cap_finds >= CAPTURE_FINDS_MAX)
        return !g->find_full_told;
    if (g->find_pending || (!first && g->cap_finds && ((uint32_t) (plat_get_ticks() - g->find_end_ms) < CAPTURE_FIND_GAP_MS))) {
        g->find_skipped++;
        return 0;
    }
    return 1;
}

/* A finding of the texture audit, on the GPU thread (the renderer's): the
   first of each kind asks for a capture at once, the rest at most one per
   CAPTURE_FIND_GAP_MS, CAPTURE_FINDS_MAX a session; while a capture
   records, a named one goes in that capture's text. */
void
vgpu98_gpu_find(vgpu98_gpu_t *g, int kind, uint32_t image, uint32_t sub, const char *why, int named)
{
    const uint32_t now   = plat_get_ticks();
    const uint32_t bit   = 1u << kind;
    const int      first = !(g->find_seen & bit);

    if (!(g->log_mask & VGPU98_LOG_CAPTURE))
        return;
    if (g->cap_active) {
        if (named || first)
            cap_head_then(g, now, "a finding", why, NULL);
        g->find_seen |= bit;
        return;
    }
    if (g->cap_finds >= CAPTURE_FINDS_MAX) {
        if (!g->find_full_told) {
            g->find_full_told = 1;
            vgpu98_gpu_log(g, VGPU98_LOG_CAPTURE,
                           "VGPU98 capture: %u snapshots written for findings, the limit for a session; findings are still "
                           "counted and named\n",
                           CAPTURE_FINDS_MAX);
        }
        return;
    }
    thread_wait_mutex(g->lock);
    /* One waits already (the thread starts it after this packet), or the
       last finding's snapshot was too recent for any but a kind's first. */
    if (g->find_pending || (!first && g->cap_finds && ((uint32_t) (now - g->find_end_ms) < CAPTURE_FIND_GAP_MS))) {
        thread_release_mutex(g->lock);
        g->find_skipped++;
        return;
    }
    g->find_pending = 1;
    g->find_image   = image;
    g->find_sub     = sub;
    snprintf(g->find_why, sizeof(g->find_why), "%s", why);
    trig_regs_locked(g, &g->find_regs);
    thread_set_event(g->wake);
    thread_release_mutex(g->lock);
    g->find_seen |= bit;
}

void
vgpu98_gpu_fail_item(vgpu98_gpu_t *g, uint32_t dword)
{
    if (g)
        g->fail_item = dword;
}

/* The name of a packet's first argument, for the rejection line; NULL where
   a name says nothing (floats, state pairs). */
static const char *
arg_name(uint32_t op)
{
    switch (op) {
        case VGPU98_OP_FENCE:
            return "value";
        case VGPU98_OP_SYNC:
        case VGPU98_OP_DISCARD:
        case VGPU98_OP_INLINE:
            return "offset";
        case VGPU98_OP_FLIP:
            return "start";
        case VGPU98_OP_FILL:
        case VGPU98_OP_BLT:
        case VGPU98_OP_UPLOAD:
        case VGPU98_OP_DOWNLOAD:
        case VGPU98_OP_TARGET:
            return "surface at";
        case VGPU98_OP_VIEWPORT:
            return "x";
        case VGPU98_OP_TEXTURE:
        case VGPU98_OP_DRAW:
            return "control";
        case VGPU98_OP_PALETTE:
            return "stage";
        case VGPU98_OP_CLEAR:
            return "flags";
        case VGPU98_OP_DECLARE:
            return "elements";
        case VGPU98_OP_STREAM:
            return "stream";
        case VGPU98_OP_INDICES:
            return "format";
        case VGPU98_OP_TRANSFORM:
            return "which";
        case VGPU98_OP_LIGHT:
            return "slot";
        case VGPU98_OP_LIGHTENABLE:
            return "mask";
        case VGPU98_OP_CLIPPLANE:
            return "index";
        case VGPU98_OP_PROGRAM:
            return "id";
        case VGPU98_OP_BIND:
            return "vertex";
        case VGPU98_OP_CONSTANTS:
            return "kind";
        default:
            return NULL;
    }
}

/* A rejected packet in words: its name and length, the failing item of a
   list (a state pair, a declaration's element) or its first argument by
   name, then its first dwords. A packet refused before it was read is named
   by its header. */
static void
packet_describe(const vgpu98_gpu_t *g, uint32_t op, uint32_t len, int read, char *buf, size_t size)
{
    const uint32_t *p;
    const char     *arg;
    uint32_t        item;
    int             n;
    uint32_t        i;

    if (op == NO_PACKET) {
        snprintf(buf, size, "no packet");
        return;
    }
    if (!read) {
        snprintf(buf, size, "header %08X (%s, %u dwords)", g->fetch_hdr, op_name(op), len);
        return;
    }
    p    = g->payload;
    item = g->fail_item;
    arg  = arg_name(op);
    n    = snprintf(buf, size, "%s (%02Xh), %u dwords", op_name(op), op, len);
    if ((item != NO_ITEM) && (item < len) && (op == VGPU98_OP_STATE) && (item + 1 < len))
        n += snprintf(buf + n, size - n, ", state %u = %08X (pair %u of %u)", p[item], p[item + 1], item / 2 + 1, len / 2);
    else if ((item != NO_ITEM) && (item < len) && (op == VGPU98_OP_DECLARE) && (item + 1 < len))
        n += snprintf(buf + n, size - n, ", element %u: %08X %08X", (item - 1) / 2, p[item], p[item + 1]);
    else if ((item != NO_ITEM) && (item < len))
        n += snprintf(buf + n, size - n, ", dword %u: %08X", item, p[item]);
    else if (arg && len && (p[0] < 0x10000))
        n += snprintf(buf + n, size - n, ", %s %u", arg, p[0]);
    else if (arg && len)
        n += snprintf(buf + n, size - n, ", %s %08X", arg, p[0]);
    for (i = 0; (i < len) && (i < 4) && (n > 0) && ((size_t) n < size); i++)
        n += snprintf(buf + n, size - n, "%s%08X", i ? " " : "; ", p[i]);
}

/* Counts a rejection of its kind (under lock); 1 if it is to be logged in
   full. */
static int
err_sig_note(vgpu98_gpu_t *g, uint32_t code, uint32_t op, uint32_t key, const char *what)
{
    const uint32_t now = plat_get_ticks();
    err_sig_t     *s   = NULL;
    err_sig_t     *old = &g->err_sig[0];
    int            i;

    for (i = 0; i < ERR_SIGS; i++) {
        err_sig_t *e = &g->err_sig[i];

        if (e->used && (e->code == code) && (e->op == op) && (e->key == key)) {
            s = e;
            break;
        }
        /* The record to reuse: a free one, else the one seen least lately. */
        if (!e->used) {
            if (old->used)
                old = e;
        } else if (old->used && ((uint32_t) (now - e->last_ms) > (uint32_t) (now - old->last_ms)))
            old = e;
    }
    if (!s) {
        s = old;
        memset(s, 0, sizeof(*s));
        s->used = 1;
        s->code = code;
        s->op   = op;
        s->key  = key;
        snprintf(s->what, sizeof(s->what), "%s", what);
    } else if ((uint32_t) (now - s->last_ms) >= ERR_QUIET_MS)
        s->logged = 0;
    s->last_ms = now;
    s->count++;
    if (s->logged < ERR_LOG_FIRST) {
        s->logged++;
        return 1;
    }
    s->unlogged++;
    return 0;
}

/* Stops the GPU on a rejected packet (under lock). `op` is NO_PACKET when
   there is none (a bad RING_TAIL); `read` says whether its payload was read. */
static void
gpu_stop_error(vgpu98_gpu_t *g, uint32_t code, uint32_t at, uint32_t op, uint32_t len, int read)
{
    char     what[160];
    uint32_t key = 0;

    /* The kind's key: a list's failing item, else the first argument. */
    if (read && len)
        key = ((g->fail_item != NO_ITEM) && (g->fail_item < len)) ? g->payload[g->fail_item] : g->payload[0];

    g->error        = 1;
    g->error_code   = code;
    g->error_offset = at;
    g->error_count++;
    status_write_locked(g);
    packet_describe(g, op, len, read, what, sizeof(what));
    g->err_quiet = !err_sig_note(g, code, op, key, what);
    if (!g->err_quiet)
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 GPU: packet at ring offset %05X rejected, error %u (%s): %s\n", at, code,
                       error_name(code), what);
    trigger_locked(g, "the GPU rejected the packet at ring offset %05X, error %u (%s): %s", at, code, error_name(code), what);
}

/* RUN with a ring the GPU cannot read (under lock): ERROR, bad argument. */
static void
gpu_refuse_ring(vgpu98_gpu_t *g)
{
    g->error        = 1;
    g->error_code   = VGPU98_ERR_ARGUMENT;
    g->error_offset = 0;
    g->error_count++;
    g->err_quiet = 0; /* logged below: the reset after it is too */
    status_write_locked(g);
    vgpu98_gpu_log(g, VGPU98_LOG_EVENTS,
                   "VGPU98 GPU: RUN refused: a ring of %u bytes at %08X in %s memory is not one the GPU can read (error %u, %s)\n",
                   g->ring_size, g->ring_base & ~0xfffu, (g->ring_base & VGPU98_RING_SYSMEM) ? "main" : "video",
                   VGPU98_ERR_ARGUMENT, error_name(VGPU98_ERR_ARGUMENT));
    trigger_locked(g, "the GPU refused its ring at RUN");
}

static void
gpu_thread(void *priv)
{
    vgpu98_gpu_t *g = (vgpu98_gpu_t *) priv;

    thread_wait_mutex(g->lock);
    while (!g->quit) {
        g->stats.errors        = g->error_count;
        g->stats.resets        = g->generation;
        g->stats.ring_low_room = g->ring_low;
        /* Before a reset drops the 3D state, so the capture can show it. */
        if (g->trig_pending) {
            char        why[sizeof(g->trig_why)];
            trig_regs_t regs = g->trig_regs;

            memcpy(why, g->trig_why, sizeof(why));
            g->trig_pending = 0;
            thread_release_mutex(g->lock);
            dog_set(g, DOG_CAPTURE, 0);
            capture_start(g, why, &regs, CAP_FAIL, 0, 0);
            dog_set(g, DOG_IDLE, 0);
            thread_wait_mutex(g->lock);
            continue;
        }
        if (g->shot_pending) {
            trig_regs_t regs = g->shot_regs;

            g->shot_pending = 0;
            thread_release_mutex(g->lock);
            dog_set(g, DOG_CAPTURE, 0);
            capture_start(g, "a screenshot (Action > Take screenshot)", &regs, CAP_SHOT, 0, 0);
            dog_set(g, DOG_IDLE, 0);
            thread_wait_mutex(g->lock);
            continue;
        }
        if (g->find_pending) {
            char           why[sizeof(g->find_why)];
            trig_regs_t    regs  = g->find_regs;
            const uint32_t image = g->find_image;
            const uint32_t sub   = g->find_sub;

            memcpy(why, g->find_why, sizeof(why));
            g->find_pending = 0;
            thread_release_mutex(g->lock);
            dog_set(g, DOG_CAPTURE, 0);
            capture_start(g, why, &regs, CAP_FIND, image, sub);
            dog_set(g, DOG_IDLE, 0);
            thread_wait_mutex(g->lock);
            continue;
        }
        if (g->reset_req) {
            const int      cause      = g->reset_cause;
            const int      forget     = g->reset_forget;
            const int      glide_end  = g->reset_glide;
            const int      ogl_end    = g->reset_ogl;
            const uint32_t generation = g->generation;
            /* The writer of RESET waits 500 ms for it: copies back stop
               short of that. */
            const uint64_t deadline   = g->reset_us + 350000;

            g->reset_req    = 0;
            g->reset_forget = 0;
            g->reset_glide  = 0;
            g->reset_ogl    = 0;
            thread_release_mutex(g->lock);
            dog_set(g, DOG_RESET, 0);
            atomic_store_explicit(&g->dog_reset, plat_get_ticks(), memory_order_relaxed);
            if (g->r3d)
                vgpu98_r3d_reset(g->r3d, deadline, forget);
            /* Glide's sessions outlast a restart and a DOS box (hidden till
               their next swap), not a mode change or a new machine. */
            if (g->glide)
                vgpu98_glide_reset(g->glide, glide_end);
            /* OpenGL's contexts and textures outlast every reset but a new
               machine's: GL 1.1 has no lost context. */
            if (g->ogl)
                vgpu98_ogl_reset(g->ogl, ogl_end);
            dog_set(g, DOG_IDLE, 0);
            thread_set_event(g->reset_ack);
            if (g->log_mask & (VGPU98_LOG_CAPTURE | VGPU98_LOG_COMMANDS)) {
                rec_t r;

                memset(&r, 0, sizeof(r));
                r.ms   = plat_get_ticks();
                r.kind = REC_RESET;
                r.v[0] = (uint32_t) cause;
                r.v[1] = generation;
                rec_put(g, &r);
            }
            thread_wait_mutex(g->lock);
            continue;
        }
        if (g->touch_served != g->touch_serial) {
            dog_set(g, DOG_TOUCH, 0);
            touch_service_locked(g);
            dog_set(g, DOG_IDLE, 0);
            continue;
        }
        if (g->sync_visible_req) {
            g->sync_visible_req = 0;
            thread_release_mutex(g->lock);
            dog_set(g, DOG_SHOW, 0);
            gpu_show_visible(g, 1);
            dog_set(g, DOG_IDLE, 0);
            thread_wait_mutex(g->lock);
            continue;
        }
        if (gpu_can_run(g) && (g->ring_head != g->ring_tail)) {
            uint32_t generation = g->generation;
            uint32_t at         = g->ring_head;
            uint32_t op;
            uint32_t len;
            int      err;

            g->fail_item = NO_ITEM;
            err          = gpu_fetch(g, &op, &len);
            if (err) {
                gpu_stop_error(g, err, at, op, len, 0);
                continue;
            }
            g->executing  = 1;
            g->batch_open = 1;
            g->ring_known = g->ring_done + ((g->ring_tail >= at) ? g->ring_tail - at : g->ring_size - at + g->ring_tail);
            thread_release_mutex(g->lock);
            count_packet(&g->stats, op, len);
            if (g->log_mask & (VGPU98_LOG_CAPTURE | VGPU98_LOG_COMMANDS))
                note_packet(g, at, op, len);
            dog_set(g, DOG_PACKET, op);
            err = gpu_execute(g, op, len, generation);
            if (g->dog) {
                const uint32_t t = plat_get_ticks();

                atomic_store_explicit(&g->dog_packet, t, memory_order_relaxed);
                if (op == VGPU98_OP_DRAW)
                    atomic_store_explicit(&g->dog_draw, t, memory_order_relaxed);
            }
            dog_set(g, DOG_IDLE, 0);
            if (g->log_mask)
                diag_tick(g);
            dog_set(g, DOG_LOCK, 0);
            thread_wait_mutex(g->lock);
            dog_set(g, DOG_IDLE, 0);
            g->executing = 0;
            if (g->generation != generation)
                continue; /* reset meanwhile */
            if (err) {
                gpu_stop_error(g, err, at, op, len, 1);
                continue;
            }
            g->ring_head = at + 4 + len * 4;
            g->ring_done += 4 + (uint64_t) len * 4;
            if (g->ring_head >= g->ring_size)
                g->ring_head = 0;
            status_write_locked(g);
            continue;
        }
        if (g->batch_open && gpu_can_run(g)) {
            uint32_t generation = g->generation;

            g->executing = 1;
            thread_release_mutex(g->lock);
            dog_set(g, DOG_BATCH, 0);
            if (g->r3d)
                vgpu98_r3d_flush(g->r3d);
            gpu_show_visible(g, 0);
            if (g->glide)
                vgpu98_glide_tick(g->glide);
            dog_set(g, DOG_IDLE, 0);
            if (g->out_len) {
                dog_set(g, DOG_LOG, 0);
                out_flush(g);
                dog_set(g, DOG_IDLE, 0);
            }
            thread_wait_mutex(g->lock);
            g->executing = 0;
            if (g->generation == generation)
                g->batch_open = 0;
            status_write_locked(g);
            continue;
        }
        {
            /* A Glide front buffer drawn into (a vertical blank woke the
               thread for it). */
            const int front = g->glide && g->glide_front && gpu_can_run(g);

            thread_release_mutex(g->lock);
            if (front) {
                dog_set(g, DOG_SHOW, 0);
                vgpu98_glide_tick(g->glide);
                dog_set(g, DOG_IDLE, 0);
            }
        }
        if (g->log_mask) {
            diag_tick(g);
            if (g->out_len) {
                dog_set(g, DOG_LOG, 0);
                out_flush(g);
                dog_set(g, DOG_IDLE, 0);
            }
        }
        {
            const uint64_t t0 = g->log_mask ? vgpu98_diag_us() : 0;

            thread_wait_event(g->wake, 100);
            thread_reset_event(g->wake);
            if (g->log_mask)
                g->stats.idle_us += us_since(t0);
        }
        thread_wait_mutex(g->lock);
    }
    thread_release_mutex(g->lock);
    if (g->cap_active)
        capture_write(g);
    if (g->out_len)
        out_flush(g);
    thread_set_event(g->exited);
}

/* --- Registers (emulation thread) -------------------------------------------- */

uint32_t
vgpu98_gpu_profile(const vgpu98_gpu_t *g)
{
    return g->profile;
}

uint32_t
vgpu98_gpu_options(const vgpu98_gpu_t *g)
{
    return g->host.options;
}

/* File 0 is the path 86Box gave, file 1 the same with ".2" before its
   extension: the cache is saved to them in turn, so a save cut short (or
   another 86Box's) leaves the other. */
FILE *
vgpu98_gpu_cache_open(const vgpu98_gpu_t *g, int which, const char *mode)
{
    char path[sizeof(g->host.pipeline_cache) + 4];

    if (!g->host.pipeline_cache[0] || !g->host.open)
        return NULL;
    vgpu98_gpu_cache_name(g, which, path, sizeof(path));
    return g->host.open(path, mode);
}

const char *
vgpu98_gpu_capture_dir(const vgpu98_gpu_t *g)
{
    return g->host.capture_dir;
}

void
vgpu98_gpu_cache_name(const vgpu98_gpu_t *g, int which, char *buf, size_t size)
{
    const char *dot = strrchr(g->host.pipeline_cache, '.');
    const size_t n  = dot ? (size_t) (dot - g->host.pipeline_cache) : strlen(g->host.pipeline_cache);

    if (!which)
        snprintf(buf, size, "%s", g->host.pipeline_cache);
    else
        snprintf(buf, size, "%.*s.2%s", (int) n, g->host.pipeline_cache, dot ? dot : "");
}

vgpu98_disp_t *
vgpu98_gpu_disp(vgpu98_gpu_t *g)
{
    return g->disp;
}

volatile uint16_t *
vgpu98_gpu_newer(vgpu98_gpu_t *g)
{
    return g->vram.newer;
}

void
vgpu98_gpu_touch(vgpu98_gpu_t *g, uint32_t addr)
{
    const uint64_t t0 = vgpu98_diag_us();
    uint32_t       serial;
    int            done = 0;
    int            i;

    thread_wait_mutex(g->lock);
    serial        = ++g->touch_serial;
    g->touch_page = addr >> VGPU98_PAGE_SHIFT;
    thread_reset_event(g->touch_done);
    thread_set_event(g->wake);
    thread_set_event(g->flip_wake);
    thread_release_mutex(g->lock);
    /* Up to about two seconds: a host GPU that hangs must not hang the
       machine (the access then sees video memory as it is). */
    for (i = 0; (i < 40) && !done; i++) {
        thread_wait_event(g->touch_done, 50);
        thread_wait_mutex(g->lock);
        done = (int32_t) (g->touch_served - serial) >= 0;
        if (!done)
            thread_reset_event(g->touch_done);
        thread_release_mutex(g->lock);
    }
    if (!done)
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS,
                       "VGPU98 GPU: a guest access to video memory at %08Xh waited 2 s for the host GPU's pixels; it went ahead without them\n",
                       addr);
    thread_wait_mutex(g->lock);
    g->touch_count++;
    g->touch_us += vgpu98_diag_us() - t0;
    thread_release_mutex(g->lock);
}

void
vgpu98_gpu_sync_visible_request(vgpu98_gpu_t *g)
{
    thread_wait_mutex(g->lock);
    g->sync_visible_req = 1;
    thread_set_event(g->wake);
    thread_release_mutex(g->lock);
}

uint32_t
vgpu98_gpu_read(vgpu98_gpu_t *g, uint32_t reg)
{
    uint32_t v = 0;

    thread_wait_mutex(g->lock);
    switch (reg) {
        case VGPU98_REG_COMMAND_SET:
            v = VGPU98_COMMAND_SET;
            break;
        case VGPU98_REG_RING_BASE:
            v = g->ring_base;
            break;
        case VGPU98_REG_RING_SIZE:
            v = g->ring_size;
            break;
        case VGPU98_REG_RING_HEAD:
            v = g->ring_head;
            break;
        case VGPU98_REG_RING_TAIL:
            v = g->ring_tail;
            break;
        case VGPU98_REG_CONTROL:
            if (g->run && !g->hold_driver && !g->hold_vdd)
                v |= VGPU98_CTRL_RUN;
            if (g->hold_driver)
                v |= VGPU98_CTRL_HOLD_DRIVER;
            if (g->hold_vdd)
                v |= VGPU98_CTRL_HOLD_VDD;
            break;
        case VGPU98_REG_STATUS:
            v = status_locked(g);
            break;
        case VGPU98_REG_ERROR_OFFSET:
            v = g->error ? g->error_offset : 0;
            break;
        case VGPU98_REG_ERROR_CODE:
            v = g->error ? g->error_code : 0;
            break;
        case VGPU98_REG_FENCE:
            v = g->fence;
            break;
        case VGPU98_REG_VBLANK_COUNT:
            v = g->vblank_count;
            break;
        case VGPU98_REG_STATUS_BASE:
            v = g->status_base;
            break;
        case VGPU98_REG_GPUVM_BASE:
            v = g->gpuvm_base;
            break;
        case VGPU98_REG_GPUVM_ENTRIES:
            v = g->gpuvm_entries;
            break;
        default:
            break;
    }
    thread_release_mutex(g->lock);
    return v;
}

/* A ring in video memory lies wholly inside it, up to 1 MB. A ring in
   system memory (RING_BASE.SYSMEM, the rest of bits 0-11 zero) lies wholly
   in guest RAM above the first megabyte, up to 4 MB. Either is a power of
   two of at least 4 KB. */
static int
ring_valid(const vgpu98_gpu_t *g)
{
    const uint64_t ram_bytes = (uint64_t) mem_size * 1024;
    const uint32_t phys      = g->ring_base & ~0xfffu;

    if ((g->ring_size < VGPU98_RING_SIZE_MIN) || (g->ring_size & (g->ring_size - 1)))
        return 0;
    if (g->ring_base & VGPU98_RING_SYSMEM)
        return ((g->ring_base & 0xfff) == VGPU98_RING_SYSMEM) && (g->ring_size <= VGPU98_RING_SYS_SIZE_MAX) &&
               ram && (phys >= 0x100000) && ((uint64_t) phys + g->ring_size <= ram_bytes);
    return (g->ring_size <= VGPU98_RING_SIZE_MAX) && !(g->ring_base & 0xfff) && (g->ring_base < g->vram.size) &&
           (g->ring_size <= g->vram.size - g->ring_base);
}

/* RESET (under lock): the GPU stops and forgets everything it holds outside
   video memory. FENCE keeps its value. */
static void
gpu_reset_locked(vgpu98_gpu_t *g, int cause)
{
    g->reset_cause  = cause;
    g->run          = 0;
    g->ring_head    = 0;
    g->ring_tail    = 0;
    g->error        = 0;
    g->error_code   = 0;
    g->error_offset = 0;
    g->flip_pending = 0;
    g->flip_queued  = 0;
    g->imm_pending  = 0;
    g->batch_open   = 0;
    g->generation++;
    /* Two RESETs the thread takes as one keep the stronger for the Glide
       device: a mode change's hold before the driver's start still ends its
       sessions. */
    {
        const int glide = ((cause == CAUSE_MACHINE) || (cause == CAUSE_HOLD_DRIVER) || (cause == CAUSE_POWER_ON)) ? 2
                          : (cause == CAUSE_HOLD_VDD)                                                          ? 1
                                                                                                                : 0;

        if (!g->reset_req) {
            g->reset_glide = 0;
            g->reset_ogl   = 0;
        }
        if (glide > g->reset_glide)
            g->reset_glide = glide;
        if ((cause == CAUSE_MACHINE) || (cause == CAUSE_POWER_ON))
            g->reset_ogl = 1;
    }
    g->reset_req = 1;
    g->reset_us  = vgpu98_diag_us();
    if ((cause == CAUSE_POWER_ON) || (cause == CAUSE_MACHINE) || (cause == CAUSE_START))
        g->reset_forget = 1;
    status_write_locked(g);
    thread_reset_event(g->reset_ack);
    thread_set_event(g->wake);
    thread_set_event(g->flip_wake);
}

void
vgpu98_gpu_write(vgpu98_gpu_t *g, uint32_t reg, uint32_t val)
{
    int wait_reset = 0;

    thread_wait_mutex(g->lock);
    switch (reg) {
        case VGPU98_REG_RING_BASE:
            if (!g->run)
                g->ring_base = val;
            break;
        case VGPU98_REG_RING_SIZE:
            if (!g->run)
                g->ring_size = val;
            break;
        case VGPU98_REG_RING_TAIL:
            if ((val & 3) || (val >= g->ring_size)) {
                if (!g->error)
                    gpu_stop_error(g, VGPU98_ERR_TAIL, g->ring_head, NO_PACKET, 0, 0);
                break;
            }
            g->ring_tail = val;
            status_write_locked(g); /* BUSY */
            if (g->log_mask && g->ring_size) {
                const uint32_t used = (val >= g->ring_head) ? val - g->ring_head : g->ring_size - g->ring_head + val;

                if (used > g->ring_peak)
                    g->ring_peak = used;
                if ((g->ring_size - used) < (g->ring_size / 8))
                    g->ring_low++;
            }
            thread_set_event(g->wake);
            break;
        case VGPU98_REG_CONTROL: {
            /* What a RESET is for: a hold it sets, else the DirectX driver
               (re)starting the GPU, which it does to a running GPU only
               after it made no progress. */
            const int was_running = g->run && !g->error && !g->hold_driver && !g->hold_vdd;
            const int was_error   = g->error;
            int       cause;

            if ((val & VGPU98_CTRL_HOLD_DRIVER_WE) && !(val & VGPU98_CTRL_HOLD_DRIVER) && g->hold_driver)
                vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 GPU: the display driver lets go of its hold\n");
            if ((val & VGPU98_CTRL_HOLD_VDD_WE) && !(val & VGPU98_CTRL_HOLD_VDD) && g->hold_vdd)
                vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 GPU: the mini-VDD lets go of its hold\n");
            if (val & VGPU98_CTRL_HOLD_DRIVER_WE)
                g->hold_driver = !!(val & VGPU98_CTRL_HOLD_DRIVER);
            if (val & VGPU98_CTRL_HOLD_VDD_WE)
                g->hold_vdd = !!(val & VGPU98_CTRL_HOLD_VDD);
            if (g->hold_driver || g->hold_vdd)
                g->run = 0;
            if (val & VGPU98_CTRL_RESET) {
                if ((val & VGPU98_CTRL_HOLD_DRIVER_WE) && (val & VGPU98_CTRL_HOLD_DRIVER))
                    cause = CAUSE_HOLD_DRIVER;
                else if ((val & VGPU98_CTRL_HOLD_VDD_WE) && (val & VGPU98_CTRL_HOLD_VDD))
                    cause = CAUSE_HOLD_VDD;
                else if (was_error)
                    cause = CAUSE_AFTER_ERROR; /* the error itself was the trigger */
                else if (was_running) {
                    cause = CAUSE_NO_PROGRESS;
                    trigger_locked(g, "%s", reset_causes[cause]);
                } else
                    cause = CAUSE_START;
                /* A reset after a rejection that went unlogged goes unlogged
                   too: the summary counts both. */
                if ((cause != CAUSE_AFTER_ERROR) || !g->err_quiet)
                    vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 GPU: reset %u: %s\n", g->generation + 1, reset_causes[cause]);
                gpu_reset_locked(g, cause);
                wait_reset = 1;
            } else if (!(val & VGPU98_CTRL_RUN))
                g->run = 0;
            else if (!g->hold_driver && !g->hold_vdd && !g->run) {
                if (ring_valid(g)) {
                    if ((g->ring_base != g->ring_shown_base) || (g->ring_size != g->ring_shown_size)) {
                        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 GPU: ring %u KB in %s memory at %08X\n",
                                       g->ring_size >> 10, (g->ring_base & VGPU98_RING_SYSMEM) ? "main" : "video",
                                       g->ring_base & ~0xfffu);
                        g->ring_shown_base = g->ring_base;
                        g->ring_shown_size = g->ring_size;
                    }
                    g->run = 1;
                    thread_set_event(g->wake);
                } else if (!g->error)
                    gpu_refuse_ring(g);
            }
            status_write_locked(g);
            break;
        }
        case VGPU98_REG_FENCE:
            if (!g->run || g->hold_driver || g->hold_vdd) {
                g->fence = val;
                status_write_locked(g);
            }
            break;
        case VGPU98_REG_STATUS_BASE:
            /* 16-byte aligned, wholly in guest RAM above the first megabyte;
               anything else turns the block off. */
            g->status_base = ((val & 0xf) || (val < 0x100000) || !ram || ((uint64_t) val + 16 > (uint64_t) mem_size * 1024)) ?
                                 0 : val;
            status_write_locked(g);
            break;
        case VGPU98_REG_GPUVM_BASE:
            /* A multiple of 4 KB in guest RAM above the first megabyte;
               anything else is no table. Each entry read is checked again. */
            g->gpuvm_base = ((val & 0xfff) || (val < 0x100000) || !ram || ((uint64_t) val + 4 > (uint64_t) mem_size * 1024)) ?
                                0 : val;
            break;
        case VGPU98_REG_GPUVM_ENTRIES:
            g->gpuvm_entries = (val > VGPU98_GPUVM_ENTRIES_MAX) ? VGPU98_GPUVM_ENTRIES_MAX : val;
            break;
        default:
            break;
    }
    thread_release_mutex(g->lock);

    /* The thread lets go of everything before the reset completes, so no
       packet still in flight writes video memory that now belongs to
       someone else. */
    if (wait_reset)
        thread_wait_event(g->reset_ack, 500);
}

int
vgpu98_gpu_vblank(vgpu98_gpu_t *g, uint32_t *start)
{
    int ret = 0;

    if (g->dog)
        atomic_store_explicit(&g->dog_vblank, plat_get_ticks(), memory_order_relaxed);
    thread_wait_mutex(g->lock);
    g->vblank_count++;
    status_write_locked(g);
    if (g->vblank_waiters)
        thread_set_event(g->flip_wake); /* a Glide swap's wait */
    if (g->glide_front)
        thread_set_event(g->wake); /* a Glide front buffer to show */
    if (g->imm_pending) {
        *start         = g->imm_start;
        g->imm_pending = 0;
        ret            = 1;
    }
    if (g->flip_queued && ((uint32_t) (g->vblank_count - g->last_flip_vblank) >= g->flip_interval)) {
        *start              = g->flip_start;
        g->flip_queued      = 0;
        g->flip_done        = 1;
        g->last_flip_vblank = g->vblank_count;
        ret                 = 1;
        thread_set_event(g->flip_wake);
    }
    thread_release_mutex(g->lock);
    return ret;
}

/* A wait for the host GPU in a watchdog line: the seconds without its work
   done, and how long ago that was checked (a long time: the host's Vulkan
   driver has not returned from the wait). Nothing before the first second. */
static void
dog_wait_text(char *buf, size_t size, uint32_t now, uint32_t seconds, uint32_t checked)
{
    if (seconds)
        snprintf(buf, size, " (not done after %u s, checked %d ms ago)", seconds, (int) (int32_t) (now - checked));
    else
        buf[0] = 0;
}

/* The watchdog: every second, a GPU thread stuck over 2 seconds in one
   packet, a batch's end, a capture, a reset, a guest access or the
   display's copy, with the step it waits in (VGPU98_STEP_*) and, while it
   matters, the display thread's, and vertical blanks stopping (the
   emulation's thread stuck, or the machine paused, which it says), each
   named again every 10 seconds while it lasts; and, once an episode,
   nothing sent for 5 minutes after 3D drawing with a device open. Reads
   the fields without the lock: a line, not a decision. */
static void
dog_thread(void *priv)
{
    static const char *const phases[9] = { "nothing", "a packet", "a batch's end", "a capture", "a reset", "a guest access",
                                           "the display's copy", "writing the log", "waiting for its lock" };
    static const char *const steps[9]  = { "",
                                           "handing work to the host GPU",
                                           "waiting for the host GPU",
                                           "copying back to video memory",
                                           "waiting for the display's layer",
                                           "making a pipeline",
                                           "saving the pipeline cache",
                                           "waiting for its lock",
                                           "waiting for the host GPU to go idle" };
    static const char *const dsteps[5] = { "waiting for a frame", "waiting for its layer", "composing",
                                           "handing work to the host GPU", "waiting for the host GPU" };
    /* How long a phase may last before it is named: a packet or a batch's
       end 1 second (the DirectX driver resets a GPU that made no progress
       for one), a capture 5 (it records for 2 seconds), the others 2. */
    static const int32_t limits[9] = { 0, 1000, 1000, 5000, 2000, 2000, 2000, 2000, 2000 };
    vgpu98_gpu_t            *g          = (vgpu98_gpu_t *) priv;
    uint32_t                 told_gpu   = 0;
    uint32_t                 told_vb    = 0;
    uint32_t                 told_since = 0; /* the episode each was named for */
    uint32_t                 told_vbat  = 0;
    uint32_t                 told_idle  = 0;

    while (!atomic_load(&g->dog_quit)) {
        uint32_t since;
        uint32_t vb;
        uint32_t now;
        int32_t  dt;
        int      phase;
        int      step;
        uint32_t step_since;
        uint32_t wait_s;
        uint32_t wait_at;
        int      dstep;
        uint32_t dsince;
        uint32_t dwait_s;
        uint32_t dwait_at;

        thread_wait_event(g->dog_wake, 1000);
        if (atomic_load(&g->dog_quit))
            break;
        /* The stored times first, the clock after: a store in between
           makes a difference negative, never a wrapped one. */
        phase      = atomic_load_explicit(&g->dog_phase, memory_order_acquire);
        since      = atomic_load_explicit(&g->dog_since, memory_order_relaxed);
        vb         = atomic_load_explicit(&g->dog_vblank, memory_order_relaxed);
        step       = atomic_load_explicit(&g->dog_step, memory_order_acquire);
        step_since = atomic_load_explicit(&g->dog_step_since, memory_order_relaxed);
        wait_s     = atomic_load_explicit(&g->dog_wait_s, memory_order_acquire);
        wait_at    = atomic_load_explicit(&g->dog_wait_at, memory_order_relaxed);
        dstep      = vgpu98_disp_watch(g->disp, &dsince, &dwait_s, &dwait_at);
        {
            const uint32_t last  = atomic_load_explicit(&g->dog_packet, memory_order_relaxed);
            const uint32_t drawn = atomic_load_explicit(&g->dog_draw, memory_order_relaxed);
            const uint32_t reset = atomic_load_explicit(&g->dog_reset, memory_order_relaxed);

            now = plat_get_ticks();
            /* Stopped while drawing: no packet for DOG_QUIET_MS, the last
               ones 3D, drawing not ended since (no RESET, which a program
               going back to its menu or the desktop by a mode change
               brings, and the DirectX driver's last Direct3D context not
               gone: its DEBUG_IDLE), vertical blanks still coming. The
               guest sends no work with a device open for longer than any
               load takes (3DMark's scenes load in well under a minute): a
               possible hang outside the DirectX driver's waits, as round
               4's stall. Not after vertical blanks stopped since (the
               machine paused: its quiet is not the guest's). A full-screen
               program's: a windowed one's desktop drawing in the meantime
               keeps it quiet (its last packets are not 3D). Once an
               episode. */
            if ((phase == DOG_IDLE) && last && drawn && ((int32_t) (last - drawn) < 1000) &&
                ((int32_t) (now - last) >= DOG_QUIET_MS) && !(reset && ((int32_t) (reset - drawn) >= 0)) && vb &&
                ((int32_t) (now - vb) < 2000) && !(told_vb && ((int32_t) (told_vb - last) >= 0)) && (last != told_idle)) {
                told_idle = last;
                vgpu98_gpu_log(g, VGPU98_LOG_EVENTS,
                               "VGPU98 GPU: watchdog: nothing sent for %d s after 3D drawing with a device open, vertical blanks "
                               "running (a possible hang outside the DirectX driver's waits)\n",
                               (int) ((now - last) / 1000));
            }
        }
        dt    = (int32_t) (now - since);
        if ((phase > DOG_IDLE) && (phase <= DOG_LOCK) && (dt >= limits[phase]) &&
            ((since != told_since) || ((int32_t) (now - told_gpu) >= 10000))) {
            char inside[160] = "";
            char shown[160]  = "";
            char w[64];

            told_gpu   = now;
            told_since = since;
            /* What it waits on in the phase; the display thread's step while
               the GPU thread waits for its layer, or once it has been in one
               a second (waiting for a frame is its rest). */
            if ((step > VGPU98_STEP_NONE) && (step <= VGPU98_STEP_IDLE)) {
                dog_wait_text(w, sizeof(w), now, (step == VGPU98_STEP_HOST_GPU) ? wait_s : 0, wait_at);
                snprintf(inside, sizeof(inside), "; inside, %s for %d ms%s", steps[step], (int) (int32_t) (now - step_since), w);
            }
            if ((dstep > VGPU98_DSTEP_FRAME) && (dstep <= VGPU98_DSTEP_HOST_GPU) &&
                ((step == VGPU98_STEP_LAYER) || ((int32_t) (now - dsince) >= 1000))) {
                dog_wait_text(w, sizeof(w), now, (dstep == VGPU98_DSTEP_HOST_GPU) ? dwait_s : 0, dwait_at);
                snprintf(shown, sizeof(shown), "; the display thread %s for %d ms%s", dsteps[dstep],
                         (int) (int32_t) (now - dsince), w);
            }
            vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 GPU: watchdog: the GPU thread has been in %s%s%s for %d ms (ring head %X, tail %X)%s%s\n",
                           phases[phase], (phase == DOG_PACKET) ? " " : "",
                           (phase == DOG_PACKET) ? op_name(atomic_load(&g->dog_op)) : "", (int) dt, g->ring_head, g->ring_tail,
                           inside, shown);
        }
        dt = (int32_t) (now - vb);
        if (vb && (dt >= 2000) && ((vb != told_vbat) || ((int32_t) (now - told_vb) >= 10000))) {
            told_vb   = now;
            told_vbat = vb;
            vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 GPU: watchdog: no vertical blank for %d ms (%s); the GPU thread in %s\n",
                           (int) dt, dopause ? "the machine is paused" : "the emulation's thread is stuck, or the display off",
                           ((phase >= DOG_IDLE) && (phase <= DOG_LOCK)) ? phases[phase] : "?");
        }
    }
    thread_set_event(g->dog_exited);
}

/* The registers' power-on state: a machine reset, or GPU_RESET from the VGA
   BIOS at POST. The status block goes first, so nothing more is written to
   memory that is no longer the drivers'. */
static void
gpu_power_on(vgpu98_gpu_t *g, int cause)
{
    thread_wait_mutex(g->lock);
    g->status_base     = 0;
    g->gpuvm_base      = 0;
    g->gpuvm_entries   = 0;
    g->ring_base       = 0;
    g->ring_size       = 0;
    g->ring_shown_base = 0;
    g->ring_shown_size = 0;
    g->hold_driver     = 0;
    g->hold_vdd        = 0;
    g->fence           = 0;
    if (cause != CAUSE_MACHINE)
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 GPU: reset %u: %s\n", g->generation + 1, reset_causes[cause]);
    /* The V-sync overrides go with a machine reset and POST, not with the
       GPU's own resets. */
    vgpu98_gpu_set_vsync(g, 0);
    gpu_reset_locked(g, cause);
    thread_release_mutex(g->lock);
    thread_wait_event(g->reset_ack, 500);
}

void
vgpu98_gpu_reset(vgpu98_gpu_t *g)
{
    gpu_power_on(g, CAUSE_MACHINE);
}

void
vgpu98_gpu_power_on(vgpu98_gpu_t *g)
{
    gpu_power_on(g, CAUSE_POWER_ON);
}

vgpu98_gpu_t *
vgpu98_gpu_init(uint8_t *vram, uint32_t vram_size, const vgpu98_host_t *host, int log_mask)
{
    vgpu98_gpu_t *g     = calloc(1, sizeof(vgpu98_gpu_t));
    uint32_t      pages = vram_size >> VGPU98_PAGE_SHIFT;

    g->host           = *host;
    g->log_mask       = log_mask;
    g->vram.data      = vram;
    g->vram.size      = vram_size;
    g->vram.cpu_pages = (uint32_t *) calloc(pages + 1, sizeof(uint32_t)) + 1;
    g->vram.gpu_pages = (uint32_t *) calloc(pages + 1, sizeof(uint32_t)) + 1;
    g->vram.newer     = (volatile uint16_t *) calloc(pages, sizeof(uint16_t));
    g->payload        = malloc(VGPU98_PKT_MAX_LEN * sizeof(uint32_t));
    g->out            = calloc(1, OUT_BLOCK);
    if (log_mask & VGPU98_LOG_CAPTURE)
        g->rec = calloc(REC_COUNT, sizeof(rec_t));
    g->summary_ms = plat_get_ticks();

    g->lock      = thread_create_mutex();
    g->wake      = thread_create_event();
    g->flip_wake = thread_create_event();
    g->reset_ack = thread_create_event();
    g->exited    = thread_create_event();
    g->touch_done = thread_create_event();

    /* Commands, 2D and reads of main memory run on the host CPU and are
       always there; 3D needs the host's GPU. */
    g->profile = VGPU98_PROFILE_COMMANDS | VGPU98_PROFILE_BLIT | VGPU98_PROFILE_SYSRING | VGPU98_PROFILE_STATUSWB |
                 VGPU98_PROFILE_GPUVM;
    /* Command set 2.3's DOWNLOAD, unless "Locks through main memory" is Off. */
    if (!(host->options & VGPU98_OPT_NO_DOWNLOAD))
        g->profile |= VGPU98_PROFILE_DOWNLOAD;
    /* Interface 3.4's hardware cursor, which the card shows whoever draws
       the frame, unless "Cursor" is Software. */
    if (!(host->options & VGPU98_OPT_SOFT_CURSOR))
        g->profile |= VGPU98_PROFILE_CURSOR;
    /* Interface 3.5's gamma table, unless "Gamma" is Off. */
    if (!(host->options & VGPU98_OPT_NO_GAMMA))
        g->profile |= VGPU98_PROFILE_GAMMA;
    /* Interface 3.6's V-sync overrides, unless "V-sync overrides" is Off. */
    if (!(host->options & VGPU98_OPT_NO_VSYNC))
        g->profile |= VGPU98_PROFILE_VSYNC;
    g->r3d     = vgpu98_r3d_init(g, &g->vram);
    if (g->r3d)
        g->profile |= VGPU98_PROFILE_RENDER3D | VGPU98_PROFILE_VERTEX;
    /* Command set 2.1, unless the setting "DirectX 8 interface" is Off. */
    if (g->r3d && !(host->options & VGPU98_OPT_NO_FIXED8))
        g->profile |= VGPU98_PROFILE_FIXED8;
    /* Command set 2.2's programs with 2.1, unless "DirectX 8 shaders" is Off. */
    if ((g->profile & VGPU98_PROFILE_FIXED8) && !(host->options & VGPU98_OPT_NO_PROGRAMS))
        g->profile |= VGPU98_PROFILE_PROGRAMS;
    g->vram.fixed8   = (g->profile & VGPU98_PROFILE_FIXED8) != 0;
    g->vram.programs = (g->profile & VGPU98_PROFILE_PROGRAMS) != 0;
    g->disp = vgpu98_disp_init(g, g->r3d, &g->vram);
    if (g->r3d)
        vgpu98_r3d_set_disp(g->r3d, g->disp);
    /* Command set 2.4's Glide device, with 3D. */
    g->glide = vgpu98_glide_init(g, g->r3d, g->disp, &g->vram);
    if (g->glide)
        g->profile |= VGPU98_PROFILE_GLIDE;
    /* Command set 2.5's OpenGL device, with 3D. */
    g->ogl = vgpu98_ogl_init(g, g->r3d, g->disp, &g->vram);
    if (g->ogl)
        g->profile |= VGPU98_PROFILE_OGL;
    g->profile &= VGPU98_PROFILE;
    vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 GPU: profile %X (3D %s%s%s%s%s%s%s)\n", g->profile,
                   g->r3d ? "on the host's GPU" : "unavailable",
                   !g->r3d                                  ? "" :
                   (g->profile & VGPU98_PROFILE_PROGRAMS) ? "; command set 2.2, DirectX 8 with shaders" :
                   (g->profile & VGPU98_PROFILE_FIXED8)   ? "; command set 2.1, DirectX 8: shaders are Off" :
                                                            "; command set 2.0: the DirectX 8 interface is Off",
                   (g->profile & VGPU98_PROFILE_GLIDE) ? "; Glide (command set 2.4)" : "",
                   (g->profile & VGPU98_PROFILE_OGL) ? "; OpenGL (command set 2.7)" : "",
                   (g->profile & VGPU98_PROFILE_CURSOR) ? "; hardware cursor" : "",
                   (g->profile & VGPU98_PROFILE_GAMMA) ? "; gamma" : "",
                   (g->profile & VGPU98_PROFILE_VSYNC) ? "; V-sync overrides" : "");

    g->thread = thread_create_named(gpu_thread, g, "VGPU98 GPU");
    if (log_mask) {
        g->dog_wake   = thread_create_event();
        g->dog_exited = thread_create_event();
        g->dog        = thread_create_named(dog_thread, g, "VGPU98 watchdog");
    }
    return g;
}

int
vgpu98_gpu_close(vgpu98_gpu_t *g)
{
    int stuck, ogl_closed;

    /* Neither thread stays in a wait for the host GPU (a hang's), and the
       pipeline cache's warm-up stops making pipelines. */
    atomic_store(&g->closing, 1);
    vgpu98_disp_closing(g->disp);
    vgpu98_r3d_closing(g->r3d);
    thread_wait_mutex(g->lock);
    g->quit = 1;
    thread_set_event(g->wake);
    thread_set_event(g->flip_wake);
    thread_release_mutex(g->lock);
    stuck = thread_wait_event(g->exited, 5000);
    if (g->dog) {
        atomic_store(&g->dog_quit, 1);
        thread_set_event(g->dog_wake);
        /* It wakes within a second. Its event, not its handle: a thread's
           handle may close as it ends (_beginthread's on Windows). */
        if (thread_wait_event(g->dog_exited, 3000))
            stuck = 1;
        else {
            thread_destroy_event(g->dog_wake);
            thread_destroy_event(g->dog_exited);
        }
    }
    /* A thread that has not ended in time (the GPU thread, the watchdog, the
       display's, a pipeline cache save) keeps everything it may use: 86Box
       is closing, and it is left rather than freed under it. */
    if (stuck)
        return 0;

    if (g->r3d)
        vgpu98_r3d_set_disp(g->r3d, NULL); /* closing the renderer drops images: not into a closed display */
    vgpu98_glide_close(g->glide);          /* before the display and the renderer: it uses both */
    ogl_closed = vgpu98_ogl_close(g->ogl); /* likewise */
    if (!vgpu98_disp_close(g->disp))       /* before the renderer: it uses its device */
        return 0;
    /* The OpenGL device's warm-up still making a pipeline keeps the
       renderer (its device and pipeline cache) and this record, through
       which it logs, as the renderer's own threads do below. */
    if (!ogl_closed)
        return 1;
    /* The renderer's own threads still running (a pipeline cache save or
       warm-up) keep the renderer and this record, through which a save
       logs and opens its file, but never read video memory or the device:
       those may go. */
    if (g->r3d && !vgpu98_r3d_close(g->r3d))
        return 1;
    thread_destroy_event(g->wake);
    thread_destroy_event(g->flip_wake);
    thread_destroy_event(g->reset_ack);
    thread_destroy_event(g->exited);
    thread_destroy_event(g->touch_done);
    thread_close_mutex(g->lock);
    free(g->vram.cpu_pages - 1);
    free(g->vram.gpu_pages - 1);
    free((void *) g->vram.newer);
    free(g->payload);
    free(g->tmp);
    free(g->out);
    free(g->rec);
    free(g->cap_head);
    free(g);
    return 1;
}
