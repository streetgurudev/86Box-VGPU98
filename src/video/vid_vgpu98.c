/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: a PCI VGA card with its own VGA BIOS and a
 *          GPU. A register-exact VGA (86Box's svga core), a display
 *          engine for packed-pixel modes at 8, 15, 16 and 32 bpp (with
 *          double scan for low resolutions) and its hardware cursor, the
 *          card's PCI configuration space (register window, linear
 *          framebuffer, expansion ROM), the virtual monitor's EDID,
 *          developer debug registers, and the GPU: a
 *          command processor (vid_vgpu98_gpu.c) drawing 3D on the host's
 *          GPU through Vulkan (vid_vgpu98_r3d.c). The card's programmer's
 *          reference lives in the VGPU98 project's spec/ folder.
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
#include <wchar.h>
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/path.h>
#include <86box/plat.h>
#include "cpu.h"
#include <86box/io.h>
#include <86box/mem.h>
#include <86box/rom.h>
#include <86box/device.h>
#include <86box/timer.h>
#include <86box/video.h>
#include <86box/vid_svga.h>
#include <86box/vid_svga_render.h>
#include <86box/pci.h>
#include <86box/machine.h>
#include <86box/plat_unused.h>
#include <86box/vid_vgpu98_card.h>
#include <86box/vid_vgpu98_gpu.h>
#include <86box/vid_vgpu98_dev.h>

#define VGPU98_DEBUG_LINE 160

#define DISP_REG(dev, sr)  ((dev)->disp_regs[(sr) - VGPU98_SR_DISP_FIRST])
#define DISP_ENABLED(dev)  (DISP_REG(dev, VGPU98_SR_DISP_CONTROL) & VGPU98_DISP_CTRL_ENABLE)
#define DISP_DSCAN(dev)    (DISP_REG(dev, VGPU98_SR_DISP_CONTROL) & VGPU98_DISP_CTRL_DSCAN)

typedef struct vgpu98_t {
    svga_t        svga;
    rom_t         bios_rom;
    int           has_rom;

    mem_mapping_t mmio_mapping;   /* BAR0: register window */
    mem_mapping_t linear_mapping; /* BAR1: framebuffer */
    mem_mapping_t window_mapping; /* A0000h-AFFFFh while the display engine is on */

    uint8_t       slot;
    uint8_t       pci_command;
    uint8_t       int_line;
    uint32_t      bar0;
    uint32_t      bar1;
    uint32_t      rom_bar;

    uint32_t      vram_size;

    /* Display engine (spec/display-engine.md) */
    int           disp_unlocked;
    uint8_t       disp_regs[VGPU98_SR_DISP_LAST - VGPU98_SR_DISP_FIRST + 1];
    uint8_t       edid_index;
    uint32_t      disp_start; /* START as last taken: when its high byte is written */

    /* The GPU (spec/gpu-registers.md) */
    vgpu98_gpu_t  *gpu;
    uint32_t      *cpu_pages;    /* per-page write counters of guest writes to video memory */
    vgpu98_disp_t *disp;         /* the display on the host's GPU, NULL without one */
    volatile uint16_t *newer;    /* the GPU-newer page map: an access there copies back first */
    int            host_display; /* it shows the display engine's frames (svga.host_display is set) */
    int            disp_restart; /* the engine went off meanwhile: everything is uploaded again */
    vgpu98_frame_t frame;        /* the frame posted at the last vertical blank */
    /* The mode and place of what the host display last put in the
       monitor's frame buffer, a picture or a black frame;
       fb_held 0 while 86Box draws it. */
    int            fb_held;
    uint32_t       fb_width, fb_height, fb_bpp, fb_pitch, fb_dscan;
    int            fb_x, fb_y;

    /* The hardware cursor: its registers as
       written, then as taken at the last vertical blank. */
    int            cursor_hw;   /* PROFILE.CURSOR: the setting "Cursor" is Hardware */
    uint32_t       cur_control;
    uint32_t       cur_pos;
    uint32_t       cur_base;
    int            cur_reload;  /* CURSOR_BASE or CURSOR_CONTROL written: the pattern is read again */
    int            cur_on;
    int32_t        cur_x;
    int32_t        cur_y;
    uint32_t       cur_bytes;   /* the pattern's bytes a pixel, the mode's when it was read */
    uint32_t       cur_serial;  /* changes with every pattern read */
    uint8_t        cur_pattern[VGPU98_CURSOR_BYTES];
    void         (*render_line)(svga_t *svga); /* the mode's renderer, which vgpu98_render_cursor wraps */

    /* The gamma table: its registers as written, then as
       taken at the last vertical blank. */
    int            gamma_hw;      /* PROFILE.GAMMA: the setting "Gamma" is On */
    int            vsync_hw;      /* PROFILE.VSYNC: the setting "V-sync overrides" is On */
    uint32_t       gam_control;
    uint32_t       gam_base;
    int            gam_reload;    /* GAMMA_BASE or GAMMA_CONTROL written: the table is read again */
    int            gam_on;        /* applied: on, the engine on, a direct colour mode */
    uint32_t       gam_serial;    /* changes with every table read */
    uint32_t       gam_table[VGPU98_GAMMA_ENTRIES];
    uint32_t       gam_logged_at; /* the log's line for a new table: plat_get_ticks of the last */
    uint32_t       gam_unlogged;  /* tables taken since it */

    /* Diagnostics "Display": its watch, NULL
       at other levels, and 86Box's timings as last seen. */
    vgpu98_dd_t   *dd;
    int            dd_dispend;
    int            dd_vsyncstart;
    int            dd_vtotal;
    int            dd_hdisp;

    /* Host diagnostic logging: VGPU98_LOG_* sources */
    int           log_mask;
    int           shader_capture; /* "DirectX 8 shaders": 1 Capture, 2 Capture with the GPU's programs (DEBUG's bits 3, 4) */
    atomic_int    shot_seen;  /* the monitor's screenshot requests when last looked (the UI thread and the emulation's) */
    char          debug_line[VGPU98_DEBUG_LINE]; /* from the debug register (SR E0h) */
    int           debug_len;
    char          dll_line[VGPU98_DEBUG_LINE];   /* from BAR0's DEBUG register */
    int           dll_len;

    /* SYSBUF_PHYS, SYSBUF_LINEAR, SYSBUF_SIZE, SYSBUF_STATUS: the
       mini-VDD's ring buffer and status page in main memory, for the
       DirectX driver (interfaces 2.1 and 2.2) */
    uint32_t      sysbuf[4];
} vgpu98_t;

static video_timings_t timing_vgpu98 = { .type = VIDEO_PCI, .write_b = 2, .write_w = 2, .write_l = 1, .read_b = 20, .read_w = 20, .read_l = 21 };

#ifdef ENABLE_VGPU98_LOG
int vgpu98_do_log = ENABLE_VGPU98_LOG;

static void
vgpu98_log(const char *fmt, ...)
{
    va_list ap;

    if (vgpu98_do_log) {
        va_start(ap, fmt);
        pclog_ex(fmt, ap);
        va_end(ap);
    }
}
#else
#    define vgpu98_log(fmt, ...)
#endif

/* The "Diagnostics" setting:
   what goes to 86Box's log. The values are stored in the machine's
   configuration; never renumber them. */
#define VGPU98_DIAG_OFF       0
#define VGPU98_DIAG_DIRECTX   1
#define VGPU98_DIAG_CAPTURE   2
#define VGPU98_DIAG_DEVELOPER 3
#define VGPU98_DIAG_AUDIT     4
#define VGPU98_DIAG_AUDIT8    5
#define VGPU98_DIAG_MINIMAL   6 /* the events and the start line alone */
#define VGPU98_DIAG_TEXTURES  7 /* DirectX + capture with the texture audit */
#define VGPU98_DIAG_GLIDE     8 /* Minimal, with the Glide DLLs' full lines */
#define VGPU98_DIAG_OPENGL    9 /* Minimal, with OpenGL's picture: the ICD's full lines, the device's report */
#define VGPU98_DIAG_DISPLAY   10 /* Minimal, with the screen watched at its transitions */

static const char *const vgpu98_diag_names[] = { "Off",
                                                 "DirectX",
                                                 "DirectX + capture",
                                                 "Developer: every packet (slow)",
                                                 "DirectX 7: vertex audit",
                                                 "DirectX 8: stream + texture audit (slow)",
                                                 "Minimal",
                                                 "DirectX: texture audit",
                                                 "Glide",
                                                 "OpenGL",
                                                 "Display" };

/* A name for every level up to the last. */
_Static_assert(sizeof(vgpu98_diag_names) / sizeof(vgpu98_diag_names[0]) == VGPU98_DIAG_DISPLAY + 1,
               "a Diagnostics level without a name");

static int
vgpu98_log_mask(int level)
{
    const int light   = VGPU98_LOG_DEBUG_REGISTER | VGPU98_LOG_EVENTS | VGPU98_LOG_SUMMARY;
    /* No summaries; the DirectX driver counts nothing and writes its event
       lines alone. */
    const int minimal = VGPU98_LOG_DEBUG_REGISTER | VGPU98_LOG_EVENTS | VGPU98_LOG_MINIMAL;

    if ((level < 0) || (level > VGPU98_DIAG_DISPLAY))
        return 0;
    switch (level) {
        case VGPU98_DIAG_MINIMAL:
            return minimal;
        case VGPU98_DIAG_DIRECTX:
            return light;
        case VGPU98_DIAG_CAPTURE:
            return light | VGPU98_LOG_CAPTURE;
        case VGPU98_DIAG_DEVELOPER:
            return light | VGPU98_LOG_CAPTURE | VGPU98_LOG_COMMANDS;
        case VGPU98_DIAG_AUDIT:
            return light | VGPU98_LOG_CAPTURE | VGPU98_LOG_AUDIT;
        case VGPU98_DIAG_AUDIT8:
            /* With the texture audit too. */
            return light | VGPU98_LOG_CAPTURE | VGPU98_LOG_AUDIT | VGPU98_LOG_AUDIT8 | VGPU98_LOG_TEXTURES;
        case VGPU98_DIAG_TEXTURES:
            return light | VGPU98_LOG_CAPTURE | VGPU98_LOG_TEXTURES;
        case VGPU98_DIAG_GLIDE:
            /* Minimal for the host and the DirectX driver; the Glide
               device's report, findings and captures, and the Glide DLLs'
               full lines. */
            return minimal | VGPU98_LOG_GLIDE;
        case VGPU98_DIAG_OPENGL:
            /* Minimal for the host, the DirectX driver and Glide; the
               OpenGL device's report, findings and captures, and the ICD's
               full lines. */
            return minimal | VGPU98_LOG_OPENGL;
        case VGPU98_DIAG_DISPLAY:
            /* Minimal for everything else; the screen's transitions, 86Box's
               screen sizes and the frames after them, captured. */
            return minimal | VGPU98_LOG_DISPLAY;
        default:
            return 0;
    }
}

/* --- Developer debug registers: SR E0h and BAR0's DEBUG --- */

static void
vgpu98_debug_flush(vgpu98_t *dev, char *line, int *len)
{
    if (*len > 0) {
        line[*len] = 0;
        if (dev->log_mask & VGPU98_LOG_DEBUG_REGISTER) {
            char stamp[24];

            vgpu98_log_stamp(stamp, sizeof(stamp));
            pclog("%s VGPU98 debug: %s\n", stamp, line);
        }
        *len = 0;
    }
}

static void
vgpu98_debug_put(vgpu98_t *dev, char *line, int *len, uint8_t val)
{
    if (val == '\n') {
        vgpu98_debug_flush(dev, line, len);
        return;
    }
    if (val == '\r')
        return;
    if ((val < 0x20) || (val > 0x7e))
        val = '?';
    line[(*len)++] = (char) val;
    if (*len >= (VGPU98_DEBUG_LINE - 1))
        vgpu98_debug_flush(dev, line, len);
}

static void
vgpu98_debug_write(vgpu98_t *dev, uint8_t val)
{
    if (dev->log_mask & VGPU98_LOG_DEBUG_REGISTER)
        vgpu98_debug_put(dev, dev->debug_line, &dev->debug_len, val);
}

/* --- Display engine (spec/display-engine.md) --- */

static const uint8_t vgpu98_edid[128] = { VGPU98_EDID_BYTES };

static uint32_t
disp_value(const vgpu98_t *dev, int sr, int bytes)
{
    uint32_t v = 0;
    int      i;

    for (i = bytes - 1; i >= 0; i--)
        v = (v << 8) | dev->disp_regs[sr - VGPU98_SR_DISP_FIRST + i];
    return v;
}

static int
disp_bpp(const vgpu98_t *dev)
{
    int bpp = DISP_REG(dev, VGPU98_SR_DISP_FORMAT);

    return bpp ? bpp : 8;
}

static uint32_t
disp_width(const vgpu98_t *dev)
{
    uint32_t w = disp_value(dev, VGPU98_SR_DISP_WIDTH, 2) & ~7u;

    if (w > VGPU98_DISP_MAX_WIDTH)
        w = VGPU98_DISP_MAX_WIDTH;
    return w ? w : 8;
}

static uint32_t
disp_height(const vgpu98_t *dev)
{
    uint32_t h = disp_value(dev, VGPU98_SR_DISP_HEIGHT, 2);

    if (h > VGPU98_DISP_MAX_HEIGHT)
        h = VGPU98_DISP_MAX_HEIGHT;
    return h ? h : 1;
}

static uint32_t
disp_pitch(const vgpu98_t *dev)
{
    uint32_t pitch = disp_value(dev, VGPU98_SR_DISP_PITCH, 2) & ~3u;

    return pitch ? pitch : disp_width(dev) * ((disp_bpp(dev) + 7) / 8);
}

static uint32_t
disp_refresh(const vgpu98_t *dev)
{
    uint32_t hz = DISP_REG(dev, VGPU98_SR_DISP_REFRESH);

    return ((hz >= VGPU98_DISP_REFRESH_MIN) && (hz <= VGPU98_DISP_REFRESH_MAX)) ? hz : 60;
}

static uint32_t
disp_bank_base(const vgpu98_t *dev)
{
    return (disp_value(dev, VGPU98_SR_DISP_BANK, 2) << 16) & (dev->vram_size - 1);
}

/* The svga core recomputes its fast write path only on SR04 and graphics
   controller writes, but it depends on fb_only, which the engine changes. */
static void
vgpu98_update_fast(svga_t *svga)
{
    svga->fast = (svga->gdcreg[8] == 0xff && !(svga->gdcreg[3] & 0x18) && !svga->gdcreg[1]) &&
                 ((svga->chain4 && (svga->packed_chain4 || svga->force_old_addr)) || svga->fb_only) &&
                 !(svga->adv_flags & FLAG_ADDR_BY8);
}

/* Double scan: one line of video memory drawn as two output pixels per
   pixel (the svga core repeats the line, linedbl). hdisp is the output width. */
static void
vgpu98_render_dscan(svga_t *svga)
{
    const vgpu98_t *dev  = (vgpu98_t *) svga->priv;
    const int       bpp  = disp_bpp(dev);
    const uint32_t  w    = svga->hdisp / 2;
    const uint32_t  mask = svga->vram_display_mask;
    const uint32_t  addr = svga->memaddr;
    const uint32_t  last = addr + w * ((bpp + 7) / 8) - 1;
    uint32_t       *p;
    uint32_t        x;

    if (((svga->displine + svga->y_add) < 0) || (svga->monitor->target_buffer == NULL) ||
        (svga->monitor->target_buffer->line[svga->displine + svga->y_add] == NULL))
        return;
    if (!svga->fullchange && !svga->changedvram[(addr & mask) >> 12] && !svga->changedvram[(last & mask) >> 12])
        return;

    p = &svga->monitor->target_buffer->line[svga->displine + svga->y_add][svga->x_add];
    if (svga->firstline_draw == 2000)
        svga->firstline_draw = svga->displine;
    svga->lastline_draw = svga->displine;

    for (x = 0; x < w; x++) {
        uint32_t c;
        uint32_t a;

        switch (bpp) {
            case 15:
            case 16:
                a = (addr + 2 * x) & mask;
                c = svga->conv_16to32(svga, svga->vram[a] | (svga->vram[(a + 1) & mask] << 8), bpp);
                break;
            case 32:
                a = (addr + 4 * x) & mask;
                c = svga_lookup_lut_ram(svga, svga->vram[a] | (svga->vram[(a + 1) & mask] << 8) |
                                                  (svga->vram[(a + 2) & mask] << 16));
                break;
            default:
                c = svga->map8[svga->vram[(addr + x) & mask] & svga->dac_mask];
                break;
        }
        p[2 * x]     = c;
        p[2 * x + 1] = c;
    }
}

/* The hardware cursor in 86Box's own line rebuild, while the display on the
   host GPU is not composing: the mode's renderer draws each line under the
   cursor whether or not its pages changed, then the cursor's rule is
   applied to the line's pixels as video memory holds them, with the
   renderer's own conversions. */
static void
vgpu98_render_cursor(svga_t *svga)
{
    vgpu98_t      *dev   = (vgpu98_t *) svga->priv;
    const uint32_t dscan = DISP_DSCAN(dev) ? 2 : 1;
    const int32_t  row   = svga->displine / (int32_t) dscan; /* the mode's row: double scan draws each twice */
    const uint32_t line  = svga->memaddr;                    /* the renderers move it along */
    const int      bpp   = disp_bpp(dev);
    const uint32_t bytes = (bpp + 7) / 8;
    const uint32_t mask  = svga->vram_display_mask;
    const uint32_t width = disp_width(dev);
    const int      full  = svga->fullchange;
    const uint8_t *and_row;
    const uint8_t *img;
    uint32_t      *p;
    int32_t        cx;

    if (!dev->cur_on || (dev->render_line == svga_render_blank) || (bytes != dev->cur_bytes) || (row < dev->cur_y) ||
        (row >= dev->cur_y + VGPU98_CURSOR_SIZE)) {
        dev->render_line(svga);
        return;
    }
    svga->fullchange = full ? full : 1;
    dev->render_line(svga);
    svga->fullchange = full;
    if (((svga->displine + svga->y_add) < 0) || (svga->monitor->target_buffer == NULL) ||
        (svga->monitor->target_buffer->line[svga->displine + svga->y_add] == NULL))
        return;

    p       = &svga->monitor->target_buffer->line[svga->displine + svga->y_add][svga->x_add];
    and_row = dev->cur_pattern + (row - dev->cur_y) * (VGPU98_CURSOR_SIZE / 8);
    img     = dev->cur_pattern + VGPU98_CURSOR_XOR + (uint32_t) (row - dev->cur_y) * VGPU98_CURSOR_SIZE * bytes;
    for (cx = 0; cx < VGPU98_CURSOR_SIZE; cx++) {
        const int32_t  x = dev->cur_x + cx;
        const uint8_t *q = img + (uint32_t) cx * bytes;
        uint32_t       a;
        uint32_t       v;
        uint32_t       xr;
        uint32_t       c;
        uint32_t       k;

        if ((x < 0) || ((uint32_t) x >= width))
            continue;
        a = line + (uint32_t) x * bytes;
        switch (bytes) {
            case 1:
                v  = svga->vram[a & mask];
                xr = q[0];
                break;
            case 2:
                v  = svga->vram[a & mask] | (svga->vram[(a + 1) & mask] << 8);
                xr = q[0] | (q[1] << 8);
                break;
            default:
                v  = svga->vram[a & mask] | (svga->vram[(a + 1) & mask] << 8) | (svga->vram[(a + 2) & mask] << 16);
                xr = q[0] | (q[1] << 8) | (q[2] << 16);
                break;
        }
        /* Windows' rule: (pixel AND mask) XOR image. */
        v = ((and_row[cx >> 3] >> (7 - (cx & 7))) & 1) ? (v ^ xr) : xr;
        switch (bpp) {
            case 15:
            case 16:
                c = svga->conv_16to32(svga, (uint16_t) v, bpp);
                break;
            case 32:
                c = svga_lookup_lut_ram(svga, v);
                break;
            default:
                c = svga->map8[v & svga->dac_mask];
                break;
        }
        for (k = 0; k < dscan; k++)
            p[(uint32_t) x * dscan + k] = c;
    }
}

/* The gamma table in 86Box's own line rebuild: after the
   mode's renderer and the cursor, each channel of the line's pixels
   through the table, on a line the renderer drew this time (it skips the
   lines whose pages did not change, which already have it). */
static void
vgpu98_render_gamma(svga_t *svga)
{
    const vgpu98_t *dev = (vgpu98_t *) svga->priv;
    uint32_t       *p;
    int             x;

    vgpu98_render_cursor(svga);
    if (!dev->gam_on || (dev->render_line == svga_render_blank) || (svga->lastline_draw != svga->displine) ||
        ((svga->displine + svga->y_add) < 0) || (svga->monitor->target_buffer == NULL) ||
        (svga->monitor->target_buffer->line[svga->displine + svga->y_add] == NULL))
        return;
    p = &svga->monitor->target_buffer->line[svga->displine + svga->y_add][svga->x_add];
    for (x = 0; x < svga->hdisp; x++) {
        const uint32_t c = p[x];

        p[x] = (c & 0xff000000u) | (dev->gam_table[(c >> 16) & 0xff] & 0xff0000u) |
               (dev->gam_table[(c >> 8) & 0xff] & 0xff00u) | (dev->gam_table[c & 0xff] & 0xffu);
    }
}

/* Called by svga_recalctimings: while the engine is on, it sets the frame. */
static void
vgpu98_recalctimings(svga_t *svga)
{
    vgpu98_t *dev = (vgpu98_t *) svga->priv;

    if (DISP_ENABLED(dev)) {
        /* Double scan shows every pixel and line twice: the output is twice the mode. */
        const uint32_t dscan  = DISP_DSCAN(dev) ? 2 : 1;
        const uint32_t w      = disp_width(dev) * dscan;
        const uint32_t h      = disp_height(dev) * dscan;
        uint32_t       vblank = h / VGPU98_DISP_V_DIV;
        uint32_t       htotal;
        uint32_t       vtotal;
        double         dotclock;

        if (vblank < VGPU98_DISP_V_BLANK)
            vblank = VGPU98_DISP_V_BLANK;
        htotal   = w + VGPU98_DISP_H_BLANK;
        vtotal   = h + vblank;
        dotclock = (double) htotal * (double) vtotal * (double) disp_refresh(dev);

        svga->char_width     = 1;
        svga->dots_per_clock = 1;
        svga->clock          = (cpuclock * (double) (1ULL << 32)) / dotclock;
        if (svga->seqregs[1] & 8)
            svga->clock /= 2.0; /* the core doubles the line time for SR01 bit 3; the engine ignores it */
        svga->hdisp          = w;
        svga->hdisp_time     = w;
        svga->htotal         = htotal;
        svga->hblankstart    = w;
        svga->hblankend      = htotal - 1;
        svga->dispend        = h;
        svga->vblankstart    = h;
        svga->vsyncstart     = h + 3;
        svga->vtotal         = vtotal;
        svga->rowcount       = 0;
        svga->linedbl        = (dscan == 2);
        svga->interlace      = 0;
        svga->lowres         = 0;
        svga->hoverride      = 1;
        svga->panning_blank  = 1;
        svga->fb_only        = 1;
        svga->adv_flags     |= FLAG_NO_SHIFT3;
        svga->split          = 0xffffff;
        svga->bpp            = disp_bpp(dev);
        svga->rowoffset      = disp_pitch(dev);
        svga->memaddr_latch  = dev->disp_start & (dev->vram_size - 1);
        svga->vram_display_mask = dev->vram_size - 1;

        if (dscan == 2)
            svga->render = vgpu98_render_dscan;
        else switch (svga->bpp) {
            case 15:
                svga->render = svga_render_15bpp_highres;
                break;
            case 16:
                svga->render = svga_render_16bpp_highres;
                break;
            case 32:
                svga->render = svga_render_32bpp_highres;
                break;
            default:
                svga->render = svga_render_8bpp_clone_highres;
                break;
        }
        if (svga->scrblank)
            svga->render = svga_render_blank; /* screen off (SR01 bit 5) still blanks */
        /* The hardware cursor and the gamma table go over whichever
           renderer the mode has. */
        dev->render_line = svga->render;
        svga->render     = vgpu98_render_gamma;
    } else {
        svga->fb_only           = 0;
        svga->adv_flags        &= ~FLAG_NO_SHIFT3;
        svga->hoverride         = 0;
        svga->panning_blank     = 0;
        svga->bpp               = 8;
        svga->vram_display_mask = 0x3ffff; /* 256 KB, as on a VGA */
    }
    vgpu98_update_fast(svga);
}

/* Marks the 4 KB page holding a written byte as changed, and the page before it:
   the renderers check only the first two pages of a line, and a 32 bpp line wider
   than 1024 pixels spans three. */
static void
vgpu98_mark_changed(svga_t *svga, uint32_t addr)
{
    svga->changedvram[addr >> 12] = svga->monitor->mon_changeframecount;
    if (addr >= 0x1000)
        svga->changedvram[(addr >> 12) - 1] = svga->monitor->mon_changeframecount;
}

/* The A0000h window: in extended mode a 64 KB bank of video memory as plain bytes. */
/* A byte of video memory for the guest's CPU: pixels the GPU holds newer
   are copied back first (spec/gpu-coherency.md). */
static void
vgpu98_vram_touch(const vgpu98_t *dev, uint32_t a)
{
    if (dev->newer[a >> 12])
        vgpu98_gpu_touch(dev->gpu, a);
    else
        atomic_thread_fence(memory_order_acquire); /* a copy back's bytes, before the count that let them go */
}

static uint8_t
vgpu98_vram_get(const vgpu98_t *dev, uint32_t a)
{
    vgpu98_vram_touch(dev, a);
    return dev->svga.vram[a];
}

static uint8_t
vgpu98_window_readb(uint32_t addr, void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_read_b;
    return vgpu98_vram_get(dev, (disp_bank_base(dev) + (addr & 0xffff)) & (dev->vram_size - 1));
}

static uint16_t
vgpu98_window_readw(uint32_t addr, void *priv)
{
    const vgpu98_t *dev  = (vgpu98_t *) priv;
    const uint32_t  base = disp_bank_base(dev);
    const uint32_t  mask = dev->vram_size - 1;

    cycles -= dev->svga.monitor->mon_video_timing_read_w;
    return vgpu98_vram_get(dev, (base + (addr & 0xffff)) & mask) |
           (vgpu98_vram_get(dev, (base + ((addr + 1) & 0xffff)) & mask) << 8);
}

static uint32_t
vgpu98_window_readl(uint32_t addr, void *priv)
{
    const vgpu98_t *dev  = (vgpu98_t *) priv;
    const uint32_t  base = disp_bank_base(dev);
    const uint32_t  mask = dev->vram_size - 1;
    uint32_t        v    = 0;
    int             i;

    cycles -= dev->svga.monitor->mon_video_timing_read_l;
    for (i = 3; i >= 0; i--)
        v = (v << 8) | vgpu98_vram_get(dev, (base + ((addr + i) & 0xffff)) & mask);
    return v;
}

static void
vgpu98_window_put(vgpu98_t *dev, uint32_t addr, uint8_t val)
{
    svga_t        *svga = &dev->svga;
    const uint32_t a    = (disp_bank_base(dev) + (addr & 0xffff)) & (dev->vram_size - 1);

    vgpu98_vram_touch(dev, a);
    vgpu98_mark_changed(svga, a);
    svga->vram[a] = val;
    /* The byte before its page's version: whoever sees the new version and
       then reads the page sees the byte (the display, the renderer). */
    atomic_thread_fence(memory_order_release);
    vgpu98_gpu_cpu_write(dev->cpu_pages, a);
}

static void
vgpu98_window_writeb(uint32_t addr, uint8_t val, void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_write_b;
    vgpu98_window_put(dev, addr, val);
}

static void
vgpu98_window_writew(uint32_t addr, uint16_t val, void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_write_w;
    vgpu98_window_put(dev, addr, val & 0xff);
    vgpu98_window_put(dev, addr + 1, val >> 8);
}

static void
vgpu98_window_writel(uint32_t addr, uint32_t val, void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;
    int       i;

    cycles -= dev->svga.monitor->mon_video_timing_write_l;
    for (i = 0; i < 4; i++)
        vgpu98_window_put(dev, addr + i, (val >> (8 * i)) & 0xff);
}

static void vgpu98_update_mem(vgpu98_t *dev);
static void vgpu98_cursor_reset(vgpu98_t *dev);
static void vgpu98_gamma_reset(vgpu98_t *dev);

/* After the card's own change of the timings (the display engine's mode,
   the engine on or off, a reset): a frame the change left with 86Box's line
   counter past the new picture's end while in its picture, or past the new
   total, is ended there, so the counter never runs on to 2047 and 86Box
   never shows a frame taller than the mode.
   VGA register writes are left to 86Box. */
static void
vgpu98_end_frame(vgpu98_t *dev)
{
    svga_t          *svga = &dev->svga;
    vgpu98_dd_scan_t s;
    vgpu98_dd_scan_t before;

    s.vc         = svga->vc;
    s.linepos    = svga->linepos;
    s.dispon     = svga->dispon;
    s.displine   = svga->displine;
    s.firstline  = svga->firstline;
    s.lastline   = svga->lastline;
    s.dispend    = svga->dispend;
    s.vsyncstart = svga->vsyncstart;
    s.vtotal     = svga->vtotal;
    before       = s;
    if (!vgpu98_dd_end_frame(&s))
        return;
    svga->vc        = s.vc;
    svga->dispon    = s.dispon;
    svga->firstline = s.firstline;
    svga->lastline  = s.lastline;
    if (dev->dd)
        vgpu98_dd_ended(dev->dd, &before, &s);
}

/* `was_on`: the engine was on before the write. */
static void
vgpu98_disp_changed(vgpu98_t *dev, int was_on)
{
    svga_t *svga = &dev->svga;

    vgpu98_update_mem(dev);
    if (!DISP_ENABLED(dev))
        svga->bpp = 8; /* the core picks the VGA renderer from bpp before calling us */
    svga->fullchange = svga->monitor->mon_changeframecount;
    svga_recalctimings(svga);
    /* The engine's change only: a CONTROL write that leaves it off (the VGA
       BIOS writes one at every mode set) leaves the VGA's frame to 86Box. */
    if (was_on || DISP_ENABLED(dev))
        vgpu98_end_frame(dev);
}

static void
vgpu98_disp_write(vgpu98_t *dev, uint8_t sr, uint8_t val)
{
    const int was_on = DISP_ENABLED(dev) ? 1 : 0;

    if (sr == VGPU98_SR_DISP_KEY) {
        dev->disp_unlocked = (val == VGPU98_DISP_KEY);
        return;
    }
    if (!dev->disp_unlocked)
        return;

    switch (sr) {
        case VGPU98_SR_DISP_FORMAT:
            if ((val != 8) && (val != 15) && (val != 16) && (val != 32))
                return; /* unsupported formats are ignored */
            DISP_REG(dev, sr) = val;
            break;
        case VGPU98_SR_DISP_CONTROL:
            if ((val & VGPU98_DISP_CTRL_ENABLE) && !DISP_ENABLED(dev))
                dev->disp_start = disp_value(dev, VGPU98_SR_DISP_START, 4) & ~3u;
            /* Off even for a moment: the VGA core may write video memory
               uncounted, so the display uploads it all again. */
            if (!(val & VGPU98_DISP_CTRL_ENABLE) && DISP_ENABLED(dev))
                dev->disp_restart = 1;
            DISP_REG(dev, sr) = val & (VGPU98_DISP_CTRL_ENABLE | VGPU98_DISP_CTRL_DSCAN);
            break;
        case VGPU98_SR_DISP_START:
        case VGPU98_SR_DISP_START + 1:
        case VGPU98_SR_DISP_START + 2:
            DISP_REG(dev, sr) = val;
            return; /* taken with the high byte */
        case VGPU98_SR_DISP_START + 3:
            DISP_REG(dev, sr) = val;
            dev->disp_start   = disp_value(dev, VGPU98_SR_DISP_START, 4) & ~3u;
            break;
        case VGPU98_SR_DISP_WIDTH:
        case VGPU98_SR_DISP_WIDTH + 1:
        case VGPU98_SR_DISP_HEIGHT:
        case VGPU98_SR_DISP_HEIGHT + 1:
        case VGPU98_SR_DISP_PITCH:
        case VGPU98_SR_DISP_PITCH + 1:
        case VGPU98_SR_DISP_REFRESH:
        case VGPU98_SR_DISP_BANK:
        case VGPU98_SR_DISP_BANK + 1:
            DISP_REG(dev, sr) = val;
            break;
        case VGPU98_SR_DISP_EDID_INDEX:
            dev->edid_index = val & 0x7f;
            return;
        case VGPU98_SR_DISP_SCRATCH:
        case VGPU98_SR_DISP_SCRATCH + 1:
            DISP_REG(dev, sr) = val;
            return; /* software's own storage: no effect on the display */
        case VGPU98_SR_DISP_GPU_RESET:
            /* The VGA BIOS at POST: whatever reset this was, a GPU the last
               boot left running writes no memory of this one. */
            if (val & VGPU98_DISP_GPU_RESET) {
                memset(dev->sysbuf, 0x00, sizeof(dev->sysbuf));
                vgpu98_cursor_reset(dev);
                vgpu98_gamma_reset(dev);
                vgpu98_gpu_power_on(dev->gpu);
            }
            return;
        default:
            return; /* read-only or reserved */
    }

    if ((sr == VGPU98_SR_DISP_BANK) || (sr == VGPU98_SR_DISP_BANK + 1))
        return; /* the window handlers read the bank directly */
    if ((sr == VGPU98_SR_DISP_CONTROL) || DISP_ENABLED(dev))
        vgpu98_disp_changed(dev, was_on);
}

static uint8_t
vgpu98_disp_read(vgpu98_t *dev, uint8_t sr)
{
    uint32_t v;

    if (sr == VGPU98_SR_DISP_KEY)
        return dev->disp_unlocked ? 0x01 : 0x00;
    if (!dev->disp_unlocked)
        return 0x00;

    switch (sr) {
        case VGPU98_SR_DISP_ID:
            return VGPU98_DISP_ID;
        case VGPU98_SR_DISP_REVISION:
            return VGPU98_DISP_REVISION;
        case VGPU98_SR_DISP_FEATURES:
            return VGPU98_DISP_FEATURES;
        case VGPU98_SR_DISP_FORMATS:
            return VGPU98_DISP_FORMATS;
        case VGPU98_SR_DISP_MAX_WIDTH:
        case VGPU98_SR_DISP_MAX_WIDTH + 1:
            return (VGPU98_DISP_MAX_WIDTH >> (8 * (sr - VGPU98_SR_DISP_MAX_WIDTH))) & 0xff;
        case VGPU98_SR_DISP_VRAM_SIZE:
        case VGPU98_SR_DISP_VRAM_SIZE + 1:
        case VGPU98_SR_DISP_VRAM_SIZE + 2:
        case VGPU98_SR_DISP_VRAM_SIZE + 3:
            return (dev->vram_size >> (8 * (sr - VGPU98_SR_DISP_VRAM_SIZE))) & 0xff;
        case VGPU98_SR_DISP_LFB_BASE:
        case VGPU98_SR_DISP_LFB_BASE + 1:
        case VGPU98_SR_DISP_LFB_BASE + 2:
        case VGPU98_SR_DISP_LFB_BASE + 3:
            v = ((dev->bar1 != 0) && (dev->bar1 != (uint32_t) ~(VGPU98_BAR1_SIZE - 1))) ? dev->bar1 : 0;
            return (v >> (8 * (sr - VGPU98_SR_DISP_LFB_BASE))) & 0xff;
        case VGPU98_SR_DISP_REGS_BASE:
        case VGPU98_SR_DISP_REGS_BASE + 1:
        case VGPU98_SR_DISP_REGS_BASE + 2:
        case VGPU98_SR_DISP_REGS_BASE + 3:
            v = ((dev->bar0 != 0) && (dev->bar0 != (uint32_t) ~(VGPU98_BAR0_SIZE - 1))) ? dev->bar0 : 0;
            return (v >> (8 * (sr - VGPU98_SR_DISP_REGS_BASE))) & 0xff;
        case VGPU98_SR_DISP_EDID_INDEX:
            return dev->edid_index;
        case VGPU98_SR_DISP_EDID_DATA:
            v               = vgpu98_edid[dev->edid_index];
            dev->edid_index = (dev->edid_index + 1) & 0x7f;
            return v;
        case VGPU98_SR_DISP_CONTROL:
        case VGPU98_SR_DISP_FORMAT:
        case VGPU98_SR_DISP_WIDTH:
        case VGPU98_SR_DISP_WIDTH + 1:
        case VGPU98_SR_DISP_HEIGHT:
        case VGPU98_SR_DISP_HEIGHT + 1:
        case VGPU98_SR_DISP_PITCH:
        case VGPU98_SR_DISP_PITCH + 1:
        case VGPU98_SR_DISP_START:
        case VGPU98_SR_DISP_START + 1:
        case VGPU98_SR_DISP_START + 2:
        case VGPU98_SR_DISP_START + 3:
        case VGPU98_SR_DISP_REFRESH:
        case VGPU98_SR_DISP_BANK:
        case VGPU98_SR_DISP_BANK + 1:
        case VGPU98_SR_DISP_SCRATCH:
        case VGPU98_SR_DISP_SCRATCH + 1:
            return DISP_REG(dev, sr);
        default:
            return 0x00;
    }
}

/* --- VGA ports --- */

/* The CRTC and status ports answer at 3Bxh or 3Dxh, whichever the miscellaneous
   output register selects; the other range is not decoded. Returns the port to
   handle (3Bxh folded onto 3Dxh), or 0 for a port the card ignores. */
static uint16_t
vgpu98_port(const svga_t *svga, uint16_t addr)
{
    if ((addr & 0xfff0) == 0x3b0)
        return (svga->miscout & 1) ? 0 : (addr ^ 0x60);
    if ((addr & 0xfff0) == 0x3d0)
        return (svga->miscout & 1) ? addr : 0;
    return addr;
}

static void
vgpu98_out_port(uint16_t addr, uint8_t val, void *priv)
{
    vgpu98_t *dev  = (vgpu98_t *) priv;
    svga_t   *svga = &dev->svga;
    uint8_t   old;

    if ((addr = vgpu98_port(svga, addr)) == 0)
        return;

    switch (addr) {
        case 0x3c5:
            if ((svga->seqaddr >= VGPU98_SR_DISP_FIRST) && (svga->seqaddr <= VGPU98_SR_DISP_LAST)) {
                vgpu98_disp_write(dev, svga->seqaddr, val);
                return;
            }
            if ((dev->log_mask & VGPU98_LOG_DEBUG_REGISTER) && (svga->seqaddr == VGPU98_DEBUG_SEQ_INDEX)) {
                vgpu98_debug_write(dev, val);
                return;
            }
            break;

        case 0x3d4:
            svga->crtcreg = val & 0x3f;
            return;

        case 0x3d5:
            if (svga->crtcreg & 0x20)
                return;
            if ((svga->crtcreg < 7) && (svga->crtc[0x11] & 0x80))
                return;
            if ((svga->crtcreg == 7) && (svga->crtc[0x11] & 0x80))
                val = (svga->crtc[7] & ~0x10) | (val & 0x10);
            old                       = svga->crtc[svga->crtcreg];
            svga->crtc[svga->crtcreg] = val;
            if (old != val) {
                if ((svga->crtcreg < 0xe) || (svga->crtcreg > 0x10)) {
                    if ((svga->crtcreg == 0xc) || (svga->crtcreg == 0xd)) {
                        if (!DISP_ENABLED(dev)) { /* the engine has its own start address */
                            svga->fullchange    = 3;
                            svga->memaddr_latch = ((svga->crtc[0xc] << 8) | svga->crtc[0xd]) + ((svga->crtc[8] & 0x60) >> 5);
                        }
                    } else {
                        svga->fullchange = svga->monitor->mon_changeframecount;
                        svga_recalctimings(svga);
                    }
                }
            }
            return;

        default:
            break;
    }

    svga_out(addr, val, svga);

    /* svga_out maps the A0000h window when GR06 changes; without memory decode, or while
       the display engine owns the window, it must stay off. */
    if ((addr == 0x3cf) && ((svga->gdcaddr & 0x0f) == 0x06) &&
        (!(dev->pci_command & PCI_COMMAND_MEM) || DISP_ENABLED(dev)))
        mem_mapping_disable(&svga->mapping);
}

/* Diagnostics "Display": a write that changed 86Box's vertical timings or
   the picture's width, with where its line counter stood. Any change of
   them comes from a port write (the VGA's registers or the display
   engine's), a machine reset or a speed change. */
static void
vgpu98_dd_check(vgpu98_t *dev)
{
    const svga_t    *svga = &dev->svga;
    vgpu98_dd_scan_t s;

    if ((svga->dispend == dev->dd_dispend) && (svga->vsyncstart == dev->dd_vsyncstart) &&
        (svga->vtotal == dev->dd_vtotal) && (svga->hdisp == dev->dd_hdisp))
        return;
    s.vc         = svga->vc;
    s.linepos    = svga->linepos;
    s.dispon     = svga->dispon;
    s.displine   = svga->displine;
    s.firstline  = svga->firstline;
    s.lastline   = svga->lastline;
    s.dispend    = svga->dispend;
    s.vsyncstart = svga->vsyncstart;
    s.vtotal     = svga->vtotal;
    vgpu98_dd_timing(dev->dd, &s, dev->dd_vtotal);
    dev->dd_dispend    = svga->dispend;
    dev->dd_vsyncstart = svga->vsyncstart;
    dev->dd_vtotal     = svga->vtotal;
    dev->dd_hdisp      = svga->hdisp;
}

static void
vgpu98_out(uint16_t addr, uint8_t val, void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;

    vgpu98_out_port(addr, val, priv);
    if (dev->dd) {
        if (addr == 0x3c9)
            vgpu98_dd_palette(dev->dd);
        vgpu98_dd_check(dev);
    }
}

static uint8_t
vgpu98_in(uint16_t addr, void *priv)
{
    vgpu98_t     *dev  = (vgpu98_t *) priv;
    svga_t       *svga = &dev->svga;
    uint8_t       ret;

    if ((addr = vgpu98_port(svga, addr)) == 0)
        return 0xff;

    switch (addr) {
        case 0x3c5:
            /* The debug register reads 01h ("ready") while it is enabled and 00h while it is off. */
            if ((svga->seqaddr >= VGPU98_SR_DISP_FIRST) && (svga->seqaddr <= VGPU98_SR_DISP_LAST))
                ret = vgpu98_disp_read(dev, svga->seqaddr);
            else if (svga->seqaddr == VGPU98_DEBUG_SEQ_INDEX)
                ret = (dev->log_mask & VGPU98_LOG_DEBUG_REGISTER) ? 0x01 : 0x00;
            else
                ret = svga_in(addr, svga);
            break;

        case 0x3d4:
            ret = svga->crtcreg;
            break;

        case 0x3d5:
            ret = 0xff;
            switch (svga->crtcreg) {
                case 0x00 ... 0x18:
                    ret = svga->crtc[svga->crtcreg];
                    break;
                case 0x22:
                    ret = svga->latch.b[svga->gdcreg[0x04] & 0x03];
                    break;
                case 0x24:
                    ret = (svga->attrff & 0x01) | ((svga->attraddr & 0x1f) << 3);
                    break;
                default:
                    break;
            }
            break;

        default:
            ret = svga_in(addr, svga);
            break;
    }

    return ret;
}

/* --- BAR0: register window (spec/registers.md, spec/gpu-registers.md) --- */

static uint32_t
vgpu98_mmio_reg(const vgpu98_t *dev, uint32_t addr)
{
    const svga_t  *svga = &dev->svga;
    const uint32_t reg  = addr & (VGPU98_BAR0_SIZE - 4);
    uint32_t       v;

    switch (reg) {
        case VGPU98_REG_ID:
            return VGPU98_ID_MAGIC;
        case VGPU98_REG_HW_VERSION:
            return VGPU98_HW_VERSION;
        case VGPU98_REG_PROFILE:
            return vgpu98_gpu_profile(dev->gpu);
        case VGPU98_REG_VRAM_SIZE:
            return dev->vram_size;
        case VGPU98_REG_DEBUG:
            /* While the host's log listens, so the DirectX driver can skip
               its diagnostics; and with the vertex audit and the texture
               audit. At Minimal the log takes its event lines alone
               (EVENTS, not LISTENING); at Glide and OpenGL too, the Glide
               DLLs or the ICD with their full lines (GLIDE, OPENGL). */
            {
                const int listening = (dev->log_mask & VGPU98_LOG_DEBUG_REGISTER) && !(dev->log_mask & VGPU98_LOG_MINIMAL);

                return (listening ? VGPU98_DEBUG_LISTENING : 0) | ((dev->log_mask & VGPU98_LOG_MINIMAL) ? VGPU98_DEBUG_EVENTS : 0) |
                       ((dev->log_mask & VGPU98_LOG_AUDIT) ? VGPU98_DEBUG_AUDIT : 0) |
                       ((dev->log_mask & VGPU98_LOG_AUDIT8) ? VGPU98_DEBUG_AUDIT8 : 0) |
                       (((dev->log_mask & VGPU98_LOG_TEXTURES) && listening) ? VGPU98_DEBUG_TEXTURES : 0) |
                       ((dev->shader_capture && listening) ? VGPU98_DEBUG_SHADERS : 0) |
                       (((dev->shader_capture == 2) && listening) ? VGPU98_DEBUG_SHADERS14 : 0) |
                       ((dev->log_mask & VGPU98_LOG_GLIDE) ? VGPU98_DEBUG_GLIDE : 0) |
                       ((dev->log_mask & VGPU98_LOG_OPENGL) ? VGPU98_DEBUG_OPENGL : 0);
            }
        case VGPU98_REG_SYSBUF_PHYS:
        case VGPU98_REG_SYSBUF_LINEAR:
        case VGPU98_REG_SYSBUF_SIZE:
        case VGPU98_REG_SYSBUF_STATUS:
            return dev->sysbuf[(reg - VGPU98_REG_SYSBUF_PHYS) >> 2];
        case VGPU98_REG_STATUS_BASE:
        case VGPU98_REG_GPUVM_BASE:
        case VGPU98_REG_GPUVM_ENTRIES:
            return vgpu98_gpu_read(dev->gpu, reg);
        case VGPU98_REG_SCANLINE:
            /* The line being scanned out, counting on through the blanking lines. */
            v = svga->vc & 0xffff;
            if (svga->vc >= svga->dispend)
                v |= VGPU98_SCANLINE_VBLANK;
            return v;
        /* The hardware cursor's, as written; 0 without PROFILE.CURSOR. */
        case VGPU98_REG_CURSOR_CONTROL:
            return dev->cursor_hw ? dev->cur_control : 0;
        case VGPU98_REG_CURSOR_POS:
            return dev->cursor_hw ? dev->cur_pos : 0;
        case VGPU98_REG_CURSOR_BASE:
            return dev->cursor_hw ? dev->cur_base : 0;
        /* The gamma table's, as written; 0 without PROFILE.GAMMA. */
        case VGPU98_REG_GAMMA_CONTROL:
            return dev->gamma_hw ? dev->gam_control : 0;
        case VGPU98_REG_GAMMA_BASE:
            return dev->gamma_hw ? dev->gam_base : 0;
        /* The V-sync overrides, as written; 0 without PROFILE.VSYNC. */
        case VGPU98_REG_VSYNC_CONTROL:
            return dev->vsync_hw ? vgpu98_gpu_vsync(dev->gpu) : 0;
        default:
            return (reg < VGPU98_REG_DEBUG) ? vgpu98_gpu_read(dev->gpu, reg) : 0x00000000;
    }
}

static uint8_t
vgpu98_mmio_byte(const vgpu98_t *dev, uint32_t addr)
{
    return (uint8_t) (vgpu98_mmio_reg(dev, addr) >> ((addr & 3) << 3));
}

static uint8_t
vgpu98_mmio_readb(uint32_t addr, void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_read_b;
    return vgpu98_mmio_byte(dev, addr);
}

static uint16_t
vgpu98_mmio_readw(uint32_t addr, void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_read_w;
    if (!(addr & 1) && ((addr & 3) != 3))
        return (uint16_t) (vgpu98_mmio_reg(dev, addr) >> ((addr & 2) << 3));
    return vgpu98_mmio_byte(dev, addr) | (vgpu98_mmio_byte(dev, addr + 1) << 8);
}

static uint32_t
vgpu98_mmio_readl(uint32_t addr, void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_read_l;
    if (!(addr & 3))
        return vgpu98_mmio_reg(dev, addr);
    return vgpu98_mmio_byte(dev, addr) | (vgpu98_mmio_byte(dev, addr + 1) << 8) |
           (vgpu98_mmio_byte(dev, addr + 2) << 16) | ((uint32_t) vgpu98_mmio_byte(dev, addr + 3) << 24);
}

/* The V-sync overrides' new value, in the log. */
static void
vgpu98_vsync_log(const vgpu98_t *dev, uint32_t val)
{
    static const char *const modes[4] = { "application's choice", "always on", "always off", "application's choice" };
    char                     stamp[24];

    if (!(dev->log_mask & VGPU98_LOG_EVENTS))
        return;
    vgpu98_log_stamp(stamp, sizeof(stamp));
    pclog("%s VGPU98: V-sync overrides: Direct3D %s, OpenGL %s, Glide %s\n", stamp,
          modes[(val >> VGPU98_VSYNC_D3D_SHIFT) & VGPU98_VSYNC_FIELD],
          modes[(val >> VGPU98_VSYNC_OGL_SHIFT) & VGPU98_VSYNC_FIELD],
          modes[(val >> VGPU98_VSYNC_GLIDE_SHIFT) & VGPU98_VSYNC_FIELD]);
}

/* Registers take aligned 32-bit writes only; byte and word writes are ignored. */
static void
vgpu98_mmio_writeb(UNUSED(uint32_t addr), UNUSED(uint8_t val), void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_write_b;
}

static void
vgpu98_mmio_writew(UNUSED(uint32_t addr), UNUSED(uint16_t val), void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_write_w;
}

static void
vgpu98_mmio_writel(uint32_t addr, uint32_t val, void *priv)
{
    vgpu98_t      *dev = (vgpu98_t *) priv;
    const uint32_t reg = addr & (VGPU98_BAR0_SIZE - 1);

    cycles -= dev->svga.monitor->mon_video_timing_write_l;
    if (reg & 3)
        return;
    if (reg == VGPU98_REG_DEBUG) {
        /* The DirectX driver caught a fault (its line just before): a
           capture, as a GPU error starts one. Its low byte, a carriage
           return, is what an older device sees. */
        if (val == VGPU98_DEBUG_IDLE) {
            /* Its last Direct3D context went: the watchdog's drawing ended. */
            if (dev->gpu)
                vgpu98_gpu_idle(dev->gpu);
        } else if (val == VGPU98_DEBUG_CAPTURE) {
            if (dev->gpu)
                vgpu98_gpu_trigger(dev->gpu, "the DirectX driver caught a fault (its line above)");
        } else if (dev->log_mask & VGPU98_LOG_DEBUG_REGISTER)
            vgpu98_debug_put(dev, dev->dll_line, &dev->dll_len, val & 0xff);
    } else if ((reg >= VGPU98_REG_SYSBUF_PHYS) && (reg <= VGPU98_REG_SYSBUF_STATUS))
        dev->sysbuf[(reg - VGPU98_REG_SYSBUF_PHYS) >> 2] = val;
    else if ((reg >= VGPU98_REG_CURSOR_CONTROL) && (reg <= VGPU98_REG_CURSOR_BASE)) {
        /* Taken at the next vertical blank (vgpu98_cursor_vblank). */
        if (!dev->cursor_hw)
            return;
        if (reg == VGPU98_REG_CURSOR_CONTROL) {
            dev->cur_control = val & VGPU98_CURSOR_ENABLE;
            dev->cur_reload  = 1;
        } else if (reg == VGPU98_REG_CURSOR_POS)
            dev->cur_pos = val;
        else {
            dev->cur_base   = val & ~(VGPU98_CURSOR_ALIGN - 1);
            dev->cur_reload = 1;
        }
    } else if ((reg == VGPU98_REG_GAMMA_CONTROL) || (reg == VGPU98_REG_GAMMA_BASE)) {
        /* Taken at the next vertical blank, the table read then
           (vgpu98_gamma_vblank). */
        if (!dev->gamma_hw)
            return;
        if (reg == VGPU98_REG_GAMMA_CONTROL)
            dev->gam_control = val & VGPU98_GAMMA_ENABLE;
        else
            dev->gam_base = val & ~(VGPU98_GAMMA_ALIGN - 1);
        dev->gam_reload = 1;
    } else if (reg == VGPU98_REG_VSYNC_CONTROL) {
        /* Taken at the next flip or swap. */
        if (!dev->vsync_hw)
            return;
        val &= (VGPU98_VSYNC_FIELD << VGPU98_VSYNC_D3D_SHIFT) | (VGPU98_VSYNC_FIELD << VGPU98_VSYNC_OGL_SHIFT) |
               (VGPU98_VSYNC_FIELD << VGPU98_VSYNC_GLIDE_SHIFT);
        if (val != vgpu98_gpu_vsync(dev->gpu))
            vgpu98_vsync_log(dev, val);
        vgpu98_gpu_set_vsync(dev->gpu, val);
    } else if (((reg >= VGPU98_REG_COMMAND_SET) && (reg < VGPU98_REG_DEBUG)) || (reg == VGPU98_REG_STATUS_BASE) ||
             (reg == VGPU98_REG_GPUVM_BASE) || (reg == VGPU98_REG_GPUVM_ENTRIES))
        vgpu98_gpu_write(dev->gpu, reg, val);
}

/* --- The GPU's view of the display, and vertical blank --- */

static void
vgpu98_gpu_display(void *priv, int *enabled, uint32_t *start, uint32_t *pitch, uint32_t *height, uint32_t *bpp)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    *enabled = DISP_ENABLED(dev) ? 1 : 0;
    *start   = dev->disp_start & (dev->vram_size - 1); /* as the display shows it */
    *pitch   = disp_pitch(dev);
    *height  = disp_height(dev);
    *bpp     = (uint32_t) disp_bpp(dev);
}

/* --- The hardware cursor --- */

/* Its registers to their power-on state: a machine reset and POST's
   GPU_RESET (not the GPU's own RESET). It goes at the next vertical blank. */
static void
vgpu98_cursor_reset(vgpu98_t *dev)
{
    dev->cur_control = 0;
    dev->cur_pos     = 0;
    dev->cur_base    = 0;
    dev->cur_reload  = 1;
}

/* 86Box's line rebuild draws the mode's rows the cursor covers from row y0
   again: their pages marked changed. */
static void
vgpu98_cursor_mark(vgpu98_t *dev, int32_t y0)
{
    svga_t        *svga   = &dev->svga;
    const uint32_t bytes  = (disp_bpp(dev) + 7) / 8;
    const uint32_t start  = dev->disp_start & (dev->vram_size - 1);
    const uint32_t pitch  = disp_pitch(dev);
    const uint32_t len    = disp_width(dev) * bytes;
    const int32_t  height = (int32_t) disp_height(dev);
    const uint32_t pages  = dev->vram_size >> 12;
    int32_t        y;

    for (y = (y0 < 0) ? 0 : y0; (y < y0 + VGPU98_CURSOR_SIZE) && (y < height); y++) {
        const uint32_t a = start + (uint32_t) y * pitch;
        uint32_t       p;

        for (p = a >> 12; p <= ((a + len - 1) >> 12); p++)
            svga->changedvram[p % pages] = svga->monitor->mon_changeframecount;
    }
}

/* The start of vertical blank: the cursor's registers taken, and its
   pattern read from video memory when CURSOR_BASE or CURSOR_CONTROL was
   written or the mode's depth changed. Shown while the display engine is
   on, but not over a Glide session's screen, as a Voodoo3's Glide turns its
   cursor off. */
static void
vgpu98_cursor_vblank(vgpu98_t *dev)
{
    const uint32_t bytes = (disp_bpp(dev) + 7) / 8;
    const uint32_t size  = VGPU98_CURSOR_XOR + VGPU98_CURSOR_SIZE * VGPU98_CURSOR_SIZE * bytes;
    const int      on    = dev->cursor_hw && (dev->cur_control & VGPU98_CURSOR_ENABLE) && DISP_ENABLED(dev) &&
                           !vgpu98_gpu_glide_screen_on(dev->gpu) && (dev->cur_base <= dev->vram_size - size);
    const int32_t  x     = (int16_t) (dev->cur_pos & 0xffff);
    const int32_t  y     = (int16_t) (dev->cur_pos >> 16);
    int            read  = 0;

    if (on && (dev->cur_reload || !dev->cur_on || (bytes != dev->cur_bytes))) {
        memcpy(dev->cur_pattern, dev->svga.vram + dev->cur_base, size);
        dev->cur_bytes = bytes;
        dev->cur_serial++;
        read = 1;
    }
    dev->cur_reload = 0;
    if ((on != dev->cur_on) || (on && (read || (x != dev->cur_x) || (y != dev->cur_y)))) {
        if (dev->cur_on)
            vgpu98_cursor_mark(dev, dev->cur_y);
        if (on)
            vgpu98_cursor_mark(dev, y);
    }
    dev->cur_on = on;
    dev->cur_x  = x;
    dev->cur_y  = y;
}

/* --- The gamma table --- */

/* Its registers to their power-on state, as the cursor's. */
static void
vgpu98_gamma_reset(vgpu98_t *dev)
{
    dev->gam_control = 0;
    dev->gam_base    = 0;
    dev->gam_reload  = 1;
}

/* The log's line for the table taken, at most one each 5 seconds: a game
   fading through gamma every frame gives a line, with a count. */
static void
vgpu98_gamma_log(vgpu98_t *dev, int force)
{
    const uint32_t now = plat_get_ticks();
    char           stamp[24];

    if (!(dev->log_mask & VGPU98_LOG_EVENTS) || !dev->gam_unlogged ||
        (!force && dev->gam_logged_at && (now - dev->gam_logged_at < 5000)))
        return;
    vgpu98_log_stamp(stamp, sizeof(stamp));
    if (dev->gam_on)
        pclog("%s VGPU98: gamma table on: red, green, blue at 64 %u/%u/%u, at 128 %u/%u/%u, at 192 %u/%u/%u%s\n", stamp,
              (dev->gam_table[64] >> 16) & 0xff, (dev->gam_table[64] >> 8) & 0xff, dev->gam_table[64] & 0xff,
              (dev->gam_table[128] >> 16) & 0xff, (dev->gam_table[128] >> 8) & 0xff, dev->gam_table[128] & 0xff,
              (dev->gam_table[192] >> 16) & 0xff, (dev->gam_table[192] >> 8) & 0xff, dev->gam_table[192] & 0xff,
              (dev->gam_unlogged > 1) ? " (the latest of several)" : "");
    else
        pclog("%s VGPU98: gamma table off\n", stamp);
    dev->gam_logged_at = now ? now : 1;
    dev->gam_unlogged  = 0;
}

/* The start of vertical blank: the gamma table's registers taken, and the
   table read from video memory when GAMMA_BASE or GAMMA_CONTROL was
   written or it is turned on again. Applied while the display engine is
   on in a direct colour mode (at 8 bpp the DAC is the palette), and not
   over a Glide session's screen, which has its own. */
static void
vgpu98_gamma_vblank(vgpu98_t *dev)
{
    const uint32_t size = VGPU98_GAMMA_ENTRIES * 4;
    const int      on   = dev->gamma_hw && (dev->gam_control & VGPU98_GAMMA_ENABLE) && DISP_ENABLED(dev) &&
                          (disp_bpp(dev) != 8) && (dev->gam_base <= dev->vram_size - size);
    const int      read = on && (dev->gam_reload || !dev->gam_on);
    uint32_t       i;

    if (read) {
        for (i = 0; i < VGPU98_GAMMA_ENTRIES; i++) {
            const uint8_t *e = &dev->svga.vram[dev->gam_base + i * 4];

            dev->gam_table[i] = e[0] | ((uint32_t) e[1] << 8) | ((uint32_t) e[2] << 16);
        }
        dev->gam_serial++;
    }
    dev->gam_reload = 0;
    if (read || (on != dev->gam_on)) {
        /* Every pixel changes: 86Box's line rebuild draws them all again. */
        dev->svga.fullchange = dev->svga.monitor->mon_changeframecount;
        dev->gam_on          = on;
        dev->gam_unlogged++;
    }
    vgpu98_gamma_log(dev, 0);
}

/* What the host display last put in the monitor's frame buffer (a picture
   or a black frame) is of frame f's mode and place, or is now. */
static int
vgpu98_fb_holds(const vgpu98_t *dev, const vgpu98_frame_t *f)
{
    return dev->fb_held && (dev->fb_width == f->width) && (dev->fb_height == f->height) && (dev->fb_bpp == f->bpp) &&
           (dev->fb_pitch == f->pitch) && (dev->fb_dscan == f->dscan) && (dev->fb_x == f->x) && (dev->fb_y == f->y);
}

static void
vgpu98_fb_hold(vgpu98_t *dev, const vgpu98_frame_t *f)
{
    dev->fb_held   = 1;
    dev->fb_width  = f->width;
    dev->fb_height = f->height;
    dev->fb_bpp    = f->bpp;
    dev->fb_pitch  = f->pitch;
    dev->fb_dscan  = f->dscan;
    dev->fb_x      = f->x;
    dev->fb_y      = f->y;
}

/* The frame's place in the monitor's frame buffer, black. */
static void
vgpu98_blank_frame(vgpu98_t *dev, const vgpu98_frame_t *f)
{
    bitmap_t      *tb = dev->svga.monitor->target_buffer;
    const uint32_t w  = f->width * f->dscan;
    const uint32_t h  = f->height * f->dscan;
    uint32_t       y;

    if (!tb || (f->x < 0) || (f->y < 0) || ((uint32_t) f->x + w > (uint32_t) tb->w) ||
        ((uint32_t) f->y + h > (uint32_t) tb->h))
        return;
    for (y = 0; y < h; y++)
        memset(&tb->line[f->y + (int) y][f->x], 0, (size_t) w * 4);
    vgpu98_fb_hold(dev, f);
}

/* The display on the host's GPU: while the display engine is on, the VGA
   core's host_display flag stops 86Box's line-by-line rebuild, and the
   frame goes to the display thread, which composes it on the host's GPU.
   Here, at the start of every vertical blank, the rows it composed since go
   into the monitor's frame buffer, which the VGA core then blits (sized,
   bordered and timed as always), and the next frame is posted: one frame
   from the vertical blank that loads START to its blit, as before. A
   pass-through card that takes the screen (a Voodoo sets the core's
   override) has it to itself: the display steps aside. */
static void
vgpu98_host_display(vgpu98_t *dev)
{
    svga_t         *svga = &dev->svga;
    monitor_t      *m    = svga->monitor;
    vgpu98_frame_t *f    = &dev->frame;
    const uint32_t  dscan = DISP_DSCAN(dev) ? 2 : 1;
    /* The frame where the VGA core would draw it, inside its 2048x2048
       frame buffer. */
    const int       want  = vgpu98_disp_ok(dev->disp) && DISP_ENABLED(dev) && !svga->dpms && !svga->override &&
                     (svga->x_add >= 0) && (svga->y_add >= 0) &&
                     ((uint32_t) svga->x_add + disp_width(dev) * dscan <= 2048) &&
                     ((uint32_t) svga->y_add + disp_height(dev) * dscan <= 2048);
    int             i;

    if (!want) {
        if (dev->host_display) {
            vgpu98_disp_stop(dev->disp);
            dev->host_display  = 0;
            dev->fb_held       = 0; /* 86Box draws the frame buffer again */
            svga->host_display = 0;
            svga->fullchange   = svga->monitor->mon_changeframecount;
            /* What the GPU drew for the screen was in the display's layer
               only: 86Box's own display reads video memory. */
            vgpu98_gpu_sync_visible_request(dev->gpu);
        }
        return;
    }
    f->start  = dev->disp_start & (dev->vram_size - 1);
    f->pitch  = disp_pitch(dev);
    f->width  = disp_width(dev);
    f->height = disp_height(dev);
    f->bpp    = (uint32_t) disp_bpp(dev);
    f->dscan  = DISP_DSCAN(dev) ? 2 : 1;
    f->blank  = svga->scrblank ? 1 : 0;
    for (i = 0; i < 256; i++)
        f->palette[i] = svga->map8[i & svga->dac_mask];

    /* Where the VGA core would draw it: its renderers write the first pixel
       of a line at x_add (the left overscan less the panning) and y_add. */
    f->x = svga->x_add;
    f->y = svga->y_add;
    f->cur_on       = dev->cur_on;
    f->cur_x        = dev->cur_x;
    f->cur_y        = dev->cur_y;
    f->cur_serial   = dev->cur_serial;
    f->gamma_on     = dev->gam_on;
    f->gamma_serial = dev->gam_serial;

    /* The VGA core waits for its last blit before it draws a frame's first
       line; a frame the card ended drew none, so the frame buffer is not
       written before that blit is done. */
    video_wait_for_buffer_monitor(svga->monitor_index);

    /* A new mode's first frame is black until the host GPU's first picture
       of it, never the mode before's:
       when the display starts, and whenever the frame buffer
       holds no picture of the mode now shown, a picture of another mode
       being held back. */
    if (!dev->host_display || dev->disp_restart) {
        dev->host_display  = 1;
        dev->disp_restart  = 0;
        svga->host_display = 1;
        vgpu98_disp_start(dev->disp);
        vgpu98_blank_frame(dev, f);
        if (dev->dd)
            vgpu98_dd_vblank(dev->dd, 1, 1, 0, NULL);
    } else {
        vgpu98_frame_t taken;
        const int      took    = vgpu98_disp_take(dev->disp, m->target_buffer, f, dev->dd ? &taken : NULL);
        int            blanked = 0;

        if (took > 0)
            vgpu98_fb_hold(dev, f);
        else if (!vgpu98_fb_holds(dev, f)) {
            vgpu98_blank_frame(dev, f);
            blanked = 1;
        }
        /* Diagnostics "Display": which frame's rows this blank's blit shows. */
        if (dev->dd)
            vgpu98_dd_vblank(dev->dd, 0, blanked, took, &taken);
    }
    vgpu98_disp_frame(dev->disp, f, dev->cur_pattern, dev->gam_table);
}

/* With captures on, a screenshot (Action > Take screenshot, raw or to the
   clipboard as well) also takes one: what the GPU did around the picture.
   The monitor counts the screenshots asked for, so a new one is seen
   however soon the renderer takes it: when it is asked for
   (vgpu98_force_redraw) or at the next vertical blank. */
static void
vgpu98_screenshot_capture(vgpu98_t *dev)
{
    const int asked = atomic_load(&dev->svga.monitor->mon_screenshot_requests);

    if (atomic_exchange(&dev->shot_seen, asked) == asked)
        return;
    if (dev->log_mask & VGPU98_LOG_CAPTURE)
        vgpu98_gpu_screenshot(dev->gpu);
    /* At Diagnostics "Glide", the Glide device's frame on the screen too. */
    if (dev->log_mask & VGPU98_LOG_GLIDE)
        vgpu98_gpu_glide_screenshot(dev->gpu);
    /* At "OpenGL", the OpenGL device's. */
    if (dev->log_mask & VGPU98_LOG_OPENGL)
        vgpu98_gpu_ogl_screenshot(dev->gpu);
    /* At "Display", the frames from the next vertical sync. */
    if (dev->dd)
        vgpu98_dd_screenshot(dev->dd);
}

/* --- Diagnostics "Display" --- */

/* The screen as the card shows it. */
static void
vgpu98_dd_state(const vgpu98_t *dev, vgpu98_dd_state_t *s)
{
    const svga_t *svga = &dev->svga;

    memset(s, 0, sizeof(*s));
    s->engine = DISP_ENABLED(dev) ? 1 : 0;
    if (s->engine) {
        s->width   = disp_width(dev);
        s->height  = disp_height(dev);
        s->bpp     = (uint32_t) disp_bpp(dev);
        s->pitch   = disp_pitch(dev);
        s->dscan   = DISP_DSCAN(dev) ? 2 : 1;
        s->refresh = disp_refresh(dev);
    } else {
        s->vga_width = svga->hdisp;
        s->vga_lines = svga->dispend;
        s->vga_text  = !(svga->gdcreg[6] & 1) && !(svga->attrregs[0x10] & 1);
    }
    s->blank    = svga->scrblank ? 1 : 0;
    s->dpms     = svga->dpms ? 1 : 0;
    s->override = svga->override ? 1 : 0;
    s->glide    = (dev->gpu && vgpu98_gpu_glide_screen_on(dev->gpu)) ? 1 : 0;
    /* As vgpu98_host_display decides it. */
    if (dev->host_display)
        s->host = VGPU98_DD_HOST_ON;
    else if (!vgpu98_disp_ok(dev->disp))
        s->host = VGPU98_DD_HOST_NONE;
    else if (!s->engine)
        s->host = VGPU98_DD_HOST_ENGINE;
    else if (s->dpms)
        s->host = VGPU98_DD_HOST_DPMS;
    else if (s->override)
        s->host = VGPU98_DD_HOST_OVERRIDE;
    else
        s->host = VGPU98_DD_HOST_SIZE;
}

/* The vertical sync, just after the VGA core's blit of the frame. */
static void
vgpu98_vsync(svga_t *svga)
{
    vgpu98_t         *dev = (vgpu98_t *) svga->priv;
    vgpu98_dd_frame_t f;
    uint32_t          palette[256];
    int               i;

    if (!dev->dd)
        return;
    vgpu98_dd_state(dev, &f.state);
    /* svga_doblit shows nothing for a pass-through card's screen, or for a
       frame of one line or none. */
    f.shown    = !svga->override && (svga->hdisp > 0) && (svga->vdisp >= 2);
    f.lines    = svga->override ? 0 : svga->vdisp;
    f.rows     = svga->dispend;
    f.screen_w = svga->monitor->mon_xsize;
    f.screen_h = svga->monitor->mon_ysize;
    /* Where the blit starts without overscan. */
    f.tb        = svga->monitor->target_buffer;
    f.x         = svga->left_overscan;
    f.y         = svga->y_add;
    f.vram      = svga->vram;
    f.vram_mask = dev->vram_size - 1;
    f.start     = dev->disp_start & (dev->vram_size - 1);
    for (i = 0; i < 256; i++)
        palette[i] = svga->map8[i & svga->dac_mask];
    f.palette = palette;
    vgpu98_dd_vsync(dev->dd, &f);
}

/* The start of vertical blank: a flip takes effect (START loads the new
   frame, shown from the next retrace), pages the GPU wrote are redrawn, and
   the frame goes to the display on the host's GPU. */
static void
vgpu98_vblank_start(svga_t *svga)
{
    vgpu98_t *dev = (vgpu98_t *) svga->priv;
    uint32_t  start;
    uint32_t  first;
    uint32_t  count;
    int       i;

    if (vgpu98_gpu_vblank(dev->gpu, &start)) {
        for (i = 0; i < 4; i++)
            DISP_REG(dev, VGPU98_SR_DISP_START + i) = (start >> (8 * i)) & 0xff;
        dev->disp_start = start & ~3u;
        if (DISP_ENABLED(dev)) {
            svga->memaddr_latch = dev->disp_start & (dev->vram_size - 1);
            svga->fullchange    = svga->monitor->mon_changeframecount;
        }
    }
    if (vgpu98_gpu_take_dirty(dev->gpu, &first, &count)) {
        uint32_t p;

        for (p = (first ? first - 1 : 0); (p < first + count) && (p < (dev->vram_size >> 12)); p++)
            svga->changedvram[p] = svga->monitor->mon_changeframecount;
    }
    vgpu98_cursor_vblank(dev);
    vgpu98_gamma_vblank(dev);
    if (dev->disp)
        vgpu98_host_display(dev);
    vgpu98_screenshot_capture(dev);
}

/* --- BAR1: framebuffer, a direct window onto VRAM --- */

static uint8_t
vgpu98_lfb_get(const vgpu98_t *dev, uint32_t addr)
{
    addr &= VGPU98_BAR1_SIZE - 1;
    if (addr >= dev->vram_size)
        return 0xff;
    return vgpu98_vram_get(dev, addr);
}

static void
vgpu98_lfb_put(vgpu98_t *dev, uint32_t addr, uint8_t val)
{
    svga_t *svga = &dev->svga;

    addr &= VGPU98_BAR1_SIZE - 1;
    if (addr >= dev->vram_size)
        return;
    vgpu98_vram_touch(dev, addr);
    vgpu98_mark_changed(svga, addr);
    svga->vram[addr] = val;
    atomic_thread_fence(memory_order_release); /* the byte before its page's version */
    vgpu98_gpu_cpu_write(dev->cpu_pages, addr);
}

static uint8_t
vgpu98_lfb_readb(uint32_t addr, void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_read_b;
    return vgpu98_lfb_get(dev, addr);
}

static uint16_t
vgpu98_lfb_readw(uint32_t addr, void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_read_w;
    return vgpu98_lfb_get(dev, addr) | (vgpu98_lfb_get(dev, addr + 1) << 8);
}

static uint32_t
vgpu98_lfb_readl(uint32_t addr, void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_read_l;
    return vgpu98_lfb_get(dev, addr) | (vgpu98_lfb_get(dev, addr + 1) << 8) | (vgpu98_lfb_get(dev, addr + 2) << 16) |
           ((uint32_t) vgpu98_lfb_get(dev, addr + 3) << 24);
}

static void
vgpu98_lfb_writeb(uint32_t addr, uint8_t val, void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_write_b;
    vgpu98_lfb_put(dev, addr, val);
}

static void
vgpu98_lfb_writew(uint32_t addr, uint16_t val, void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_write_w;
    vgpu98_lfb_put(dev, addr, val & 0xff);
    vgpu98_lfb_put(dev, addr + 1, val >> 8);
}

static void
vgpu98_lfb_writel(uint32_t addr, uint32_t val, void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;

    cycles -= dev->svga.monitor->mon_video_timing_write_l;
    vgpu98_lfb_put(dev, addr, val & 0xff);
    vgpu98_lfb_put(dev, addr + 1, (val >> 8) & 0xff);
    vgpu98_lfb_put(dev, addr + 2, (val >> 16) & 0xff);
    vgpu98_lfb_put(dev, addr + 3, val >> 24);
}

/* --- PCI configuration space (VGPU98 spec: pci-config.md) --- */

static void
vgpu98_update_io(vgpu98_t *dev)
{
    io_removehandler(0x03a0, 0x0040, vgpu98_in, NULL, NULL, vgpu98_out, NULL, NULL, dev);
    if (dev->pci_command & PCI_COMMAND_IO) {
        if (!(dev->svga.miscout & 0x01))
            io_sethandler(0x03a0, 0x0020, vgpu98_in, NULL, NULL, vgpu98_out, NULL, NULL, dev);
        io_sethandler(0x03c0, 0x0020, vgpu98_in, NULL, NULL, vgpu98_out, NULL, NULL, dev);
    }
}

static void
vgpu98_update_mem(vgpu98_t *dev)
{
    int mem = dev->pci_command & PCI_COMMAND_MEM;

    mem_mapping_disable(&dev->svga.mapping);
    mem_mapping_disable(&dev->window_mapping);
    mem_mapping_disable(&dev->mmio_mapping);
    mem_mapping_disable(&dev->linear_mapping);
    if (dev->has_rom)
        mem_mapping_disable(&dev->bios_rom.mapping);

    if (!mem)
        return;

    /* The A0000h window: the VGA core's, or the display engine's 64 KB bank. */
    if (DISP_ENABLED(dev))
        mem_mapping_enable(&dev->window_mapping);
    else
        mem_mapping_enable(&dev->svga.mapping);
    /* A BAR holding 0 or the all-ones sizing probe is not mapped. */
    if ((dev->bar0 != 0) && (dev->bar0 != (uint32_t) ~(VGPU98_BAR0_SIZE - 1)))
        mem_mapping_set_addr(&dev->mmio_mapping, dev->bar0, VGPU98_BAR0_SIZE);
    if ((dev->bar1 != 0) && (dev->bar1 != (uint32_t) ~(VGPU98_BAR1_SIZE - 1)))
        mem_mapping_set_addr(&dev->linear_mapping, dev->bar1, VGPU98_BAR1_SIZE);
    if (dev->has_rom && (dev->rom_bar & 0x01)) {
        uint32_t rom_addr = dev->rom_bar & ~(VGPU98_ROM_BAR_SIZE - 1);

        if ((rom_addr != 0) && (rom_addr != (uint32_t) ~(VGPU98_ROM_BAR_SIZE - 1)))
            mem_mapping_set_addr(&dev->bios_rom.mapping, rom_addr, VGPU98_ROM_SIZE);
    }
}

static uint8_t
vgpu98_pci_read(int func, int addr, UNUSED(int len), void *priv)
{
    const vgpu98_t *dev = (vgpu98_t *) priv;

    if (func != 0)
        return 0xff;

    switch (addr) {
        case 0x00:
            return VGPU98_PCI_VENDOR_ID & 0xff;
        case 0x01:
            return VGPU98_PCI_VENDOR_ID >> 8;
        case 0x02:
            return VGPU98_PCI_DEVICE_ID & 0xff;
        case 0x03:
            return VGPU98_PCI_DEVICE_ID >> 8;
        case 0x04:
            return dev->pci_command;
        case 0x06:
            return 0x00;
        case 0x07:
            return 0x02; /* DEVSEL timing: medium */
        case 0x08:
            return VGPU98_PCI_REVISION;
        case 0x09:
            return VGPU98_PCI_CLASS_CODE & 0xff;
        case 0x0a:
            return (VGPU98_PCI_CLASS_CODE >> 8) & 0xff;
        case 0x0b:
            return VGPU98_PCI_CLASS_CODE >> 16;
        case 0x10 ... 0x13:
            return (uint8_t) (dev->bar0 >> ((addr & 3) << 3));
        case 0x14:
            return 0x08; /* prefetchable 32-bit memory */
        case 0x15 ... 0x17:
            return (uint8_t) (dev->bar1 >> ((addr & 3) << 3));
        case 0x2c:
            return VGPU98_PCI_SUBSYS_VENDOR_ID & 0xff;
        case 0x2d:
            return VGPU98_PCI_SUBSYS_VENDOR_ID >> 8;
        case 0x2e:
            return VGPU98_PCI_SUBSYS_ID & 0xff;
        case 0x2f:
            return VGPU98_PCI_SUBSYS_ID >> 8;
        case 0x30 ... 0x33:
            return (uint8_t) (dev->rom_bar >> ((addr & 3) << 3));
        case 0x3c:
            return dev->int_line;
        case 0x3d:
            return PCI_INTA;
        default:
            return 0x00;
    }
}

/* Writes one byte of a BAR, keeping only the address bits a BAR of this size decodes. */
static uint32_t
vgpu98_bar_write(uint32_t bar, int addr, uint8_t val, uint32_t mask)
{
    int shift = (addr & 3) << 3;

    bar = (bar & ~(0xffu << shift)) | ((uint32_t) val << shift);
    return bar & mask;
}

static void
vgpu98_pci_write(int func, int addr, UNUSED(int len), uint8_t val, void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;

    if (func != 0)
        return;

    switch (addr) {
        case 0x04:
            dev->pci_command = val & (PCI_COMMAND_IO | PCI_COMMAND_MEM);
            vgpu98_update_io(dev);
            vgpu98_update_mem(dev);
            break;
        case 0x10 ... 0x13:
            dev->bar0 = vgpu98_bar_write(dev->bar0, addr, val, ~(VGPU98_BAR0_SIZE - 1));
            vgpu98_update_mem(dev);
            break;
        case 0x14 ... 0x17:
            dev->bar1 = vgpu98_bar_write(dev->bar1, addr, val, ~(VGPU98_BAR1_SIZE - 1));
            vgpu98_update_mem(dev);
            break;
        case 0x30 ... 0x33:
            dev->rom_bar = vgpu98_bar_write(dev->rom_bar, addr, val, ~(VGPU98_ROM_BAR_SIZE - 1) | 0x01);
            vgpu98_update_mem(dev);
            break;
        case 0x3c:
            dev->int_line = val;
            break;
        default:
            break;
    }
}

/* --- Device --- */

static void
vgpu98_reset(void *priv)
{
    vgpu98_t *dev  = (vgpu98_t *) priv;
    svga_t   *svga = &dev->svga;
    int       i;

    /* Display engine off, registers cleared and locked. */
    memset(dev->disp_regs, 0x00, sizeof(dev->disp_regs));
    dev->disp_unlocked = 0;
    dev->edid_index    = 0;
    dev->disp_start    = 0;

    /* VGA registers to zero. The sequencer and graphics controller go through
       svga_out so the state derived from them (write mode, read plane, chaining,
       the A0000h window) follows. */
    memset(svga->crtc, 0x00, sizeof(svga->crtc));
    memset(svga->attrregs, 0x00, sizeof(svga->attrregs));
    svga->attrff              = 0;
    svga->attraddr            = 0;
    svga->attr_palette_enable = 0;
    svga->crtcreg             = 0;
    for (i = 0; i < 5; i++) {
        svga_out(0x3c4, i, svga);
        svga_out(0x3c5, 0x00, svga);
    }
    svga_out(0x3c4, 0x00, svga);
    for (i = 0; i < 9; i++) {
        svga_out(0x3ce, i, svga);
        svga_out(0x3cf, 0x00, svga);
    }
    svga_out(0x3ce, 0x00, svga);
    svga->crtc[0]     = 63;
    svga->crtc[6]     = 255;
    svga->miscout     = 0;
    svga->bpp         = 8;
    svga->dispontime  = 1000ULL << 32;
    svga->dispofftime = 1000ULL << 32;
    svga_recalctimings(svga);
    vgpu98_end_frame(dev);

    /* PCI power-on state. Like 86Box's other PCI video cards, the card decodes the
       legacy VGA ports and memory from power-on (command = I/O + memory), so the
       machine BIOS can use it before it configures the bus. The BARs start
       unassigned and stay unmapped until the BIOS programs them. */
    dev->pci_command = PCI_COMMAND_IO | PCI_COMMAND_MEM;
    dev->bar0        = 0;
    dev->bar1        = 0;
    dev->rom_bar     = 0;
    dev->int_line    = 0;
    vgpu98_update_io(dev);
    vgpu98_update_mem(dev);

    /* The buffer the mailbox named belonged to the last boot. */
    memset(dev->sysbuf, 0x00, sizeof(dev->sysbuf));
    vgpu98_cursor_reset(dev);
    vgpu98_gamma_reset(dev);
    vgpu98_gpu_reset(dev->gpu);

    /* A line the BIOS had not finished is logged as it stands, not lost. */
    vgpu98_debug_flush(dev, dev->debug_line, &dev->debug_len);
    vgpu98_debug_flush(dev, dev->dll_line, &dev->dll_len);
    if (dev->dd)
        vgpu98_dd_check(dev);
}

static void *
vgpu98_init(const device_t *info)
{
    vgpu98_t *dev = calloc(1, sizeof(vgpu98_t));

    vgpu98_host_t host;

    dev->vram_size = device_get_config_int("memory") << 20;
    dev->log_mask  = vgpu98_log_mask(device_get_config_int("diagnostics"));
    /* "DirectX 8 shaders" (dx8_shaders): On 3, Off 0, and the Capture values
       1 and 2, which act as On and with Diagnostics at a full level (not
       Minimal or Glide) log each shader's instructions, and with 2 its GPU
       program too. */
    dev->shader_capture = device_get_config_int("dx8_shaders");
    if ((dev->shader_capture != 1) && (dev->shader_capture != 2))
        dev->shader_capture = 0;
    if (dev->log_mask) {
        char stamp[24];
        char box[256];

        /* The machine, which nothing else in the log records:
           the board, the processor and its
           speed (whole MHz, as 86Box's own names: 366, not 367), the
           memory (in KB unless whole MB). */
        const int whole_mb = mem_size && !(mem_size & 1023);
        /* Shader capture needs a full level: not Minimal, Glide, OpenGL or
           Display, at which the DirectX driver keeps to its event lines. */
        const char *capture_note =
            (dev->log_mask & VGPU98_LOG_MINIMAL) ? " (not at Minimal, Glide, OpenGL or Display)" : "";

        if (cpu_f && cpu_s)
            snprintf(box, sizeof(box), "; %s, %s %s at %u MHz, %u %s of memory", machine_getname(machine), cpu_f->manufacturer,
                     cpu_f->name, (unsigned) (cpu_s->rspeed / 1000000), whole_mb ? mem_size >> 10 : mem_size,
                     whole_mb ? "MB" : "KB");
        else
            snprintf(box, sizeof(box), "; %s, %u KB of memory", machine_getname(machine), mem_size);
        vgpu98_log_stamp(stamp, sizeof(stamp));
        pclog("%s VGPU98: diagnostics \"%s\", %u MB video memory%s%s%s%s%s%s%s%s%s%s%s\n", stamp,
              vgpu98_diag_names[device_get_config_int("diagnostics")], dev->vram_size >> 20,
              device_get_config_int("depth16") ? ", 16-bit depth precision \"Full\"" : "",
              device_get_config_int("dx8") ? ", DirectX 8 interface Off" : "",
              !device_get_config_int("dx8_shaders") ? ", DirectX 8 shaders Off"
              : (dev->shader_capture == 2)        ? ", DirectX 8 shaders \"Capture, with GPU programs\""
              : dev->shader_capture               ? ", DirectX 8 shaders \"Capture\""
                                                  : "",
              dev->shader_capture ? capture_note : "",
              (device_get_config_int("pipeline_cache") == 1)   ? ", pipeline cache \"Start empty\""
              : (device_get_config_int("pipeline_cache") == 2) ? ", pipeline cache Off"
                                                               : "",
              device_get_config_int("mainmem_locks") ? ", Locks through main memory Off" : "",
              device_get_config_int("ogl_clamp") ? ", OpenGL GL_CLAMP \"Exact\"" : "",
              device_get_config_int("cursor") ? ", cursor \"Software\"" : "",
              device_get_config_int("gamma") ? ", gamma Off" : "",
              device_get_config_int("vsync") ? ", V-sync overrides Off" : "", box);
    }

    dev->has_rom = (rom_init(&dev->bios_rom, VGPU98_ROM_PATH, 0xc0000, VGPU98_ROM_SIZE, VGPU98_ROM_SIZE - 1, 0,
                             MEM_MAPPING_EXTERNAL) == 0);

    video_inform(VIDEO_FLAG_TYPE_SPECIAL, &timing_vgpu98);

    svga_init(info, &dev->svga, dev, dev->vram_size, vgpu98_recalctimings, vgpu98_in, vgpu98_out, NULL, NULL);
    dev->svga.decode_mask  = dev->vram_size - 1;
    dev->svga.vblank_start = vgpu98_vblank_start;
    /* The monitor outlives a hard reset: its screenshots so far are old. */
    atomic_init(&dev->shot_seen, atomic_load(&dev->svga.monitor->mon_screenshot_requests));
    /* The card starts in a standard VGA mode, so the display address wraps at
       256 KB as on a VGA (as Cirrus and S3 cards do in their VGA modes). */
    dev->svga.vram_display_mask = 0x3ffff;

    mem_mapping_add(&dev->mmio_mapping, 0, 0,
                    vgpu98_mmio_readb, vgpu98_mmio_readw, vgpu98_mmio_readl,
                    vgpu98_mmio_writeb, vgpu98_mmio_writew, vgpu98_mmio_writel,
                    NULL, MEM_MAPPING_EXTERNAL, dev);
    mem_mapping_add(&dev->linear_mapping, 0, 0,
                    vgpu98_lfb_readb, vgpu98_lfb_readw, vgpu98_lfb_readl,
                    vgpu98_lfb_writeb, vgpu98_lfb_writew, vgpu98_lfb_writel,
                    NULL, MEM_MAPPING_EXTERNAL, dev);
    mem_mapping_add(&dev->window_mapping, 0xa0000, 0x10000,
                    vgpu98_window_readb, vgpu98_window_readw, vgpu98_window_readl,
                    vgpu98_window_writeb, vgpu98_window_writew, vgpu98_window_writel,
                    NULL, MEM_MAPPING_EXTERNAL, dev);
    mem_mapping_disable(&dev->window_mapping);

    pci_add_card(PCI_ADD_NORMAL, vgpu98_pci_read, vgpu98_pci_write, dev, &dev->slot);

    memset(&host, 0, sizeof(host));
    host.priv      = dev;
    host.display   = vgpu98_gpu_display;
    /* The host GPU's pipeline cache, beside 86Box's global configuration
       (shared by every machine: Vulkan checks it is the host GPU's). */
    {
        char dir[1024];

        if (global_cfg_path[0])
            path_get_dirname(dir, global_cfg_path);
        else
            snprintf(dir, sizeof(dir), "%s", usr_path);
        path_append_filename(host.pipeline_cache, dir, "vgpu98-pipelines.bin");
        host.open = plat_fopen;
    }
    /* The texture audit's dumps and the Glide and OpenGL devices' captures,
       in the machine's own folder (at Diagnostics "Glide" and "OpenGL" too). */
    if (dev->log_mask & (VGPU98_LOG_TEXTURES | VGPU98_LOG_GLIDE | VGPU98_LOG_OPENGL))
        path_append_filename(host.capture_dir, usr_path, "vgpu98-textures");
    /* "Pipeline cache" (pipeline_cache): On 0, Start empty 1, Off 2; the
       cache is shared by every machine, the setting is each machine's. */
    host.options   = (device_get_config_int("depth16") ? VGPU98_OPT_DEPTH16_FULL : 0) |
                   (device_get_config_int("dx8") ? VGPU98_OPT_NO_FIXED8 : 0) |
                   (!device_get_config_int("dx8_shaders") ? VGPU98_OPT_NO_PROGRAMS : 0) |
                   ((device_get_config_int("pipeline_cache") == 1) ? VGPU98_OPT_PCACHE_EMPTY : 0) |
                   ((device_get_config_int("pipeline_cache") == 2) ? VGPU98_OPT_PCACHE_OFF : 0) |
                   (device_get_config_int("mainmem_locks") ? VGPU98_OPT_NO_DOWNLOAD : 0) |
                   (device_get_config_int("ogl_clamp") ? VGPU98_OPT_OGL_CLAMP_EXACT : 0) |
                   (device_get_config_int("cursor") ? VGPU98_OPT_SOFT_CURSOR : 0) |
                   (device_get_config_int("gamma") ? VGPU98_OPT_NO_GAMMA : 0) |
                   (device_get_config_int("vsync") ? VGPU98_OPT_NO_VSYNC : 0);
    dev->gpu       = vgpu98_gpu_init(dev->svga.vram, dev->vram_size, &host, dev->log_mask);
    dev->cursor_hw = (vgpu98_gpu_profile(dev->gpu) & VGPU98_PROFILE_CURSOR) != 0;
    dev->gamma_hw  = (vgpu98_gpu_profile(dev->gpu) & VGPU98_PROFILE_GAMMA) != 0;
    dev->vsync_hw  = (vgpu98_gpu_profile(dev->gpu) & VGPU98_PROFILE_VSYNC) != 0;
    dev->cpu_pages = vgpu98_gpu_cpu_pages(dev->gpu);
    dev->disp      = vgpu98_gpu_disp(dev->gpu);
    dev->newer     = vgpu98_gpu_newer(dev->gpu);
    /* At Diagnostics "Display", the screen's watch; its captures in the
       machine's folder with the others'. */
    if (dev->log_mask & VGPU98_LOG_DISPLAY) {
        char base[1024];

        path_append_filename(base, usr_path, "vgpu98-textures");
        dev->dd = vgpu98_dd_init(dev->disp, base);
        if (dev->dd)
            dev->svga.vsync_callback = vgpu98_vsync;
    }

    vgpu98_reset(dev);

    vgpu98_log("VGPU98: %u MB VRAM, ROM %s\n", dev->vram_size >> 20, dev->has_rom ? "loaded" : "missing");

    return dev;
}

static void
vgpu98_close(void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;
    int       freed;

    if (dev->host_display) {
        vgpu98_disp_stop(dev->disp);
        dev->svga.host_display = 0;
    }
    /* Before the display goes with the GPU: its last line asks the layer. */
    vgpu98_dd_close(dev->dd);
    dev->dd                  = NULL;
    dev->svga.vsync_callback = NULL;
    freed = vgpu98_gpu_close(dev->gpu);
    vgpu98_debug_flush(dev, dev->debug_line, &dev->debug_len);
    vgpu98_debug_flush(dev, dev->dll_line, &dev->dll_len);
    /* A GPU thread still running keeps video memory and the device, which
       it reads (86Box is closing: left rather than freed under it). */
    if (!freed)
        dev->svga.vram = NULL;
    svga_close(&dev->svga);
    if (freed)
        free(dev);
}

static int
vgpu98_available(void)
{
    return rom_present(VGPU98_ROM_PATH);
}

static void
vgpu98_speed_changed(void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;

    svga_recalctimings(&dev->svga);
    if (dev->dd)
        vgpu98_dd_check(dev);
}

static void
vgpu98_force_redraw(void *priv)
{
    vgpu98_t *dev = (vgpu98_t *) priv;

    dev->svga.fullchange = dev->svga.monitor->mon_changeframecount;
    vgpu98_screenshot_capture(dev);
}

// clang-format off
static const device_config_t vgpu98_config[] = {
    {
        .name           = "memory",
        .description    = "Memory size",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = VGPU98_VRAM_DEFAULT_MB,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            VGPU98_VRAM_SELECTION
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        .name           = "diagnostics",
        .description    = "Diagnostics (86Box's log, not inside the VM)",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = VGPU98_DIAG_MINIMAL,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "Off",                            .value = VGPU98_DIAG_OFF },
            { .description = "Minimal",                        .value = VGPU98_DIAG_MINIMAL },
            { .description = "DirectX",                        .value = VGPU98_DIAG_DIRECTX },
            { .description = "DirectX + capture",                        .value = VGPU98_DIAG_CAPTURE },
            { .description = "DirectX: texture audit",                   .value = VGPU98_DIAG_TEXTURES },
            { .description = "DirectX 7: vertex audit",                  .value = VGPU98_DIAG_AUDIT },
            { .description = "DirectX 8: stream + texture audit (slow)", .value = VGPU98_DIAG_AUDIT8 },
            { .description = "Glide",                                    .value = VGPU98_DIAG_GLIDE },
            { .description = "OpenGL",                                   .value = VGPU98_DIAG_OPENGL },
            { .description = "Display",                                  .value = VGPU98_DIAG_DISPLAY },
            { .description = "Developer: every packet (slow)",           .value = VGPU98_DIAG_DEVELOPER },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        .name           = "depth16",
        .description    = "16-bit depth precision",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 0,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "As the guest's", .value = 0 },
            { .description = "Full",           .value = 1 },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        .name           = "dx8",
        .description    = "DirectX 8 interface",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 0,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "On",  .value = 0 },
            { .description = "Off", .value = 1 },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        .name           = "dx8_shaders",
        .description    = "DirectX 8 shaders",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 3,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "On",                          .value = 3 },
            { .description = "Off",                         .value = 0 },
            { .description = "Capture",                     .value = 1 },
            { .description = "Capture, with GPU programs",  .value = 2 },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        .name           = "mainmem_locks",
        .description    = "Locks through main memory",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 0,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "On",  .value = 0 },
            { .description = "Off", .value = 1 },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        /* The OpenGL device's GL_CLAMP */
        .name           = "ogl_clamp",
        .description    = "OpenGL GL_CLAMP",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 0,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "As period drivers (clamp to edge)", .value = 0 },
            { .description = "Exact (OpenGL 1.1)",                .value = 1 },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        /* The hardware cursor */
        .name           = "cursor",
        .description    = "Cursor",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 0,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "Hardware",                  .value = 0 },
            { .description = "Software (the DIB engine's)", .value = 1 },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        /* The gamma table's off switch: Off takes
           PROFILE.GAMMA away, so Windows, DirectDraw and the card's tab
           see a card without gamma. */
        .name           = "gamma",
        .description    = "Gamma",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 0,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "On",  .value = 0 },
            { .description = "Off", .value = 1 },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        /* The V-sync overrides' off switch: Off takes
           PROFILE.VSYNC away, so programs decide their waits as they ask
           and the card's tab says the overrides are off. */
        .name           = "vsync",
        .description    = "V-sync overrides",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 0,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "On",  .value = 0 },
            { .description = "Off", .value = 1 },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    {
        .name           = "pipeline_cache",
        .description    = "Pipeline cache (shared by every machine)",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 0,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "On",          .value = 0 },
            { .description = "Start empty", .value = 1 },
            { .description = "Off",         .value = 2 },
            { .description = "" }
        },
        .bios           = { { 0 } }
    },
    { .name = "", .description = "", .type = CONFIG_END }
};
// clang-format on

const device_t vgpu98_pci_device = {
    .name          = VGPU98_CARD_NAME,
    .internal_name = VGPU98_CARD_INTERNAL_NAME,
    .flags         = DEVICE_PCI,
    .local         = 0,
    .init          = vgpu98_init,
    .close         = vgpu98_close,
    .reset         = vgpu98_reset,
    .available     = vgpu98_available,
    .speed_changed = vgpu98_speed_changed,
    .force_redraw  = vgpu98_force_redraw,
    .config        = vgpu98_config
};
