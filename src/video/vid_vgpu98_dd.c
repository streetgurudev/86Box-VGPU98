/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: Diagnostics "Display".
 *          The screen is watched at its
 *          transitions only: a mode change of the display engine or the
 *          VGA, the engine on or off, the host GPU's display starting or
 *          stopping, a Glide session's or a pass-through card's screen, the
 *          monitor's sleep. For 30 vertical syncs after each, what 86Box
 *          did with the screen is logged (the lines it counted against the
 *          mode's, its screen's size, the host GPU's pictures, the palette,
 *          the layer), and the frames that differ from the last kept are
 *          copied, to be written as pictures on a thread of their own.
 *          Nothing is checked while nothing changes but 86Box's screen
 *          size.
 *
 *          Everything runs on the emulation thread but the screenshot's
 *          flag and the captures' writers.
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
#include <png.h>
#include <86box/86box.h>
#include <86box/path.h>
#include <86box/plat.h>
#include <86box/thread.h>
#include <86box/video.h>
#include <86box/vid_vgpu98_card.h>
#include <86box/vid_vgpu98_gpu.h>
#include <86box/vid_vgpu98_dev.h>

#define DD_WINDOW     30          /* vertical syncs watched after a transition */
#define DD_LOGGED     200         /* transitions a session logged in full */
#define DD_KEEP       6           /* frames a capture keeps */
#define DD_KEEP_BYTES (96u << 20) /* and their bytes at most */
#define DD_CAPTURES   24          /* a session */
#define DD_WRITERS    3           /* captures being written at once */
#define DD_TEXT       16384       /* a capture's transition.txt */

/* A frame kept: 86Box's pixels as it blitted them, and in the engine's
   modes the frame's rows of video memory, packed, with the palette. */
typedef struct dd_kept_t {
    uint32_t  frame;
    uint32_t *shown;
    uint32_t  w, h;
    uint8_t  *mem;
    uint32_t  mw, mh, mbpp;
    uint32_t  palette[256];
} dd_kept_t;

typedef struct dd_capture_t {
    char      base[1024];
    char      name[128]; /* its folder's, from the transition's time */
    uint32_t  serial;
    uint32_t  transition;
    uint32_t  n;
    size_t    bytes;
    dd_kept_t kept[DD_KEEP];
    char      text[DD_TEXT]; /* the transition's lines, then a line a frame kept */
    size_t    len;
} dd_capture_t;

struct vgpu98_dd_t {
    vgpu98_disp_t    *disp;
    char              base[1024];
    int               started; /* the first vertical sync seen */
    vgpu98_dd_state_t cur;     /* the screen at the last */
    int               screen_w, screen_h;

    /* Since the last vertical sync */
    uint32_t         timing;     /* timing changes */
    vgpu98_dd_scan_t scan;       /* at the last of them */
    int              old_vtotal; /* the frame's total before the first */
    int              ended;      /* frames the card ended */
    vgpu98_dd_scan_t end_scan;   /* the scan the last of them was ended from */
    int              palette;    /* the DAC's colours written */
    int              restarted;  /* the host display restarted at the vertical blank */
    int              took;       /* and took the frame `taken` */
    vgpu98_frame_t   taken;
    int              blanked;    /* and blacked the frame out: a start, or a picture of the mode before */
    atomic_int       shot;

    /* The transition watched */
    int               open;
    uint32_t          serial;
    uint32_t          frame;  /* vertical syncs since */
    int               logged; /* its lines go to the log */
    vgpu98_dd_state_t to;     /* the screen it went to */
    uint32_t          to_start;
    uint32_t          stretched;
    int               tallest;
    uint32_t          restarts;
    uint32_t          host_new_at; /* the frame of the host GPU's first picture of the new mode */
    uint32_t          host_old;    /* frames that showed its picture of the old mode */
    uint32_t          black;       /* frames shown black before it */
    int               black_on;    /* the frame buffer is black: blacked out, no picture since */
    uint32_t          palette_at;
    uint32_t          layer_first;
    uint64_t          last_hash;
    int               capturing; /* a capture is made at its first frame kept */
    char              cap_name[96];
    dd_capture_t     *cap;
    char              text[DD_TEXT]; /* its lines, for the capture */
    size_t            text_len;

    /* The session's */
    uint32_t transitions;
    uint32_t stretched_all;
    uint32_t with_stretched; /* transitions with a stretched frame */
    uint32_t ended_all;      /* vertical syncs after a frame the card ended */
    uint32_t captures;
    int      captures_told;
};

static atomic_int dd_writers;

static const char *const dd_host_why[] = { "it composes them",
                                           "the display engine is off",
                                           "the monitor sleeps",
                                           "a pass-through card has the screen",
                                           "the frame does not fit 86Box's frame buffer",
                                           "no display on the host GPU" };

/* A session's line: always logged. */
static void
dd_say(const char *fmt, ...)
{
    char    line[1024];
    char    stamp[24];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    vgpu98_log_stamp(stamp, sizeof(stamp));
    pclog("%s VGPU98 Display: %s\n", stamp, line);
}

/* A text of `size` bytes, `*len` of them used, grows by a line. */
static void
dd_text(char *text, size_t *len, size_t size, const char *fmt, ...)
{
    va_list ap;
    int     n;

    if (*len + 1 >= size)
        return;
    va_start(ap, fmt);
    n = vsnprintf(text + *len, size - *len, fmt, ap);
    va_end(ap);
    if (n > 0)
        *len += ((size_t) n < size - *len) ? (size_t) n : size - *len - 1;
}

/* A transition's line: to the log while it is logged in full, and to the
   text its capture takes. */
static void
dd_line(vgpu98_dd_t *dd, const char *fmt, ...)
{
    char    line[1024];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (dd->logged)
        dd_say("%s", line);
    dd_text(dd->text, &dd->text_len, sizeof(dd->text), "%s\n", line);
}

static void
dd_cat(char *buf, size_t size, const char *fmt, ...)
{
    const size_t n = strlen(buf);
    va_list      ap;

    if (n + 1 >= size)
        return;
    va_start(ap, fmt);
    vsnprintf(buf + n, size - n, fmt, ap);
    va_end(ap);
}

static void
dd_mode(const vgpu98_dd_state_t *s, char *buf, size_t size)
{
    buf[0] = 0;
    if (s->engine) {
        dd_cat(buf, size, "%ux%ux%u at %u Hz", s->width, s->height, s->bpp, s->refresh);
        if (s->dscan == 2)
            dd_cat(buf, size, ", double scan");
        if (s->pitch != s->width * ((s->bpp + 7) / 8))
            dd_cat(buf, size, ", pitch %u", s->pitch);
    } else
        dd_cat(buf, size, "VGA %dx%d %s", s->vga_width, s->vga_lines, s->vga_text ? "text" : "graphics");
    if (s->blank)
        dd_cat(buf, size, ", blanked");
}

static int
dd_same_mode(const vgpu98_dd_state_t *a, const vgpu98_dd_state_t *b)
{
    if (a->engine != b->engine)
        return 0;
    if (a->engine)
        return (a->width == b->width) && (a->height == b->height) && (a->bpp == b->bpp) && (a->pitch == b->pitch) &&
               (a->dscan == b->dscan) && (a->refresh == b->refresh);
    return (a->vga_width == b->vga_width) && (a->vga_lines == b->vga_lines) && (a->vga_text == b->vga_text);
}

/* What changed from screen a to screen b, each part after "; ", or "". */
static void
dd_changes(const vgpu98_dd_state_t *a, const vgpu98_dd_state_t *b, char *buf, size_t size)
{
    buf[0] = 0;
    if (a->engine != b->engine)
        dd_cat(buf, size, b->engine ? "; the display engine on" : "; the display engine off");
    else if (!dd_same_mode(a, b))
        dd_cat(buf, size, b->engine ? "; the engine's mode" : "; the VGA's mode");
    if (a->dpms != b->dpms)
        dd_cat(buf, size, b->dpms ? "; the monitor asleep" : "; the monitor awake");
    if (a->override != b->override)
        dd_cat(buf, size,
               b->override ? "; a pass-through card took the screen" : "; the pass-through card gave the screen back");
    if (a->glide != b->glide)
        dd_cat(buf, size, b->glide ? "; a Glide session took the screen" : "; the desktop back from Glide");
    if ((a->host == VGPU98_DD_HOST_ON) != (b->host == VGPU98_DD_HOST_ON)) {
        if (b->host == VGPU98_DD_HOST_ON)
            dd_cat(buf, size, "; the host GPU's display started");
        else
            dd_cat(buf, size, "; the host GPU's display stopped (%s)", dd_host_why[b->host]);
    }
}

/* The opening line's second: where the scan was when the timings last
   changed (or, when the card ended the frame, where it was ended from),
   and what 86Box's next sync was predicted and seen to count. */
static void
dd_scan_line(vgpu98_dd_t *dd, const vgpu98_dd_frame_t *f)
{
    const vgpu98_dd_scan_t *s = dd->ended ? &dd->end_scan : &dd->scan;
    char                    where[192];
    char                    next[64];
    char                    seen[64];
    char                    writes[32];
    int                     lines;
    const int               predicted = vgpu98_dd_next_sync(&dd->scan, &lines);

    if (s->vc >= s->vtotal)
        snprintf(where, sizeof(where), "past the new frame's total (%d)%s", s->vtotal,
                 dd->ended ? "" : ": 86Box's line counter runs on to 2047 and wraps");
    else if (s->vc >= s->vsyncstart)
        snprintf(where, sizeof(where), "past the new sync (%d), before the new total (%d)", s->vsyncstart, s->vtotal);
    else if (s->vc >= s->dispend)
        snprintf(where, sizeof(where), "past the new picture's end (%d)", s->dispend);
    else
        snprintf(where, sizeof(where), "inside the new frame (%d lines)", s->vtotal);
    if (dd->ended)
        dd_cat(where, sizeof(where), ": the card ended the frame there");
    if (dd->timing)
        snprintf(writes, sizeof(writes), "%u write%s", dd->timing, (dd->timing == 1) ? "" : "s");
    else
        snprintf(writes, sizeof(writes), "no timing changed");
    if (predicted < 0)
        snprintf(next, sizeof(next), "no sync");
    else if (!predicted)
        snprintf(next, sizeof(next), "no frame, after %d lines", lines);
    else
        snprintf(next, sizeof(next), "%d lines, after %d", predicted, lines);
    if (!f->shown)
        snprintf(seen, sizeof(seen), "no frame");
    else
        snprintf(seen, sizeof(seen), "%d lines", f->lines);
    dd_line(dd,
            "transition %u: the timings changed with the scan at line %d of %d (%s), %s; %s; "
            "the next sync predicted %s, 86Box showed %s",
            dd->serial, s->vc, dd->old_vtotal, s->dispon ? "in the picture" : "in the blanking", where, writes, next,
            seen);
}

static void
dd_free_capture(dd_capture_t *c)
{
    uint32_t i;

    if (!c)
        return;
    for (i = 0; i < c->n; i++) {
        free(c->kept[i].shown);
        free(c->kept[i].mem);
    }
    free(c);
}

static void
dd_open(vgpu98_dd_t *dd, const vgpu98_dd_frame_t *f, const char *why)
{
    char from[128];
    char to[128];

    dd->open        = 1;
    dd->serial      = ++dd->transitions;
    dd->frame       = 0;
    dd->to          = f->state;
    dd->to_start    = f->start;
    dd->stretched   = 0;
    dd->tallest     = 0;
    dd->restarts    = 0;
    dd->host_new_at = 0;
    dd->host_old    = 0;
    dd->black       = 0;
    dd->black_on    = 0;
    dd->palette_at  = 0;
    dd->layer_first = 0;
    dd->last_hash   = 0;
    dd->text_len    = 0;
    dd->logged      = (dd->serial <= DD_LOGGED);
    if (dd->serial == DD_LOGGED + 1)
        dd_say("transitions from %u on are counted, not logged", dd->serial);

    /* The capture's folder is named for the transition's time, local, as
       the log's: "2026-10-27 21-05-12" (no colons in a file name). */
    dd->cap       = NULL;
    dd->capturing = dd->base[0] && (dd->captures < DD_CAPTURES);
    if (dd->capturing) {
        char  stamp[24];
        char *p;

        vgpu98_log_stamp(stamp, sizeof(stamp));
        for (p = stamp; *p; p++) {
            if (*p == ':')
                *p = '-';
        }
        snprintf(dd->cap_name, sizeof(dd->cap_name), "%s display capture", stamp);
    } else if (dd->base[0] && !dd->captures_told) {
        dd->captures_told = 1;
        dd_say("no more captures this session (%u made); transitions are still logged", dd->captures);
    }

    dd_mode(&dd->cur, from, sizeof(from));
    dd_mode(&f->state, to, sizeof(to));
    if (strcmp(from, to))
        dd_line(dd, "transition %u: %s -> %s; %s", dd->serial, from, to, why);
    else
        dd_line(dd, "transition %u: %s; %s", dd->serial, to, why);
    if (dd->timing || dd->ended)
        dd_scan_line(dd, f);
}

/* Copies the frame 86Box blitted when it differs from the last kept, and
   in the engine's modes while the host GPU composes, its rows of video
   memory. */
static void
dd_keep(vgpu98_dd_t *dd, const vgpu98_dd_frame_t *f)
{
    dd_capture_t            *c;
    const bitmap_t          *tb = f->tb;
    const vgpu98_dd_state_t *s  = &f->state;
    dd_kept_t               *k;
    uint32_t                 w;
    uint32_t                 h;
    uint32_t                 y;
    uint64_t                 hash = VGPU98_DD_HASH_START;
    size_t                   need;
    size_t                   mneed = 0;

    if (!f->shown || !tb || (dd->cap && (dd->cap->n >= DD_KEEP)) || (f->x < 0) || (f->y < 0) || (f->screen_w <= 0) ||
        (f->screen_h <= 0) || (f->x >= tb->w) || (f->y >= tb->h))
        return;
    if (!dd->cap) {
        dd->cap = calloc(1, sizeof(dd_capture_t));
        if (!dd->cap) {
            dd->capturing = 0;
            return;
        }
        dd->cap->serial     = ++dd->captures;
        dd->cap->transition = dd->serial;
        snprintf(dd->cap->base, sizeof(dd->cap->base), "%s", dd->base);
        snprintf(dd->cap->name, sizeof(dd->cap->name), "%s %u", dd->cap_name, dd->cap->serial);
    }
    c = dd->cap;
    w = (uint32_t) f->screen_w;
    h = (uint32_t) f->screen_h;
    if ((uint32_t) f->x + w > (uint32_t) tb->w)
        w = (uint32_t) (tb->w - f->x);
    if ((uint32_t) f->y + h > (uint32_t) tb->h)
        h = (uint32_t) (tb->h - f->y);
    for (y = 0; y < h; y++)
        hash = vgpu98_dd_hash(hash, &tb->line[f->y + (int) y][f->x], w);
    if (c->n && (hash == dd->last_hash))
        return;
    need = (size_t) w * h * 4;
    if (s->engine && (s->host == VGPU98_DD_HOST_ON) && !s->glide && f->vram)
        mneed = (size_t) s->width * ((s->bpp + 7) / 8) * s->height;
    if (c->bytes + need + mneed > DD_KEEP_BYTES)
        return;
    k        = &c->kept[c->n];
    k->shown = malloc(need);
    if (!k->shown)
        return;
    dd->last_hash = hash;
    for (y = 0; y < h; y++)
        memcpy(k->shown + (size_t) y * w, &tb->line[f->y + (int) y][f->x], (size_t) w * 4);
    k->w     = w;
    k->h     = h;
    k->frame = dd->frame;
    if (mneed) {
        k->mem = malloc(mneed);
        if (k->mem) {
            const uint32_t row = s->width * ((s->bpp + 7) / 8);
            uint32_t       i;

            for (y = 0; y < s->height; y++) {
                const uint32_t a = (f->start + y * s->pitch) & f->vram_mask;

                if (a + row - 1 <= f->vram_mask)
                    memcpy(k->mem + (size_t) y * row, f->vram + a, row);
                else
                    for (i = 0; i < row; i++)
                        k->mem[(size_t) y * row + i] = f->vram[(a + i) & f->vram_mask];
            }
            k->mw   = s->width;
            k->mh   = s->height;
            k->mbpp = s->bpp;
            memcpy(k->palette, f->palette, sizeof(k->palette));
        } else
            mneed = 0;
    }
    c->bytes += need + mneed;
    c->n++;
    /* Its line in transition.txt only. */
    dd_text(dd->text, &dd->text_len, sizeof(dd->text), "frame %u kept: 86Box's screen %ux%u, %d lines counted; ",
            k->frame, w, h, f->lines);
    if (!s->engine || (s->host != VGPU98_DD_HOST_ON))
        dd_text(dd->text, &dd->text_len, sizeof(dd->text), "86Box drew it from video memory\n");
    else {
        if (dd->black_on)
            dd_text(dd->text, &dd->text_len, sizeof(dd->text), "black until the host GPU's first picture of the mode");
        else if (dd->took)
            dd_text(dd->text, &dd->text_len, sizeof(dd->text), "the host GPU's picture of %ux%ux%u%s", dd->taken.width,
                    dd->taken.height, dd->taken.bpp, (dd->taken.dscan == 2) ? ", double scan" : "");
        else
            dd_text(dd->text, &dd->text_len, sizeof(dd->text), "no new picture from the host GPU");
        dd_text(dd->text, &dd->text_len, sizeof(dd->text), "%s\n",
                k->mem ? "; memory.png: video memory at this sync" : "");
    }
}

static int
dd_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h)
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

/* A capture's pictures and text, on a thread of its own. */
static void
dd_writer(void *priv)
{
    dd_capture_t *c = (dd_capture_t *) priv;
    char          dir[1024];
    char          path[1024];
    char          name[64];
    uint32_t      files = 0;
    uint32_t      i;
    FILE         *fp;

    if (strlen(c->base) + 1 + strlen(c->name) + sizeof("\\frame 30 memory.png") >= 260) {
        dd_say("capture %u (transition %u): not written, the captures' folder's path is too long (%s)", c->serial,
               c->transition, c->base);
        goto done;
    }
    snprintf(dir, sizeof(dir), "%s", c->base);
    if (!plat_dir_check(dir) && plat_dir_create(dir))
        goto failed;
    path_append_filename(path, dir, c->name);
    snprintf(dir, sizeof(dir), "%s", path);
    if (!plat_dir_check(dir) && plat_dir_create(dir))
        goto failed;
    for (i = 0; i < c->n; i++) {
        const dd_kept_t *k     = &c->kept[i];
        const size_t     big   = (size_t) ((k->w * k->h > k->mw * k->mh) ? k->w * k->h : k->mw * k->mh) * 4;
        uint8_t         *rgba  = malloc(big);
        uint32_t         p;
        uint32_t         y;

        if (!rgba)
            continue;
        for (p = 0; p < k->w * k->h; p++)
            vgpu98_dd_rgba(k->shown[p], rgba + (size_t) p * 4);
        snprintf(name, sizeof(name), "frame %02u shown.png", k->frame);
        path_append_filename(path, dir, name);
        files += dd_png(path, rgba, k->w, k->h);
        if (k->mem) {
            const uint32_t row = k->mw * ((k->mbpp + 7) / 8);

            for (y = 0; y < k->mh; y++)
                vgpu98_dd_decode_row(k->mem, 0xffffffffu, y * row, k->mw, k->mbpp, k->palette,
                                     rgba + (size_t) y * k->mw * 4);
            snprintf(name, sizeof(name), "frame %02u memory.png", k->frame);
            path_append_filename(path, dir, name);
            files += dd_png(path, rgba, k->mw, k->mh);
        }
        free(rgba);
    }
    path_append_filename(path, dir, "transition.txt");
    fp = plat_fopen(path, "w");
    if (fp) {
        fwrite(c->text, 1, c->len, fp);
        files += fclose(fp) == 0;
    }
    dd_say("capture %u (transition %u): %u files in %s", c->serial, c->transition, files, dir);
    goto done;
failed:
    dd_say("capture %u (transition %u): not written, no folder in %s", c->serial, c->transition, c->base);
done:
    dd_free_capture(c);
    atomic_fetch_sub(&dd_writers, 1);
}

/* The window's capture goes to a writer of its own. */
static void
dd_hand_off(vgpu98_dd_t *dd)
{
    dd_capture_t *c = dd->cap;

    dd->cap = NULL;
    if (!c)
        return;
    if (!c->n) {
        dd_free_capture(c);
        return;
    }
    memcpy(c->text, dd->text, dd->text_len);
    c->len = dd->text_len;
    if (atomic_fetch_add(&dd_writers, 1) >= DD_WRITERS) {
        atomic_fetch_sub(&dd_writers, 1);
        dd_say("capture %u (transition %u) dropped: %d captures are still being written", c->serial, c->transition,
               DD_WRITERS);
        dd_free_capture(c);
        return;
    }
    if (!thread_create_named(dd_writer, c, "VGPU98 display capture")) {
        atomic_fetch_sub(&dd_writers, 1);
        dd_free_capture(c);
    }
}

/* The window ends: its closing line, and its capture written. `by` is the
   transition that ended it early, 0 at its end. */
static void
dd_end(vgpu98_dd_t *dd, uint32_t by)
{
    const vgpu98_dd_state_t *s = &dd->to;
    char                     line[1024];
    uint32_t                 layer_end = 0;

    snprintf(line, sizeof(line), "transition %u: %u frame%s watched", dd->serial, dd->frame,
             (dd->frame == 1) ? "" : "s");
    if (by)
        dd_cat(line, sizeof(line), ", ended by transition %u", by);
    dd_cat(line, sizeof(line), "; 86Box's screen %dx%d", dd->screen_w, dd->screen_h);
    if (dd->stretched)
        dd_cat(line, sizeof(line), "; %u stretched frame%s (the tallest %d lines)", dd->stretched,
               (dd->stretched == 1) ? "" : "s", dd->tallest);
    if (s->engine) {
        if (s->host == VGPU98_DD_HOST_ON) {
            if (dd->restarts)
                dd_cat(line, sizeof(line), "; the host GPU's display restarted");
            if (dd->black)
                dd_cat(line, sizeof(line), "; %u frame%s black", dd->black, (dd->black == 1) ? "" : "s");
            if (dd->host_new_at)
                dd_cat(line, sizeof(line), "; its first picture of the new mode at frame %u", dd->host_new_at);
            else
                dd_cat(line, sizeof(line), "; no picture of the new mode from the host GPU yet");
            if (dd->host_old)
                dd_cat(line, sizeof(line), "; %u frame%s showed its picture of the old mode", dd->host_old,
                       (dd->host_old == 1) ? "" : "s");
            if (dd->disp)
                layer_end = vgpu98_disp_layer_over(dd->disp, dd->to_start, s->pitch * s->height, (s->bpp + 7) / 8);
            if (dd->layer_first || layer_end)
                dd_cat(line, sizeof(line),
                       "; the layer: %u rectangle%s of GPU-newer pixels over the frame at frame 1, %u at the end",
                       dd->layer_first, (dd->layer_first == 1) ? "" : "s", layer_end);
        } else
            dd_cat(line, sizeof(line), "; 86Box drew the frames (%s)", dd_host_why[s->host]);
        if (s->bpp == 8) {
            if (dd->palette_at)
                dd_cat(line, sizeof(line), "; the palette written at frame %u", dd->palette_at);
            else
                dd_cat(line, sizeof(line), "; the palette not written");
        }
    }
    if (dd->cap)
        dd_cat(line, sizeof(line), "; %u frame%s kept for capture %u", dd->cap->n, (dd->cap->n == 1) ? "" : "s",
               dd->cap->serial);
    dd_line(dd, "%s", line);
    if (dd->stretched) {
        dd->stretched_all += dd->stretched;
        dd->with_stretched++;
    }
    dd_hand_off(dd);
    dd->open = 0;
}

/* A vertical sync in the window. */
static void
dd_watch(vgpu98_dd_t *dd, const vgpu98_dd_frame_t *f)
{
    const vgpu98_dd_state_t *s         = &f->state;
    const int                mode_w    = s->engine ? (int) (s->width * s->dscan) : s->vga_width;
    const int                stretched = f->shown && vgpu98_dd_stretched(f->lines, f->rows);

    dd->frame++;
    if ((f->screen_w != dd->screen_w) || (f->screen_h != dd->screen_h)) {
        char note[96];

        if (stretched)
            snprintf(note, sizeof(note), "a stretched frame, %d lines counted for the mode's %d", f->lines, f->rows);
        else if ((f->screen_w == mode_w) && (f->screen_h == f->rows))
            snprintf(note, sizeof(note), "the mode's");
        else
            snprintf(note, sizeof(note), "the mode is %dx%d", mode_w, f->rows);
        dd_line(dd, "transition %u, frame %u: 86Box's screen %dx%d -> %dx%d (%s)", dd->serial, dd->frame, dd->screen_w,
                dd->screen_h, f->screen_w, f->screen_h, note);
    } else if (stretched)
        dd_line(dd, "transition %u, frame %u: a stretched frame, %d lines counted for the mode's %d", dd->serial,
                dd->frame, f->lines, f->rows);
    if (stretched) {
        dd->stretched++;
        if (f->lines > dd->tallest)
            dd->tallest = f->lines;
    }
    dd->screen_w = f->screen_w;
    dd->screen_h = f->screen_h;

    /* The host GPU's display at the vertical blank before this sync: the
       rows it took are what this sync's blit shows. */
    if (dd->restarted)
        dd->restarts++;
    /* Black from the frame blacked out to the host GPU's next picture, the
       frames between with no new picture too. */
    if (dd->blanked)
        dd->black_on = 1;
    if (dd->took || !s->engine || (s->host != VGPU98_DD_HOST_ON))
        dd->black_on = 0;
    if (dd->black_on && f->shown)
        dd->black++;
    if (dd->took && s->engine) {
        if ((dd->taken.width == s->width) && (dd->taken.height == s->height) && (dd->taken.bpp == s->bpp) &&
            (dd->taken.dscan == s->dscan)) {
            if (!dd->host_new_at)
                dd->host_new_at = dd->frame;
        } else
            dd->host_old++;
    }
    if (dd->palette && !dd->palette_at)
        dd->palette_at = dd->frame;
    if ((dd->frame == 1) && s->engine && (s->host == VGPU98_DD_HOST_ON) && dd->disp)
        dd->layer_first = vgpu98_disp_layer_over(dd->disp, f->start, s->pitch * s->height, (s->bpp + 7) / 8);
    if (dd->capturing)
        dd_keep(dd, f);
    if (dd->frame >= DD_WINDOW)
        dd_end(dd, 0);
}

void
vgpu98_dd_vsync(vgpu98_dd_t *dd, const vgpu98_dd_frame_t *f)
{
    char      changes[512];
    int       shot;
    int       fresh;
    int       resized;

    if (!dd)
        return;
    shot = atomic_exchange(&dd->shot, 0);
    if (!dd->started) {
        char mode[128];

        dd->started = 1;
        dd_mode(&f->state, mode, sizeof(mode));
        dd_say("watching the screen: %s; 86Box's screen %dx%d", mode, f->screen_w, f->screen_h);
    } else {
        dd_changes(&dd->cur, &f->state, changes, sizeof(changes));
        fresh   = changes[0] || dd->timing || dd->ended;
        resized = (f->screen_w != dd->screen_w) || (f->screen_h != dd->screen_h);
        if (dd->ended)
            dd->ended_all++;
        if (dd->open && fresh)
            dd_end(dd, dd->transitions + 1);
        if (!dd->open && (fresh || shot || resized)) {
            const char *why = changes[0]   ? changes + 2
                              : dd->timing ? "its timings changed"
                              : dd->ended  ? "the card ended a frame"
                              : shot       ? "an 86Box screenshot"
                                           : "86Box's screen size alone";

            dd_open(dd, f, why);
        }
        if (dd->open)
            dd_watch(dd, f);
    }
    dd->cur       = f->state;
    dd->screen_w  = f->screen_w;
    dd->screen_h  = f->screen_h;
    dd->timing    = 0;
    dd->ended     = 0;
    dd->palette   = 0;
    dd->restarted = 0;
    dd->took      = 0;
    dd->blanked   = 0;
}

void
vgpu98_dd_timing(vgpu98_dd_t *dd, const vgpu98_dd_scan_t *scan, int old_vtotal)
{
    if (!dd || !dd->started)
        return;
    if (!dd->timing)
        dd->old_vtotal = old_vtotal;
    dd->scan = *scan;
    dd->timing++;
}

void
vgpu98_dd_ended(vgpu98_dd_t *dd, const vgpu98_dd_scan_t *before, const vgpu98_dd_scan_t *after)
{
    if (!dd || !dd->started)
        return;
    /* Without a timing change since the last sync (one whose port write
       comes next records itself), the scan is the ended frame's. */
    if (!dd->timing && !dd->ended) {
        dd->old_vtotal = before->vtotal;
        dd->scan       = *after;
    }
    dd->end_scan = *before;
    dd->ended++;
}

void
vgpu98_dd_palette(vgpu98_dd_t *dd)
{
    if (dd)
        dd->palette = 1;
}

void
vgpu98_dd_vblank(vgpu98_dd_t *dd, int restarted, int blanked, int took, const vgpu98_frame_t *taken)
{
    if (!dd)
        return;
    if (restarted)
        dd->restarted = 1;
    /* A start, or no picture of the mode shown in the frame buffer: the
       frame is black. */
    if (blanked)
        dd->blanked = 1;
    if ((took > 0) && taken) {
        dd->took  = 1;
        dd->taken = *taken;
    }
}

void
vgpu98_dd_screenshot(vgpu98_dd_t *dd)
{
    if (dd)
        atomic_store(&dd->shot, 1);
}

vgpu98_dd_t *
vgpu98_dd_init(vgpu98_disp_t *disp, const char *capture_base)
{
    vgpu98_dd_t *dd = calloc(1, sizeof(vgpu98_dd_t));

    if (!dd)
        return NULL;
    dd->disp = disp;
    snprintf(dd->base, sizeof(dd->base), "%s", capture_base ? capture_base : "");
    atomic_init(&dd->shot, 0);
    return dd;
}

void
vgpu98_dd_close(vgpu98_dd_t *dd)
{
    if (!dd)
        return;
    /* 86Box is closing: the capture is not written, and the layer is not
       asked (a GPU thread that hangs may hold its lock). */
    dd_free_capture(dd->cap);
    dd->cap  = NULL;
    dd->disp = NULL;
    if (dd->open)
        dd_end(dd, 0);
    if (dd->started)
        dd_say("%u transition%s; %u stretched frame%s in %u of them; %u frame%s ended by the card; %u capture%s",
               dd->transitions, (dd->transitions == 1) ? "" : "s", dd->stretched_all,
               (dd->stretched_all == 1) ? "" : "s", dd->with_stretched, dd->ended_all, (dd->ended_all == 1) ? "" : "s",
               dd->captures, (dd->captures == 1) ? "" : "s");
    free(dd);
}
