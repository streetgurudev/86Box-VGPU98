/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the display on the host's GPU. While the display
 *          engine is on, the screen is composed on the 3D renderer's
 *          Vulkan device instead of being rebuilt line by line on the
 *          emulation thread:
 *
 *          - a GPU-side mirror of video memory, into which only the pages
 *            of the visible frame that changed since they were last
 *            uploaded are copied (their versions are the card's per-page
 *            write counters);
 *          - a compute shader (vgpu98/display.comp) that turns the frame
 *            into 86Box's 32-bit pixels exactly as its own renderers do;
 *          - a layer: GPU-newer pixels the 3D renderer publishes straight
 *            from its images (vgpu98_r3d_publish), at their video memory
 *            offsets, with the rectangles that hold them; the shader
 *            takes those pixels from the layer, the rest from the mirror;
 *          - the rows that changed, read back for the device;
 *          - the hardware cursor's pattern, which the shader lays over the
 *            frame;
 *          - the gamma table, which the shader applies to the direct
 *            colour modes' pixels.
 *
 *          At the start of every vertical blank the device (vid_vgpu98.c,
 *          the emulation thread) copies the rows of the frame composed
 *          since into the monitor's frame buffer, where 86Box's VGA core
 *          blits them as it blits every card's frame (it renders no lines
 *          meanwhile), and posts the next frame. This module's thread
 *          composes; it never touches 86Box's frame buffer or its blit.
 *          Queue access is shared with the 3D renderer under the device's
 *          queue lock.
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
#include <86box/plat.h>
#include <86box/thread.h>
#include <86box/video.h>
#include <86box/vid_vgpu98_card.h>
#include <86box/vid_vgpu98_gpu.h>
#include <86box/vid_vgpu98_dev.h>

#ifndef VGPU98_NO_VULKAN

#    include <86box/vid_vgpu98_vk.h>
#    include "vgpu98_display.h"
#    include "vgpu98_glide_show.h"

#    define OUT_MAX_W   2048 /* 86Box's frame buffer */
#    define OUT_MAX_H   2048
#    define FRAME_BYTES ((VkDeviceSize) OUT_MAX_W * OUT_MAX_H * 4)
#    define PAGE        4096u
#    define LAYER_RECTS 32 /* display.comp, Rects */
#    define DISP_BINDINGS 7 /* display.comp's buffers: Mirror to Gamma */
#    define CHANGES     256 /* layer changes remembered between frames; more redraw it all */

/* display.comp's Rect, and the Rects block. */
typedef struct gpu_rect_t {
    uint32_t offset, pitch, bytes, x0, y0, x1, y1, pad;
} gpu_rect_t;

typedef struct gpu_rects_t {
    uint32_t   n, pad[3];
    gpu_rect_t r[LAYER_RECTS];
} gpu_rects_t;

/* Video memory bytes a layer rectangle covers: [lo, hi). */
typedef struct span_t {
    uint32_t lo, hi;
} span_t;

/* glide_show.comp's push constants (Show). */
typedef struct glide_push_t {
    uint32_t width;
    uint32_t height;
    uint32_t pitch;
    uint32_t out_w;
    uint32_t out_h;
    uint32_t out_pitch;
    uint32_t blank;
} glide_push_t;

#    define GLIDE_FB_BYTES ((VkDeviceSize) OUT_MAX_W * OUT_MAX_H * 2)

/* The compute shader's push constants (display.comp, Frame). */
typedef struct disp_push_t {
    uint32_t start;
    uint32_t pitch;
    uint32_t width;
    uint32_t height;
    uint32_t bpp;
    uint32_t dscan;
    uint32_t blank;
    uint32_t mask;
    uint32_t out_pitch;
    int32_t  cur_x; /* the hardware cursor's pattern's top-left corner, in the mode's pixels */
    int32_t  cur_y;
    uint32_t cur_on;
    uint32_t gamma_on; /* the gamma table applies */
} disp_push_t;

struct vgpu98_disp_t {
    vgpu98_gpu_t        *gpu;
    vgpu98_vk_t         *vk;
    const vgpu98_vram_t *vram;
    uint32_t             pages;
    volatile int         ok; /* 0 once the host GPU failed: the device shows frames itself again */

    thread_t *thread;
    event_t  *wake;
    event_t  *exited; /* the thread's end (its handle may close as it ends) */
    mutex_t  *lock; /* the request, the composed frame's hand-over, started, quit */
    int       quit;
    int       started;
    int       pending;
    int       restart; /* every page uploaded and every row written again */
    uint32_t  gen;     /* changes at every start and stop: older work is dropped */
    uint32_t  req_gen;
    vgpu98_frame_t req;
    vgpu98_frame_t work; /* the request being composed (display thread) */
    /* A composed frame waiting for the device: readback and row_new hold it,
       and the display thread leaves them alone until the device took it. */
    int            ready;

    /* The display thread's own */
    vgpu98_frame_t cur; /* the frame last composed */
    int            have_cur;
    uint32_t      *page_ver;   /* the mirror's version of each page */
    uint8_t       *page_valid;
    uint8_t       *page_new;   /* uploaded for this frame */
    uint8_t       *row_new;    /* output rows to write */
    VkBufferCopy  *copies;

    VkCommandPool         pool;
    VkCommandBuffer       cmd;
    VkFence               fence;
    VkShaderModule        cs;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      layout;
    VkPipeline            pipe;
    VkDescriptorPool      dpool;
    VkDescriptorSet       set;
    vgpu98_vk_buffer_t    mirror;   /* video memory, on the host GPU */
    vgpu98_vk_buffer_t    out;      /* the composed frame, on the host GPU */
    vgpu98_vk_buffer_t    pal;      /* host-visible */
    vgpu98_vk_buffer_t    staging;  /* host-visible: pages on their way up */
    vgpu98_vk_buffer_t    readback; /* host-visible: rows on their way down */
    vgpu98_vk_buffer_t    layer;    /* GPU-newer pixels at their video memory offsets, on the host GPU */
    vgpu98_vk_buffer_t    rects;    /* host-visible: the layer's rectangles, for the shader */
    vgpu98_vk_buffer_t    cursor;   /* host-visible: the hardware cursor's pattern, for the shader */
    vgpu98_vk_buffer_t    gamma;    /* host-visible: the gamma table, for the shader */

    /* The hardware cursor's pattern: the device's
       copy of the latest (under lock), and the one in the buffer (the
       display thread's), by their serials. */
    uint8_t              *req_cursor;
    uint32_t              req_cursor_serial;
    uint32_t              cursor_loaded;
    /* The gamma table likewise. */
    uint32_t              req_gamma[VGPU98_GAMMA_ENTRIES];
    uint32_t              req_gamma_serial;
    uint32_t              gamma_loaded;

    /* The layer's rectangles and the bytes that changed since the last
       frame (under layer_lock) */
    mutex_t             *layer_lock;
    vgpu98_layer_rect_t  rect[LAYER_RECTS];
    uint32_t             n_rect;
    span_t               change[CHANGES];
    uint32_t             n_change;
    int                  change_all;

    /* The Glide screen, under layer_lock:
       while glide_on, frames are composed from the Glide device's front
       buffer instead of video memory. */
    int                   glide_ok; /* its objects were made: without them Glide's frames go to video memory */
    int                   glide_on;
    uint32_t              glide_w;
    uint32_t              glide_h;
    uint32_t              glide_serial; /* changes with every frame and gamma table */
    uint32_t              glide_shown;  /* the serial last composed (display thread) */
    vgpu98_vk_buffer_t    glide_fb;     /* 16-bit pixels, rows of glide_w, on the host GPU */
    vgpu98_vk_buffer_t    glide_gamma;  /* host-visible: Glide's gamma table */
    VkShaderModule        glide_cs;
    VkDescriptorSetLayout glide_dsl;
    VkPipelineLayout      glide_layout;
    VkPipeline            glide_pipe;
    VkDescriptorPool      glide_dpool;
    VkDescriptorSet       glide_set;

    vgpu98_disp_stats_t stats;

    /* The display thread's step for the watchdog (VGPU98_DSTEP_*), since
       when, and in a wait for the host GPU the seconds without its work
       done and when that was last checked; closing gives such a wait up. */
    atomic_int  step;
    atomic_uint step_since;
    atomic_uint wait_s;
    atomic_uint wait_at;
    atomic_int  closing;
};

static span_t
rect_span(const vgpu98_layer_rect_t *q)
{
    span_t s;

    s.lo = q->offset + q->y0 * q->pitch + q->x0 * q->bytes;
    s.hi = q->offset + (q->y1 - 1) * q->pitch + q->x1 * q->bytes;
    return s;
}

/* Under layer_lock. */
static void
change_add(vgpu98_disp_t *d, const vgpu98_layer_rect_t *q)
{
    if (d->n_change < CHANGES)
        d->change[d->n_change++] = rect_span(q);
    else
        d->change_all = 1;
}

/* The display thread's step, for the watchdog. */
static void
dstep(vgpu98_disp_t *d, int step)
{
    atomic_store_explicit(&d->wait_s, 0, memory_order_relaxed);
    atomic_store_explicit(&d->step_since, plat_get_ticks(), memory_order_relaxed);
    atomic_store_explicit(&d->step, step, memory_order_release);
}

/* A second more of waiting for the host GPU (vgpu98_vk_wait_fence's
   `second`): given up when 86Box closes. */
static int
disp_waited(void *priv, uint32_t seconds)
{
    vgpu98_disp_t *d = (vgpu98_disp_t *) priv;

    atomic_store_explicit(&d->wait_at, plat_get_ticks(), memory_order_relaxed);
    atomic_store_explicit(&d->wait_s, seconds, memory_order_release);
    return atomic_load(&d->closing);
}

static void
disp_fail(vgpu98_disp_t *d, const char *what, VkResult res)
{
    if (d->ok)
        vgpu98_gpu_log(d->gpu, VGPU98_LOG_EVENTS,
                       "VGPU98 display: %s failed (VkResult %d); the display engine's frames are drawn by 86Box again\n",
                       what, (int) res);
    d->ok = 0;
}

static int
same_geometry(const vgpu98_frame_t *a, const vgpu98_frame_t *b)
{
    return (a->start == b->start) && (a->pitch == b->pitch) && (a->width == b->width) && (a->height == b->height) &&
           (a->bpp == b->bpp) && (a->dscan == b->dscan) && (a->blank == b->blank) && (a->x == b->x) && (a->y == b->y);
}

/* Uploads the pages of the frame that changed since they were last
   uploaded. Returns the number of bytes, with d->page_new marking them. */
static VkDeviceSize
upload(vgpu98_disp_t *d, const vgpu98_frame_t *f, uint32_t *n_copies)
{
    const uint32_t bytes_px = (f->bpp + 7) / 8;
    const uint64_t last     = (uint64_t) f->start + (uint64_t) (f->height - 1) * f->pitch + (uint64_t) f->width * bytes_px - 1;
    const uint32_t first_p  = f->start / PAGE;
    const uint32_t count    = (uint32_t) (last / PAGE) - first_p + 1;
    uint8_t       *stage    = (uint8_t *) d->staging.map;
    VkDeviceSize   used     = 0;
    uint32_t       i;

    *n_copies = 0;
    for (i = 0; (i < count) && (i < d->pages); i++)
        d->page_new[(first_p + i) % d->pages] = 0;
    for (i = 0; (i < count) && (i < d->pages); i++) {
        const uint32_t p = (first_p + i) % d->pages; /* the display wraps at the end of video memory */
        const uint32_t v = vgpu98_page_version(d->vram, p);

        if (d->page_valid[p] && (d->page_ver[p] == v))
            continue;
        /* The version before the bytes (writers store the byte first): a
           write during the copy changes it again, and the page goes up
           again next frame. */
        atomic_thread_fence(memory_order_acquire);
        if (used + PAGE > d->staging.size)
            break; /* the rest next frame */
        d->page_ver[p]   = v;
        d->page_valid[p] = 1;
        d->page_new[p]   = 1;
        memcpy(stage + used, d->vram->data + (size_t) p * PAGE, PAGE);
        if (*n_copies && (d->copies[*n_copies - 1].srcOffset + d->copies[*n_copies - 1].size == used) &&
            (d->copies[*n_copies - 1].dstOffset + d->copies[*n_copies - 1].size == (VkDeviceSize) p * PAGE))
            d->copies[*n_copies - 1].size += PAGE;
        else {
            d->copies[*n_copies].srcOffset = used;
            d->copies[*n_copies].dstOffset = (VkDeviceSize) p * PAGE;
            d->copies[*n_copies].size      = PAGE;
            (*n_copies)++;
        }
        used += PAGE;
    }
    return used;
}

/* Marks the output rows whose source pages were uploaded; returns how many. */
static uint32_t
rows_changed(vgpu98_disp_t *d, const vgpu98_frame_t *f)
{
    const uint32_t bytes_px = (f->bpp + 7) / 8;
    uint32_t       n        = 0;
    uint32_t       y;

    for (y = 0; y < f->height; y++) {
        const uint64_t a  = (uint64_t) f->start + (uint64_t) y * f->pitch;
        const uint32_t p0 = (uint32_t) (a / PAGE);
        const uint32_t p1 = (uint32_t) ((a + (uint64_t) f->width * bytes_px - 1) / PAGE);
        uint32_t       p;
        int            hit = 0;

        for (p = p0; (p <= p1) && !hit; p++)
            hit = d->page_new[p % d->pages];
        memset(d->row_new + y * f->dscan, hit, f->dscan);
        n += hit;
    }
    return n;
}

/* Marks the rows over the layer's changed bytes (under layer_lock); returns
   how many rows that were not marked already. */
static uint32_t
rows_layer(vgpu98_disp_t *d, const vgpu98_frame_t *f)
{
    const uint64_t vs = f->start;
    const uint64_t ve = (uint64_t) f->start + (uint64_t) f->height * f->pitch;
    uint32_t       n  = 0;
    uint32_t       i;
    uint64_t       shift;

    /* Each change wherever the frame shows it: past the end of video memory
       the frame wraps to its start, as often as it is longer. */
    for (i = 0; i < d->n_change; i++) {
        for (shift = 0; d->change[i].lo + shift < ve; shift += d->vram->size) {
            uint64_t lo = d->change[i].lo + shift;
            uint64_t hi = d->change[i].hi + shift;
            uint32_t y0;
            uint32_t y1;
            uint32_t y;

            if (hi <= vs)
                continue;
            if (lo < vs)
                lo = vs;
            if (hi > ve)
                hi = ve;
            y0 = (uint32_t) ((lo - vs) / f->pitch);
            y1 = (uint32_t) ((hi - 1 - vs) / f->pitch);
            for (y = y0; (y <= y1) && (y < f->height); y++) {
                if (!d->row_new[y * f->dscan]) {
                    memset(d->row_new + y * f->dscan, 1, f->dscan);
                    n++;
                }
            }
        }
    }
    return n;
}

/* Marks the output rows of the hardware cursor's rows from y0 that were not
   marked already; returns how many. */
static uint32_t
rows_cursor(vgpu98_disp_t *d, const vgpu98_frame_t *f, int32_t y0)
{
    uint32_t n = 0;
    int32_t  y;

    for (y = (y0 < 0) ? 0 : y0; (y < y0 + VGPU98_CURSOR_SIZE) && (y < (int32_t) f->height); y++) {
        if (!d->row_new[y * f->dscan]) {
            memset(d->row_new + y * f->dscan, 1, f->dscan);
            n++;
        }
    }
    return n;
}

/* Composes the frame on the host GPU and reads back the rows marked in
   d->row_new. */
static int
compose(vgpu98_disp_t *d, const vgpu98_frame_t *f, uint32_t n_copies)
{
    vgpu98_vk_t             *vk    = d->vk;
    const uint32_t           out_w = f->width * f->dscan;
    const uint32_t           out_h = f->height * f->dscan;
    VkCommandBufferBeginInfo bi    = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VkSubmitInfo             si    = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkMemoryBarrier          mb    = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    VkBufferCopy             rows[64];
    disp_push_t              pc;
    uint32_t                 n_rows = 0;
    uint32_t                 y;
    VkResult                 res;

    memcpy(d->pal.map, f->palette, sizeof(f->palette));
    {
        gpu_rects_t *gr = (gpu_rects_t *) d->rects.map;
        uint32_t     i;

        gr->n = d->n_rect;
        for (i = 0; i < d->n_rect; i++) {
            gr->r[i].offset = d->rect[i].offset;
            gr->r[i].pitch  = d->rect[i].pitch;
            gr->r[i].bytes  = d->rect[i].bytes;
            gr->r[i].x0     = d->rect[i].x0;
            gr->r[i].y0     = d->rect[i].y0;
            gr->r[i].x1     = d->rect[i].x1;
            gr->r[i].y1     = d->rect[i].y1;
            gr->r[i].pad    = 0;
        }
    }
    vk->vkResetCommandBuffer(d->cmd, 0);
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((res = vk->vkBeginCommandBuffer(d->cmd, &bi)) != VK_SUCCESS) {
        disp_fail(d, "vkBeginCommandBuffer", res);
        return 0;
    }
    if (n_copies)
        vk->vkCmdCopyBuffer(d->cmd, d->staging.buffer, d->mirror.buffer, n_copies, d->copies);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vk->vkCmdPipelineBarrier(d->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    pc.start     = f->start;
    pc.pitch     = f->pitch;
    pc.width     = f->width;
    pc.height    = f->height;
    pc.bpp       = f->bpp;
    pc.dscan     = f->dscan;
    pc.blank     = f->blank;
    pc.mask      = d->vram->size - 1;
    pc.out_pitch = out_w;
    pc.cur_x     = f->cur_x;
    pc.cur_y     = f->cur_y;
    pc.cur_on    = f->cur_on ? 1 : 0;
    pc.gamma_on  = (f->gamma_on && (f->bpp != 8)) ? 1 : 0;
    vk->vkCmdBindPipeline(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipe);
    vk->vkCmdBindDescriptorSets(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->layout, 0, 1, &d->set, 0, NULL);
    vk->vkCmdPushConstants(d->cmd, d->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk->vkCmdDispatch(d->cmd, (f->width + 15) / 16, (f->height + 15) / 16, 1);

    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vk->vkCmdPipelineBarrier(d->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                             NULL, 0, NULL);
    /* The changed rows, in runs. */
    for (y = 0; y < out_h; y++) {
        const VkDeviceSize off = (VkDeviceSize) y * out_w * 4;

        if (!d->row_new[y])
            continue;
        if (n_rows && (rows[n_rows - 1].srcOffset + rows[n_rows - 1].size == off))
            rows[n_rows - 1].size += (VkDeviceSize) out_w * 4;
        else {
            if (n_rows == 64) {
                vk->vkCmdCopyBuffer(d->cmd, d->out.buffer, d->readback.buffer, n_rows, rows);
                n_rows = 0;
            }
            rows[n_rows].srcOffset = off;
            rows[n_rows].dstOffset = off;
            rows[n_rows].size      = (VkDeviceSize) out_w * 4;
            n_rows++;
        }
    }
    if (n_rows)
        vk->vkCmdCopyBuffer(d->cmd, d->out.buffer, d->readback.buffer, n_rows, rows);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vk->vkCmdPipelineBarrier(d->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0,
                             NULL);
    if ((res = vk->vkEndCommandBuffer(d->cmd)) != VK_SUCCESS) {
        disp_fail(d, "vkEndCommandBuffer", res);
        return 0;
    }
    vk->vkResetFences(vk->dev, 1, &d->fence);
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &d->cmd;
    dstep(d, VGPU98_DSTEP_SUBMIT);
    res = vgpu98_vk_submit(vk, &si, d->fence);
    dstep(d, VGPU98_DSTEP_COMPOSE);
    if (res != VK_SUCCESS) {
        disp_fail(d, "vkQueueSubmit", res);
        return 0;
    }
    dstep(d, VGPU98_DSTEP_HOST_GPU);
    res = vgpu98_vk_wait_fence(vk, d->fence, disp_waited, d);
    dstep(d, VGPU98_DSTEP_COMPOSE);
    if (res == VK_TIMEOUT) {
        /* Given up: 86Box closes. The fence may still be in use, so the
           display composes no more. */
        d->ok = 0;
        return 0;
    }
    if (res != VK_SUCCESS) {
        disp_fail(d, "vkWaitForFences", res);
        return 0;
    }
    return 1;
}

/* The Glide screen (under layer_lock): the Glide device's front buffer
   through glide_show.comp, every row read back. */
static int
compose_glide(vgpu98_disp_t *d, const vgpu98_frame_t *f)
{
    vgpu98_vk_t             *vk    = d->vk;
    const uint32_t           out_w = f->width * f->dscan;
    const uint32_t           out_h = f->height * f->dscan;
    VkCommandBufferBeginInfo bi    = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VkSubmitInfo             si    = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkMemoryBarrier          mb    = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    VkBufferCopy             all;
    glide_push_t             pc;
    VkResult                 res;

    vk->vkResetCommandBuffer(d->cmd, 0);
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((res = vk->vkBeginCommandBuffer(d->cmd, &bi)) != VK_SUCCESS) {
        disp_fail(d, "vkBeginCommandBuffer", res);
        return 0;
    }
    /* The front buffer was copied in by the GPU thread's submission, and the
       gamma table written by the host, before this. */
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vk->vkCmdPipelineBarrier(d->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    pc.width     = d->glide_w;
    pc.height    = d->glide_h;
    pc.pitch     = d->glide_w;
    pc.out_w     = out_w;
    pc.out_h     = out_h;
    pc.out_pitch = out_w;
    pc.blank     = f->blank;
    vk->vkCmdBindPipeline(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->glide_pipe);
    vk->vkCmdBindDescriptorSets(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->glide_layout, 0, 1, &d->glide_set, 0, NULL);
    vk->vkCmdPushConstants(d->cmd, d->glide_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk->vkCmdDispatch(d->cmd, (out_w + 15) / 16, (out_h + 15) / 16, 1);
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vk->vkCmdPipelineBarrier(d->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                             NULL, 0, NULL);
    all.srcOffset = 0;
    all.dstOffset = 0;
    all.size      = (VkDeviceSize) out_w * out_h * 4;
    vk->vkCmdCopyBuffer(d->cmd, d->out.buffer, d->readback.buffer, 1, &all);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vk->vkCmdPipelineBarrier(d->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0,
                             NULL);
    if ((res = vk->vkEndCommandBuffer(d->cmd)) != VK_SUCCESS) {
        disp_fail(d, "vkEndCommandBuffer", res);
        return 0;
    }
    vk->vkResetFences(vk->dev, 1, &d->fence);
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &d->cmd;
    dstep(d, VGPU98_DSTEP_SUBMIT);
    res = vgpu98_vk_submit(vk, &si, d->fence);
    dstep(d, VGPU98_DSTEP_COMPOSE);
    if (res != VK_SUCCESS) {
        disp_fail(d, "vkQueueSubmit", res);
        return 0;
    }
    dstep(d, VGPU98_DSTEP_HOST_GPU);
    res = vgpu98_vk_wait_fence(vk, d->fence, disp_waited, d);
    dstep(d, VGPU98_DSTEP_COMPOSE);
    if (res == VK_TIMEOUT) {
        d->ok = 0;
        return 0;
    }
    if (res != VK_SUCCESS) {
        disp_fail(d, "vkWaitForFences", res);
        return 0;
    }
    return 1;
}

/* One frame: upload, compose, read back. Returns 1 when rows changed: the
   frame is then ready for the device. */
static int
show(vgpu98_disp_t *d, const vgpu98_frame_t *f, int restart)
{
    const uint64_t t0    = vgpu98_diag_us();
    const uint32_t out_w = f->width * f->dscan;
    const uint32_t out_h = f->height * f->dscan;
    int            all;
    uint32_t       n_copies;
    VkDeviceSize   up;
    uint32_t       changed;

    if (!f->width || !f->height || (out_w > OUT_MAX_W) || (out_h > OUT_MAX_H) || (f->x < 0) || (f->y < 0) ||
        ((uint32_t) f->x + out_w > (uint32_t) OUT_MAX_W) || ((uint32_t) f->y + out_h > (uint32_t) OUT_MAX_H))
        return 0;
    if (restart)
        memset(d->page_valid, 0, d->pages);
    /* The layer is not written while it is composed from. */
    dstep(d, VGPU98_DSTEP_LAYER);
    thread_wait_mutex(d->layer_lock);
    dstep(d, VGPU98_DSTEP_COMPOSE);
    if (d->glide_on) {
        /* A Glide session shows: a new frame, gamma table or screen is
           composed whole; video memory waits for the desktop's return. */
        changed = 0;
        if (restart || !d->have_cur || !same_geometry(f, &d->cur) || (f->blank != d->cur.blank) ||
            (d->glide_serial != d->glide_shown)) {
            if (!compose_glide(d, f)) {
                thread_release_mutex(d->layer_lock);
                return 0;
            }
            memset(d->row_new, 1, out_h);
            changed          = out_h;
            d->glide_shown   = d->glide_serial;
            d->stats.frames++;
        }
        thread_release_mutex(d->layer_lock);
        d->cur      = *f;
        d->have_cur = 1;
        d->stats.busy_us += vgpu98_diag_us() - t0;
        return changed != 0;
    }
    /* A new palette (8 bpp) or gamma table (the others) changes every
       pixel. */
    all = restart || d->change_all || !d->have_cur || !same_geometry(f, &d->cur) ||
          ((f->bpp == 8) && memcmp(f->palette, d->cur.palette, sizeof(f->palette))) ||
          ((f->bpp != 8) &&
           ((f->gamma_on != d->cur.gamma_on) || (f->gamma_on && (f->gamma_serial != d->cur.gamma_serial))));
    up  = f->blank ? 0 : upload(d, f, &n_copies);
    if (f->blank)
        n_copies = 0;
    if (all) {
        memset(d->row_new, 1, out_h);
        changed = out_h;
    } else {
        memset(d->row_new, 0, out_h);
        changed = up ? rows_changed(d, f) : 0;
        changed += rows_layer(d, f);
        /* The hardware cursor's rows, where it was and where it is. */
        if ((f->cur_on != d->cur.cur_on) ||
            (f->cur_on && ((f->cur_x != d->cur.cur_x) || (f->cur_y != d->cur.cur_y) || (f->cur_serial != d->cur.cur_serial)))) {
            if (d->cur.cur_on)
                changed += rows_cursor(d, f, d->cur.cur_y);
            if (f->cur_on)
                changed += rows_cursor(d, f, f->cur_y);
        }
    }
    d->n_change   = 0;
    d->change_all = 0;
    /* Pages staged count as uploaded: they go up even when no row shows
       them yet (a virtual screen panned later). */
    if (changed || n_copies) {
        if (!compose(d, f, n_copies)) {
            thread_release_mutex(d->layer_lock);
            return 0;
        }
        d->stats.frames++;
        d->stats.upload_bytes += up;
    }
    thread_release_mutex(d->layer_lock);
    d->cur      = *f;
    d->have_cur = 1;
    d->stats.busy_us += vgpu98_diag_us() - t0;
    return changed != 0;
}

static void
disp_thread(void *priv)
{
    vgpu98_disp_t *d = (vgpu98_disp_t *) priv;

    for (;;) {
        uint32_t gen;
        int      restart;
        int      changed;

        dstep(d, VGPU98_DSTEP_FRAME);
        thread_wait_event(d->wake, -1);
        thread_reset_event(d->wake);
        thread_wait_mutex(d->lock);
        if (d->quit) {
            thread_release_mutex(d->lock);
            break;
        }
        /* A frame the device has not taken yet holds the readback. */
        if (!d->pending || !d->started || !d->ok || d->ready) {
            thread_release_mutex(d->lock);
            continue;
        }
        d->pending = 0;
        restart    = d->restart;
        d->restart = 0;
        gen        = d->req_gen;
        d->work    = d->req; /* the device may post the next one meanwhile */
        /* The cursor's new pattern into the shader's buffer: the last
           frame's work on the host GPU is done. */
        if (d->req.cur_serial != d->cursor_loaded) {
            memcpy(d->cursor.map, d->req_cursor, VGPU98_CURSOR_BYTES);
            d->cursor_loaded = d->req.cur_serial;
        }
        if (d->req.gamma_serial != d->gamma_loaded) {
            memcpy(d->gamma.map, d->req_gamma, sizeof(d->req_gamma));
            d->gamma_loaded = d->req.gamma_serial;
        }
        thread_release_mutex(d->lock);
        changed = show(d, &d->work, restart);
        thread_wait_mutex(d->lock);
        if (changed && d->started && (gen == d->gen))
            d->ready = 1;
        else if (restart && (gen == d->gen))
            d->restart = 1; /* not composed: the next frame starts afresh */
        thread_release_mutex(d->lock);
    }
    thread_set_event(d->exited);
}

/* --- Interface (emulation thread) ---------------------------------------- */

int
vgpu98_disp_ok(const vgpu98_disp_t *d)
{
    return d && d->ok;
}

void
vgpu98_disp_start(vgpu98_disp_t *d)
{
    thread_wait_mutex(d->lock);
    d->started = 1;
    d->restart = 1;
    d->pending = 0;
    d->ready   = 0;
    d->gen++;
    thread_release_mutex(d->lock);
}

void
vgpu98_disp_stop(vgpu98_disp_t *d)
{
    thread_wait_mutex(d->lock);
    d->started = 0;
    d->pending = 0;
    d->ready   = 0;
    d->gen++;
    thread_release_mutex(d->lock);
}

int
vgpu98_disp_take(vgpu98_disp_t *d, bitmap_t *tb, const vgpu98_frame_t *now, vgpu98_frame_t *taken)
{
    const vgpu98_frame_t *f;
    const uint8_t        *src = (const uint8_t *) d->readback.map;
    const int             x   = now->x;
    const int             y   = now->y;
    uint32_t              out_w;
    uint32_t              out_h;
    uint32_t              r;
    int                   ready;
    int                   other;

    thread_wait_mutex(d->lock);
    ready = d->ready && d->started;
    thread_release_mutex(d->lock);
    if (!ready)
        return 0;
    /* The display thread leaves the readback, the rows and the composed
       frame alone while ready is set. */
    f     = &d->cur;
    if (taken)
        *taken = *f;
    out_w = f->width * f->dscan;
    out_h = f->height * f->dscan;
    /* A picture of the mode before, or of the frame's place before (the
       overscan changed), is held back;
       the next frame is composed whole, its geometry being
       new. */
    other = (f->width != now->width) || (f->height != now->height) || (f->bpp != now->bpp) ||
            (f->pitch != now->pitch) || (f->dscan != now->dscan) || (f->x != x) || (f->y != y);
    if (!other && tb && ((uint32_t) y + out_h <= (uint32_t) tb->h) && ((uint32_t) x + out_w <= (uint32_t) tb->w)) {
        for (r = 0; r < out_h; r++) {
            if (!d->row_new[r])
                continue;
            memcpy(&tb->line[y + (int) r][x], src + (size_t) r * out_w * 4, (size_t) out_w * 4);
            d->stats.read_bytes += (uint64_t) out_w * 4;
        }
    } else if (!other) {
        /* No frame buffer, or the frame does not fit it: the next frame is
           composed again, every row of it. */
        thread_wait_mutex(d->lock);
        d->restart = 1;
        thread_release_mutex(d->lock);
    }
    thread_wait_mutex(d->lock);
    d->ready = 0;
    thread_release_mutex(d->lock);
    thread_set_event(d->wake);
    return other ? -1 : 1;
}

void
vgpu98_disp_frame(vgpu98_disp_t *d, const vgpu98_frame_t *f, const uint8_t *cursor, const uint32_t *gamma)
{
    thread_wait_mutex(d->lock);
    if (d->started) {
        d->req     = *f;
        d->req_gen = d->gen;
        d->pending = 1;
        if (f->cur_serial != d->req_cursor_serial) {
            memcpy(d->req_cursor, cursor, VGPU98_CURSOR_BYTES);
            d->req_cursor_serial = f->cur_serial;
        }
        if (f->gamma_serial != d->req_gamma_serial) {
            memcpy(d->req_gamma, gamma, sizeof(d->req_gamma));
            d->req_gamma_serial = f->gamma_serial;
        }
    }
    thread_release_mutex(d->lock);
    thread_set_event(d->wake);
}

void
vgpu98_disp_stats(const vgpu98_disp_t *d, vgpu98_disp_stats_t *s)
{
    if (d)
        *s = d->stats;
    else
        memset(s, 0, sizeof(*s));
}

int
vgpu98_disp_watch(const vgpu98_disp_t *d, uint32_t *since, uint32_t *seconds, uint32_t *checked)
{
    vgpu98_disp_t *w = (vgpu98_disp_t *) d;
    int            step;

    if (!w) {
        *since = *seconds = *checked = 0;
        return VGPU98_DSTEP_FRAME;
    }
    step     = atomic_load_explicit(&w->step, memory_order_acquire);
    *since   = atomic_load_explicit(&w->step_since, memory_order_relaxed);
    *seconds = atomic_load_explicit(&w->wait_s, memory_order_acquire);
    *checked = atomic_load_explicit(&w->wait_at, memory_order_relaxed);
    return step;
}

void
vgpu98_disp_closing(vgpu98_disp_t *d)
{
    if (d)
        atomic_store(&d->closing, 1);
}

int
vgpu98_disp_showing(const vgpu98_disp_t *d)
{
    return d && d->ok && *(volatile const int *) &d->started;
}

/* --- The layer (the command processor's thread) ---------------------------- */

void
vgpu98_disp_layer_lock(vgpu98_disp_t *d)
{
    thread_wait_mutex(d->layer_lock);
}

void
vgpu98_disp_layer_unlock(vgpu98_disp_t *d)
{
    thread_release_mutex(d->layer_lock);
}

uint64_t
vgpu98_disp_layer_buffer(vgpu98_disp_t *d)
{
    return (uint64_t) (uintptr_t) d->layer.buffer;
}

int
vgpu98_disp_layer_set(vgpu98_disp_t *d, const vgpu98_layer_rect_t *q)
{
    uint32_t i;

    for (i = 0; i < d->n_rect; i++) {
        if (d->rect[i].id == q->id) {
            if (memcmp(&d->rect[i], q, sizeof(*q)))
                change_add(d, &d->rect[i]); /* where it was, unless it is where it is */
            d->rect[i] = *q;
            change_add(d, q);
            return 1;
        }
    }
    if (d->n_rect == LAYER_RECTS)
        return 0;
    d->rect[d->n_rect++] = *q;
    change_add(d, q);
    return 1;
}

void
vgpu98_disp_layer_drop(vgpu98_disp_t *d, uint32_t id)
{
    uint32_t i;
    int      step;

    /* The command processor's thread (the renderer's): its wait is named. */
    step = vgpu98_gpu_step(d->gpu, VGPU98_STEP_LAYER);
    thread_wait_mutex(d->layer_lock);
    vgpu98_gpu_step(d->gpu, step);
    for (i = 0; i < d->n_rect; i++) {
        if (d->rect[i].id == id) {
            change_add(d, &d->rect[i]);
            d->rect[i] = d->rect[--d->n_rect];
            break;
        }
    }
    thread_release_mutex(d->layer_lock);
}

uint32_t
vgpu98_disp_layer_over(vgpu98_disp_t *d, uint32_t start, uint32_t len, uint32_t bytes)
{
    const uint64_t end = (uint64_t) start + len;
    uint32_t       n   = 0;
    uint32_t       i;

    thread_wait_mutex(d->layer_lock);
    for (i = 0; i < d->n_rect; i++) {
        const span_t s = rect_span(&d->rect[i]);

        if ((d->rect[i].bytes == bytes) && (s.lo < end) && (s.hi > start))
            n++;
    }
    thread_release_mutex(d->layer_lock);
    return n;
}

/* --- The Glide screen (the command processor's thread) ----------------------- */

uint64_t
vgpu98_disp_glide_buffer(vgpu98_disp_t *d)
{
    return (d && d->glide_ok) ? (uint64_t) (uintptr_t) d->glide_fb.buffer : 0;
}

void
vgpu98_disp_glide_frame(vgpu98_disp_t *d, uint32_t width, uint32_t height, const uint32_t *gamma)
{
    d->glide_on = 1;
    d->glide_w  = width;
    d->glide_h  = height;
    memcpy(d->glide_gamma.map, gamma, 256 * sizeof(uint32_t));
    d->glide_serial++;
}

void
vgpu98_disp_glide_end(vgpu98_disp_t *d)
{
    int step;

    if (!d)
        return;
    step = vgpu98_gpu_step(d->gpu, VGPU98_STEP_LAYER);
    thread_wait_mutex(d->layer_lock);
    vgpu98_gpu_step(d->gpu, step);
    if (d->glide_on) {
        /* The desktop back: video memory's frame again, every row of it. */
        d->glide_on   = 0;
        d->change_all = 1;
    }
    thread_release_mutex(d->layer_lock);
}

int
vgpu98_disp_glide_on(const vgpu98_disp_t *d)
{
    return d && *(volatile const int *) &d->glide_on;
}

/* glide_show.comp's pipeline: the Glide frame, the gamma table, the output. */
static int
glide_objects(vgpu98_disp_t *d)
{
    vgpu98_vk_t                    *vk = d->vk;
    VkShaderModuleCreateInfo        smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkDescriptorSetLayoutBinding    b[3];
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPushConstantRange             pcr;
    VkPipelineLayoutCreateInfo      pli  = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkComputePipelineCreateInfo     cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    VkDescriptorPoolSize            ps;
    VkDescriptorPoolCreateInfo      dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    VkDescriptorSetAllocateInfo     dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorBufferInfo          dbi[3];
    VkWriteDescriptorSet            w[3];
    uint32_t                       *lut;
    int                             i;

    if (!vgpu98_vk_buffer_create_local(vk, &d->glide_fb, GLIDE_FB_BYTES,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT) ||
        !vgpu98_vk_buffer_create(vk, &d->glide_gamma, 256 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
        return 0;
    /* No gamma until a session sets its own. */
    lut = (uint32_t *) d->glide_gamma.map;
    for (i = 0; i < 256; i++)
        lut[i] = (uint32_t) i * 0x010101u;
    smi.codeSize = sizeof(vgpu98_glide_show);
    smi.pCode    = vgpu98_glide_show;
    if (vk->vkCreateShaderModule(vk->dev, &smi, NULL, &d->glide_cs) != VK_SUCCESS)
        return 0;
    memset(b, 0, sizeof(b));
    for (i = 0; i < 3; i++) {
        b[i].binding         = (uint32_t) i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    dli.bindingCount = 3;
    dli.pBindings    = b;
    if (vk->vkCreateDescriptorSetLayout(vk->dev, &dli, NULL, &d->glide_dsl) != VK_SUCCESS)
        return 0;
    pcr.stageFlags             = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset                 = 0;
    pcr.size                   = sizeof(glide_push_t);
    pli.setLayoutCount         = 1;
    pli.pSetLayouts            = &d->glide_dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges    = &pcr;
    if (vk->vkCreatePipelineLayout(vk->dev, &pli, NULL, &d->glide_layout) != VK_SUCCESS)
        return 0;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = d->glide_cs;
    cpci.stage.pName  = "main";
    cpci.layout       = d->glide_layout;
    if (vk->vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &d->glide_pipe) != VK_SUCCESS)
        return 0;
    ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 3;
    dpi.maxSets        = 1;
    dpi.poolSizeCount  = 1;
    dpi.pPoolSizes     = &ps;
    if (vk->vkCreateDescriptorPool(vk->dev, &dpi, NULL, &d->glide_dpool) != VK_SUCCESS)
        return 0;
    dai.descriptorPool     = d->glide_dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts        = &d->glide_dsl;
    if (vk->vkAllocateDescriptorSets(vk->dev, &dai, &d->glide_set) != VK_SUCCESS)
        return 0;
    dbi[0].buffer = d->glide_fb.buffer;
    dbi[1].buffer = d->glide_gamma.buffer;
    dbi[2].buffer = d->out.buffer;
    memset(w, 0, sizeof(w));
    for (i = 0; i < 3; i++) {
        dbi[i].offset        = 0;
        dbi[i].range         = VK_WHOLE_SIZE;
        w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet          = d->glide_set;
        w[i].dstBinding      = (uint32_t) i;
        w[i].descriptorCount = 1;
        w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo     = &dbi[i];
    }
    vk->vkUpdateDescriptorSets(vk->dev, 3, w, 0, NULL);
    return 1;
}

/* --- Start-up ---------------------------------------------------------------- */

static int
disp_objects(vgpu98_disp_t *d)
{
    vgpu98_vk_t                 *vk = d->vk;
    VkCommandPoolCreateInfo      cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    VkCommandBufferAllocateInfo  cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    VkFenceCreateInfo            fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkShaderModuleCreateInfo     smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkDescriptorSetLayoutBinding b[DISP_BINDINGS];
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPushConstantRange          pcr;
    VkPipelineLayoutCreateInfo   pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkComputePipelineCreateInfo  cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    VkDescriptorPoolSize         ps;
    VkDescriptorPoolCreateInfo   dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    VkDescriptorSetAllocateInfo  dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorBufferInfo       dbi[DISP_BINDINGS];
    VkWriteDescriptorSet         w[DISP_BINDINGS];
    int                          i;

    cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = vk->queue_family;
    if (vk->vkCreateCommandPool(vk->dev, &cpi, NULL, &d->pool) != VK_SUCCESS)
        return 0;
    cai.commandPool        = d->pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vk->vkAllocateCommandBuffers(vk->dev, &cai, &d->cmd) != VK_SUCCESS)
        return 0;
    if (vk->vkCreateFence(vk->dev, &fci, NULL, &d->fence) != VK_SUCCESS)
        return 0;

    if (!vgpu98_vk_buffer_create_local(vk, &d->mirror, d->vram->size,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT) ||
        !vgpu98_vk_buffer_create_local(vk, &d->out, FRAME_BYTES,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
        !vgpu98_vk_buffer_create(vk, &d->pal, 256 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        !vgpu98_vk_buffer_create(vk, &d->staging, FRAME_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
        !vgpu98_vk_buffer_create(vk, &d->readback, FRAME_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT) ||
        !vgpu98_vk_buffer_create_local(vk, &d->layer, d->vram->size,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT) ||
        !vgpu98_vk_buffer_create(vk, &d->rects, sizeof(gpu_rects_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        !vgpu98_vk_buffer_create(vk, &d->cursor, VGPU98_CURSOR_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        !vgpu98_vk_buffer_create(vk, &d->gamma, VGPU98_GAMMA_ENTRIES * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
        return 0;

    smi.codeSize = sizeof(vgpu98_display);
    smi.pCode    = vgpu98_display;
    if (vk->vkCreateShaderModule(vk->dev, &smi, NULL, &d->cs) != VK_SUCCESS)
        return 0;
    memset(b, 0, sizeof(b));
    for (i = 0; i < DISP_BINDINGS; i++) {
        b[i].binding         = (uint32_t) i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    dli.bindingCount = DISP_BINDINGS;
    dli.pBindings    = b;
    if (vk->vkCreateDescriptorSetLayout(vk->dev, &dli, NULL, &d->dsl) != VK_SUCCESS)
        return 0;
    pcr.stageFlags             = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset                 = 0;
    pcr.size                   = sizeof(disp_push_t);
    pli.setLayoutCount         = 1;
    pli.pSetLayouts            = &d->dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges    = &pcr;
    if (vk->vkCreatePipelineLayout(vk->dev, &pli, NULL, &d->layout) != VK_SUCCESS)
        return 0;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = d->cs;
    cpci.stage.pName  = "main";
    cpci.layout       = d->layout;
    if (vk->vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &d->pipe) != VK_SUCCESS)
        return 0;

    ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = DISP_BINDINGS;
    dpi.maxSets        = 1;
    dpi.poolSizeCount  = 1;
    dpi.pPoolSizes     = &ps;
    if (vk->vkCreateDescriptorPool(vk->dev, &dpi, NULL, &d->dpool) != VK_SUCCESS)
        return 0;
    dai.descriptorPool     = d->dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts        = &d->dsl;
    if (vk->vkAllocateDescriptorSets(vk->dev, &dai, &d->set) != VK_SUCCESS)
        return 0;
    dbi[0].buffer = d->mirror.buffer;
    dbi[1].buffer = d->pal.buffer;
    dbi[2].buffer = d->out.buffer;
    dbi[3].buffer = d->layer.buffer;
    dbi[4].buffer = d->rects.buffer;
    dbi[5].buffer = d->cursor.buffer;
    dbi[6].buffer = d->gamma.buffer;
    memset(w, 0, sizeof(w));
    for (i = 0; i < DISP_BINDINGS; i++) {
        dbi[i].offset          = 0;
        dbi[i].range           = VK_WHOLE_SIZE;
        w[i].sType             = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet            = d->set;
        w[i].dstBinding        = (uint32_t) i;
        w[i].descriptorCount   = 1;
        w[i].descriptorType    = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo       = &dbi[i];
    }
    vk->vkUpdateDescriptorSets(vk->dev, DISP_BINDINGS, w, 0, NULL);
    /* Glide's are its own: without them, DirectX's frames still show. */
    d->glide_ok = glide_objects(d);
    if (!d->glide_ok)
        vgpu98_gpu_log(d->gpu, VGPU98_LOG_EVENTS,
                       "VGPU98 display: no Glide screen on the host GPU; Glide's frames go to video memory\n");
    return 1;
}

static void
disp_free(vgpu98_disp_t *d)
{
    vgpu98_vk_t *vk = d->vk;

    if (vk) {
        vgpu98_vk_wait_idle(vk);
        if (d->glide_dpool)
            vk->vkDestroyDescriptorPool(vk->dev, d->glide_dpool, NULL);
        if (d->glide_pipe)
            vk->vkDestroyPipeline(vk->dev, d->glide_pipe, NULL);
        if (d->glide_layout)
            vk->vkDestroyPipelineLayout(vk->dev, d->glide_layout, NULL);
        if (d->glide_dsl)
            vk->vkDestroyDescriptorSetLayout(vk->dev, d->glide_dsl, NULL);
        if (d->glide_cs)
            vk->vkDestroyShaderModule(vk->dev, d->glide_cs, NULL);
        vgpu98_vk_buffer_destroy(vk, &d->glide_fb);
        vgpu98_vk_buffer_destroy(vk, &d->glide_gamma);
        if (d->dpool)
            vk->vkDestroyDescriptorPool(vk->dev, d->dpool, NULL);
        if (d->pipe)
            vk->vkDestroyPipeline(vk->dev, d->pipe, NULL);
        if (d->layout)
            vk->vkDestroyPipelineLayout(vk->dev, d->layout, NULL);
        if (d->dsl)
            vk->vkDestroyDescriptorSetLayout(vk->dev, d->dsl, NULL);
        if (d->cs)
            vk->vkDestroyShaderModule(vk->dev, d->cs, NULL);
        if (d->fence)
            vk->vkDestroyFence(vk->dev, d->fence, NULL);
        if (d->pool)
            vk->vkDestroyCommandPool(vk->dev, d->pool, NULL);
        vgpu98_vk_buffer_destroy(vk, &d->mirror);
        vgpu98_vk_buffer_destroy(vk, &d->out);
        vgpu98_vk_buffer_destroy(vk, &d->pal);
        vgpu98_vk_buffer_destroy(vk, &d->staging);
        vgpu98_vk_buffer_destroy(vk, &d->readback);
        vgpu98_vk_buffer_destroy(vk, &d->layer);
        vgpu98_vk_buffer_destroy(vk, &d->rects);
        vgpu98_vk_buffer_destroy(vk, &d->cursor);
        vgpu98_vk_buffer_destroy(vk, &d->gamma);
    }
    if (d->layer_lock)
        thread_close_mutex(d->layer_lock);
    free(d->req_cursor);
    free(d->page_ver);
    free(d->page_valid);
    free(d->page_new);
    free(d->row_new);
    free(d->copies);
    free(d);
}

vgpu98_disp_t *
vgpu98_disp_init(vgpu98_gpu_t *g, vgpu98_r3d_t *r, const vgpu98_vram_t *vram)
{
    vgpu98_disp_t *d;

    if (!r || !vgpu98_r3d_vk(r)) {
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 display: no 3D on the host GPU, so 86Box draws the frames\n");
        return NULL;
    }
    d             = calloc(1, sizeof(vgpu98_disp_t));
    d->gpu        = g;
    d->vk         = vgpu98_r3d_vk(r);
    d->vram       = vram;
    d->pages      = vram->size / PAGE;
    d->page_ver   = calloc(d->pages, sizeof(uint32_t));
    d->page_valid = calloc(d->pages, 1);
    d->page_new   = calloc(d->pages, 1);
    d->row_new    = calloc(OUT_MAX_H, 1);
    d->copies     = calloc(d->pages, sizeof(VkBufferCopy));
    d->req_cursor = calloc(1, VGPU98_CURSOR_BYTES);
    d->layer_lock = thread_create_mutex();
    if (!disp_objects(d)) {
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 display: setting up the host GPU failed; 86Box draws the frames\n");
        disp_free(d);
        return NULL;
    }
    d->ok      = 1;
    d->wake    = thread_create_event();
    d->exited  = thread_create_event();
    d->lock    = thread_create_mutex();
    d->thread  = thread_create_named(disp_thread, d, "VGPU98 display");
    vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 display: on the host GPU\n");
    return d;
}

int
vgpu98_disp_close(vgpu98_disp_t *d)
{
    if (!d)
        return 1;
    atomic_store(&d->closing, 1);
    thread_wait_mutex(d->lock);
    d->quit = 1;
    thread_release_mutex(d->lock);
    thread_set_event(d->wake);
    /* A wait for the host GPU gives up within a second of closing. A thread
       that has not ended by the limit keeps what it uses (86Box is closing:
       it is left, not freed under it). */
    if (thread_wait_event(d->exited, 5000))
        return 0;
    thread_destroy_event(d->wake);
    thread_destroy_event(d->exited);
    thread_close_mutex(d->lock);
    disp_free(d);
    return 1;
}

#else /* VGPU98_NO_VULKAN */

vgpu98_disp_t *
vgpu98_disp_init(vgpu98_gpu_t *g, vgpu98_r3d_t *r, const vgpu98_vram_t *vram)
{
    (void) g;
    (void) r;
    (void) vram;
    return NULL;
}

int
vgpu98_disp_close(vgpu98_disp_t *d)
{
    (void) d;
    return 1;
}

int
vgpu98_disp_ok(const vgpu98_disp_t *d)
{
    (void) d;
    return 0;
}

void
vgpu98_disp_start(vgpu98_disp_t *d)
{
    (void) d;
}

void
vgpu98_disp_stop(vgpu98_disp_t *d)
{
    (void) d;
}

int
vgpu98_disp_take(vgpu98_disp_t *d, bitmap_t *tb, const vgpu98_frame_t *now, vgpu98_frame_t *taken)
{
    (void) d;
    (void) tb;
    (void) now;
    (void) taken;
    return 0;
}

void
vgpu98_disp_frame(vgpu98_disp_t *d, const vgpu98_frame_t *f, const uint8_t *cursor, const uint32_t *gamma)
{
    (void) d;
    (void) f;
    (void) cursor;
    (void) gamma;
}

void
vgpu98_disp_stats(const vgpu98_disp_t *d, vgpu98_disp_stats_t *s)
{
    (void) d;
    memset(s, 0, sizeof(*s));
}

int
vgpu98_disp_watch(const vgpu98_disp_t *d, uint32_t *since, uint32_t *seconds, uint32_t *checked)
{
    (void) d;
    *since = *seconds = *checked = 0;
    return VGPU98_DSTEP_FRAME;
}

void
vgpu98_disp_closing(vgpu98_disp_t *d)
{
    (void) d;
}

int
vgpu98_disp_showing(const vgpu98_disp_t *d)
{
    (void) d;
    return 0;
}

void
vgpu98_disp_layer_lock(vgpu98_disp_t *d)
{
    (void) d;
}

void
vgpu98_disp_layer_unlock(vgpu98_disp_t *d)
{
    (void) d;
}

uint64_t
vgpu98_disp_layer_buffer(vgpu98_disp_t *d)
{
    (void) d;
    return 0;
}

int
vgpu98_disp_layer_set(vgpu98_disp_t *d, const vgpu98_layer_rect_t *q)
{
    (void) d;
    (void) q;
    return 0;
}

void
vgpu98_disp_layer_drop(vgpu98_disp_t *d, uint32_t id)
{
    (void) d;
    (void) id;
}

uint32_t
vgpu98_disp_layer_over(vgpu98_disp_t *d, uint32_t start, uint32_t len, uint32_t bytes)
{
    (void) d;
    (void) start;
    (void) len;
    (void) bytes;
    return 0;
}

uint64_t
vgpu98_disp_glide_buffer(vgpu98_disp_t *d)
{
    (void) d;
    return 0;
}

void
vgpu98_disp_glide_frame(vgpu98_disp_t *d, uint32_t width, uint32_t height, const uint32_t *gamma)
{
    (void) d;
    (void) width;
    (void) height;
    (void) gamma;
}

void
vgpu98_disp_glide_end(vgpu98_disp_t *d)
{
    (void) d;
}

int
vgpu98_disp_glide_on(const vgpu98_disp_t *d)
{
    (void) d;
    return 0;
}

#endif
