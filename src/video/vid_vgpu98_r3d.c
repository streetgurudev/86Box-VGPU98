/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          VGPU98 2000: the 3D renderer, drawing the GPU's 3D packets
 *          on the host's GPU through Vulkan.
 *
 *          Surfaces in the card's video memory get host images, keyed
 *          by their description, kept coherent with video memory as
 *          the project's spec/gpu-coherency.md defines: uploaded when
 *          the pages under them change, copied back when something
 *          reads the bytes they hold newer (the guest's CPU through the
 *          page map, a Lock's SYNC, a 2D packet, another image), and
 *          published to the display on the host GPU without leaving it.
 *          Formats follow spec/gpu-formats.md (P8 through its
 *          palette, DXT decoded on the CPU, bump formats as floats, a
 *          depth buffer W-buffered into as a 32-bit float holding W).
 *
 *          Vertices run on the host GPU (command set 2.0's vertex
 *          pipeline): each draw hands vgpu98/vgpu98.vert a copy of the
 *          stream bytes its vertices occupy and the declaration
 *          resolved (vid_vgpu98_vtx.c), and the shader fetches,
 *          transforms and places them. Plain triangles and points use
 *          the host GPU's own topologies and culling, which assemble and
 *          cull as Direct3D does. What needs a whole primitive (lines and
 *          wireframe by Direct3D's line rule, point fill, WRAPn) draws
 *          from a list of the draw's primitives built here, one shader
 *          invocation per output vertex. The fixed-function pixel
 *          pipeline is vgpu98/vgpu98.frag.
 *
 *          Runs on the command processor's thread only.
 *
 * Authors: VGPU98 Project contributors
 *
 *          Copyright 2026 VGPU98 Project contributors.
 */
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <86box/thread.h>
#include <86box/vid_vgpu98_card.h>
#include <86box/vid_vgpu98_gpu.h>
#include <86box/vid_vgpu98_prog.h>
#include <86box/vid_vgpu98_dev.h>

#ifndef VGPU98_NO_VULKAN

#    include <png.h>
#    include <time.h>
#    include <86box/86box.h>
#    include <86box/path.h>
#    include <86box/plat.h>
#    include <86box/vid_vgpu98_vk.h>
#    include "vgpu98_vert.h"
#    include "vgpu98_frag.h"
#    include "vgpu98_frag_depth.h"

#    define SLOTS          3
_Static_assert(SLOTS == VGPU98_R3D_SLOTS, "the Glide device keeps a descriptor pool for each slot");
#    define SETS_PER_SLOT  1024
#    define STAGING_MIN    (16u << 20)
#    define VBUF_MIN       (8u << 20) /* vertex copies, primitive and index lists */
#    define UBUF_MIN       (4u << 20)
#    define MAX_LEVELS     VGPU98_TEX_MAX_LEVELS
#    define N_FORMATS      0x23
#    define N_STAGES       VGPU98_MAX_STAGES
#    define MAX_CUSTOM_BORDERS 24

/* Host format conversions */
enum {
    CONV_COPY,   /* the same bits */
    CONV_F16,    /* an RGB format through R16G16B16A16_SFLOAT */
    CONV_BUMP,   /* bump formats: (du, dv, l, 1) as R16G16B16A16_SFLOAT */
    CONV_WIDE,   /* 2.2's 16- and 10-bit formats: as R32G32B32A32_SFLOAT (their values exactly) */
    CONV_P8,     /* through the stage's palette, to B8G8R8A8 */
    CONV_DXT,    /* decoded to B8G8R8A8 */
    CONV_D16,    /* D16_UNORM */
    CONV_D24S8,  /* D24_UNORM_S8_UINT */
    CONV_D24F32, /* D32_SFLOAT_S8_UINT, holding 24-bit values or W (depth_in) */
    CONV_D16F32, /* D32_SFLOAT, holding 16-bit values or W */
    CONV_COUNT
};

typedef struct hfmt_t {
    VkFormat           vk;
    VkComponentMapping swz;
    int                conv;
    int                bytes;  /* host bytes per texel (colour aspect) */
    int                render; /* usable as a colour or depth attachment */
} hfmt_t;

/* What a draw's vertex shader invocations draw (vgpu98.vert, SHAPE_*). */
enum {
    SHAPE_PLAIN,  /* one vertex each */
    SHAPE_WRAP,   /* a corner of a triangle with WRAPn */
    SHAPE_POINTS, /* a vertex of a triangle in point fill */
    SHAPE_WIRE,   /* a corner of the band of a triangle's edge in wireframe */
    SHAPE_LINE,   /* a corner of the band of a line */
    SHAPE_SPRITE, /* a corner of a point's square (2.1: sized points, sprites) */
    SHAPE_FILLSPRITE /* a corner of the square of a triangle's vertex in point fill (2.1) */
};
#    define BAND_VERTICES   18 /* a line's band: three quads */
#    define SPRITE_VERTICES 6  /* a point's square: two triangles */

/* The vertex shader's uniform block (std140; vgpu98.vert, Vertex). */
typedef struct r3d_light_ubo_t {
    float diffuse[4];
    float specular[4];
    float ambient[4];
    float position[4];
    float direction[4];
    float atten[4];
    float spot[4];
} r3d_light_ubo_t;

typedef struct r3d_vubo_t {
    uint32_t        slot[VGPU98_SLOTS][4];
    int32_t         range[4];
    float           fb[4];
    float           vp[4];
    float           zrange[4];
    uint32_t        mode[4];
    uint32_t        tci[2][4];
    uint32_t        ttff[2][4];
    float           wvp[4][16];
    float           wv[4][16];
    float           wvinv[4][16];
    float           tex[8][16];
    float           plane[6][4];
    float           world[4][16];
    r3d_light_ubo_t light[8];
    float           mat_diffuse[4];
    float           mat_ambient[4];
    float           mat_specular[4];
    float           mat_emissive[4];
    float           material[4];
    float           ambient[4];
    float           fog[4];
    uint32_t        tnl[4];
    uint32_t        sources[4];
    /* Command set 2.1 */
    float           pt[4];      /* POINTSIZE, POINTSIZE_MIN, the largest size, the viewport's height */
    float           ptscale[4]; /* POINTSCALE_A, B, C, TWEENFACTOR */
    uint32_t        dx8[4];     /* DX8_* flags, the palette's entries */
    /* Command set 2.2 */
    uint32_t        vprog[4];   /* the vertex program's instructions (0: none), the constants copied, the outputs it writes, VPF_* */
} r3d_vubo_t;
_Static_assert(sizeof(r3d_vubo_t) == 3152, "vertex uniform block layout");

/* vprog[3] flags (vgpu98.vert) */
#    define VPF_PIXEL 1u /* a pixel program is bound: coordinate sets read as DirectX 8's texcoord pads them */

/* The world matrix palette's entry (vgpu98.vert, Pal; std430). */
typedef struct r3d_pal_t {
    float wvp[16];
    float wv[16];
    float wvinv[16];
    float world[16];
} r3d_pal_t;
_Static_assert(sizeof(r3d_pal_t) == 256, "palette entry layout");

#    define VF_LASTPIXEL 2u
#    define VF_CLIPPING  4u

/* dx8[0] flags (vgpu98.vert) */
#    define DX8_INDEXED 1u
#    define DX8_TWEEN   2u
#    define DX8_SCALE   4u
#    define DX8_SPRITE  8u

/* tnl.y flags (vgpu98.vert) */
#    define TNL_LIGHTING  1u
#    define TNL_SPECULAR  2u
#    define TNL_LOCAL     4u
#    define TNL_NORMALIZE 8u
#    define TNL_RANGEFOG  16u

/* The fragment shader's uniform block (std140; vgpu98.frag, State). */
typedef struct r3d_stage_ubo_t {
    uint32_t ops[4];
    uint32_t ops2[4];
    float    bump[4];
    float    misc[4];
    float    border[4];
    uint32_t ops3[4]; /* 2.1: COLORARG0, ALPHAARG0, RESULTARG */
} r3d_stage_ubo_t;

typedef struct r3d_ubo_t {
    r3d_stage_ubo_t st[N_STAGES];
    float           tfactor[4];
    float           fogcolor[4];
    float           fog[4];
    uint32_t        mode[4];
    float           wrange[4];
    float           dither[4];
    float           zrange[4]; /* minz, maxz - minz: ZRANGE, for a computed depth (pkt_draw) */
    /* Command set 2.2 */
    uint32_t        pprog[4];  /* the pixel program's instructions (0: none), the stages it samples, its VGPU98_PROGF_* flags */
    float           prange[4]; /* its range */
    float           pconst[VGPU98_PROG_PCONSTS][4];
} r3d_ubo_t;
_Static_assert(sizeof(r3d_ubo_t) == 1040, "uniform block layout");

#    define SF_BOUND    1u
#    define SF_KEYED    2u
#    define SF_BORDER_U 4u
#    define SF_BORDER_V 8u
#    define SF_CUBE     16u /* a cube map: binding 6 (and 7 for its key companion) */
#    define SF_KEYZ     32u /* COLORKEYBLENDENABLE: the texture with matching texels zeroed is bound */
#    define SF_VOLUME   64u /* 2.1: a volume texture: binding 9 */
#    define SF_BORDER_W 128u /* 2.1: BORDER on a volume's third coordinate, in the shader */
#    define MF_SPECULAR 1u
#    define MF_COLORKEY 2u
#    define MF_FLAT     4u
#    define MF_DITHER   8u
#    define MF_LINEAR   16u

/* Things whose destruction waits for the command buffers that use them. */
enum { DEAD_IMAGE, DEAD_VIEW, DEAD_MEMORY, DEAD_FRAMEBUFFER, DEAD_BUFFER };

typedef struct dead_t {
    int      kind;
    uint64_t handle;
} dead_t;

typedef struct slot_t {
    VkCommandBuffer    cmd;
    VkFence            fence;
    int                pending;
    int                finished;   /* its pending submission is known to be done (res_poll) */
    uint64_t           sub_serial; /* the serial its last submission had */
    uint64_t           gen;        /* moves when its descriptor pool is reset (set_cache) */
    VkDescriptorPool   pool;
    uint32_t           sets;
    vgpu98_vk_buffer_t staging;
    VkDeviceSize       staging_used;
    vgpu98_vk_buffer_t vbuf;
    VkDeviceSize       vbuf_used;
    vgpu98_vk_buffer_t ubuf;
    VkDeviceSize       ubuf_used;
    dead_t            *dead;
    size_t             n_dead;
    size_t             cap_dead;
} slot_t;

#define IMG_HASH      4096 /* image lookup buckets */
#define CENSUS_KINDS  16   /* draw kinds a census describes, with generated coordinates or a transform, and others */
#define CENSUS_SEEN   256  /* draw kinds a census tells apart exactly */
#define CENSUS_BITS   4096 /* and past those, by a bit each */
#define CENSUS_MAX    (128u << 10) /* bytes of a census's text at most */
#define MIP_LOGGED    16   /* levels the mip check names in the log in a session */
#define IMG_MAX       4096 /* host images kept at most */
#define STALE_SECONDS 10   /* a clean image no draw used for this long goes */
#define MAX_SUBS      (MAX_LEVELS * VGPU98_CUBE_FACES)

struct r3d_image_t;

/* A level of a face of an image: what a texture level, a cube face, a
   render target or a depth buffer is in video memory, and the GPU-newer
   pixels it holds. */
typedef struct r3d_sub_t {
    struct r3d_image_t *im;
    uint32_t            index; /* in the image's sub[]: face * levels + level */
    uint32_t            level;
    uint32_t            face;
    uint32_t            width;
    uint32_t            height;
    uint32_t            offset;
    uint32_t            pitch;
    uint32_t            lo; /* video memory bytes covered: [lo, hi) */
    uint32_t            hi;
    int                 dirty;   /* has GPU-newer pixels */
    uint32_t            dx0, dy0, dx1, dy1; /* the rectangle that holds them */
    int                 span_on; /* its dirty rectangle's pages are counted in the page map */
    uint32_t            span_p0, span_p1;
    VkImageView         att_view;  /* as an attachment (no swizzle), made when first drawn into */
    struct r3d_sub_t   *hash_next; /* the next level or face in its lookup bucket */
    /* The texture audit: a hash of its rows in
       video memory, made when its pages' versions were these (sub_vsig). */
    uint64_t            ahash;
    uint32_t            avsig;
    int                 aset;
} r3d_sub_t;

/* A host image: one per resource, a texture with its levels, a cube map with
   six faces of levels, or a surface drawn into on its own. A target or depth
   buffer that is a level or face of a texture is that texture's image. */
typedef struct r3d_image_t {
    uint32_t   id;
    uint32_t   format; /* VGPU98_FMT_* */
    uint32_t   width;  /* level 0 */
    uint32_t   height;
    uint32_t   levels;
    uint32_t   faces; /* 1, or 6 for a cube map */
    uint32_t   n_sub; /* levels * faces */
    r3d_sub_t *sub;   /* face by face, levels from 0 */
    uint32_t   lo;    /* video memory bytes from its first level or face to its last: [lo, hi), gaps included */
    uint32_t   hi;

    const hfmt_t  *hf;
    VkImage        image;
    VkDeviceMemory memory;
    VkImageView    view; /* sampled: every level and face (a cube view for a cube map), with the format's swizzle */
    VkDeviceSize   bytes;

    /* Colour keys: the companion, 1 where a texel matches the key (R8), for
       COLORKEYENABLE; the texture with matching texels (0, 0, 0, 0)
       (B8G8R8A8), for COLORKEYBLENDENABLE. */
    VkImage        key_image;
    VkDeviceMemory key_memory;
    VkImageView    key_view;
    uint32_t       key_value;
    int            key_valid;
    VkImage        kz_image;
    VkDeviceMemory kz_memory;
    VkImageView    kz_view;
    uint32_t       kz_value;
    int            kz_valid;

    uint32_t pal_sig; /* P8: the palette its texels went through */
    int      depth_w; /* a depth image's depth is W (W-buffering), not Z: how its clears and conversions read it */
    /* A depth image in a float format: its bytes in video memory as of its
       last upload or copy back (from sub[0].offset, rows its pitch apart),
       so a stale upload takes only the pixels someone else wrote and the
       rest keeps the precision video memory's form lacks (depth_resync). */
    uint8_t *shadow;
    int      mip_checked; /* the mip check looked at its levels (mip_check_image) */

    uint32_t  first_page;
    uint32_t  n_pages;
    uint32_t *versions; /* page versions as of the last upload */
    uint8_t  *covered;  /* 1 for the pages a level or face lies on (not the gaps between them) */
    uint32_t  epoch;    /* video memory's epoch when the versions were last found current */
    int       valid;    /* uploaded (or fully cleared) at least once */
    int       fresh;    /* not prepared yet: it takes over the images of its levels (image_merge) */
    int       merging;  /* image_merge is moving its pixels into another image */
    int       dirty;    /* levels and faces with GPU-newer pixels */
    size_t    dirty_at; /* its place in the renderer's dirty list plus one, 0: not there */
    int       pub_new;  /* drawn since its dirty rectangle last went to the display's layer */
    uint64_t  last_use;
    uint64_t  tex_use;   /* the last draw that sampled it (the renderer's tick), 0: none */
    uint64_t  a_checked; /* the texture audit's last check of it (us) ... */
    int       a_suspect; /* ... found its bytes changed, its versions not */
    int       a_named;   /* named stale since its last upload */
    int       a_zero_told; /* found drawn through a palette of zeros since its last upload */

    struct r3d_image_t *hash_next; /* the next image in its lookup bucket */
} r3d_image_t;

typedef struct pipe_key_t {
    uint32_t color_vk;
    uint32_t depth_vk;
    uint8_t  shape;    /* SHAPE_* */
    uint8_t  program;  /* 0 pre-transformed, 1 fixed function, 2 a vertex program (2.2) */
    uint8_t  topology; /* VkPrimitiveTopology */
    uint8_t  cull;     /* VkCullModeFlags */
    uint8_t  computed_depth;
    uint8_t  blend;
    uint8_t  src;
    uint8_t  dst;
    uint8_t  depth_test;
    uint8_t  depth_write;
    uint8_t  depth_func;
    uint8_t  stencil;
    uint8_t  s_fail;
    uint8_t  s_zfail;
    uint8_t  s_pass;
    uint8_t  s_func;
    uint8_t  blend_op;   /* 2.1: VkBlendOp (0, add, without blending) */
    uint8_t  write_mask; /* 2.1: COLORWRITEENABLE as VkColorComponentFlags (the same bits) */
    uint8_t  pixel_program; /* 2.2: a pixel program runs instead of the stages */
    uint8_t  pad[1];
} pipe_key_t;
_Static_assert(sizeof(pipe_key_t) == 28, "pipeline key without padding");

typedef struct pipe_t {
    pipe_key_t key;
    VkPipeline pipeline;
} pipe_t;

typedef struct pass_t {
    VkFormat     color;
    VkFormat     depth;
    VkRenderPass pass;
} pass_t;

typedef struct fb_t {
    uint32_t      color_id;
    uint32_t      depth_id;
    uint32_t      color_sub; /* the level or face of each */
    uint32_t      depth_sub;
    VkFramebuffer fb;
} fb_t;

typedef struct samp_key_t {
    uint8_t  mag, min, mip, addr_u, addr_v, aniso, border, addr_w; /* addr_w 0: clamped (2D textures, cube maps) */
    float    min_lod;
    float    max_lod;
    uint32_t border_color;
} samp_key_t;

typedef struct samp_t {
    samp_key_t key;
    VkSampler  sampler;
} samp_t;

/* A depth surface W-buffered into (vgpu98_r3d_t.drec). */
#define DEPTH_RECS 8
typedef struct depth_rec_t {
    int      used;
    uint32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t offset;
    uint32_t pitch;
    int      depth_w; /* what its image held when it last went: W or Z */
    uint64_t tick;    /* the renderer's tick at its last W-buffered use: the least recent goes first */
} depth_rec_t;

/* A texture's levels as the mip check found them (mip_results): each
   level's distance from a downsample of the one above, what unrelated
   content scores and the downsample's spread (-1: not measured), whether
   its span overlaps another level's, and its bytes. */
#define MIP_RECS 1024
typedef struct mip_rec_t {
    uint32_t key; /* the texture's description (mip_key) */
    uint32_t sig; /* and roughly what it holds (mip_sig) */
    uint8_t  again; /* times measured again since, its content changed */
    uint8_t  stale; /* (in a copy) its content changed since, past MIP_AGAIN */
    uint32_t lo;  /* all its levels' bytes, for a quick test of a DISCARD */
    uint32_t hi;
    uint8_t  levels;
    uint8_t  overlap[MAX_LEVELS];
    float    score[MAX_LEVELS];
    float    base[MAX_LEVELS];
    float    spread[MAX_LEVELS];
    float    point[MAX_LEVELS]; /* off the nearest of the four point samples of the level above */
    uint32_t lev_lo[MAX_LEVELS]; /* each level's bytes, pitch gaps included */
    uint32_t lev_hi[MAX_LEVELS];
} mip_rec_t;

/* The texture audit's churn: by
   texture, its first level's address, format and size, what happened to it
   since the last summary. Open-addressed, three quarters full at most. */
#define CHURN_RECS 1024
typedef struct churn_t {
    uint32_t offset; /* 0 with format 0: a free record */
    uint32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t uploads;
    uint64_t bytes;
    uint32_t made;    /* host images made for it */
    uint32_t evicted; /* and evicted */
} churn_t;

/* The texture audit's palettes, kept by signature (an 8-bit image's
   pal_sig) as P8 textures upload, so a dump decodes such a texture with
   the palette its texels went through. Replaced in turn. */
#define PAL_KEPT 2048
typedef struct pal_kept_t {
    int      used;
    uint32_t sig;
    int      alpha; /* VGPU98_TEX_PALETTE_ALPHA */
    uint32_t c[256];
} pal_kept_t;

/* The coordinate checks' kinds of draw, by a hash, with when each was last
   read (coord_check). */
#define COORD_KINDS 1024
typedef struct coord_kind_t {
    int      used;
    uint32_t key;
    uint32_t ms;
    int      mx_set; /* the texture-matrix checks' (matrix_check): the view and texture matrices last read for it */
    uint32_t view;
    uint32_t tex;
} coord_kind_t;

/* The display check: frames drawn mostly into
   the surface the last FLIP showed, this many running, are a finding. */
#define SHOWN_FRAMES 8

/* The vertex race detector's race, held for the Locks that came after it
   (vsum_check). The buffer's later
   notes (from later draws) carry its Locks since the one before; those up
   to the first the driver wrote after the GPU's read (at or past `until`,
   the ring bytes it had written then) hold every Lock given between the
   draw's note and the read, one of which wrote what the draw read (no Lock
   comes inside a DrawPrimitives2 call). The race's line is written when it
   is found (a capture shows that moment), a second line with the Locks
   when they are all in, or after VSUM_PEND_FRAMES frames or VSUM_PEND_NOTES
   notes with those that came. Keyed by the buffer's audit id (a rename
   keeps it), else its end. The summary counts races by those Locks
   (VSUM_AFTER_*, the first of these the word holds). */
#define VSUM_PENDING        64
#define VSUM_PEND_FRAMES    4
#define VSUM_PEND_NOTES     16384
#define VSUM_AFTER_NOOW     0 /* NOOVERWRITE given while a draw that read the buffer was pending */
#define VSUM_AFTER_UNWAITED 1 /* another given so, without a wait (a buffer in system memory) */
#define VSUM_AFTER_IDLE     2 /* one given with no draw pending: the buffer's fence had passed */
#define VSUM_AFTER_WAITED   3 /* one that waited for the pending draw */
#define VSUM_AFTER_RENAMED  4 /* new memory only */
#define VSUM_AFTER_OPEN     5 /* none, but a Lock open at a draw */
#define VSUM_AFTER_NONE     6 /* none */
#define VSUM_AFTER_UNSEEN   7 /* no later draw of the buffer */
#define VSUM_AFTERS         8
typedef struct vsum_pend_t {
    int      used;
    int      named; /* its line was written: the Locks' line follows */
    int      later; /* a later draw's note of the buffer came */
    uint64_t key;
    uint64_t until;
    uint32_t seq;   /* the race note's sequence: both lines name it */
    uint32_t draw;  /* vsum_draw, flip_frame and note_count when found */
    uint32_t frame;
    uint32_t notes;
    uint32_t lock;    /* the draw's own note's word */
    uint32_t after;   /* the later notes' words together ... */
    uint32_t n_after; /* ... and their Locks */
    char     proc[20];
} vsum_pend_t;

/* The buffers the notes named in memory, by its end (GPU address): the
   last, and the one before with when it changed, so a race in memory that
   another buffer held lately names it (a pointer kept into memory renamed
   away, then reused). VSUM_OWNERS at most, the least lately seen replaced;
   "lately" is VSUM_OWNER_NOTES notes. */
#define VSUM_OWNERS      512
#define VSUM_OWNER_NOTES 65536
typedef struct vsum_owner_t {
    int      used;
    uint32_t end;
    uint32_t id;
    uint32_t prev;  /* 0: none */
    uint32_t moved; /* note_count when it changed ... */
    uint32_t seen;  /* ... and at the last note */
} vsum_owner_t;

/* The texture audit's findings named of each kind, then a quiet spell
   (as the vertex audit's stale buffers: STALE_LOGGED, STALE_QUIET_MS). */
#define FIND_LOGGED   3
#define FIND_QUIET_MS 10000
/* The stale-texture check's budget (tex_check): bytes of video memory it
   hashes a second and a draw at most (a texture is begun while they are
   not spent), the longest its interval grows, and the largest image it
   checks. */
#define TEX_AUDIT_BYTES     (32u << 20)
#define TEX_AUDIT_DRAW      (2u << 20)
#define TEX_AUDIT_EVERY_MAX 32000000u
#define TEX_AUDIT_BIG       (8u << 20)

/* A texture as the TEXTURE packet binds it. */
typedef struct tex_t {
    uint32_t format; /* 0: none */
    uint32_t control;
    uint32_t width;
    uint32_t height;
    uint32_t levels;
    uint32_t faces; /* 1, or 6 for a cube map */
    uint32_t colorkey;
    uint32_t offset[MAX_SUBS]; /* face by face, MAX_LEVELS apart */
    uint32_t pitch[MAX_SUBS];
    uint32_t depth;                            /* 2.1: a volume's (its levels in offset[] and pitch[]); 0 otherwise */
    uint32_t slice[VGPU98_TEX_VOLUME_LEVELS]; /* and their slice pitches */
} tex_t;

/* A draw's descriptor set 0 (its textures, samplers, uniform buffers and
   the arena; set 1 holds its own buffers): reused by a later draw of the
   same slot that binds the same, SET_CACHE of them kept. Handles are
   destroyed only when their slot comes round again, which resets its pool,
   so a set reused never names an object gone. */
#define SET_CACHE 16

typedef struct set_key_t {
    VkImageView view[5 * N_STAGES];
    VkSampler   samp[N_STAGES];
    VkBuffer    ubuf;
    VkBuffer    arena;
} set_key_t;

typedef struct set_cache_t {
    set_key_t       key;
    VkDescriptorSet set;
    int             slot;
    uint64_t        gen;
    int             valid;
} set_cache_t;

/* Resident buffers: buffers at GPU addresses held whole in one host
   buffer, the arena, by where they end and their version (which name their
   contents, spec/gpu-commands.md STREAM), so draws read them there instead
   of copying their vertices or indices from guest RAM again. A record
   holds a buffer from the lowest start a draw named (the DLL names a
   DirectX 7 buffer from the call's dwVertexOffset, a DirectX 8 one from a
   draw's first vertex): a higher start of the same end is read in it; a
   lower one puts it in again from there. A buffer goes
   in when a draw of a later recording than the first that read it reads it
   again, if it is at most RES_BIG and mapped: a dynamic buffer, whose
   version moves at every Lock, is drawn within one recording and never
   copied whole. An older version at the same address goes then (versions
   are never reused). For room, the least recently used of those no pending
   submission reads goes, with the idle ones beside it until the gap
   holds the new one. */
#define RES_BYTES   (64u << 20)
#define RES_ENTRIES 4096
#define RES_HASH    4096
#define RES_SEEN    1024
#define RES_BIG     (8u << 20)
#define RES_ALIGN   256

typedef struct res_t {
    uint32_t     address;
    uint32_t     size;
    uint32_t     version;
    int          used;      /* 0: a free record */
    VkDeviceSize at;        /* its bytes in the arena */
    VkDeviceSize bytes;     /* the room it takes */
    uint64_t     serial;    /* the last recording that reads it */
    uint64_t     tick;      /* the last draw that read it */
    int          hash_next; /* the next record in its bucket, plus one */
    int          stale;     /* the vertex audit found guest RAM changed, its version not: no draw reads it */
    uint64_t     checked;   /* when the vertex audit last compared it (us) */
} res_t;

/* The pipeline cache file's header (spec/pipeline-cache.md), then
   Vulkan's data, then the key list. */
typedef struct pcache_hdr_t {
    char     magic[8];     /* PCACHE_MAGIC */
    uint32_t header_bytes; /* sizeof(pcache_hdr_t) */
    uint32_t number;       /* the save's: one past the newest on disk when it was made */
    uint32_t session;      /* sessions started from this cache: the keys' age */
    uint32_t stamp;        /* the GPU's shaders and pipeline code (pcache_stamp) */
    uint32_t vk_bytes;     /* Vulkan's data (0: none) */
    uint32_t vk_hash;
    uint32_t key_version;  /* PCACHE_KEY_VERSION */
    uint32_t key_bytes;    /* a key record's */
    uint32_t key_count;
    uint32_t key_hash;
    uint32_t header_hash;  /* of the header's bytes before it */
} pcache_hdr_t;
_Static_assert(sizeof(pcache_hdr_t) == 52, "pipeline cache header without padding");

/* A pipeline's key in the list, with the session it was last used in. */
typedef struct pcache_key_t {
    pipe_key_t key;
    uint32_t   session;
} pcache_key_t;
_Static_assert(sizeof(pcache_key_t) == 32, "pipeline cache key without padding");

/* A save of the pipeline cache on its own thread (pcache_save). */
typedef struct pcache_job_t {
    vgpu98_vk_t    *vk;
    VkPipelineCache cache;
    vgpu98_gpu_t   *gpu;
    uint32_t        stamp;   /* what the save carries */
    uint32_t        session;
    uint32_t        last;    /* the number loaded or last saved */
    pipe_key_t     *sess;    /* this session's pipelines' keys (the save's own copy) */
    uint32_t        n_sess;
    const pcache_key_t *keys; /* the key list loaded (the renderer's, unchanged after start) */
    uint32_t        n_keys;
    int             keys_only; /* past the cap: the keys alone */
    int             which;   /* out: the file it went to */
    uint32_t        number;  /* out: its number */
    int             capped;  /* out: Vulkan's data was past the cap */
    uint32_t        told;    /* what was named so far (PCACHE_TOLD_*), on the saving thread */
    int             ok;
    atomic_int      done;
    event_t        *ended;   /* set just before `done`, on its own thread */
} pcache_job_t;

/* A buffer seen, not resident: by its key, the recording that first read
   it, or that it cannot go in (not mapped). */
typedef struct res_seen_t {
    uint64_t key;
    uint64_t serial;
    int      failed;
} res_seen_t;

/* A volume texture as the host holds it (2.1): a 3D image, uploaded whole
   from video memory whenever a page under it changed (a volume is never
   drawn into, so it never holds GPU-newer texels), and kept while draws
   use it, MAX_VOLS at most, their memory at most a quarter of the images'
   budget. */
#define MAX_VOLS 64

typedef struct r3d_vol_t {
    uint32_t       format;
    uint32_t       width;
    uint32_t       height;
    uint32_t       depth;
    uint32_t       levels;
    uint32_t       offset[VGPU98_TEX_VOLUME_LEVELS];
    uint32_t       pitch[VGPU98_TEX_VOLUME_LEVELS];
    uint32_t       slice[VGPU98_TEX_VOLUME_LEVELS];
    uint32_t       lo; /* video memory bytes from its first level's to its last's: [lo, hi) */
    uint32_t       hi;
    uint32_t       lev_lo[VGPU98_TEX_VOLUME_LEVELS]; /* each level's: [lev_lo, lev_hi) */
    uint32_t       lev_hi[VGPU98_TEX_VOLUME_LEVELS];
    const hfmt_t  *hf;
    VkImage        image;
    VkDeviceMemory memory;
    VkImageView    view;
    VkDeviceSize   bytes;
    uint32_t       first_page;
    uint32_t       n_pages;
    uint32_t      *versions; /* page versions as of the last upload */
    uint32_t       epoch;
    int            valid;
    uint64_t       last_use;
} r3d_vol_t;

/* The copy the current submission holds of a program or of the vertex
   constants: draws of the same submission read it while it stays current. */
typedef struct r3d_up_t {
    uint64_t     gen;
    uint64_t     serial;
    VkBuffer     buf;
    VkDeviceSize off;
    VkDeviceSize size;
} r3d_up_t;

/* A program (2.2), as PROGRAM made it. */
typedef struct r3d_prog_t {
    uint32_t           kind;  /* VGPU98_PROG_*, 0: no program */
    uint32_t           range; /* the pixel registers' bound (float bits) */
    uint32_t           n;     /* instructions */
    uint32_t           cneed; /* a vertex program's constants: past the highest it reads, all with a relative source */
    vgpu98_prog_info_t info;
    uint32_t          *ins;   /* n * 4 dwords */
    uint64_t           gen;   /* names its contents: moves with every PROGRAM */
    r3d_up_t           up;    /* its copy in the current submission */
} r3d_prog_t;

struct vgpu98_r3d_t {
    vgpu98_gpu_t        *gpu;
    vgpu98_stats_t      *stats; /* the command processor's counters */
    const vgpu98_vram_t *vram;
    vgpu98_vk_t         *vk;
    int                  broken;
    uint32_t             past_logged; /* draws reading past their streams named since the last quiet spell */
    uint32_t             past_ms;     /* when the last was seen */
    uint32_t             past_seen[64]; /* kinds of them named (stream address, count, base, lacking): each new one is */
    uint32_t             n_past_seen;
    int                  drop_why;    /* why prepare_draw failed: VGPU98_DROP_* */

    hfmt_t fmt[N_FORMATS];
    int    has_fmt[N_FORMATS];
    int    vol_fmt[N_FORMATS]; /* 2.1: fmt[] takes 3D images up to 256 each way with 9 levels */
    hfmt_t vol_f16;            /* the half-float format a volume of another falls back to ... */
    int    vol_f16_ok;         /* ... when the host takes it so */
    hfmt_t wfmt[N_FORMATS]; /* the 32-bit float formats a depth image moves to for W-buffering (depth_sub) */
    int    has_wfmt[N_FORMATS];
    int    d16_full;        /* "16-bit depth precision" Full: D16 in a float image without the grid */
    size_t n_shadowed;      /* images with a shadow (float depth images) */

    /* The draw census while a capture records (vgpu98_r3d_census_start). */
    int      census_on;
    uint64_t census_us;                     /* when it began */
    uint32_t census_seen[CENSUS_SEEN]; /* the kinds seen, by signature */
    uint32_t n_census_seen;
    uint32_t census_bits[CENSUS_BITS / 32]; /* and past those, a bit each */
    int      census_lost;   /* more kinds than census_seen holds: two may share a bit */
    uint32_t census_gen;    /* kinds described with generated coordinates or a texture transform */
    uint32_t census_other;  /* and the others */
    uint32_t census_untold; /* kinds past those, not described */
    char    *census_text;
    size_t   census_len;
    size_t   census_cap;
    mip_rec_t *mip_recs;   /* the textures the mip check measured this session */
    uint32_t   n_mip_recs;
    uint32_t   mip_rec_next; /* with the records full, the one replaced next */
    uint32_t   mip_logged; /* levels the mip check named in the log */
    /* The texture audit. */
    uint64_t ta_second;                /* the start of the hashing budget's second (us) ... */
    uint64_t ta_bytes;                 /* ... the bytes the checks hashed in it ... */
    uint32_t ta_waited;                /* ... and the checks put off for the budget */
    uint64_t ta_draw_bytes;            /* hashed for the draw being prepared */
    uint64_t ta_every;                 /* a texture's checks at most this far apart (us; 0: not begun) */
    uint32_t find_logged[VGPU98_FINDS]; /* each kind's findings named since its last quiet spell */
    uint32_t find_ms[VGPU98_FINDS];     /* when each kind's last was found */
    /* and its summary's figures, cleared at each (vgpu98_r3d_tex_summary) */
    uint32_t ta_fmt[N_FORMATS];  /* stages drawn, by format ... */
    uint32_t ta_conv[CONV_COUNT]; /* ... and by the host's conversion */
    uint32_t ta_checks;          /* the stale check's checks */
    uint32_t ta_big;             /* textures past TEX_AUDIT_BIG it passed over */
    uint64_t ta_hash_up;         /* bytes it hashed at uploads ... */
    uint64_t ta_hash_check;      /* ... and at checks */
    uint32_t ta_suspects;        /* textures found changed once, to be looked at again */
    uint32_t ta_coord_draws;     /* draws the coordinate checks read (coord_check) ... */
    uint32_t ta_one_coord;       /* ... those whose primitives all read one coordinate ... */
    uint32_t ta_wrap_short;      /* ... their triangles that WRAPn takes the short way round ... */
    uint32_t ta_smeared;         /* ... pre-transformed draws smearing a texel one way only ... */
    uint32_t ta_pre_xform;       /* ... and pre-transformed draws with a texture transform set */
    uint32_t ta_finds[VGPU98_FINDS];
    churn_t *churn;              /* CHURN_RECS */
    uint32_t n_churn;
    uint32_t churn_lost;         /* events of textures past the table's three quarters, not kept */
    pal_kept_t *pal_kept;        /* PAL_KEPT, for the dumps ... */
    uint32_t   *pal_sigs;        /* ... their signatures, to search */
    uint32_t    pal_next;        /* the record replaced next */
    uint32_t    dump_count;      /* pictures dumped this session */
    coord_kind_t *coord_kinds;   /* the coordinate checks': COORD_KINDS ... */
    uint32_t    coord_second;    /* ... the start of their second (ms) ... */
    uint32_t    coord_read;      /* ... and the draws read in it */
    uint32_t    ta_coord_skipped; /* the summary's: draws not read (their kind's read lately, or the second's are) */
    /* The display check (shown_draw, vgpu98_r3d_frame): frames ended
       (FLIPs; the vertex race detector's ageing too), the start the last
       FLIP showed (once one came), the frame's draws and those into that
       surface, the frames in a row drawn mostly there, and the summary's
       frames looked at and those drawn mostly on display. */
    uint32_t    flip_frame;
    uint32_t    shown_start;
    int         shown_set;
    uint32_t    shown_draws;
    uint32_t    shown_on;
    uint32_t    shown_run;
    uint32_t    ta_frames;
    uint32_t    ta_shown;
    /* The texture-matrix checks' (matrix_check): draws read, matrices
       mapping to one direction, one-way smears edge-on to a planar mapping
       and others, kinds whose matrix moved under a still view, stages with
       normal or reflection coordinates not followed. */
    uint32_t    ta_mx_draws;
    uint32_t    ta_mx_collapsed;
    uint32_t    ta_mx_edge;
    uint32_t    ta_mx_smeared;
    uint32_t    ta_mx_moving;
    uint32_t    ta_mx_unfollowed;
    /* The vertex race detector (vgpu98_r3d_note):
       the last note's sequence, the notes taken, a window's bytes, the races
       waiting for their buffer's next note, and the summary's figures:
       windows checked and their bytes, those the draw read changed, races
       told by the Locks after (VSUM_AFTER_*), windows changed in guest RAM
       only (the draw read the arena's copy, made before), arena copies
       differing, VSUM notes naming no stream, notes damaged or out of
       sequence. */
    uint32_t    note_seq;
    uint32_t    note_count;
    uint32_t    vsum_draw;       /* DRAWs: a buffer in two of a draw's streams has two notes, one draw's Locks */
    char        proc_name[20];   /* the drawing process's executable, from the last PROC note ("" none) */
    uint8_t    *vsum_buf;
    vsum_pend_t vsum_pend[VSUM_PENDING];
    uint32_t    ta_vsum_checked;
    uint64_t    ta_vsum_bytes;
    uint32_t    ta_vsum_races;
    uint32_t    ta_vsum_after[VSUM_AFTERS];
    uint32_t    ta_vsum_ram;
    uint32_t    ta_vsum_arena;
    uint32_t    ta_vsum_refused;
    uint32_t    ta_note_refused;
    vsum_owner_t *vsum_owners;   /* VSUM_OWNERS */
    uint32_t    ta_vsum_whole;   /* races changing every part of the window ... */
    uint32_t    ta_vsum_reused;  /* ... in memory another buffer held lately */
    uint32_t    ta_vsum_early;   /* races told before their Locks were all in, for want of room ... */
    uint32_t    ta_vsum_dropped; /* ... and dropped by a reset that starts a new session */
    /* Depth surfaces W-buffered into: their images are made float, holding
       what they last held, until the program discards the surface or the
       GPU is reset (depth_rec_*). */
    depth_rec_t drec[DEPTH_RECS];

    VkDescriptorSetLayout dsl;      /* set 0: textures, uniforms, the arena */
    VkDescriptorSetLayout dsl_draw; /* set 1: a draw's own buffers */
    VkPipelineLayout      layout;
    set_cache_t           set_cache[SET_CACHE];
    VkShaderModule        vs;
    VkShaderModule        fs;
    VkShaderModule        fs_depth;
    VkCommandPool         cmdpool;
    slot_t                slot[SLOTS];
    int                   cur;
    int                   recording;
    VkCommandBuffer       sink;  /* with 3D stopped, what is recorded goes here, never submitted */
    int                   idle;  /* the host GPU is idle: dead objects need no new submission */
    int                   timed; /* inside a timed upload or copy back: its waits are its own */
    uint64_t              serial;

    pass_t *passes;
    size_t  n_passes;
    fb_t   *fbs;
    size_t  n_fbs;
    pipe_t *pipes;
    size_t  n_pipes;
    samp_t *samps;
    size_t  n_samps;
    /* Pipelines and samplers by their keys' hash: an open-addressed table of
       their numbers plus one, twice their number in size at least. */
    uint32_t       *pipe_tab;
    uint32_t        pipe_tab_size;
    uint32_t       *samp_tab;
    uint32_t        samp_tab_size;
    VkPipelineCache pcache;     /* kept on disk between sessions (pcache_save) */
    uint32_t        pcache_new; /* pipelines made since it was last saved */
    uint64_t        pcache_us;  /* when it was */
    uint32_t        pcache_number;  /* the number loaded or last saved */
    uint32_t        pcache_stamp;   /* this 86Box's (pcache_stamp) */
    uint32_t        pcache_session; /* this session's: one past the file's */
    int             pcache_mode;    /* the "Pipeline cache" setting: PCACHE_ON, _EMPTY, _OFF */
    pcache_key_t   *pcache_keys;    /* the file's key list (checked, most recently used first) */
    uint32_t        pcache_n_keys;
    int             pcache_capped;  /* past the cap: saves carry the keys alone */
    /* The warm-up: the first warm_n
       keys made by threads of its own, each taking the next. */
    uint32_t        warm_n;
    int             warm_cold;      /* Vulkan's data was not used: every pipeline it makes is new */
    size_t          warm_size0;     /* else the cache's data at its start: it is saved for what it adds */
    int             warm_threads;   /* started */
    int             warm_told;      /* its end named */
    atomic_uint     warm_next;
    atomic_uint     warm_made;
    atomic_uint     warm_failed;
    atomic_uint     warm_start;     /* milliseconds (vgpu98_diag_us) at its start, at its last thread's end */
    atomic_uint     warm_end;
    atomic_int      warm_live;      /* threads not ended: decremented last of all */
    atomic_int      warm_stop;
    atomic_int      warm_dirty;     /* pipelines made since the last save */
    event_t        *warm_ev;        /* set as each thread ends */
    pcache_job_t    pcache_job;     /* a save on its own thread ... */
    int             pcache_running; /* ... while this is set */
    event_t        *pcache_ev;      /* set as a save's thread ends */
    int     n_custom_borders;

    r3d_image_t **img;
    size_t        n_img;
    uint32_t      next_id;
    VkDeviceSize  img_bytes;
    VkDeviceSize  img_budget;
    uint64_t      tick;
    r3d_image_t  *img_hash[IMG_HASH]; /* image_find's index: by format, size, first offset and pitch */
    r3d_sub_t    *sub_hash[IMG_HASH]; /* sub_find's: every level of every face, by format, size, offset and pitch */
    vgpu98_disp_t *disp;             /* the display on the host GPU: published pixels go to its layer */
    r3d_image_t **dirty;              /* the images with GPU-newer pixels */
    size_t        n_dirty;
    size_t        cap_dirty;
    uint64_t      stale_us;                  /* when evict_stale last looked */
    uint64_t      tick_hist[STALE_SECONDS]; /* tick at each of the last seconds, oldest first */

    VkImage        dummy;
    VkDeviceMemory dummy_mem;
    VkImageView    dummy_view;
    VkImage        dummy_cube; /* the same as a cube map, for stages without one */
    VkDeviceMemory dummy_cube_mem;
    VkImageView    dummy_cube_view;
    VkImage        dummy_vol; /* and as a volume (2.1) */
    VkDeviceMemory dummy_vol_mem;
    VkImageView    dummy_vol_view;

    /* Resident buffers */
    vgpu98_vk_buffer_t res_buf;    /* the arena, made at the first buffer that goes in */
    int                res_failed; /* it could not be made */
    res_t             *res;        /* RES_ENTRIES records */
    int                res_hash[RES_HASH]; /* a bucket's first record, plus one */
    int               *res_order;  /* the records in use by their place in the arena */
    uint32_t           n_res;
    res_seen_t         res_seen[RES_SEEN]; /* buffers seen, not resident */
    uint64_t           done_serial; /* every submission up to this serial has finished */
    res_t             *res_used[VGPU98_MAX_STREAMS + 1]; /* the draw's records, stamped as it is recorded */
    uint32_t           n_res_used;
    res_t             *res_stream[VGPU98_MAX_STREAMS]; /* the draw's, by stream (NULL: not resident) ... */
    res_t             *res_ib;                         /* ... and its index buffer's */
    VkDeviceSize       res_ib_at;                      /* where the draw's index buffer starts in the arena */
    int                res_logged;                     /* the arena's failure was logged */
    uint8_t           *res_scratch; /* the vertex audit's copy of a resident buffer's guest bytes */
    uint32_t           stale_logged;
    uint32_t           stale_ms;

    /* Volume textures (2.1) */
    r3d_vol_t   *vols[MAX_VOLS];
    size_t       n_vols;
    VkDeviceSize vol_bytes;
    r3d_vol_t   *vol_bound[N_STAGES]; /* the draw's, by stage (prepare_draw) */

    /* GPU state (spec/gpu-commands.md), command set 2.1's included */
    uint32_t      rs[VGPU98_RS_GLOBAL_COUNT8];
    uint32_t      tss[N_STAGES][VGPU98_TSS_COUNT8];
    vgpu98_surf_t target;
    vgpu98_surf_t depth;
    int           have_target;
    int           have_depth;
    uint32_t      vp[4];
    float         wnear;
    float         wfar;
    float         wnorm;     /* the reference's normalization of W: 1 / (wfar - wnear), or 1 (w_norm) */
    float         zrange[2]; /* ZRANGE: minz, maxz */
    tex_t         tex[N_STAGES];
    uint32_t      pal[N_STAGES][256];
    vgpu98_vin_t  vin;       /* the vertex input */

    /* Command set 2.2: the programs by id (VGPU98_PROG_MAX_ID + 1, 0
       unused), the binding, the constants (pixel ones clamped to -1..1),
       and the copies the current submission holds. */
    r3d_prog_t *progs;
    uint32_t    bind_v;
    uint32_t    bind_p;
    float       vconst[VGPU98_PROG_VCONSTS][4];
    float       pconst[VGPU98_PROG_PCONSTS][4];
    uint32_t    vconst_hi;  /* registers set since the reset: 0 to vconst_hi - 1 */
    uint64_t    vconst_gen; /* moves with every change of them */
    uint64_t    prog_gen;   /* the last generation a PROGRAM took */
    r3d_up_t    up_vprog;
    r3d_up_t    up_pprog;
    r3d_up_t    up_vconst;

    /* The fixed-function state: TRANSFORM, LIGHT, LIGHTENABLE, MATERIAL,
       CLIPPLANE as sent */
    float      xform[VGPU98_XFORM_COUNT][16]; /* Direct3D's matrix order */
    uint32_t   light[VGPU98_MAX_LIGHTS][26];  /* D3DLIGHT7 */
    uint32_t   light_mask;
    float      material[17];                  /* D3DMATERIAL7 */
    float      plane[VGPU98_MAX_CLIPPLANES][4];
    int        proj_set;                      /* a TRANSFORM of the projection since the reset (DirectX 7 only sends one) */
    int        ff_dirty;                      /* `ff` needs computing again */
    r3d_vubo_t ff;                            /* its part of the vertex uniforms (ff_update) */
    /* The world matrix palette (2.1): world matrices 0-255 (0-3 are xform's
       world matrices, kept equal), the highest set since the reset, and its
       entries for the vertex shader (pal_update), the last an identity world
       matrix; a draw with indexed blending takes a copy of entries 0 to
       pal_max + 1, unless the last copy, made in the same submission, is
       current. */
    float        pal_world[VGPU98_PALETTE_SIZE][16];
    uint32_t     pal_max;
    int          pal_dirty;                                     /* some entry needs computing again ... */
    int          pal_all;                                       /* ... every one (the view or projection changed) */
    uint32_t     pal_bits[(VGPU98_PALETTE_SIZE + 1 + 31) / 32]; /* ... or these */
    r3d_pal_t   *pal_ff;        /* VGPU98_PALETTE_SIZE + 1 entries */
    uint64_t     pal_gen;       /* moves with every change of them */
    uint64_t     pal_up_gen;    /* the last copy made: of this generation ... */
    uint64_t     pal_up_serial; /* ... in this submission ... */
    VkBuffer     pal_up_buf;    /* ... here */
    VkDeviceSize pal_up_off;
    VkDeviceSize pal_up_size;

    /* Recording */
    int          in_pass;
    r3d_sub_t   *pass_color; /* the level or face the render pass draws into */
    r3d_sub_t   *pass_depth;
    VkPipeline   bound_pipe;

    /* The Glide device (vid_vgpu98_glide.c)
       and the OpenGL device (vid_vgpu98_ogl.c) record into these command buffers
       too: a device's render pass is ended through its hook whenever the
       renderer records for itself (cmd, outside the devices' own calls:
       in_glide) or submits, and when the other device records. */
    void (*glide_end)(void *priv);
    void  *glide_priv;
    void (*ogl_end)(void *priv);
    void  *ogl_priv;
    int    in_glide;
};

static void r3d_log(vgpu98_r3d_t *r, int source, const char *fmt, ...);
static void copy_back(vgpu98_r3d_t *r, r3d_image_t **list, size_t n, uint32_t lo, uint32_t hi, uint32_t skip_lo,
                      uint32_t skip_hi, int cause);
static void mip_check_image(vgpu98_r3d_t *r, r3d_image_t *im, const uint32_t *palette, int pal_alpha);
static int      pkt_program(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len);
static int      pkt_bind(vgpu98_r3d_t *r, const uint32_t *p);
static int      pkt_constants(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len);
static uint32_t stage_mask(const vgpu98_r3d_t *r);
static uint32_t vconst_count(const vgpu98_r3d_t *r, const r3d_prog_t *vp);
static void census_draw(vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d);
static void pal_keep(vgpu98_r3d_t *r, uint32_t sig, const uint32_t *palette, int alpha);
static int  stage_reads_texture(const vgpu98_r3d_t *r, int s);
static atomic_int dump_busy; /* the texture dumps' thread is writing (dump_thread) */
static void mip_recs_drop(vgpu98_r3d_t *r, uint32_t lo, uint32_t hi);

/* --- Small helpers ----------------------------------------------------------- */

static void
r3d_log(vgpu98_r3d_t *r, int source, const char *fmt, ...)
{
    char    line[1024];
    va_list ap;

    if (!vgpu98_gpu_logging(r->gpu, source))
        return;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    vgpu98_gpu_log(r->gpu, source, "%s", line);
}

static void
vk_log_cb(void *priv, const char *fmt, ...)
{
    vgpu98_gpu_t *g = (vgpu98_gpu_t *) priv;
    char          line[512];
    va_list       ap;
    size_t        n;

    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    /* A line cut short still ends its line. */
    n = strlen(line);
    if (!n || (line[n - 1] != '\n')) {
        if (n == sizeof(line) - 1)
            n--;
        line[n++] = '\n';
        line[n]   = 0;
    }
    vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "%s", line);
}

static void
r3d_fail(vgpu98_r3d_t *r, const char *what, VkResult res)
{
    if (!r->broken) {
        char why[160];

        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: %s failed (VkResult %d); 3D stops until the next reset\n", what, (int) res);
        vgpu98_gpu_note(r->gpu, VGPU98_NOTE_FAILURE, (uint32_t) res, 0, 0, 0, 0);
        snprintf(why, sizeof(why), "the host GPU failed: %s (VkResult %d)", what, (int) res);
        r->broken = 1;
        vgpu98_gpu_trigger(r->gpu, why);
    }
    r->broken = 1;
}

static uint64_t
us_since(uint64_t t0)
{
    const uint64_t t = vgpu98_diag_us();

    return (t > t0) ? t - t0 : 0;
}

static float
u2f(uint32_t u)
{
    float f;

    memcpy(&f, &u, 4);
    return f;
}

static uint16_t
f2h(float f)
{
    uint32_t x;
    uint32_t sign;
    int32_t  e;
    uint32_t m;

    memcpy(&x, &f, 4);
    sign = (x >> 16) & 0x8000;
    e    = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    m    = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff)
        return (uint16_t) (sign | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 31)
        return (uint16_t) (sign | 0x7c00);
    if (e <= 0) {
        if (e < -10)
            return (uint16_t) sign;
        m |= 0x800000;
        {
            uint32_t shift = (uint32_t) (14 - e);
            uint32_t half  = m >> shift;
            uint32_t rem   = m & ((1u << shift) - 1);
            uint32_t mid   = 1u << (shift - 1);

            if ((rem > mid) || ((rem == mid) && (half & 1)))
                half++;
            return (uint16_t) (sign | half);
        }
    }
    {
        uint32_t h   = sign | ((uint32_t) e << 10) | (m >> 13);
        uint32_t rem = m & 0x1fff;

        if ((rem > 0x1000) || ((rem == 0x1000) && (h & 1)))
            h++;
        return (uint16_t) h;
    }
}

static float
h2f(uint16_t h)
{
    uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t e    = (h >> 10) & 0x1f;
    uint32_t m    = h & 0x3ff;
    uint32_t x;
    float    f;

    if (e == 0) {
        f = (float) m / 16777216.0f; /* 2^-24 */
        return sign ? -f : f;
    }
    if (e == 31)
        x = sign | 0x7f800000 | (m << 13);
    else
        x = sign | ((e - 15 + 127) << 23) | (m << 13);
    memcpy(&f, &x, 4);
    return f;
}

static uint32_t
fnv(const void *p, size_t n, uint32_t h)
{
    const uint8_t *b = p;

    while (n--)
        h = (h ^ *b++) * 16777619u;
    return h;
}

static uint32_t
rd16(const uint8_t *p)
{
    return p[0] | (p[1] << 8);
}

static uint32_t
rd32(const uint8_t *p)
{
    return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t) p[3] << 24);
}

static void
wr16(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t) v;
    p[1] = (uint8_t) (v >> 8);
}

static void
wr32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t) v;
    p[1] = (uint8_t) (v >> 8);
    p[2] = (uint8_t) (v >> 16);
    p[3] = (uint8_t) (v >> 24);
}

static uint32_t
max1(uint32_t v)
{
    return v ? v : 1;
}

/* --- Formats ---------------------------------------------------------------- */

static const VkComponentMapping SWZ_ID = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
static const VkComponentMapping SWZ_X = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                          VK_COMPONENT_SWIZZLE_ONE };

/* The host format for each GPU format (spec/gpu-formats.md, "Host
   representation"): native where the host has the format for its uses,
   R16G16B16A16_SFLOAT otherwise. */
static void
formats_init(vgpu98_r3d_t *r)
{
    const VkFormatFeatureFlags tex = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                     VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                     VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    const VkFormatFeatureFlags rt  = tex | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
    const VkFormatFeatureFlags ds  = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    const vgpu98_vk_t *vk = r->vk;
    hfmt_t             f16;
    int                i;

    memset(r->fmt, 0, sizeof(r->fmt));
    memset(r->has_fmt, 0, sizeof(r->has_fmt));
    memset(r->wfmt, 0, sizeof(r->wfmt));
    memset(r->has_wfmt, 0, sizeof(r->has_wfmt));
    f16.vk     = VK_FORMAT_R16G16B16A16_SFLOAT;
    f16.swz    = SWZ_ID;
    f16.conv   = CONV_F16;
    f16.bytes  = 8;
    f16.render = 1;

#    define SET(code, vkf, swizzle, conversion, nbytes, rendering) \
        do {                                                       \
            r->fmt[code].vk     = (vkf);                           \
            r->fmt[code].swz    = (swizzle);                       \
            r->fmt[code].conv   = (conversion);                    \
            r->fmt[code].bytes  = (nbytes);                        \
            r->fmt[code].render = (rendering);                     \
            r->has_fmt[code]    = 1;                               \
        } while (0)

    /* Render target formats. */
    if (vgpu98_vk_format_has(vk, VK_FORMAT_R5G6B5_UNORM_PACK16, rt))
        SET(VGPU98_FMT_R5G6B5, VK_FORMAT_R5G6B5_UNORM_PACK16, SWZ_ID, CONV_COPY, 2, 1);
    else
        r->fmt[VGPU98_FMT_R5G6B5] = f16, r->has_fmt[VGPU98_FMT_R5G6B5] = 1;
    if (vgpu98_vk_format_has(vk, VK_FORMAT_A1R5G5B5_UNORM_PACK16, rt)) {
        SET(VGPU98_FMT_X1R5G5B5, VK_FORMAT_A1R5G5B5_UNORM_PACK16, SWZ_X, CONV_COPY, 2, 1);
        SET(VGPU98_FMT_A1R5G5B5, VK_FORMAT_A1R5G5B5_UNORM_PACK16, SWZ_ID, CONV_COPY, 2, 1);
    } else {
        r->fmt[VGPU98_FMT_X1R5G5B5] = f16, r->has_fmt[VGPU98_FMT_X1R5G5B5] = 1;
        r->fmt[VGPU98_FMT_A1R5G5B5] = f16, r->has_fmt[VGPU98_FMT_A1R5G5B5] = 1;
    }
    if (vgpu98_vk_format_has(vk, VK_FORMAT_B8G8R8A8_UNORM, rt)) {
        SET(VGPU98_FMT_X8R8G8B8, VK_FORMAT_B8G8R8A8_UNORM, SWZ_X, CONV_COPY, 4, 1);
        SET(VGPU98_FMT_A8R8G8B8, VK_FORMAT_B8G8R8A8_UNORM, SWZ_ID, CONV_COPY, 4, 1);
    } else {
        r->fmt[VGPU98_FMT_X8R8G8B8] = f16, r->has_fmt[VGPU98_FMT_X8R8G8B8] = 1;
        r->fmt[VGPU98_FMT_A8R8G8B8] = f16, r->has_fmt[VGPU98_FMT_A8R8G8B8] = 1;
    }

    /* Texture-only formats. */
    if (vgpu98_vk_format_has(vk, VK_FORMAT_B4G4R4A4_UNORM_PACK16, tex)) {
        /* B4G4R4A4 holds our A, R, G, B in its B, G, R, A. */
        VkComponentMapping m = { VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_A,
                                 VK_COMPONENT_SWIZZLE_B };
        SET(VGPU98_FMT_A4R4G4B4, VK_FORMAT_B4G4R4A4_UNORM_PACK16, m, CONV_COPY, 2, 0);
    } else
        r->fmt[VGPU98_FMT_A4R4G4B4] = f16, r->fmt[VGPU98_FMT_A4R4G4B4].render = 0, r->has_fmt[VGPU98_FMT_A4R4G4B4] = 1;
    if (vgpu98_vk_format_has(vk, VK_FORMAT_R8_UNORM, tex)) {
        VkComponentMapping m = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
                                 VK_COMPONENT_SWIZZLE_ONE };
        SET(VGPU98_FMT_L8, VK_FORMAT_R8_UNORM, m, CONV_COPY, 1, 0);
    } else
        r->fmt[VGPU98_FMT_L8] = f16, r->fmt[VGPU98_FMT_L8].render = 0, r->has_fmt[VGPU98_FMT_L8] = 1;
    if (vgpu98_vk_format_has(vk, VK_FORMAT_R8G8_UNORM, tex)) {
        VkComponentMapping m = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
                                 VK_COMPONENT_SWIZZLE_G };
        SET(VGPU98_FMT_A8L8, VK_FORMAT_R8G8_UNORM, m, CONV_COPY, 2, 0);
    } else
        r->fmt[VGPU98_FMT_A8L8] = f16, r->fmt[VGPU98_FMT_A8L8].render = 0, r->has_fmt[VGPU98_FMT_A8L8] = 1;
    SET(VGPU98_FMT_P8, VK_FORMAT_B8G8R8A8_UNORM, SWZ_ID, CONV_P8, 4, 0);
    /* Command set 2.1's: A8 as (0, 0, 0, A), X4R4G4B4 as A4R4G4B4 with
       alpha 1, A4L4 and Q8W8V8U8 converted (spec/gpu-formats.md). */
    if (vgpu98_vk_format_has(vk, VK_FORMAT_R8_UNORM, tex)) {
        VkComponentMapping m = { VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO,
                                 VK_COMPONENT_SWIZZLE_R };
        SET(VGPU98_FMT_A8, VK_FORMAT_R8_UNORM, m, CONV_COPY, 1, 0);
    } else
        r->fmt[VGPU98_FMT_A8] = f16, r->fmt[VGPU98_FMT_A8].render = 0, r->has_fmt[VGPU98_FMT_A8] = 1;
    if (vgpu98_vk_format_has(vk, VK_FORMAT_B4G4R4A4_UNORM_PACK16, tex)) {
        VkComponentMapping m = { VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_A,
                                 VK_COMPONENT_SWIZZLE_ONE };
        SET(VGPU98_FMT_X4R4G4B4, VK_FORMAT_B4G4R4A4_UNORM_PACK16, m, CONV_COPY, 2, 0);
    } else
        r->fmt[VGPU98_FMT_X4R4G4B4] = f16, r->fmt[VGPU98_FMT_X4R4G4B4].render = 0, r->has_fmt[VGPU98_FMT_X4R4G4B4] = 1;
    if (vgpu98_vk_format_has(vk, VK_FORMAT_R4G4_UNORM_PACK8, tex)) {
        /* R4G4 holds A4L4's A in R, its L in G. */
        VkComponentMapping m = { VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_G,
                                 VK_COMPONENT_SWIZZLE_R };
        SET(VGPU98_FMT_A4L4, VK_FORMAT_R4G4_UNORM_PACK8, m, CONV_COPY, 1, 0);
    } else
        r->fmt[VGPU98_FMT_A4L4] = f16, r->fmt[VGPU98_FMT_A4L4].render = 0, r->has_fmt[VGPU98_FMT_A4L4] = 1;
    SET(VGPU98_FMT_Q8W8V8U8, VK_FORMAT_R16G16B16A16_SFLOAT, SWZ_ID, CONV_BUMP, 8, 0);
    SET(VGPU98_FMT_V8U8, VK_FORMAT_R16G16B16A16_SFLOAT, SWZ_ID, CONV_BUMP, 8, 0);
    SET(VGPU98_FMT_L6V5U5, VK_FORMAT_R16G16B16A16_SFLOAT, SWZ_ID, CONV_BUMP, 8, 0);
    SET(VGPU98_FMT_X8L8V8U8, VK_FORMAT_R16G16B16A16_SFLOAT, SWZ_ID, CONV_BUMP, 8, 0);
    /* Command set 2.2's: 32-bit floats hold their 16- and 10-bit values
       exactly; half floats where the host cannot filter those (within a
       step of 1/2048). */
    for (i = VGPU98_FMT_V16U16; i <= VGPU98_FMT_A2W10V10U10; i++) {
        if (vgpu98_vk_format_has(vk, VK_FORMAT_R32G32B32A32_SFLOAT, tex))
            SET(i, VK_FORMAT_R32G32B32A32_SFLOAT, SWZ_ID, CONV_WIDE, 16, 0);
        else
            SET(i, VK_FORMAT_R16G16B16A16_SFLOAT, SWZ_ID, CONV_BUMP, 8, 0);
    }
    for (i = VGPU98_FMT_DXT1; i <= VGPU98_FMT_DXT5; i++)
        SET(i, VK_FORMAT_B8G8R8A8_UNORM, SWZ_ID, CONV_DXT, 4, 0);

    /* Depth. */
    if (vgpu98_vk_format_has(vk, VK_FORMAT_D16_UNORM, ds))
        SET(VGPU98_FMT_D16, VK_FORMAT_D16_UNORM, SWZ_ID, CONV_D16, 2, 1);
    if (vgpu98_vk_format_has(vk, VK_FORMAT_D24_UNORM_S8_UINT, ds)) {
        SET(VGPU98_FMT_D24X8, VK_FORMAT_D24_UNORM_S8_UINT, SWZ_ID, CONV_D24S8, 4, 1);
        SET(VGPU98_FMT_D24S8, VK_FORMAT_D24_UNORM_S8_UINT, SWZ_ID, CONV_D24S8, 4, 1);
    } else if (vgpu98_vk_format_has(vk, VK_FORMAT_D32_SFLOAT_S8_UINT, ds)) {
        SET(VGPU98_FMT_D24X8, VK_FORMAT_D32_SFLOAT_S8_UINT, SWZ_ID, CONV_D24F32, 4, 1);
        SET(VGPU98_FMT_D24S8, VK_FORMAT_D32_SFLOAT_S8_UINT, SWZ_ID, CONV_D24F32, 4, 1);
    }
#    undef SET

    /* The float formats a depth image moves to when something W-buffers
       into it (depth_sub): W keeps its precision there whatever the W
       range. */
#    define WSET(code, vkf, conversion)               \
        do {                                          \
            r->wfmt[code].vk     = (vkf);             \
            r->wfmt[code].swz    = SWZ_ID;            \
            r->wfmt[code].conv   = (conversion);      \
            r->wfmt[code].bytes  = 4;                 \
            r->wfmt[code].render = 1;                 \
            r->has_wfmt[code]    = r->has_fmt[code];  \
        } while (0)
    if (vgpu98_vk_format_has(vk, VK_FORMAT_D32_SFLOAT, ds))
        WSET(VGPU98_FMT_D16, VK_FORMAT_D32_SFLOAT, CONV_D16F32);
    if (vgpu98_vk_format_has(vk, VK_FORMAT_D32_SFLOAT_S8_UINT, ds)) {
        WSET(VGPU98_FMT_D24X8, VK_FORMAT_D32_SFLOAT_S8_UINT, CONV_D24F32);
        WSET(VGPU98_FMT_D24S8, VK_FORMAT_D32_SFLOAT_S8_UINT, CONV_D24F32);
    }
#    undef WSET

    /* Volumes (2.1): the formats the host takes as 3D images of the
       largest size, with every level; others go through half floats. */
    {
        const VkImageUsageFlags  use = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VkImageFormatProperties ip;

        r->vol_f16      = f16;
        r->vol_f16.conv = CONV_F16;
        r->vol_f16_ok   = (vk->vkGetPhysicalDeviceImageFormatProperties(vk->phys, f16.vk, VK_IMAGE_TYPE_3D,
                                                                        VK_IMAGE_TILING_OPTIMAL, use, 0, &ip) == VK_SUCCESS) &&
                        (ip.maxExtent.width >= VGPU98_TEX_VOLUME_MAX) && (ip.maxExtent.height >= VGPU98_TEX_VOLUME_MAX) &&
                        (ip.maxExtent.depth >= VGPU98_TEX_VOLUME_MAX) && (ip.maxMipLevels >= VGPU98_TEX_VOLUME_LEVELS);
        memset(r->vol_fmt, 0, sizeof(r->vol_fmt));
        for (i = 0; i < N_FORMATS; i++) {
            if (!r->has_fmt[i] || vgpu98_fmt_is_depth((uint32_t) i) || (i == VGPU98_FMT_P8) || vgpu98_fmt_is_dxt((uint32_t) i))
                continue;
            r->vol_fmt[i] = (vk->vkGetPhysicalDeviceImageFormatProperties(vk->phys, r->fmt[i].vk, VK_IMAGE_TYPE_3D,
                                                                          VK_IMAGE_TILING_OPTIMAL, use, 0, &ip) == VK_SUCCESS) &&
                            (ip.maxExtent.width >= VGPU98_TEX_VOLUME_MAX) && (ip.maxExtent.height >= VGPU98_TEX_VOLUME_MAX) &&
                            (ip.maxExtent.depth >= VGPU98_TEX_VOLUME_MAX) && (ip.maxMipLevels >= VGPU98_TEX_VOLUME_LEVELS);
        }
    }

    /* "16-bit depth precision" Full: D16 in its float format from the
       start, without the 16-bit grid (depth_grid), so Z compares at the
       host's precision. */
    r->d16_full = 0;
    if (r->has_wfmt[VGPU98_FMT_D16] && (vgpu98_gpu_options(r->gpu) & VGPU98_OPT_DEPTH16_FULL)) {
        r->fmt[VGPU98_FMT_D16] = r->wfmt[VGPU98_FMT_D16];
        r->d16_full            = 1;
    }
}

static int
fmt_has_stencil(uint32_t f)
{
    return f == VGPU98_FMT_D24S8;
}

/* --- Depth in 32-bit float images -------------------------------------------- */

/* A depth image in a 32-bit float format: one of a surface W-buffered into
   (depth_sub), or a 24-bit one on a host without D24_UNORM_S8_UINT. Such an
   image holds W, as W * 2^-64 (exact, ordered, the same precision at every
   distance, whatever the W range), or Z on the guest buffer's grid, step k
   as k / 2^bits (exact again, so the fragment shader, clears, uploads and
   copy-backs agree to the bit). vgpu98.frag computes both. */
#    define W_SCALE   5.42101086242752217e-20f /* 2^-64 */
#    define W_UNSCALE 18446744073709551616.0f  /* 2^64 */

static int
depth_float(const hfmt_t *hf)
{
    return (hf->conv == CONV_D16F32) || (hf->conv == CONV_D24F32);
}

/* Does the image hold Z on the guest buffer's grid (a float image, but for
   D16 with "16-bit depth precision" Full), so a Z-buffered draw computes
   its depth? */
static int
depth_grid(const vgpu98_r3d_t *r, const r3d_image_t *im)
{
    return depth_float(im->hf) && !(r->d16_full && (im->format == VGPU98_FMT_D16));
}

/* The guest buffer's bits of depth: 16 or 24. */
static int
depth_bits(uint32_t format)
{
    return (format == VGPU98_FMT_D16) ? 16 : 24;
}

/* A value (0-1) on the grid of `bits` bits: the step nearest to c, halves
   up, as the reference converts to its buffer's format. */
static uint32_t
depth_q(double c, int bits)
{
    const uint32_t max = (1u << bits) - 1;

    if (!(c > 0.0))
        return 0;
    if (c >= 1.0)
        return max;
    return (uint32_t) (c * max + 0.5);
}

/* Z (0-1) on the grid as vgpu98.frag rounds a fragment's: the product in
   float, rounded to the nearest step, halves to even. A clear rounds the
   same way, so a draw at the cleared depth meets it exactly. */
static uint32_t
depth_grid_q(float d, int bits)
{
    if (!(d > 0.0f))
        return 0;
    if (d >= 1.0f)
        return (1u << bits) - 1;
    return (uint32_t) rintf(d * (float) ((1u << bits) - 1));
}

/* The W range, and the reference rasterizer's normalization of W for its
   depth buffer, (W - wnear) times wnorm: 1 / (wfar - wnear), 0 for an
   infinite range, 1 for an empty one. */
static void
wrange_set(vgpu98_r3d_t *r, float wnear, float wfar)
{
    const float span = wfar - wnear;

    r->wnear = wnear;
    r->wfar  = wfar;
    r->wnorm = (span != 0.0f) ? 1.0f / span : 1.0f;
}

static float
w_norm(const vgpu98_r3d_t *r)
{
    return r->wnorm;
}

/* W as a float image holds it (vgpu98.frag computes the same). */
static float
w_held(float w)
{
    if (!(w > 0.0f))
        return 0.0f;
    if (w >= W_UNSCALE)
        return 1.0f;
    return w * W_SCALE;
}

/* A depth value of `bits` bits from video memory as the float image holds
   it. W comes back from the reference's (W - wnear) / (wfar - wnear) with
   the W range now; the largest value is the far end, past every W. */
static float
depth_in(const vgpu98_r3d_t *r, const r3d_image_t *im, uint32_t v, int bits)
{
    const uint32_t max = (1u << bits) - 1;

    if (im->depth_w) {
        const float norm = w_norm(r);

        if (v >= max)
            return 1.0f;
        if (norm == 0.0f)
            return v ? 1.0f : w_held(r->wnear);
        return w_held((float) (r->wnear + ((double) v / (double) max) / norm));
    }
    if (!depth_grid(r, im))
        return (float) v / (float) max;
    return (float) v * (1.0f / (float) (1u << bits));
}

/* And back: a float image's depth value as `bits` bits for video memory. */
static uint32_t
depth_out(const vgpu98_r3d_t *r, const r3d_image_t *im, float f, int bits)
{
    const uint32_t max = (1u << bits) - 1;

    if (im->depth_w) {
        if (f >= 1.0f)
            return max;
        return depth_q(((double) f * W_UNSCALE - r->wnear) * w_norm(r), bits);
    }
    if (!depth_grid(r, im))
        return depth_q(f, bits);
    if (!(f > 0.0f))
        return 0;
    {
        const double k = (double) f * (double) (1u << bits) + 0.5;

        return (k >= (double) max) ? max : (uint32_t) k;
    }
}

/* A clear's depth value (0-1) as the image holds it. */
static float
depth_clear(const vgpu98_r3d_t *r, const r3d_image_t *im, float d)
{
    const int bits = depth_bits(im->format);

    if (!depth_float(im->hf))
        return d; /* the host GPU converts to D16_UNORM or D24_UNORM_S8_UINT */
    if (im->depth_w)
        return (d >= 1.0f) ? 1.0f : depth_in(r, im, depth_q(d, bits), bits);
    if (!depth_grid(r, im))
        return d;
    return (float) depth_grid_q(d, bits) * (1.0f / (float) (1u << bits));
}

/* The depth surfaces W-buffered into (vgpu98_r3d_t.drec): an image made for
   one is float and holds what its last image held, so an image evicted and
   made again, or moved, keeps W. A surface leaves the records when a
   DISCARD overlaps it (the program freed it) and at RESET; a depth buffer
   only ever Z-buffered keeps the host's own depth format. */
static depth_rec_t *
depth_rec_find(vgpu98_r3d_t *r, uint32_t format, uint32_t w, uint32_t h, uint32_t offset, uint32_t pitch)
{
    int i;

    for (i = 0; i < DEPTH_RECS; i++) {
        depth_rec_t *d = &r->drec[i];

        if (d->used && (d->format == format) && (d->width == w) && (d->height == h) && (d->offset == offset) &&
            (d->pitch == pitch))
            return d;
    }
    return NULL;
}

/* Records a surface, or updates its record, with what its image holds, as
   used now. With every record in use, the least recently used goes. */
static void
depth_rec_set(vgpu98_r3d_t *r, uint32_t format, uint32_t w, uint32_t h, uint32_t offset, uint32_t pitch, int depth_w)
{
    depth_rec_t *d = depth_rec_find(r, format, w, h, offset, pitch);
    int          i;

    if (!d) {
        d = &r->drec[0];
        for (i = 1; (i < DEPTH_RECS) && d->used; i++) {
            if (!r->drec[i].used || (r->drec[i].tick < d->tick))
                d = &r->drec[i];
        }
        d->used   = 1;
        d->format = format;
        d->width  = w;
        d->height = h;
        d->offset = offset;
        d->pitch  = pitch;
    }
    d->depth_w = depth_w;
    d->tick    = r->tick;
}

/* A DISCARD of [lo, hi): the surfaces it overlaps leave the records. */
static void
depth_rec_drop(vgpu98_r3d_t *r, uint32_t lo, uint32_t hi)
{
    int i;

    for (i = 0; i < DEPTH_RECS; i++) {
        depth_rec_t   *d   = &r->drec[i];
        const uint32_t end = d->offset + (d->height - 1) * d->pitch + d->width * (uint32_t) vgpu98_fmt_bpp(d->format);

        if (d->used && (d->offset < hi) && (lo < end))
            d->used = 0;
    }
}

static VkImageAspectFlags
fmt_aspects(const hfmt_t *hf)
{
    switch (hf->conv) {
        case CONV_D16:
        case CONV_D16F32:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case CONV_D24S8:
        case CONV_D24F32:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

/* Unpacks a texel of an RGB render target format to floats, and back (for
   the R16G16B16A16_SFLOAT host representation). */
static void
rgb_unpack(uint32_t f, uint32_t v, float *c)
{
    switch (f) {
        case VGPU98_FMT_R5G6B5:
            c[0] = ((v >> 11) & 31) / 31.0f, c[1] = ((v >> 5) & 63) / 63.0f, c[2] = (v & 31) / 31.0f, c[3] = 1.0f;
            break;
        case VGPU98_FMT_X1R5G5B5:
        case VGPU98_FMT_A1R5G5B5:
            c[0] = ((v >> 10) & 31) / 31.0f, c[1] = ((v >> 5) & 31) / 31.0f, c[2] = (v & 31) / 31.0f;
            c[3] = (f == VGPU98_FMT_A1R5G5B5) ? (float) ((v >> 15) & 1) : 1.0f;
            break;
        case VGPU98_FMT_A4R4G4B4:
        case VGPU98_FMT_X4R4G4B4:
            c[0] = ((v >> 8) & 15) / 15.0f, c[1] = ((v >> 4) & 15) / 15.0f, c[2] = (v & 15) / 15.0f;
            c[3] = (f == VGPU98_FMT_A4R4G4B4) ? ((v >> 12) & 15) / 15.0f : 1.0f;
            break;
        case VGPU98_FMT_A8:
            c[0] = c[1] = c[2] = 0.0f, c[3] = (v & 255) / 255.0f;
            break;
        case VGPU98_FMT_A4L4:
            c[0] = c[1] = c[2] = (v & 15) / 15.0f, c[3] = ((v >> 4) & 15) / 15.0f;
            break;
        case VGPU98_FMT_L8:
            c[0] = c[1] = c[2] = (v & 255) / 255.0f, c[3] = 1.0f;
            break;
        case VGPU98_FMT_A8L8:
            c[0] = c[1] = c[2] = (v & 255) / 255.0f, c[3] = ((v >> 8) & 255) / 255.0f;
            break;
        default:
            c[0] = ((v >> 16) & 255) / 255.0f, c[1] = ((v >> 8) & 255) / 255.0f, c[2] = (v & 255) / 255.0f;
            c[3] = (f == VGPU98_FMT_A8R8G8B8) ? ((v >> 24) & 255) / 255.0f : 1.0f;
            break;
    }
}

static uint32_t
q(float c, uint32_t max)
{
    if (!(c > 0.0f))
        return 0;
    if (c >= 1.0f)
        return max;
    return (uint32_t) (c * (float) max + 0.5f);
}

static uint32_t
rgb_pack(uint32_t f, const float *c)
{
    switch (f) {
        case VGPU98_FMT_R5G6B5:
            return (q(c[0], 31) << 11) | (q(c[1], 63) << 5) | q(c[2], 31);
        case VGPU98_FMT_X1R5G5B5:
            return (q(c[0], 31) << 10) | (q(c[1], 31) << 5) | q(c[2], 31);
        case VGPU98_FMT_A1R5G5B5:
            return (q(c[3], 1) << 15) | (q(c[0], 31) << 10) | (q(c[1], 31) << 5) | q(c[2], 31);
        case VGPU98_FMT_X8R8G8B8:
            return (q(c[0], 255) << 16) | (q(c[1], 255) << 8) | q(c[2], 255);
        default:
            return (q(c[3], 255) << 24) | (q(c[0], 255) << 16) | (q(c[1], 255) << 8) | q(c[2], 255);
    }
}

/* --- DXT decoding (spec/gpu-formats.md: the reference rasterizer's rounding,
   three-colour mode for DXT1 only) ------------------------------------------- */

static void
dxt_colors(const uint8_t *b, int dxt1, uint32_t *out)
{
    uint32_t c0 = rd16(b);
    uint32_t c1 = rd16(b + 2);
    uint32_t r[4], g[4], bl[4];
    int      i;

    r[0]  = ((c0 >> 11) & 31) << 3 | ((c0 >> 11) & 31) >> 2;
    g[0]  = ((c0 >> 5) & 63) << 2 | ((c0 >> 5) & 63) >> 4;
    bl[0] = (c0 & 31) << 3 | (c0 & 31) >> 2;
    r[1]  = ((c1 >> 11) & 31) << 3 | ((c1 >> 11) & 31) >> 2;
    g[1]  = ((c1 >> 5) & 63) << 2 | ((c1 >> 5) & 63) >> 4;
    bl[1] = (c1 & 31) << 3 | (c1 & 31) >> 2;
    out[0] = 0xff000000u | (r[0] << 16) | (g[0] << 8) | bl[0];
    out[1] = 0xff000000u | (r[1] << 16) | (g[1] << 8) | bl[1];
    if (!dxt1 || (c0 > c1)) {
        for (i = 0; i < 2; i++) {
            uint32_t a  = (i == 0) ? 2 : 1;
            uint32_t bb = 3 - a;

            out[2 + i] = 0xff000000u | (((a * r[0] + bb * r[1] + 1) / 3) << 16) | (((a * g[0] + bb * g[1] + 1) / 3) << 8) |
                         ((a * bl[0] + bb * bl[1] + 1) / 3);
        }
    } else {
        out[2] = 0xff000000u | (((r[0] + r[1]) / 2) << 16) | (((g[0] + g[1]) / 2) << 8) | ((bl[0] + bl[1]) / 2);
        out[3] = 0x00000000u;
    }
}

/* Decodes the DXT surface (w x h pixels, rows of blocks `pitch` apart) into
   B8G8R8A8 texels, `w` per row. */
static void
dxt_decode(uint32_t fmt, const uint8_t *src, uint32_t pitch, uint32_t w, uint32_t h, uint8_t *dst)
{
    const int dxt1  = (fmt == VGPU98_FMT_DXT1);
    const int bsize = dxt1 ? 8 : 16;
    uint32_t  bx;
    uint32_t  by;

    for (by = 0; by < (h + 3) / 4; by++) {
        for (bx = 0; bx < (w + 3) / 4; bx++) {
            const uint8_t *blk = src + (size_t) by * pitch + (size_t) bx * bsize;
            const uint8_t *cb  = dxt1 ? blk : blk + 8;
            uint32_t       col[4];
            uint32_t       alpha[16];
            uint32_t       idx = rd32(cb + 4);
            int            i;

            dxt_colors(cb, dxt1, col);
            for (i = 0; i < 16; i++)
                alpha[i] = col[(idx >> (2 * i)) & 3] >> 24;
            if ((fmt == VGPU98_FMT_DXT2) || (fmt == VGPU98_FMT_DXT3)) {
                for (i = 0; i < 16; i++) {
                    uint32_t a4 = (blk[i / 2] >> ((i & 1) * 4)) & 15;

                    alpha[i] = a4 | (a4 << 4);
                }
            } else if ((fmt == VGPU98_FMT_DXT4) || (fmt == VGPU98_FMT_DXT5)) {
                uint32_t a[8];
                uint64_t bits = 0;
                int      k;

                a[0] = blk[0];
                a[1] = blk[1];
                if (a[0] > a[1]) {
                    for (k = 1; k < 7; k++)
                        a[1 + k] = ((7 - k) * a[0] + k * a[1]) / 7;
                } else {
                    for (k = 1; k < 5; k++)
                        a[1 + k] = ((5 - k) * a[0] + k * a[1]) / 5;
                    a[6] = 0;
                    a[7] = 255;
                }
                for (k = 0; k < 6; k++)
                    bits |= (uint64_t) blk[2 + k] << (8 * k);
                for (i = 0; i < 16; i++)
                    alpha[i] = a[(bits >> (3 * i)) & 7];
            }
            for (i = 0; i < 16; i++) {
                uint32_t x = bx * 4 + (i & 3);
                uint32_t y = by * 4 + (i >> 2);
                uint32_t c;

                if ((x >= w) || (y >= h))
                    continue;
                c = (col[(idx >> (2 * i)) & 3] & 0x00ffffffu) | (alpha[i] << 24);
                wr32(dst + ((size_t) y * w + x) * 4, c);
            }
        }
    }
}

/* --- Command buffers, slots and memory ----------------------------------------- */

static VkCommandBuffer cmd(vgpu98_r3d_t *r);

/* Queues an object for destruction once the host GPU is done with it. It
   goes on the recording slot's list, freed when that slot's fence has
   signalled: after everything submitted before it. With nothing recording,
   recording starts, so the object never waits only on a slot whose last
   submission is older than one still running. */
static void
dead_add(vgpu98_r3d_t *r, int kind, uint64_t handle)
{
    slot_t *s;

    if (!handle)
        return;
    if (!r->recording && !r->idle)
        cmd(r);
    s = &r->slot[r->cur];
    if (s->n_dead == s->cap_dead) {
        s->cap_dead = s->cap_dead ? s->cap_dead * 2 : 64;
        s->dead     = realloc(s->dead, s->cap_dead * sizeof(dead_t));
    }
    s->dead[s->n_dead].kind   = kind;
    s->dead[s->n_dead].handle = handle;
    s->n_dead++;
}

static void
dead_free(vgpu98_r3d_t *r, slot_t *s)
{
    vgpu98_vk_t *vk = r->vk;
    size_t       i;

    for (i = 0; i < s->n_dead; i++) {
        uint64_t h = s->dead[i].handle;

        switch (s->dead[i].kind) {
            case DEAD_IMAGE:
                vk->vkDestroyImage(vk->dev, (VkImage) (uintptr_t) h, NULL);
                break;
            case DEAD_VIEW:
                vk->vkDestroyImageView(vk->dev, (VkImageView) (uintptr_t) h, NULL);
                break;
            case DEAD_MEMORY:
                vk->vkFreeMemory(vk->dev, (VkDeviceMemory) (uintptr_t) h, NULL);
                break;
            case DEAD_FRAMEBUFFER:
                vk->vkDestroyFramebuffer(vk->dev, (VkFramebuffer) (uintptr_t) h, NULL);
                break;
            case DEAD_BUFFER: {
                vgpu98_vk_buffer_t *b = (vgpu98_vk_buffer_t *) (uintptr_t) h;

                vgpu98_vk_buffer_destroy(vk, b);
                free(b);
                break;
            }
        }
    }
    s->n_dead = 0;
}

/* Waits for a submission's fence, a second at a time (the seconds noted for
   the watchdog). 0 when it failed, which stops 3D, or was given up because
   86Box closes (quietly: 3D stops with it). */
static int
fence_wait(vgpu98_r3d_t *r, VkFence fence)
{
    const int      step = vgpu98_gpu_step(r->gpu, VGPU98_STEP_HOST_GPU);
    const VkResult res  = vgpu98_vk_wait_fence(r->vk, fence, vgpu98_gpu_waited, r->gpu);

    vgpu98_gpu_step(r->gpu, step);
    if (res == VK_SUCCESS)
        return 1;
    if (res == VK_TIMEOUT)
        r->broken = 1;
    else
        r3d_fail(r, "vkWaitForFences", res);
    return 0;
}

/* Starts recording in the current slot, first waiting for its last use. */
static int
slot_begin(vgpu98_r3d_t *r)
{
    vgpu98_vk_t             *vk = r->vk;
    slot_t                  *s  = &r->slot[r->cur];
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VkResult                 res;

    /* 3D stopped (a failure, or a wait given up): no wait again, and no
       recording into a slot the host GPU may still run. */
    if (r->broken)
        return 0;
    if (s->pending) {
        const uint64_t t0 = vgpu98_diag_us();
        const int      ok = fence_wait(r, s->fence);

        if (!r->timed)
            r->stats->gpu_wait_us += us_since(t0);
        if (!ok)
            return 0;
        s->pending = 0;
        if (s->sub_serial > r->done_serial)
            r->done_serial = s->sub_serial;
    }
    vk->vkResetFences(vk->dev, 1, &s->fence);
    dead_free(r, s);
    vk->vkResetDescriptorPool(vk->dev, s->pool, 0);
    s->gen++;
    s->sets         = 0;
    s->staging_used = 0;
    s->vbuf_used    = 0;
    s->ubuf_used    = 0;
    vk->vkResetCommandBuffer(s->cmd, 0);
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res      = vk->vkBeginCommandBuffer(s->cmd, &bi);
    if (res != VK_SUCCESS) {
        r3d_fail(r, "vkBeginCommandBuffer", res);
        return 0;
    }
    r->recording  = 1;
    r->bound_pipe = VK_NULL_HANDLE;
    return 1;
}

/* With 3D stopped, the commands the work in progress still records go to a
   buffer of their own, begun once and never submitted (dropped at the next
   reset): none goes into a slot's buffer the host GPU may still be running. */
static VkCommandBuffer
sink(vgpu98_r3d_t *r)
{
    vgpu98_vk_t                *vk = r->vk;
    VkCommandBufferAllocateInfo ca = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    VkCommandBufferBeginInfo    bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };

    if (!r->sink) {
        ca.commandPool        = r->cmdpool;
        ca.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        if (vk->vkAllocateCommandBuffers(vk->dev, &ca, &r->sink) != VK_SUCCESS)
            r->sink = VK_NULL_HANDLE;
        else if (vk->vkBeginCommandBuffer(r->sink, &bi) != VK_SUCCESS) {
            vk->vkFreeCommandBuffers(vk->dev, r->cmdpool, 1, &r->sink);
            r->sink = VK_NULL_HANDLE;
        }
    }
    /* Without one (out of host memory), the slot's, as before. */
    return r->sink ? r->sink : r->slot[r->cur].cmd;
}

/* The buffer being recorded: the slot's while recording, else the sink. */
static VkCommandBuffer
rec_cmd(vgpu98_r3d_t *r)
{
    return r->recording ? r->slot[r->cur].cmd : sink(r);
}

static VkCommandBuffer
cmd(vgpu98_r3d_t *r)
{
    /* The devices' render passes, if open, end before the renderer records
       anything of its own. */
    if (r->glide_end && !r->in_glide)
        r->glide_end(r->glide_priv);
    if (r->ogl_end && !r->in_glide)
        r->ogl_end(r->ogl_priv);
    if (!r->recording)
        slot_begin(r);
    return rec_cmd(r);
}

/* A full memory barrier: every earlier command's writes visible to every
   later command. The renderer keeps every image in GENERAL layout. */
static void
barrier(vgpu98_r3d_t *r)
{
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };

    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    r->vk->vkCmdPipelineBarrier(cmd(r), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                                &mb, 0, NULL, 0, NULL);
}

static void
pass_end(vgpu98_r3d_t *r)
{
    if (r->in_pass) {
        r->vk->vkCmdEndRenderPass(rec_cmd(r));
        r->in_pass    = 0;
        r->pass_color = NULL;
        r->pass_depth = NULL;
        barrier(r);
    }
}

/* Submits what was recorded; with `wait`, until the host GPU has done it. */
static int
submit(vgpu98_r3d_t *r, int wait)
{
    vgpu98_vk_t *vk = r->vk;
    slot_t      *s  = &r->slot[r->cur];
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkResult     res;
    int          step;

    /* Nothing recorded; with 3D stopped, nothing done either (what the sink
       holds never runs): a copy back has no pixels to write. */
    if (!r->recording)
        return !r->broken;
    if (r->glide_end)
        r->glide_end(r->glide_priv); /* the Glide device's pass, whoever submits */
    if (r->ogl_end)
        r->ogl_end(r->ogl_priv);     /* and the OpenGL device's */
    pass_end(r);
    res = vk->vkEndCommandBuffer(s->cmd);
    r->recording = 0;
    if (res != VK_SUCCESS) {
        r3d_fail(r, "vkEndCommandBuffer", res);
        return 0;
    }
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &s->cmd;
    step                  = vgpu98_gpu_step(r->gpu, VGPU98_STEP_SUBMIT);
    res                   = vgpu98_vk_submit(vk, &si, s->fence);
    vgpu98_gpu_step(r->gpu, step);
    if (res != VK_SUCCESS) {
        r3d_fail(r, "vkQueueSubmit", res);
        return 0;
    }
    s->pending    = 1;
    s->finished   = 0;
    s->sub_serial = r->serial;
    r->serial++;
    r->stats->submits++;
    if (wait) {
        if (!fence_wait(r, s->fence))
            return 0;
        s->finished = 1;
        if (s->sub_serial > r->done_serial)
            r->done_serial = s->sub_serial;
    }
    r->cur = (r->cur + 1) % SLOTS;
    return 1;
}

/* Room in one of the current slot's host buffers. When it is full the
   recording is submitted and the next slot used; a request larger than the
   buffer gets a larger one. Returns the mapped pointer, or NULL. */
static void *
slot_alloc(vgpu98_r3d_t *r, int which, VkDeviceSize size, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off)
{
    static const VkBufferUsageFlags usage[3] = {
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    };
    static const VkDeviceSize minimum[3] = { STAGING_MIN, VBUF_MIN, UBUF_MIN };
    int                       tries;

    /* With 3D stopped, none: the slot's memory may still be read by a
       submission the host GPU runs. */
    if (r->broken)
        return NULL;
    /* At most: the slot is full (submit), the next slot's buffer is too
       small (grow), then it fits. */
    for (tries = 0; tries < 3; tries++) {
        slot_t             *s;
        vgpu98_vk_buffer_t *b;
        VkDeviceSize       *used;
        VkDeviceSize        at;

        cmd(r);
        s    = &r->slot[r->cur];
        b    = (which == 0) ? &s->staging : (which == 1) ? &s->vbuf : &s->ubuf;
        used = (which == 0) ? &s->staging_used : (which == 1) ? &s->vbuf_used : &s->ubuf_used;
        at   = (*used + align - 1) / align * align;
        if (b->buffer && (at + size <= b->size)) {
            *used = at + size;
            *buf  = b->buffer;
            *off  = at;
            return (uint8_t *) b->map + at;
        }
        if (!b->buffer || (size > b->size)) {
            VkDeviceSize n = minimum[which];

            while (n < size)
                n *= 2;
            if (b->buffer) {
                /* Commands already recorded may use the old buffer. */
                vgpu98_vk_buffer_t *old = malloc(sizeof(*old));

                *old = *b;
                dead_add(r, DEAD_BUFFER, (uint64_t) (uintptr_t) old);
                memset(b, 0, sizeof(*b));
            }
            if (!vgpu98_vk_buffer_create(r->vk, b, n, usage[which])) {
                r3d_fail(r, "allocating a host buffer", VK_ERROR_OUT_OF_HOST_MEMORY);
                return NULL;
            }
            *used = 0;
            continue;
        }
        /* Full: the next slot's buffers are free once its last use is done. */
        if (r->in_pass)
            return NULL; /* the caller ends the pass and retries */
        if (!submit(r, 0))
            return NULL;
    }
    return NULL;
}

/* --- Images ---------------------------------------------------------------- */

static int
image_memory(vgpu98_r3d_t *r, VkImage image, VkDeviceMemory *mem, VkDeviceSize *bytes)
{
    vgpu98_vk_t         *vk = r->vk;
    VkMemoryRequirements req;
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    int                  type;

    vk->vkGetImageMemoryRequirements(vk->dev, image, &req);
    type = vgpu98_vk_memory_type(vk, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type < 0)
        type = vgpu98_vk_memory_type(vk, req.memoryTypeBits, 0);
    if (type < 0)
        return 0;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = (uint32_t) type;
    if (vk->vkAllocateMemory(vk->dev, &ai, NULL, mem) != VK_SUCCESS)
        return 0;
    if (vk->vkBindImageMemory(vk->dev, image, *mem, 0) != VK_SUCCESS) {
        vk->vkFreeMemory(vk->dev, *mem, NULL);
        *mem = VK_NULL_HANDLE;
        return 0;
    }
    if (bytes)
        *bytes = req.size;
    return 1;
}

/* GENERAL layout for a new image, recorded before its first use, outside a
   render pass (its subpass has no dependency on itself). */
static void
image_init_layout(vgpu98_r3d_t *r, VkImage image, VkImageAspectFlags aspects, uint32_t levels, uint32_t layers)
{
    VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };

    pass_end(r);
    ib.srcAccessMask               = 0;
    ib.dstAccessMask               = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib.oldLayout                   = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout                   = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    ib.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    ib.image                       = image;
    ib.subresourceRange.aspectMask = aspects;
    ib.subresourceRange.levelCount = levels;
    ib.subresourceRange.layerCount = layers;
    r->vk->vkCmdPipelineBarrier(cmd(r), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                NULL, 0, NULL, 1, &ib);
}

/* The lookup bucket of an image's description (image_find), or of a level
   of a face (sub_find): by format, size, offset and pitch. */
static uint32_t
image_bucket(uint32_t format, uint32_t w, uint32_t h, uint32_t offset, uint32_t pitch)
{
    uint32_t x = (offset * 0x9e3779b1u) ^ (pitch * 0x85ebca6bu) ^ (((w << 16) | h) * 0xc2b2ae35u) ^ format;

    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return x & (IMG_HASH - 1);
}

/* The dirty list: images with GPU-newer pixels in any level or face, so
   synchronizing a range looks at those few rather than every image. */
static void
dirty_add(vgpu98_r3d_t *r, r3d_image_t *im)
{
    if (im->dirty_at)
        return;
    if (r->n_dirty == r->cap_dirty) {
        r->cap_dirty = r->cap_dirty ? 2 * r->cap_dirty : 64;
        r->dirty     = realloc(r->dirty, r->cap_dirty * sizeof(*r->dirty));
    }
    r->dirty[r->n_dirty++] = im;
    im->dirty_at           = r->n_dirty;
}

/* The GPU-newer page map (vgpu98_vram_t.newer): a level or face with
   GPU-newer pixels counts the pages its dirty rectangle spans, from the
   first dirty row's first pixel to the last one's last, so the guest's CPU
   never reads or writes them unseen. */
static void
span_clear(vgpu98_r3d_t *r, r3d_sub_t *s)
{
    uint32_t p;

    if (!s->span_on)
        return;
    /* The bytes copied back before the counts that let the guest's CPU at
       them (vgpu98_vram_get). */
    atomic_thread_fence(memory_order_release);
    for (p = s->span_p0; p <= s->span_p1; p++)
        r->vram->newer[p]--;
    s->span_on = 0;
}

/* The bytes a rectangle of a level or face spans: [*lo, *hi). */
static void
sub_rect_span(const r3d_image_t *im, const r3d_sub_t *s, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t *lo,
              uint32_t *hi)
{
    if (vgpu98_fmt_is_dxt(im->format)) {
        const uint32_t block = (im->format == VGPU98_FMT_DXT1) ? 8 : 16;

        *lo = s->offset + (y0 / 4) * s->pitch + (x0 / 4) * block;
        *hi = s->offset + ((y1 + 3) / 4 - 1) * s->pitch + ((x1 + 3) / 4) * block;
    } else {
        const uint32_t bpp = (uint32_t) vgpu98_fmt_bpp(im->format);

        *lo = s->offset + y0 * s->pitch + x0 * bpp;
        *hi = s->offset + (y1 - 1) * s->pitch + x1 * bpp;
    }
}

static void
span_set(vgpu98_r3d_t *r, const r3d_image_t *im, r3d_sub_t *s)
{
    uint32_t lo;
    uint32_t hi;
    uint32_t p0;
    uint32_t p1;
    uint32_t p;

    sub_rect_span(im, s, s->dx0, s->dy0, s->dx1, s->dy1, &lo, &hi);
    p0 = lo >> VGPU98_PAGE_SHIFT;
    p1 = (hi - 1) >> VGPU98_PAGE_SHIFT;
    if (s->span_on && (p0 == s->span_p0) && (p1 == s->span_p1))
        return;
    /* The new pages first, so no page of either span reads 0 in between. */
    for (p = p0; p <= p1; p++) {
        if (!s->span_on || (p < s->span_p0) || (p > s->span_p1))
            r->vram->newer[p]++;
    }
    if (s->span_on) {
        for (p = s->span_p0; p <= s->span_p1; p++) {
            if ((p < p0) || (p > p1))
                r->vram->newer[p]--;
        }
    }
    s->span_on = 1;
    s->span_p0 = p0;
    s->span_p1 = p1;
}

/* A level or face no longer GPU-newer. */
static void
sub_clean(vgpu98_r3d_t *r, r3d_image_t *im, r3d_sub_t *s)
{
    if (!s->dirty)
        return;
    s->dirty = 0;
    span_clear(r, s);
    if (!--im->dirty && im->dirty_at) {
        const size_t at = im->dirty_at;

        if (r->disp)
            vgpu98_disp_layer_drop(r->disp, im->id);
        r->dirty[at - 1]           = r->dirty[--r->n_dirty];
        r->dirty[at - 1]->dirty_at = at;
        im->dirty_at               = 0;
    }
}

/* Every level and face no longer GPU-newer (their pixels are back in video
   memory, or lost). */
static void
image_clean(vgpu98_r3d_t *r, r3d_image_t *im)
{
    uint32_t i;

    for (i = 0; im->dirty && (i < im->n_sub); i++)
        sub_clean(r, im, &im->sub[i]);
}

static void
sub_unhash(vgpu98_r3d_t *r, r3d_sub_t *s)
{
    r3d_sub_t **pp = &r->sub_hash[image_bucket(s->im->format, s->width, s->height, s->offset, s->pitch)];

    while (*pp && (*pp != s))
        pp = &(*pp)->hash_next;
    if (*pp)
        *pp = s->hash_next;
}

static void
image_destroy(vgpu98_r3d_t *r, r3d_image_t *im)
{
    r3d_image_t **pp = &r->img_hash[image_bucket(im->format, im->width, im->height, im->sub[0].offset, im->sub[0].pitch)];
    size_t        i;
    uint32_t      k;

    while (*pp && (*pp != im))
        pp = &(*pp)->hash_next;
    if (*pp)
        *pp = im->hash_next;
    image_clean(r, im);

    /* A float depth image that held W, or whose surface was W-buffered into:
       the record keeps what it held, for the next image of the surface. */
    if (vgpu98_fmt_is_depth(im->format) && depth_float(im->hf) && (im->n_sub == 1)) {
        depth_rec_t *rec = depth_rec_find(r, im->format, im->width, im->height, im->sub[0].offset, im->sub[0].pitch);

        if (rec)
            rec->depth_w = im->depth_w;
        else if (im->depth_w)
            depth_rec_set(r, im->format, im->width, im->height, im->sub[0].offset, im->sub[0].pitch, 1);
    }

    /* Framebuffers that use it go with it. */
    for (i = 0; i < r->n_fbs;) {
        if ((r->fbs[i].color_id == im->id) || (r->fbs[i].depth_id == im->id)) {
            dead_add(r, DEAD_FRAMEBUFFER, (uint64_t) (uintptr_t) r->fbs[i].fb);
            r->fbs[i] = r->fbs[--r->n_fbs];
        } else
            i++;
    }
    if ((r->pass_color && (r->pass_color->im == im)) || (r->pass_depth && (r->pass_depth->im == im)))
        pass_end(r);
    for (k = 0; k < im->n_sub; k++) {
        sub_unhash(r, &im->sub[k]);
        dead_add(r, DEAD_VIEW, (uint64_t) (uintptr_t) im->sub[k].att_view);
    }
    dead_add(r, DEAD_VIEW, (uint64_t) (uintptr_t) im->view);
    dead_add(r, DEAD_IMAGE, (uint64_t) (uintptr_t) im->image);
    dead_add(r, DEAD_MEMORY, (uint64_t) (uintptr_t) im->memory);
    dead_add(r, DEAD_VIEW, (uint64_t) (uintptr_t) im->key_view);
    dead_add(r, DEAD_IMAGE, (uint64_t) (uintptr_t) im->key_image);
    dead_add(r, DEAD_MEMORY, (uint64_t) (uintptr_t) im->key_memory);
    dead_add(r, DEAD_VIEW, (uint64_t) (uintptr_t) im->kz_view);
    dead_add(r, DEAD_IMAGE, (uint64_t) (uintptr_t) im->kz_image);
    dead_add(r, DEAD_MEMORY, (uint64_t) (uintptr_t) im->kz_memory);
    r->img_bytes -= im->bytes;
    r->n_shadowed -= (im->shadow != NULL);
    free(im->versions);
    free(im->covered);
    free(im->shadow);
    free(im->sub);
    free(im);
}

static void
image_remove(vgpu98_r3d_t *r, size_t i)
{
    r3d_image_t *im = r->img[i];

    r->img[i] = r->img[--r->n_img];
    image_destroy(r, im);
}

static void
image_remove_ptr(vgpu98_r3d_t *r, r3d_image_t *im)
{
    size_t i;

    for (i = 0; i < r->n_img; i++) {
        if (r->img[i] == im) {
            image_remove(r, i);
            return;
        }
    }
}

/* Levels a w x h image has at most: down to 1x1. */
static uint32_t
levels_max(uint32_t w, uint32_t h)
{
    uint32_t n = 1;

    while (((w >> n) | (h >> n)) && (n < MAX_LEVELS))
        n++;
    return n;
}

/* The image of a description: every level of every face where it is. */
static r3d_image_t *
image_find(vgpu98_r3d_t *r, uint32_t format, uint32_t w, uint32_t h, uint32_t levels, uint32_t faces,
           const uint32_t *offset, const uint32_t *pitch)
{
    r3d_image_t *im;
    uint32_t     k;

    /* As image_create keeps them: levels past the 1x1 one are none. */
    if (levels > levels_max(w, h))
        levels = levels_max(w, h);
    for (im = r->img_hash[image_bucket(format, w, h, offset[0], pitch[0])]; im; im = im->hash_next) {
        if ((im->format != format) || (im->width != w) || (im->height != h) || (im->levels != levels) ||
            (im->faces != faces))
            continue;
        for (k = 0; k < im->n_sub; k++) {
            const uint32_t at = (k / levels) * MAX_LEVELS + (k % levels);

            if ((im->sub[k].offset != offset[at]) || (im->sub[k].pitch != pitch[at]))
                break;
        }
        if (k == im->n_sub)
            return im;
    }
    return NULL;
}

/* A level or face of any image that is the surface described: a target or
   depth buffer drawn into, or what UPLOAD writes. `except`: an image not to
   look in. */
static r3d_sub_t *
sub_find(const vgpu98_r3d_t *r, uint32_t format, uint32_t w, uint32_t h, uint32_t offset, uint32_t pitch,
         const r3d_image_t *except)
{
    r3d_sub_t *s;

    for (s = r->sub_hash[image_bucket(format, w, h, offset, pitch)]; s; s = s->hash_next) {
        if ((s->im != except) && (s->im->format == format) && (s->width == w) && (s->height == h) && (s->offset == offset) &&
            (s->pitch == pitch))
            return s;
    }
    return NULL;
}

/* Bytes of one level's row and its number of rows in video memory. */
static void
level_layout(uint32_t format, uint32_t w, uint32_t h, uint32_t *row, uint32_t *rows)
{
    if (vgpu98_fmt_is_dxt(format)) {
        *row  = ((w + 3) / 4) * ((format == VGPU98_FMT_DXT1) ? 8 : 16);
        *rows = (h + 3) / 4;
    } else {
        *row  = w * (uint32_t) vgpu98_fmt_bpp(format);
        *rows = h;
    }
}

/* A host image: `levels` levels of `faces` faces (1, or 6 for a cube map),
   their offsets and pitches face by face (MAX_LEVELS apart). The image of a
   depth surface W-buffered into is float, holding what its last image held
   (depth_rec_find). */
static r3d_image_t *
image_create(vgpu98_r3d_t *r, uint32_t format, uint32_t w, uint32_t h, uint32_t levels, uint32_t faces,
             const uint32_t *offset, const uint32_t *pitch)
{
    vgpu98_vk_t          *vk  = r->vk;
    const depth_rec_t    *rec = vgpu98_fmt_is_depth(format) ? depth_rec_find(r, format, w, h, offset[0], pitch[0]) : NULL;
    const hfmt_t         *hf  = (rec && r->has_wfmt[format]) ? &r->wfmt[format] : &r->fmt[format];
    r3d_image_t          *im;
    VkImageCreateInfo     ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    VkImageAspectFlags    aspects;
    uint32_t              b;
    uint32_t              k;

    if (!r->has_fmt[format])
        return NULL;
    if (levels > levels_max(w, h))
        levels = levels_max(w, h);

    /* Host memory: without it the packet is dropped (the GPU goes on). */
    {
        r3d_image_t **list = realloc(r->img, (r->n_img + 1) * sizeof(*r->img));

        if (!list)
            return NULL;
        r->img = list;
    }
    im = calloc(1, sizeof(r3d_image_t));
    if (!im)
        return NULL;
    im->depth_w = rec ? rec->depth_w : 0;
    im->id     = ++r->next_id;
    im->format = format;
    im->width  = w;
    im->height = h;
    im->levels = levels;
    im->faces  = faces;
    im->n_sub  = levels * faces;
    im->sub    = calloc(im->n_sub, sizeof(r3d_sub_t));
    if (!im->sub) {
        free(im);
        return NULL;
    }
    im->hf     = hf;
    im->lo     = 0xffffffffu;
    im->hi     = 0;
    im->fresh  = 1;
    for (k = 0; k < im->n_sub; k++) {
        r3d_sub_t     *s  = &im->sub[k];
        const uint32_t at = (k / levels) * MAX_LEVELS + (k % levels);
        uint32_t       row;
        uint32_t       rows;

        s->im     = im;
        s->index  = k;
        s->level  = k % levels;
        s->face   = k / levels;
        s->width  = max1(w >> s->level);
        s->height = max1(h >> s->level);
        s->offset = offset[at];
        s->pitch  = pitch[at];
        level_layout(format, s->width, s->height, &row, &rows);
        s->lo = s->offset;
        s->hi = s->offset + (rows - 1) * s->pitch + row;
        if (s->lo < im->lo)
            im->lo = s->lo;
        if (s->hi > im->hi)
            im->hi = s->hi;
    }
    im->first_page = im->lo >> VGPU98_PAGE_SHIFT;
    im->n_pages    = ((im->hi - 1) >> VGPU98_PAGE_SHIFT) - im->first_page + 1;
    im->versions   = calloc(im->n_pages, sizeof(uint32_t));
    im->covered    = calloc(im->n_pages, 1);
    if (!im->versions || !im->covered) {
        free(im->versions);
        free(im->covered);
        free(im->sub);
        free(im);
        return NULL;
    }
    for (k = 0; k < im->n_sub; k++) {
        uint32_t p;

        for (p = im->sub[k].lo >> VGPU98_PAGE_SHIFT; p <= ((im->sub[k].hi - 1) >> VGPU98_PAGE_SHIFT); p++)
            im->covered[p - im->first_page] = 1;
    }
    if (vgpu98_fmt_is_depth(format) && depth_float(hf) && (im->n_sub == 1)) {
        im->shadow = calloc(1, im->sub[0].hi - im->sub[0].lo);
        r->n_shadowed += (im->shadow != NULL);
    }

    aspects          = fmt_aspects(hf);
    ci.flags         = (faces == VGPU98_CUBE_FACES) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = hf->vk;
    ci.extent.width  = w;
    ci.extent.height = h;
    ci.extent.depth  = 1;
    ci.mipLevels     = levels;
    ci.arrayLayers   = faces;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (aspects & VK_IMAGE_ASPECT_COLOR_BIT) {
        /* Any level or face of a texture may be drawn into. */
        ci.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (hf->render)
            ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    } else
        ci.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vk->vkCreateImage(vk->dev, &ci, NULL, &im->image) != VK_SUCCESS)
        goto fail;
    if (!image_memory(r, im->image, &im->memory, &im->bytes))
        goto fail;
    if (aspects & VK_IMAGE_ASPECT_COLOR_BIT) {
        vi.image                           = im->image;
        vi.viewType                        = (faces == VGPU98_CUBE_FACES) ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        vi.format                          = hf->vk;
        vi.components                      = hf->swz;
        vi.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount     = levels;
        vi.subresourceRange.layerCount     = faces;
        if (vk->vkCreateImageView(vk->dev, &vi, NULL, &im->view) != VK_SUCCESS)
            goto fail;
    }
    image_init_layout(r, im->image, aspects, levels, faces);

    r->img_bytes += im->bytes;
    r->img[r->n_img++] = im; /* room made at the start */
    b              = image_bucket(format, w, h, im->sub[0].offset, im->sub[0].pitch);
    im->hash_next  = r->img_hash[b];
    r->img_hash[b] = im;
    for (k = 0; k < im->n_sub; k++) {
        r3d_sub_t *s = &im->sub[k];

        b              = image_bucket(format, s->width, s->height, s->offset, s->pitch);
        s->hash_next   = r->sub_hash[b];
        r->sub_hash[b] = s;
    }
    r->stats->new_images++;
    vgpu98_gpu_note(r->gpu, VGPU98_NOTE_NEW_IMAGE, im->id, format | (levels << 8) | (faces << 16), (w << 16) | (h & 0xffff),
                    im->lo, im->hi);
    return im;

fail:
    r3d_fail(r, "creating an image", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    if (im->view)
        vk->vkDestroyImageView(vk->dev, im->view, NULL);
    if (im->image)
        vk->vkDestroyImage(vk->dev, im->image, NULL);
    if (im->memory)
        vk->vkFreeMemory(vk->dev, im->memory, NULL);
    r->n_shadowed -= (im->shadow != NULL);
    free(im->versions);
    free(im->covered);
    free(im->shadow);
    free(im->sub);
    free(im);
    return NULL;
}

/* A level or face as an attachment: a view made the first time it is drawn
   into. */
static VkImageView
sub_att_view(vgpu98_r3d_t *r, r3d_sub_t *s)
{
    vgpu98_vk_t          *vk = r->vk;
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    const hfmt_t         *hf = s->im->hf;

    if (s->att_view)
        return s->att_view;
    vi.image                           = s->im->image;
    vi.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    vi.format                          = hf->vk;
    vi.components                      = SWZ_ID;
    vi.subresourceRange.aspectMask     = (fmt_aspects(hf) & VK_IMAGE_ASPECT_COLOR_BIT) ? VK_IMAGE_ASPECT_COLOR_BIT : fmt_aspects(hf);
    vi.subresourceRange.baseMipLevel   = s->level;
    vi.subresourceRange.levelCount     = 1;
    vi.subresourceRange.baseArrayLayer = s->face;
    vi.subresourceRange.layerCount     = 1;
    if (vk->vkCreateImageView(vk->dev, &vi, NULL, &s->att_view) != VK_SUCCESS) {
        s->att_view = VK_NULL_HANDLE;
        r3d_fail(r, "vkCreateImageView", VK_ERROR_OUT_OF_DEVICE_MEMORY);
        return VK_NULL_HANDLE;
    }
    r->stats->views++;
    return s->att_view;
}

static int
sub_overlaps(const r3d_sub_t *s, uint32_t lo, uint32_t hi)
{
    return (s->lo < hi) && (lo < s->hi);
}

/* Does a level or face of the image (with `dirty`, one with GPU-newer
   pixels) lie in [lo, hi)? The gaps between them are other surfaces'. */
static int
image_overlaps(const r3d_image_t *im, uint32_t lo, uint32_t hi, int dirty)
{
    uint32_t k;

    if ((im->hi <= lo) || (hi <= im->lo) || (dirty && !im->dirty))
        return 0;
    for (k = 0; k < im->n_sub; k++) {
        if ((!dirty || im->sub[k].dirty) && sub_overlaps(&im->sub[k], lo, hi))
            return 1;
    }
    return 0;
}

/* Have the pages under a level or face changed since the image's last
   upload? */
static int
sub_stale(vgpu98_r3d_t *r, const r3d_image_t *im, const r3d_sub_t *s)
{
    uint32_t p;

    for (p = s->lo >> VGPU98_PAGE_SHIFT; p <= ((s->hi - 1) >> VGPU98_PAGE_SHIFT); p++) {
        if (im->versions[p - im->first_page] != vgpu98_page_version(r->vram, p))
            return 1;
    }
    return 0;
}

/* Have the pages under any of its levels and faces changed since its last
   upload? */
static int
image_stale(vgpu98_r3d_t *r, r3d_image_t *im)
{
    const uint32_t epoch = vgpu98_vram_epoch(r->vram);
    uint32_t       p;

    if (!im->valid)
        return 1;
    if (im->epoch == epoch)
        return 0; /* nothing in video memory was written since */
    for (p = 0; p < im->n_pages; p++) {
        if (im->covered[p] && (im->versions[p] != vgpu98_page_version(r->vram, im->first_page + p)))
            return 1;
    }
    im->epoch = epoch;
    return 0;
}

static void
image_snapshot(vgpu98_r3d_t *r, r3d_image_t *im)
{
    uint32_t p;

    im->epoch = vgpu98_vram_epoch(r->vram);
    for (p = 0; p < im->n_pages; p++)
        im->versions[p] = vgpu98_page_version(r->vram, im->first_page + p);
    /* The versions before the bytes the upload reads (writers store the
       byte first). */
    atomic_thread_fence(memory_order_acquire);
}

/* The same, noting the pages whose version changed since the last
   (`changed`, by page from the image's first) from the very reads it keeps,
   so a write between the two can be neither missed nor kept as seen. */
static void
image_snapshot_changed(vgpu98_r3d_t *r, r3d_image_t *im, uint8_t *changed)
{
    uint32_t p;

    im->epoch = vgpu98_vram_epoch(r->vram);
    for (p = 0; p < im->n_pages; p++) {
        const uint32_t v = vgpu98_page_version(r->vram, im->first_page + p);

        changed[p]      = (v != im->versions[p]);
        im->versions[p] = v;
    }
    atomic_thread_fence(memory_order_acquire);
}

/* GPU-newer pixels in a rectangle of a level or face. */
static void
image_add_dirty(vgpu98_r3d_t *r, r3d_sub_t *s, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
{
    r3d_image_t *im = s->im;

    if (x1 > s->width)
        x1 = s->width;
    if (y1 > s->height)
        y1 = s->height;
    if ((x0 >= x1) || (y0 >= y1))
        return;
    im->pub_new = 1;
    /* Its colour key images no longer show its pixels. */
    im->key_valid = 0;
    im->kz_valid  = 0;
    if (!s->dirty) {
        s->dirty = 1;
        im->dirty++;
        dirty_add(r, im);
        s->dx0 = x0;
        s->dy0 = y0;
        s->dx1 = x1;
        s->dy1 = y1;
        span_set(r, im, s);
        return;
    }
    if (x0 < s->dx0)
        s->dx0 = x0;
    if (y0 < s->dy0)
        s->dy0 = y0;
    if (x1 > s->dx1)
        s->dx1 = x1;
    if (y1 > s->dy1)
        s->dy1 = y1;
    span_set(r, im, s);
}

/* --- Uploads ------------------------------------------------------------------ */

/* A texel of a colour format other than P8 and DXT as the shaders sample it:
   (r, g, b, a), or a bump map's (du, dv, l, 1). */
static void
texel_rgba(uint32_t format, const uint8_t *s, float *c)
{
    const int bpp = vgpu98_fmt_bpp(format);

    switch (format) {
        case VGPU98_FMT_V8U8:
            c[0] = (int8_t) s[0] / 128.0f;
            c[1] = (int8_t) s[1] / 128.0f;
            c[2] = 1.0f;
            break;
        case VGPU98_FMT_L6V5U5: {
            const uint32_t v  = rd16(s);
            const int32_t  u5 = (int32_t) (v & 31);
            const int32_t  v5 = (int32_t) ((v >> 5) & 31);

            c[0] = (float) ((u5 >= 16) ? u5 - 32 : u5) / 16.0f;
            c[1] = (float) ((v5 >= 16) ? v5 - 32 : v5) / 16.0f;
            c[2] = ((v >> 10) & 63) / 63.0f;
            break;
        }
        case VGPU98_FMT_X8L8V8U8:
            c[0] = (int8_t) s[0] / 128.0f;
            c[1] = (int8_t) s[1] / 128.0f;
            c[2] = s[2] / 255.0f;
            break;
        case VGPU98_FMT_Q8W8V8U8: /* 2.1: (U, V, W, Q), all signed */
            c[0] = (int8_t) s[0] / 128.0f;
            c[1] = (int8_t) s[1] / 128.0f;
            c[2] = (int8_t) s[2] / 128.0f;
            c[3] = (int8_t) s[3] / 128.0f;
            return;
        case VGPU98_FMT_V16U16: /* 2.2: (U, V, 1, 1), signed */
            c[0] = (int16_t) rd16(s) / 32768.0f;
            c[1] = (int16_t) rd16(s + 2) / 32768.0f;
            c[2] = 1.0f;
            break;
        case VGPU98_FMT_G16R16: /* 2.2: (R, G, 1, 1) */
            c[0] = rd16(s) / 65535.0f;
            c[1] = rd16(s + 2) / 65535.0f;
            c[2] = 1.0f;
            break;
        case VGPU98_FMT_A2W10V10U10: { /* 2.2: (U, V, W, A), U, V and W signed */
            const uint32_t v = rd32(s);
            int            k;

            for (k = 0; k < 3; k++)
                c[k] = (float) ((int32_t) (((v >> (10 * k)) & 0x3ffu) ^ 0x200u) - 0x200) / 512.0f;
            c[3] = (v >> 30) / 3.0f;
            return;
        }
        default:
            rgb_unpack(format, (bpp == 4) ? rd32(s) : (bpp == 2) ? rd16(s) : s[0], c);
            return;
    }
    c[3] = 1.0f;
}

/* Converts a w x h rectangle of video memory at `src` (rows `pit` apart; for
   DXT, rows of blocks) into host texels, tightly packed. */
static void
convert_rect(const vgpu98_r3d_t *r, const r3d_image_t *im, const uint8_t *src, uint32_t pit, uint32_t w, uint32_t h,
             const uint32_t *palette, int pal_alpha, uint8_t *dst, uint8_t *stencil)
{
    const int bpp = vgpu98_fmt_bpp(im->format);
    uint32_t  x;
    uint32_t  y;

    switch (im->hf->conv) {
        case CONV_COPY:
        case CONV_D16:
            for (y = 0; y < h; y++)
                memcpy(dst + (size_t) y * w * bpp, src + (size_t) y * pit, (size_t) w * bpp);
            break;
        case CONV_F16:
        case CONV_BUMP:
            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++) {
                    float c[4];
                    int   k;

                    texel_rgba(im->format, src + (size_t) y * pit + x * bpp, c);
                    for (k = 0; k < 4; k++)
                        wr16(dst + ((size_t) y * w + x) * 8 + 2 * k, f2h(c[k]));
                }
            }
            break;
        case CONV_WIDE:
            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++) {
                    float c[4];

                    texel_rgba(im->format, src + (size_t) y * pit + x * bpp, c);
                    memcpy(dst + ((size_t) y * w + x) * 16, c, sizeof(c));
                }
            }
            break;
        case CONV_P8:
            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++) {
                    uint32_t c = palette[src[(size_t) y * pit + x]];

                    if (!pal_alpha)
                        c |= 0xff000000u;
                    wr32(dst + ((size_t) y * w + x) * 4, c);
                }
            }
            break;
        case CONV_DXT:
            dxt_decode(im->format, src, pit, w, h, dst);
            break;
        case CONV_D24S8:
        case CONV_D24F32:
            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++) {
                    uint32_t v = rd32(src + (size_t) y * pit + x * 4);

                    if (im->hf->conv == CONV_D24S8)
                        wr32(dst + ((size_t) y * w + x) * 4, v >> 8);
                    else {
                        float f = depth_in(r, im, v >> 8, 24);

                        memcpy(dst + ((size_t) y * w + x) * 4, &f, 4);
                    }
                    stencil[(size_t) y * w + x] = (im->format == VGPU98_FMT_D24S8) ? (uint8_t) v : 0;
                }
            }
            break;
        case CONV_D16F32:
            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++) {
                    float f = depth_in(r, im, rd16(src + (size_t) y * pit + x * 2), 16);

                    memcpy(dst + ((size_t) y * w + x) * 4, &f, 4);
                }
            }
            break;
    }
}

static void
copy_region(VkBufferImageCopy *c, VkDeviceSize off, uint32_t row, VkImageAspectFlags aspect, const r3d_sub_t *s, uint32_t x,
            uint32_t y, uint32_t w, uint32_t h)
{
    memset(c, 0, sizeof(*c));
    c->bufferOffset                    = off;
    c->bufferRowLength                 = row;
    c->bufferImageHeight               = h;
    c->imageSubresource.aspectMask     = aspect;
    c->imageSubresource.mipLevel       = s ? s->level : 0;
    c->imageSubresource.baseArrayLayer = s ? s->face : 0;
    c->imageSubresource.layerCount     = 1;
    c->imageOffset.x                   = (int32_t) x;
    c->imageOffset.y                   = (int32_t) y;
    c->imageExtent.width               = w;
    c->imageExtent.height              = h;
    c->imageExtent.depth               = 1;
}

/* The bytes of a rectangle of a level or face to convert: video memory's,
   or for a shadowed image (a float depth image) its shadow's, after copying
   them there, so the shadow holds exactly what the image was made from
   (the guest may write meanwhile). */
static const uint8_t *
sub_source(vgpu98_r3d_t *r, r3d_image_t *im, const r3d_sub_t *s, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
{
    const size_t bpp = (size_t) vgpu98_fmt_bpp(im->format);
    uint32_t     y;

    if (vgpu98_fmt_is_dxt(im->format))
        return r->vram->data + s->offset + (y0 / 4) * s->pitch + (x0 / 4) * ((im->format == VGPU98_FMT_DXT1) ? 8 : 16);
    if (!im->shadow)
        return r->vram->data + s->offset + y0 * s->pitch + x0 * bpp;
    for (y = y0; y < y1; y++)
        memcpy(im->shadow + (size_t) y * s->pitch + x0 * bpp, r->vram->data + s->offset + (size_t) y * s->pitch + x0 * bpp,
               (size_t) (x1 - x0) * bpp);
    return im->shadow + (size_t) y0 * s->pitch + x0 * bpp;
}

/* Uploads a rectangle of a level or face from video memory (for DXT, x0 and
   y0 multiples of 4). Returns the bytes sent, or 0 when the staging buffer
   could not be had. */
static uint64_t
sub_upload(vgpu98_r3d_t *r, r3d_image_t *im, const r3d_sub_t *s, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
           const uint32_t *palette, int pal_alpha)
{
    const uint32_t     w     = x1 - x0;
    const uint32_t     h     = y1 - y0;
    const VkDeviceSize color = (VkDeviceSize) w * h * im->hf->bytes;
    const VkDeviceSize sten  = (fmt_aspects(im->hf) & VK_IMAGE_ASPECT_STENCIL_BIT) ? (VkDeviceSize) w * h : 0;
    const uint8_t     *src;
    VkBuffer           buf;
    VkDeviceSize       off;
    uint8_t           *p;
    VkBufferImageCopy  c[2];
    int                n = 1;

    /* At most half a slot's first staging a copy: a larger rectangle (a
       2048x2048 level of 2.2's 16-byte texels is 64 MB) goes in bands of
       rows, DXT's in whole rows of blocks, so a full slot is submitted
       rather than its buffer grown. */
    if ((h > 1) && (color + sten > STAGING_MIN / 2)) {
        const int      dxt  = vgpu98_fmt_is_dxt(im->format);
        uint32_t       band = (uint32_t) ((STAGING_MIN / 2) / ((color + sten) / h));
        uint64_t       sent = 0;
        uint32_t       y;

        if (dxt)
            band &= ~3u;
        if (band < (dxt ? 4u : 1u))
            band = dxt ? 4u : 1u;
        for (y = y0; y < y1; y += band) {
            const uint64_t b = sub_upload(r, im, s, x0, y, x1, (y1 - y > band) ? y + band : y1, palette, pal_alpha);

            if (!b)
                return 0;
            sent += b;
        }
        return sent;
    }
    p = slot_alloc(r, 0, color + sten + 4, 16, &buf, &off);
    if (!p)
        return 0;
    src = sub_source(r, im, s, x0, y0, x1, y1);
    convert_rect(r, im, src, s->pitch, w, h, palette, pal_alpha, p, p + ((color + 3) & ~(VkDeviceSize) 3));
    copy_region(&c[0], off, w, (fmt_aspects(im->hf) & VK_IMAGE_ASPECT_COLOR_BIT) ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT,
                s, x0, y0, w, h);
    if (sten) {
        copy_region(&c[1], off + ((color + 3) & ~(VkDeviceSize) 3), w, VK_IMAGE_ASPECT_STENCIL_BIT, s, x0, y0, w, h);
        n = 2;
    }
    r->vk->vkCmdCopyBufferToImage(cmd(r), buf, im->image, VK_IMAGE_LAYOUT_GENERAL, (uint32_t) n, c);
    return (color + sten) ? (color + sten) : 1;
}

/* --- The texture audit ------------------------------------ */

static int
tex_audit_on(const vgpu98_r3d_t *r)
{
    return vgpu98_gpu_logging(r->gpu, VGPU98_LOG_TEXTURES);
}

/* A format's name, for the texture audit's lines. */
static const char *
fmt_name(uint32_t format)
{
    static const char *const names[N_FORMATS] = {
        [VGPU98_FMT_R5G6B5] = "R5G6B5",     [VGPU98_FMT_X1R5G5B5] = "X1R5G5B5", [VGPU98_FMT_A1R5G5B5] = "A1R5G5B5",
        [VGPU98_FMT_A4R4G4B4] = "A4R4G4B4", [VGPU98_FMT_X8R8G8B8] = "X8R8G8B8", [VGPU98_FMT_A8R8G8B8] = "A8R8G8B8",
        [VGPU98_FMT_P8] = "P8",             [VGPU98_FMT_L8] = "L8",             [VGPU98_FMT_A8L8] = "A8L8",
        [VGPU98_FMT_V8U8] = "V8U8",         [VGPU98_FMT_L6V5U5] = "L6V5U5",     [VGPU98_FMT_X8L8V8U8] = "X8L8V8U8",
        [VGPU98_FMT_A8] = "A8",             [VGPU98_FMT_X4R4G4B4] = "X4R4G4B4", [VGPU98_FMT_A4L4] = "A4L4",
        [VGPU98_FMT_DXT1] = "DXT1",         [VGPU98_FMT_DXT2] = "DXT2",         [VGPU98_FMT_DXT3] = "DXT3",
        [VGPU98_FMT_DXT4] = "DXT4",         [VGPU98_FMT_DXT5] = "DXT5",         [VGPU98_FMT_Q8W8V8U8] = "Q8W8V8U8",
        [VGPU98_FMT_V16U16] = "V16U16",     [VGPU98_FMT_G16R16] = "G16R16",     [VGPU98_FMT_A2W10V10U10] = "A2W10V10U10",
        [VGPU98_FMT_D16] = "D16",           [VGPU98_FMT_D24X8] = "D24X8",       [VGPU98_FMT_D24S8] = "D24S8",
    };

    return ((format < N_FORMATS) && names[format]) ? names[format] : "format ?";
}

/* A host conversion's name (CONV_*). */
static const char *
conv_name(int conv)
{
    static const char *const names[CONV_COUNT] = { "copied", "half float", "bump", "wide", "through the palette", "DXT decoded",
                                                   "D16", "D24S8", "24-bit depth as float", "16-bit depth as float" };

    return ((conv >= 0) && (conv < CONV_COUNT)) ? names[conv] : "?";
}

/* A finding (VGPU98_FIND_*): counted; the first FIND_LOGGED of its kind
   named, and again after FIND_QUIET_MS without one; every one asks for a
   capture about host image `image` (0: none) and its level or face `sub`
   (index plus one, 0: none), which vgpu98_gpu_find's limits decide. Its
   line names the drawing process, when the DirectX driver said which
   (a PROC note). Returns whether it was named (the caller may add lines). */
static int
tex_find_v(vgpu98_r3d_t *r, int kind, uint32_t image, uint32_t sub, const char *fmt, va_list ap)
{
    const uint32_t ms = (uint32_t) (vgpu98_diag_us() / 1000);
    char           line[VGPU98_FIND_WHY];
    size_t         n = 0;
    int            named;

    r->ta_finds[kind]++;
    if (r->find_logged[kind] && ((uint32_t) (ms - r->find_ms[kind]) >= FIND_QUIET_MS))
        r->find_logged[kind] = 0;
    r->find_ms[kind] = ms;
    named            = (r->find_logged[kind] < FIND_LOGGED);
    if (named)
        r->find_logged[kind]++;
    else if (!vgpu98_gpu_find_wants(r->gpu, kind))
        return 0; /* counted; no line, no capture: nothing to format */
    if (r->proc_name[0])
        n = (size_t) snprintf(line, sizeof(line), "[%s] ", r->proc_name);
    vsnprintf(line + n, sizeof(line) - n, fmt, ap);
    if (named)
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: texture audit: %s\n", line);
    vgpu98_gpu_find(r->gpu, kind, image, sub, line, named);
    return named;
}

static int
tex_find(vgpu98_r3d_t *r, int kind, uint32_t image, const char *fmt, ...)
{
    va_list ap;
    int     named;

    va_start(ap, fmt);
    named = tex_find_v(r, kind, image, 0, fmt, ap);
    va_end(ap);
    return named;
}

static int
tex_find_at(vgpu98_r3d_t *r, int kind, uint32_t image, uint32_t sub, const char *fmt, ...)
{
    va_list ap;
    int     named;

    va_start(ap, fmt);
    named = tex_find_v(r, kind, image, sub, fmt, ap);
    va_end(ap);
    return named;
}

/* A level's or face's rows in video memory as the texture audit hashes
   them (the rows alone, not the pitch's padding); *bytes grows by them. */
static uint64_t
sub_hash(const vgpu98_r3d_t *r, const r3d_image_t *im, const r3d_sub_t *s, uint64_t *bytes)
{
    uint32_t row;
    uint32_t rows;
    uint32_t y;
    uint64_t h;

    level_layout(im->format, s->width, s->height, &row, &rows);
    h = 0x9e3779b97f4a7c15ull ^ ((uint64_t) row << 32) ^ rows;
    for (y = 0; y < rows; y++) {
        const uint8_t *p = r->vram->data + s->offset + (size_t) y * s->pitch;
        uint32_t       x = 0;
        uint64_t       w;

        for (; x + 8 <= row; x += 8) {
            memcpy(&w, p + x, 8);
            h = (h ^ w) * 0xff51afd7ed558ccdull;
            h ^= h >> 29;
        }
        for (; x < row; x++) {
            h = (h ^ p[x]) * 0xff51afd7ed558ccdull;
            h ^= h >> 29;
        }
    }
    *bytes += (uint64_t) row * rows;
    return h;
}

/* The versions the image holds for the pages a level or face lies on. */
static uint32_t
sub_vsig(const r3d_image_t *im, const r3d_sub_t *s)
{
    const uint32_t p1 = ((s->hi - 1) >> VGPU98_PAGE_SHIFT) - im->first_page;
    uint32_t       p  = (s->lo >> VGPU98_PAGE_SHIFT) - im->first_page;
    uint32_t       h  = 0x811c9dc5u;

    for (; p <= p1; p++)
        h = (h ^ im->versions[p]) * 0x01000193u;
    return h;
}

/* The churn table's record of a texture (by its first level's address,
   format and size), made if new; NULL when the table is three quarters
   full or was not made. */
static churn_t *
churn_rec(vgpu98_r3d_t *r, const r3d_image_t *im)
{
    const uint32_t offset = im->sub[0].offset;
    uint32_t       i;

    if (!r->churn && !(r->churn = calloc(CHURN_RECS, sizeof(churn_t))))
        return NULL;
    i = ((offset >> 4) * 0x9e3779b1u ^ (im->format * 0x85ebca6bu) ^ (im->width << 16) ^ im->height) & (CHURN_RECS - 1);
    for (;; i = (i + 1) & (CHURN_RECS - 1)) {
        churn_t *c = &r->churn[i];

        if ((c->offset == offset) && (c->format == im->format) && (c->width == im->width) && (c->height == im->height))
            return c;
        if (!c->format && !c->offset) {
            if (r->n_churn >= CHURN_RECS * 3 / 4) {
                r->churn_lost++;
                return NULL;
            }
            r->n_churn++;
            c->offset = offset;
            c->format = im->format;
            c->width  = im->width;
            c->height = im->height;
            return c;
        }
    }
}

/* Is an 8-bit image's palette signature (pal_sig) a palette of zeros's,
   read with or without its alpha? */
static int
pal_zero(uint32_t sig)
{
    static uint32_t zero_sig[2];
    static int      made;

    if (!made) {
        const uint32_t z[256] = { 0 };

        zero_sig[0] = fnv(z, 1024, 0x01000193u);
        zero_sig[1] = fnv(z, 1024, 0x811c9dc5u);
        made        = 1;
    }
    return (sig == zero_sig[0]) || (sig == zero_sig[1]);
}

/* A texture's image evicted (for room, or unused STALE_SECONDS). */
static void
churn_evicted(vgpu98_r3d_t *r, const r3d_image_t *im)
{
    churn_t *c;

    if (im->tex_use && tex_audit_on(r) && (c = churn_rec(r, im)))
        c->evicted++;
}

/* An image's description for the texture audit's lines. */
static void
tex_text(char *buf, size_t size, const r3d_image_t *im)
{
    snprintf(buf, size, "%s%ux%u %s, %u level%s, at %08X", (im->faces == VGPU98_CUBE_FACES) ? "cube map " : "", im->width,
             im->height, fmt_name(im->format), im->levels, (im->levels == 1) ? "" : "s", im->sub[0].offset);
}

/* The stale-texture check, after a draw's textures are made
   ready. A texture is checked at most once per ta_every (1 s, doubled up to
   TEX_AUDIT_EVERY_MAX while a second's TEX_AUDIT_BYTES run out and textures
   wait, halved again when under half are hashed), at most TEX_AUDIT_DRAW
   bytes begun a draw; one past TEX_AUDIT_BIG is not checked. A level or
   face with GPU-newer pixels is skipped. One not hashed since video memory
   and the host's copy last agreed (an upload hashes it; a copy back, an
   UPLOAD's rectangle or a packet overwriting it clear the hash) is hashed;
   one hashed is compared whatever its pages' recorded versions say: they
   move with an upload of it, which hashes it again, and moved without one
   the host may hold a level it never uploaded. A changed one makes the
   texture a suspect, named when the next check finds the same. Nothing is
   repaired: the host draws what it holds until the pages' versions move. */
static void
tex_check(vgpu98_r3d_t *r, r3d_image_t *im)
{
    const uint64_t now   = vgpu98_diag_us();
    uint64_t       bytes = 0;
    r3d_sub_t     *bad   = NULL;
    int            moved = 0;
    uint32_t       k;

    if (now - r->ta_second >= 1000000) {
        if ((r->ta_bytes >= TEX_AUDIT_BYTES) && r->ta_waited)
            r->ta_every = (r->ta_every * 2 > TEX_AUDIT_EVERY_MAX) ? TEX_AUDIT_EVERY_MAX : r->ta_every * 2;
        else if ((r->ta_bytes < TEX_AUDIT_BYTES / 2) && (r->ta_every > 1000000))
            r->ta_every /= 2;
        r->ta_second = now;
        r->ta_bytes  = 0;
        r->ta_waited = 0;
    }
    if (!r->ta_every)
        r->ta_every = 1000000;
    if (im->a_named || !im->valid || im->shadow || vgpu98_fmt_is_depth(im->format) || (now - im->a_checked < r->ta_every))
        return;
    if (im->hi - im->lo > TEX_AUDIT_BIG) {
        im->a_checked = now;
        r->ta_big++;
        return;
    }
    if ((r->ta_bytes >= TEX_AUDIT_BYTES) || (r->ta_draw_bytes >= TEX_AUDIT_DRAW)) {
        r->ta_waited++;
        return;
    }
    im->a_checked = now;
    for (k = 0; k < im->n_sub; k++) {
        r3d_sub_t     *s = &im->sub[k];
        const uint32_t v = sub_vsig(im, s);
        uint64_t       h;

        if (s->dirty)
            continue;
        h = sub_hash(r, im, s, &bytes);
        if (!s->aset) {
            s->ahash = h;
            s->avsig = v;
            s->aset  = 1;
        } else if ((h != s->ahash) && !bad) {
            bad   = s;
            moved = (v != s->avsig);
        }
    }
    r->ta_bytes += bytes;
    r->ta_draw_bytes += bytes;
    r->ta_hash_check += bytes;
    r->ta_checks++;
    if (!bad) {
        im->a_suspect = 0;
        return;
    }
    if (!im->a_suspect) {
        im->a_suspect = 1;
        r->ta_suspects++;
        return;
    }
    {
        char t[96];
        char face[24];

        im->a_suspect = 0;
        im->a_named   = 1;
        tex_text(t, sizeof(t), im);
        face[0] = 0;
        if (im->faces == VGPU98_CUBE_FACES)
            snprintf(face, sizeof(face), ", face %u", bad->face);
        tex_find_at(r, VGPU98_FIND_STALE, im->id, bad->index + 1,
                    "possible stale texture: %s: level %u%s at %08X (%ux%u, pitch %u) changed in video memory %s; the host GPU "
                    "draws what it holds",
                    t, bad->level, face, bad->offset, bad->width, bad->height, bad->pitch,
                    moved ? "since its upload, its pages' versions recorded as current without an upload of it"
                          : "without its pages' versions moving");
    }
}

/* Uploads the image's levels and faces from video memory: every one, or
   those `which` marks (the others are current, or filled otherwise). */
static int
image_upload(vgpu98_r3d_t *r, r3d_image_t *im, const uint32_t *palette, int pal_alpha, const uint8_t *which)
{
    const int audit = tex_audit_on(r) && !vgpu98_fmt_is_depth(im->format);
    uint64_t  sent  = 0;
    uint32_t  k;

    pass_end(r);
    barrier(r);
    /* Before video memory is read: a guest write during the conversion
       makes the image stale again. */
    image_snapshot(r, im);
    for (k = 0; k < im->n_sub; k++) {
        r3d_sub_t *s = &im->sub[k];
        uint64_t   b;

        if (which && !which[k])
            continue;
        b = sub_upload(r, im, s, 0, 0, s->width, s->height, palette, pal_alpha);
        if (!b) {
            im->valid = 0;
            return 0;
        }
        sent += b;
        /* The texture audit's hash, with the versions of the snapshot: a
           write since moved them, and the image uploads again. */
        if (audit) {
            s->ahash = sub_hash(r, im, s, &r->ta_hash_up);
            s->avsig = sub_vsig(im, s);
            s->aset  = 1;
        }
    }
    barrier(r);
    im->valid       = 1;
    im->a_suspect   = 0;
    im->a_named     = 0;
    im->a_zero_told = 0;
    if (audit)
        im->a_checked = vgpu98_diag_us(); /* just hashed: its first check is an interval on */
    if (im->format == VGPU98_FMT_P8) {
        im->pal_sig = fnv(palette, 1024, pal_alpha ? 0x811c9dc5u : 0x01000193u);
        if (audit)
            pal_keep(r, im->pal_sig, palette, pal_alpha);
    }
    im->key_valid = 0;
    im->kz_valid  = 0;
    r->stats->uploads++;
    r->stats->upload_bytes += sent;
    if (audit && im->tex_use) {
        churn_t *c = churn_rec(r, im);

        if (c) {
            c->uploads++;
            c->bytes += sent;
        }
    }
    vgpu98_gpu_note(r->gpu, VGPU98_NOTE_UPLOAD, im->id, im->lo, im->hi, (uint32_t) sent, 0);
    return 1;
}

/* --- Resident buffers ------------------------------------------------------------ */

/* By a buffer's end (address + size, below 4 GB) and version. */
static uint32_t
res_bucket(uint32_t end, uint32_t version)
{
    uint32_t x = (end * 0x9e3779b1u) ^ (version * 0xc2b2ae35u);

    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return x & (RES_HASH - 1);
}

/* The record holding [address, end) of this version, and *delta, where
   address is in it. */
static res_t *
res_find(vgpu98_r3d_t *r, uint32_t address, uint64_t end, uint32_t version, uint32_t *delta)
{
    int i;

    for (i = r->res_hash[res_bucket((uint32_t) end, version)]; i; i = r->res[i - 1].hash_next) {
        res_t *e = &r->res[i - 1];

        if (((uint64_t) e->address + e->size == end) && (e->version == version) && (e->address <= address) && !e->stale) {
            *delta = address - e->address;
            return e;
        }
    }
    return NULL;
}

/* Record k of the arena's order goes. */
static void
res_remove(vgpu98_r3d_t *r, uint32_t k)
{
    const int idx = r->res_order[k];
    res_t    *e   = &r->res[idx];
    int      *pp  = &r->res_hash[res_bucket(e->address + e->size, e->version)];

    while (*pp && (*pp != idx + 1))
        pp = &r->res[*pp - 1].hash_next;
    if (*pp)
        *pp = e->hash_next;
    e->used = 0;
    memmove(r->res_order + k, r->res_order + k + 1, (r->n_res - k - 1) * sizeof(int));
    r->n_res--;
}

/* The pending submissions that have finished, asked once before records
   are judged idle. */
static void
res_poll(vgpu98_r3d_t *r)
{
    int i;

    for (i = 0; i < SLOTS; i++) {
        slot_t *s = &r->slot[i];

        if (s->pending && !s->finished && (r->vk->vkGetFenceStatus(r->vk->dev, s->fence) == VK_SUCCESS)) {
            s->finished = 1;
            if (s->sub_serial > r->done_serial)
                r->done_serial = s->sub_serial;
        }
    }
}

/* Has the submission that read record e last finished? The recording being
   made has not; a pending slot carrying its serial has if its fence said so;
   otherwise its slot was begun again, after waiting for it. (Each
   submission's own fence: Vulkan does not order separate submissions'
   signals.) Serials start at 1. */
static int
res_idle(const vgpu98_r3d_t *r, const res_t *e)
{
    int i;

    if (e->serial >= r->serial)
        return 0;
    for (i = 0; i < SLOTS; i++) {
        if (r->slot[i].pending && (r->slot[i].sub_serial == e->serial))
            return r->slot[i].finished;
    }
    return 1;
}

/* Room for `bytes` by eviction: the least recently used idle record goes,
   with its idle neighbours in the arena (the less recently used first)
   until the gap they leave holds it. 1 if one was made. */
static int
res_evict_for(vgpu98_r3d_t *r, VkDeviceSize bytes)
{
    uint32_t tried[RES_ENTRIES / 32];
    int      round;

    memset(tried, 0, sizeof(tried));
    for (round = 0; round < 16; round++) {
        uint32_t     k;
        int          oldest = -1;
        uint32_t     lo;
        uint32_t     hi;
        VkDeviceSize start;
        VkDeviceSize end;

        for (k = 0; k < r->n_res; k++) {
            const res_t *e = &r->res[r->res_order[k]];

            if (!((tried[k / 32] >> (k % 32)) & 1) && res_idle(r, e) &&
                ((oldest < 0) || (e->tick < r->res[r->res_order[oldest]].tick)))
                oldest = (int) k;
        }
        if (oldest < 0)
            return 0;
        tried[oldest / 32] |= 1u << (oldest % 32);
        lo = hi = (uint32_t) oldest;
        for (;;) {
            const res_t *l = (lo > 0) ? &r->res[r->res_order[lo - 1]] : NULL;
            const res_t *h = (hi + 1 < r->n_res) ? &r->res[r->res_order[hi + 1]] : NULL;
            const int    lok = l && res_idle(r, l);
            const int    hok = h && res_idle(r, h);

            start = l ? l->at + l->bytes : 0;
            end   = h ? h->at : RES_BYTES;
            if (end - start >= bytes)
                break;
            if (lok && (!hok || (l->tick <= h->tick)))
                lo--;
            else if (hok)
                hi++;
            else
                break;
        }
        if (end - start < bytes)
            continue;
        for (k = hi + 1; k-- > lo;) {
            res_remove(r, k);
            r->stats->res_evictions++;
        }
        return 1;
    }
    return 0;
}

/* First fit between the records: the place in the order and the offset. */
static int
res_fit(const vgpu98_r3d_t *r, VkDeviceSize bytes, uint32_t *pos, VkDeviceSize *at)
{
    VkDeviceSize prev = 0;
    uint32_t     k;

    if (r->n_res >= RES_ENTRIES)
        return 0;
    for (k = 0; k <= r->n_res; k++) {
        const VkDeviceSize next = (k < r->n_res) ? r->res[r->res_order[k]].at : RES_BYTES;

        if (next - prev >= bytes) {
            *pos = k;
            *at  = prev;
            return 1;
        }
        if (k < r->n_res)
            prev = r->res[r->res_order[k]].at + r->res[r->res_order[k]].bytes;
    }
    return 0;
}

/* A buffer into the arena: room for it (by first fit between the records,
   else by eviction), its bytes read from guest RAM through the page table
   (`kind` for the summary). NULL if it cannot go in; *unmapped when the
   page table does not map it all. */
static res_t *
res_insert(vgpu98_r3d_t *r, uint32_t address, uint32_t size, uint32_t version, int kind, int *unmapped)
{
    const VkDeviceSize bytes = ((VkDeviceSize) size + RES_ALIGN - 1) & ~(VkDeviceSize) (RES_ALIGN - 1);
    int                idx   = -1;
    uint32_t           pos   = 0;
    VkDeviceSize       at    = 0;
    res_t             *e;
    uint32_t           k;
    int                i;

    if (!r->res_buf.buffer) {
        if (r->res_failed)
            return NULL;
        if (!r->res && !(r->res = calloc(RES_ENTRIES, sizeof(res_t)))) {
            r->res_failed = 1;
            return NULL;
        }
        if (!r->res_order && !(r->res_order = calloc(RES_ENTRIES, sizeof(int)))) {
            r->res_failed = 1;
            return NULL;
        }
        if (!vgpu98_vk_buffer_create(r->vk, &r->res_buf, RES_BYTES,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) {
            r->res_failed = 1;
            if (!r->res_logged++)
                r3d_log(r, VGPU98_LOG_EVENTS,
                        "VGPU98 3D: no arena for resident buffers (%u MB of host memory); buffers are read per draw\n",
                        RES_BYTES >> 20);
            return NULL;
        }
    }
    /* Mapped first (it costs no copy): a buffer the table does not map
       evicts nothing. */
    if (((uint64_t) address + size > 0x100000000ull) || vgpu98_gpu_gpuvm_read(r->gpu, address, size, NULL, -1)) {
        *unmapped = 1;
        return NULL;
    }
    res_poll(r);
    /* The buffer's other records (by its end) that nothing reads go: older
       versions, and this version's from a higher start, which this one
       holds. */
    for (k = r->n_res; k-- > 0;) {
        const res_t *o = &r->res[r->res_order[k]];

        if (((uint64_t) o->address + o->size == (uint64_t) address + size) && ((o->version != version) || (o->address > address)) &&
            res_idle(r, o)) {
            res_remove(r, k);
            r->stats->res_evictions++;
        }
    }
    if (!res_fit(r, bytes, &pos, &at)) {
        /* A record first, when all are in use; then the room. */
        if ((r->n_res >= RES_ENTRIES) && !res_evict_for(r, 0))
            return NULL;
        if (!res_fit(r, bytes, &pos, &at) && (!res_evict_for(r, bytes) || !res_fit(r, bytes, &pos, &at)))
            return NULL;
    }
    for (i = 0; i < RES_ENTRIES; i++) {
        if (!r->res[i].used) {
            idx = i;
            break;
        }
    }
    if ((idx < 0) || vgpu98_gpu_gpuvm_read(r->gpu, address, size, (uint8_t *) r->res_buf.map + at, kind))
        return NULL;
    e            = &r->res[idx];
    e->address   = address;
    e->size      = size;
    e->version   = version;
    e->used      = 1;
    e->stale     = 0;
    e->checked   = vgpu98_diag_us();
    e->at        = at;
    e->bytes     = bytes;
    e->serial    = r->serial;
    e->tick      = r->tick;
    e->hash_next = r->res_hash[res_bucket(address + size, version)];
    r->res_hash[res_bucket(address + size, version)] = idx + 1;
    memmove(r->res_order + pos + 1, r->res_order + pos, (r->n_res - pos) * sizeof(int));
    r->res_order[pos] = idx;
    r->n_res++;
    r->stats->res_inserts++;
    r->stats->res_insert_bytes += size;
    return e;
}

/* A buffer at a GPU address for a draw (`kind`: vertices or indices): its
   record if the arena holds it (*delta: where address is in it), or puts
   it there now if a draw of an earlier recording read it too. The record is
   the draw's (res_used), and stamped with the recording that draws it
   (res_stamp). */
static res_t *
res_get(vgpu98_r3d_t *r, uint32_t address, uint32_t size, uint32_t version, int kind, uint32_t *delta)
{
    const uint64_t end = (uint64_t) address + size;
    res_t         *e   = (r->res && (end <= 0x100000000ull)) ? res_find(r, address, end, version, delta) : NULL;
    const uint64_t key = ((uint64_t) (uint32_t) end << 32) | version; /* the buffer's end and version */
    res_seen_t    *seen;
    int            unmapped = 0;

    if (!e) {
        r->stats->res_misses++;
        if (!size || (size > RES_BIG) || r->res_failed || (end > 0x100000000ull))
            return NULL;
        seen = &r->res_seen[res_bucket((uint32_t) end, version) & (RES_SEEN - 1)];
        if (seen->key != key) {
            /* The first time: read per draw. */
            seen->key    = key;
            seen->serial = r->serial;
            seen->failed = 0;
            return NULL;
        }
        if (seen->failed || (seen->serial == r->serial))
            return NULL;
        e = res_insert(r, address, size, version, kind, &unmapped);
        if (!e) {
            /* Not mapped: never; no room now: again in a later recording. */
            seen->failed = unmapped;
            seen->serial = r->serial;
            return NULL;
        }
        seen->key = 0;
        *delta    = 0;
    } else
        r->stats->res_hits++;
    e->serial = r->serial;
    e->tick   = r->tick;
    if (r->n_res_used < VGPU98_MAX_STREAMS + 1)
        r->res_used[r->n_res_used++] = e;
    return e;
}

/* The draw's records are read by the recording it goes into, which may
   have moved on since res_draw (a submission for room). */
static void
res_stamp(vgpu98_r3d_t *r)
{
    uint32_t i;

    for (i = 0; i < r->n_res_used; i++)
        r->res_used[i]->serial = r->serial;
}

/* The vertex audit's check of the arena (each record at most once a
   second): the draw's window of each resident stream, and its indices,
   against guest RAM. A guest that changed them without a new version broke
   the rule (a NOOVERWRITE Lock's writes elsewhere in the buffer are not in
   the window): the record is stale, no draw reads it again (it goes when it
   is evicted), named the first three times and again after 10 seconds
   without one. This draw has read it already. */
#define STALE_LOGGED   3
#define STALE_QUIET_MS 10000

static void
res_check_draw(vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d)
{
    const vgpu98_vin_t *v   = &r->vin;
    const uint64_t      now = vgpu98_diag_us();
    uint32_t            s;

    if (!r->res_scratch && !(r->res_scratch = malloc(RES_BIG)))
        return;
    for (s = 0; s <= VGPU98_MAX_STREAMS; s++) {
        res_t   *e = (s < VGPU98_MAX_STREAMS) ? r->res_stream[s] : r->res_ib;
        uint64_t addr;
        uint64_t len;
        uint64_t at;
        uint64_t rend;
        uint32_t ms;

        if (!e || e->stale || (now - e->checked < 1000000))
            continue;
        if (s < VGPU98_MAX_STREAMS) {
            if (!((v->res_mask >> s) & 1))
                continue;
            addr = (uint64_t) v->stream[s].address + v->copy[s].lo;
            len  = (uint64_t) v->copy[s].words * 4;
            at   = (uint64_t) v->res_word[s] * 4 + v->copy[s].lo;
        } else {
            const uint64_t have = v->index.size / v->index.isize;

            if (!d->indexed || (d->start >= have))
                continue;
            addr = (uint64_t) v->index.address + (uint64_t) d->start * v->index.isize;
            len  = ((have - d->start < d->count) ? have - d->start : d->count) * v->index.isize;
            at   = r->res_ib_at + (uint64_t) d->start * v->index.isize;
        }
        /* Within the record. */
        rend = (uint64_t) e->address + e->size;
        if (addr + len > rend)
            len = (addr < rend) ? rend - addr : 0;
        if (!len || (len > RES_BIG))
            continue;
        e->checked = now;
        if (vgpu98_gpu_gpuvm_read(r->gpu, (uint32_t) addr, (uint32_t) len, r->res_scratch, -1) ||
            !memcmp(r->res_scratch, (const uint8_t *) r->res_buf.map + at, (size_t) len))
            continue;
        e->stale = 1;
        r->stats->res_stale++;
        ms = (uint32_t) (now / 1000);
        if (r->stale_logged && ((uint32_t) (ms - r->stale_ms) >= STALE_QUIET_MS))
            r->stale_logged = 0;
        r->stale_ms = ms;
        if (r->stale_logged < STALE_LOGGED) {
            r->stale_logged++;
            r3d_log(r, VGPU98_LOG_EVENTS,
                    "VGPU98 3D: vertex audit: a draw's %s at GPU address %08X (%u bytes, version %08X) changed in guest RAM "
                    "without a new version; draws read it from guest RAM again\n",
                    (s < VGPU98_MAX_STREAMS) ? "vertices" : "indices", (uint32_t) addr, (uint32_t) len, e->version);
        }
    }
}

/* Every record goes (a reset); the arena stays. */
static void
res_forget(vgpu98_r3d_t *r)
{
    uint32_t i;

    if (r->res) {
        for (i = 0; i < RES_ENTRIES; i++)
            r->res[i].used = 0;
    }
    memset(r->res_hash, 0, sizeof(r->res_hash));
    memset(r->res_seen, 0, sizeof(r->res_seen));
    r->n_res      = 0;
    r->n_res_used = 0;
    /* An arena that could not be made is tried again. */
    if (!r->res_buf.buffer)
        r->res_failed = 0;
}

/* Which of the draw's buffers the arena holds (the vertex input reads those
   there), and whether the draw can come straight from it: transformed
   vertices (no screen bounds to find), one vertex per invocation, every
   stream and the index buffer resident, the draw's indices within their
   buffer (32-bit ones only where the host takes any 32-bit index), no
   census recording its window and no vertex audit checking it. Then the
   host draws with the arena's indices and the draw's base, and no index is
   read on the CPU (but the texture audit's coord_check, which reads its
   first primitives' there). */
static int
res_draw(vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d, int program, int shape, res_t **ibe)
{
    vgpu98_vin_t *v   = &r->vin;
    int           all = 1;
    uint32_t      s;
    uint32_t      delta;

    v->res_mask   = 0;
    v->index_res  = NULL;
    *ibe          = NULL;
    r->res_ib     = NULL;
    r->n_res_used = 0;
    for (s = 0; s < VGPU98_MAX_STREAMS; s++) {
        res_t *e;

        r->res_stream[s] = NULL;
        if (!((v->streams_used >> s) & 1))
            continue;
        if ((v->stream[s].source != VGPU98_SRC_GPUVM) ||
            !(e = res_get(r, v->stream[s].address, v->stream[s].size, v->stream[s].version, VGPU98_GPUVM_VERTICES, &delta))) {
            all = 0;
            continue;
        }
        /* STREAM addresses are dwords apart, so is delta (a record a
           packet of another kind named otherwise is not read here). */
        if (delta & 3) {
            all = 0;
            continue;
        }
        v->res_mask |= 1u << s;
        v->res_word[s]   = (uint32_t) ((e->at + delta) / 4);
        r->res_stream[s] = e;
    }
    if (d->indexed) {
        res_t *e = (v->index.source == VGPU98_SRC_GPUVM)
                       ? res_get(r, v->index.address, v->index.size, v->index.version, VGPU98_GPUVM_INDICES, &delta)
                       : NULL;

        if (e && (delta % v->index.isize))
            e = NULL; /* the host's index buffer offset is a whole index */
        if (e) {
            /* An index apart (INDICES' rule): the host's index buffer offset. */
            *ibe         = e;
            r->res_ib    = e;
            r->res_ib_at = e->at + delta;
            if (((uint64_t) d->start + d->count > v->index.size / v->index.isize) || ((v->index.isize == 4) && !r->vk->full_index32))
                all = 0;
        } else
            all = 0;
    }
    v->res_map = r->res_buf.map;
    if (*ibe)
        v->index_res = (const uint8_t *) r->res_buf.map + r->res_ib_at;
    return all && v->res_mask && program && (shape == SHAPE_PLAIN) && !r->census_on &&
           !vgpu98_gpu_logging(r->gpu, VGPU98_LOG_AUDIT);
}

/* --- Volume textures (2.1) ---------------------------------------------------- */

static void
vol_remove(vgpu98_r3d_t *r, size_t i)
{
    r3d_vol_t *v = r->vols[i];

    r->vols[i] = r->vols[--r->n_vols];
    dead_add(r, DEAD_VIEW, (uint64_t) (uintptr_t) v->view);
    dead_add(r, DEAD_IMAGE, (uint64_t) (uintptr_t) v->image);
    dead_add(r, DEAD_MEMORY, (uint64_t) (uintptr_t) v->memory);
    r->vol_bytes -= v->bytes;
    free(v->versions);
    free(v);
}

/* Room for a volume of `bytes`: the least recently used go while the
   volumes hold MAX_VOLS or a quarter of the images' budget, but none this
   draw binds. */
static void
vol_room(vgpu98_r3d_t *r, VkDeviceSize bytes)
{
    while (r->n_vols && ((r->n_vols >= MAX_VOLS) || (r->vol_bytes + bytes > r->img_budget / 4))) {
        size_t i;
        size_t oldest = 0;

        for (i = 1; i < r->n_vols; i++) {
            if (r->vols[i]->last_use < r->vols[oldest]->last_use)
                oldest = i;
        }
        if (r->vols[oldest]->last_use >= r->tick)
            return; /* every one is this draw's */
        r->stats->evictions++;
        vol_remove(r, oldest);
    }
}

static r3d_vol_t *
vol_find(vgpu98_r3d_t *r, const tex_t *t)
{
    size_t i;

    for (i = 0; i < r->n_vols; i++) {
        r3d_vol_t *v = r->vols[i];

        if ((v->format == t->format) && (v->width == t->width) && (v->height == t->height) && (v->depth == t->depth) &&
            (v->levels == t->levels) && !memcmp(v->offset, t->offset, t->levels * sizeof(uint32_t)) &&
            !memcmp(v->pitch, t->pitch, t->levels * sizeof(uint32_t)) &&
            !memcmp(v->slice, t->slice, t->levels * sizeof(uint32_t)))
            return v;
    }
    return NULL;
}

static r3d_vol_t *
vol_create(vgpu98_r3d_t *r, const tex_t *t)
{
    vgpu98_vk_t          *vk  = r->vk;
    const uint32_t        bpp = (uint32_t) vgpu98_fmt_bpp(t->format);
    VkImageCreateInfo     ci  = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    VkImageViewCreateInfo vi  = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    r3d_vol_t            *v;
    uint32_t              l;

    if (!r->has_fmt[t->format])
        return NULL;
    v = calloc(1, sizeof(r3d_vol_t));
    if (!v)
        return NULL;
    v->format = t->format;
    v->width  = t->width;
    v->height = t->height;
    v->depth  = t->depth;
    v->levels = t->levels;
    v->lo     = 0xffffffffu;
    for (l = 0; l < t->levels; l++) {
        const uint32_t h   = max1(t->height >> l);
        const uint32_t d   = max1(t->depth >> l);
        const uint32_t end = t->offset[l] + t->slice[l] * (d - 1) + t->pitch[l] * (h - 1) + max1(t->width >> l) * bpp;

        v->offset[l] = t->offset[l];
        v->pitch[l]  = t->pitch[l];
        v->slice[l]  = t->slice[l];
        v->lev_lo[l] = t->offset[l];
        v->lev_hi[l] = end;
        if (t->offset[l] < v->lo)
            v->lo = t->offset[l];
        if (end > v->hi)
            v->hi = end;
    }
    /* The host format, if the host takes it as a volume (7: a 3D image of
       the volume's size and levels); else half floats. */
    v->hf = r->vol_fmt[t->format] ? &r->fmt[t->format] : &r->vol_f16;
    if (!r->vol_fmt[t->format] && !r->vol_f16_ok) {
        free(v);
        return NULL;
    }
    v->first_page = v->lo >> VGPU98_PAGE_SHIFT;
    v->n_pages    = ((v->hi - 1) >> VGPU98_PAGE_SHIFT) - v->first_page + 1;
    v->versions   = calloc(v->n_pages, sizeof(uint32_t));
    ci.imageType     = VK_IMAGE_TYPE_3D;
    ci.format        = v->hf->vk;
    ci.extent.width  = v->width;
    ci.extent.height = v->height;
    ci.extent.depth  = v->depth;
    ci.mipLevels     = v->levels;
    ci.arrayLayers   = 1;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; /* read back by the texture audit's dumps */
    if (!v->versions || (vk->vkCreateImage(vk->dev, &ci, NULL, &v->image) != VK_SUCCESS)) {
        free(v->versions);
        free(v);
        return NULL;
    }
    if (!image_memory(r, v->image, &v->memory, &v->bytes)) {
        vk->vkDestroyImage(vk->dev, v->image, NULL);
        free(v->versions);
        free(v);
        return NULL;
    }
    vi.image                       = v->image;
    vi.viewType                    = VK_IMAGE_VIEW_TYPE_3D;
    vi.format                      = v->hf->vk;
    vi.components                  = v->hf->swz;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = v->levels;
    vi.subresourceRange.layerCount = 1;
    if (vk->vkCreateImageView(vk->dev, &vi, NULL, &v->view) != VK_SUCCESS) {
        vk->vkFreeMemory(vk->dev, v->memory, NULL);
        vk->vkDestroyImage(vk->dev, v->image, NULL);
        free(v->versions);
        free(v);
        return NULL;
    }
    image_init_layout(r, v->image, VK_IMAGE_ASPECT_COLOR_BIT, v->levels, 1);
    r->vols[r->n_vols++] = v;
    r->vol_bytes += v->bytes;
    r->stats->new_images++;
    return v;
}

/* The levels (bit l) whose pages changed since the volume's last upload
   (every level before its first): only a level's own pages count, not the
   gaps between them. */
static uint32_t
vol_stale(vgpu98_r3d_t *r, r3d_vol_t *v)
{
    const uint32_t epoch = vgpu98_vram_epoch(r->vram);
    uint32_t       stale = 0;
    uint32_t       l;

    if (!v->valid)
        return (1u << v->levels) - 1;
    if (v->epoch == epoch)
        return 0;
    for (l = 0; l < v->levels; l++) {
        uint32_t p;

        for (p = v->lev_lo[l] >> VGPU98_PAGE_SHIFT; p <= ((v->lev_hi[l] - 1) >> VGPU98_PAGE_SHIFT); p++) {
            if (v->versions[p - v->first_page] != vgpu98_page_version(r->vram, p)) {
                stale |= 1u << l;
                break;
            }
        }
    }
    if (!stale)
        v->epoch = epoch;
    return stale;
}

/* Does a DISCARD of [lo, hi) reach one of the volume's levels? */
static int
vol_overlaps(const r3d_vol_t *v, uint32_t lo, uint32_t hi)
{
    uint32_t l;

    if ((v->hi <= lo) || (hi <= v->lo))
        return 0;
    for (l = 0; l < v->levels; l++) {
        if ((v->lev_lo[l] < hi) && (lo < v->lev_hi[l]))
            return 1;
    }
    return 0;
}

/* Uploads the levels `which` names from video memory, converted as a 2D
   texture's level is (convert_rect, through a stand-in image of the
   volume's format): a level of up to a megabyte in one copy, a larger one
   slice by slice. 0 if the staging buffer could not be had. */
static int
vol_upload(vgpu98_r3d_t *r, r3d_vol_t *v, uint32_t which)
{
    const uint32_t bpp = (uint32_t) vgpu98_fmt_bpp(v->format);
    r3d_image_t    as;
    uint64_t       sent = 0;
    uint32_t       l;
    uint32_t       p;

    memset(&as, 0, sizeof(as));
    as.format = v->format;
    as.hf     = v->hf;
    pass_end(r);
    barrier(r);
    /* The versions before the bytes (a guest write meanwhile makes the
       volume stale again). */
    v->epoch = vgpu98_vram_epoch(r->vram);
    for (p = 0; p < v->n_pages; p++)
        v->versions[p] = vgpu98_page_version(r->vram, v->first_page + p);
    atomic_thread_fence(memory_order_acquire);
    for (l = 0; l < v->levels; l++) {
        const uint32_t     w     = max1(v->width >> l);
        const uint32_t     h     = max1(v->height >> l);
        const uint32_t     d     = max1(v->depth >> l);
        const VkDeviceSize slice = (VkDeviceSize) w * h * v->hf->bytes;
        const uint32_t     per   = (slice * d <= ((VkDeviceSize) 1 << 20)) ? d : 1; /* slices a copy */
        uint32_t           z;

        if (!((which >> l) & 1))
            continue;
        for (z = 0; z < d; z += per) {
            VkBuffer          buf;
            VkDeviceSize      off;
            uint8_t          *dst = slot_alloc(r, 0, slice * per + 4, 16, &buf, &off);
            VkBufferImageCopy c;
            uint32_t          k;

            if (!dst) {
                v->valid = 0;
                return 0;
            }
            for (k = 0; k < per; k++)
                convert_rect(r, &as, r->vram->data + v->offset[l] + (size_t) (z + k) * v->slice[l], v->pitch[l], w, h, NULL,
                             0, dst + k * slice, NULL);
            copy_region(&c, off, w, VK_IMAGE_ASPECT_COLOR_BIT, NULL, 0, 0, w, h);
            c.imageSubresource.mipLevel = l;
            c.imageOffset.z             = (int32_t) z;
            c.imageExtent.depth         = per;
            r->vk->vkCmdCopyBufferToImage(cmd(r), buf, v->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
            sent += (uint64_t) w * h * bpp * per;
        }
    }
    barrier(r);
    v->valid = 1;
    r->stats->uploads++;
    r->stats->vol_uploads++;
    r->stats->upload_bytes += sent;
    vgpu98_gpu_note(r->gpu, VGPU98_NOTE_UPLOAD, 0, v->lo, v->hi, (uint32_t) sent, 0);
    return 1;
}

/* A volume stage's host image, current (prepare_draw, before the pass). */
static r3d_vol_t *
vol_prepare(vgpu98_r3d_t *r, const tex_t *t)
{
    r3d_vol_t *v = vol_find(r, t);
    uint32_t   stale;

    if (!v) {
        /* Its levels' bytes (8 a texel in the half-float fallback). */
        vol_room(r, (VkDeviceSize) t->width * t->height * t->depth * (r->vol_fmt[t->format] ? r->fmt[t->format].bytes : 8) * 8 /
                        7);
        v = vol_create(r, t);
        if (!v)
            return NULL;
    }
    v->last_use = r->tick;
    stale       = vol_stale(r, v);
    if (stale) {
        const uint64_t t0 = vgpu98_diag_us();
        int            ok;

        r->timed++;
        ok = vol_upload(r, v, stale);
        r->timed--;
        if (!r->timed)
            r->stats->upload_us += us_since(t0);
        if (!ok)
            return NULL;
    }
    return v;
}

/* A texture's colour key images (the reference rasterizer compares all of a
   texel's bits): the companion, 1 where a texel matches (R8), for
   COLORKEYENABLE; with `zeroed`, the texture with every matching texel
   (0, 0, 0, 0) (B8G8R8A8), for COLORKEYBLENDENABLE. Made from video memory,
   which is current after image_prepare. */
static int
image_key_make(vgpu98_r3d_t *r, r3d_image_t *im, VkFormat format, VkImage *image, VkDeviceMemory *memory, VkImageView *view)
{
    vgpu98_vk_t          *vk = r->vk;
    VkImageCreateInfo     ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    VkDeviceSize          bytes;

    if (*image)
        return 1;
    ci.flags         = (im->faces == VGPU98_CUBE_FACES) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = format;
    ci.extent.width  = im->width;
    ci.extent.height = im->height;
    ci.extent.depth  = 1;
    ci.mipLevels     = im->levels;
    ci.arrayLayers   = im->faces;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (vk->vkCreateImage(vk->dev, &ci, NULL, image) != VK_SUCCESS) {
        *image = VK_NULL_HANDLE;
        return 0;
    }
    if (!image_memory(r, *image, memory, &bytes))
        goto fail;
    vi.image                       = *image;
    vi.viewType                    = (im->faces == VGPU98_CUBE_FACES) ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    vi.format                      = format;
    vi.components                  = SWZ_ID;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = im->levels;
    vi.subresourceRange.layerCount = im->faces;
    if (vk->vkCreateImageView(vk->dev, &vi, NULL, view) != VK_SUCCESS)
        goto fail;
    im->bytes += bytes;
    r->img_bytes += bytes;
    image_init_layout(r, *image, VK_IMAGE_ASPECT_COLOR_BIT, im->levels, im->faces);
    return 1;

fail:
    /* Nothing half made stays: the next try starts afresh. Neither was used. */
    *view = VK_NULL_HANDLE;
    if (*memory)
        vk->vkFreeMemory(vk->dev, *memory, NULL);
    vk->vkDestroyImage(vk->dev, *image, NULL);
    *memory = VK_NULL_HANDLE;
    *image  = VK_NULL_HANDLE;
    return 0;
}

static int
image_key(vgpu98_r3d_t *r, r3d_image_t *im, uint32_t key, int zeroed, const uint32_t *palette, int pal_alpha)
{
    vgpu98_vk_t *vk  = r->vk;
    const int    bpp = vgpu98_fmt_bpp(im->format);
    /* The zeroed copy holds texels as the texture does: 8 bits a channel
       where the texture has them (8-bit RGB, P8), else half floats, so
       16-bit, luminance and bump formats keep their values exactly. */
    const int    f16 = zeroed && (im->hf->vk != VK_FORMAT_B8G8R8A8_UNORM);
    uint32_t     k;

    if (zeroed ? (im->kz_valid && (im->kz_value == key)) : (im->key_valid && (im->key_value == key)))
        return 1;
    if (zeroed ? !image_key_make(r, im, f16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_B8G8R8A8_UNORM, &im->kz_image,
                                 &im->kz_memory, &im->kz_view)
               : !image_key_make(r, im, VK_FORMAT_R8_UNORM, &im->key_image, &im->key_memory, &im->key_view))
        return 0;
    pass_end(r);
    barrier(r);
    for (k = 0; k < im->n_sub; k++) {
        const r3d_sub_t  *s  = &im->sub[k];
        const uint32_t    w  = s->width;
        const uint32_t    h  = s->height;
        const uint32_t    tb = !zeroed ? 1 : f16 ? 8 : 4;
        VkBuffer          buf;
        VkDeviceSize      off;
        uint8_t          *p = slot_alloc(r, 0, (VkDeviceSize) w * h * tb, 4, &buf, &off);
        uint32_t          x;
        uint32_t          y;
        VkBufferImageCopy c;

        if (!p)
            return 0;
        for (y = 0; y < h; y++) {
            const uint8_t *row = r->vram->data + s->offset + (size_t) y * s->pitch;

            for (x = 0; x < w; x++) {
                const uint32_t v = (bpp == 4) ? rd32(row + 4 * x) : (bpp == 2) ? rd16(row + 2 * x) : row[x];
                uint8_t       *t = p + ((size_t) y * w + x) * tb;
                float          c4[4];
                int            i;

                if (!zeroed)
                    *t = (v == key) ? 255 : 0;
                else if (v == key)
                    memset(t, 0, tb);
                else if (im->format == VGPU98_FMT_P8)
                    wr32(t, palette[v & 255] | (pal_alpha ? 0 : 0xff000000u));
                else {
                    texel_rgba(im->format, row + (size_t) x * bpp, c4);
                    if (f16) {
                        for (i = 0; i < 4; i++)
                            wr16(t + 2 * i, f2h(c4[i]));
                    } else
                        wr32(t, (q(c4[3], 255) << 24) | (q(c4[0], 255) << 16) | (q(c4[1], 255) << 8) | q(c4[2], 255));
                }
            }
        }
        copy_region(&c, off, w, VK_IMAGE_ASPECT_COLOR_BIT, s, 0, 0, w, h);
        vk->vkCmdCopyBufferToImage(cmd(r), buf, zeroed ? im->kz_image : im->key_image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    barrier(r);
    if (zeroed) {
        im->kz_value = key;
        im->kz_valid = 1;
    } else {
        im->key_value = key;
        im->key_valid = 1;
    }
    return 1;
}

/* --- Copies back --------------------------------------------------------------- */

typedef struct readback_t {
    r3d_image_t     *im;
    const r3d_sub_t *s;
    uint8_t         *data; /* in the staging buffer */
    uint8_t         *stencil;
    uint32_t         x0, y0, x1, y1;
} readback_t;

/* Writes a copied-back rectangle of a level or face into video memory,
   skipping the bytes in [skip_lo, skip_hi) (a DISCARD's range). 0 if a
   reset dropped it meanwhile. */
static int
readback_write(vgpu98_r3d_t *r, const readback_t *rb, uint32_t skip_lo, uint32_t skip_hi, uint32_t generation)
{
    r3d_image_t     *im  = rb->im;
    const r3d_sub_t *s   = rb->s;
    const int        bpp = vgpu98_fmt_bpp(im->format);
    const uint32_t   w   = rb->x1 - rb->x0;
    uint8_t         *vram = r->vram->data;
    uint32_t         start;
    uint32_t         len;
    uint32_t         x;
    uint32_t         y;
    uint8_t         *was_current;
    uint32_t         p;
    r3d_image_t     *near_im[8]; /* other images current on the first or last page written, their bytes untouched */
    uint32_t         near_at[8];
    size_t           n_near = 0;
    size_t           j;

    if (vgpu98_gpu_reset_since(r->gpu, generation))
        return 0; /* a reset dropped this data: video memory belongs to someone else now */

    start = s->offset + rb->y0 * s->pitch + rb->x0 * bpp;
    len   = (rb->y1 - rb->y0 - 1) * s->pitch + w * bpp;
    was_current = calloc(im->n_pages, 1); /* without host memory, the image goes stale instead */
    for (p = 0; was_current && (p < im->n_pages); p++)
        was_current[p] = (im->versions[p] == vgpu98_page_version(r->vram, im->first_page + p));
    /* A surface sharing a page with this one: the copy back leaves its
       bytes alone, so it stays current (side by side, two images would
       otherwise make each other stale with every copy back). */
    for (j = 0; len && (j < r->n_img) && (n_near < 8); j++) {
        r3d_image_t   *o  = r->img[j];
        const uint32_t pf = start >> VGPU98_PAGE_SHIFT;
        const uint32_t pl = (start + len - 1) >> VGPU98_PAGE_SHIFT;
        uint32_t       e;

        if ((o == im) || image_overlaps(o, start, start + len, 0))
            continue;
        for (e = pf; e <= pl; e += (pl > pf) ? pl - pf : 1) {
            if ((e >= o->first_page) && (e < o->first_page + o->n_pages) && o->covered[e - o->first_page] &&
                (o->versions[e - o->first_page] == vgpu98_page_version(r->vram, e)) && (n_near < 8)) {
                near_im[n_near]   = o;
                near_at[n_near++] = e - o->first_page;
            }
        }
    }

    for (y = rb->y0; y < rb->y1; y++) {
        uint8_t       *row = vram + s->offset + y * s->pitch;
        uint8_t       *shd = im->shadow ? im->shadow + (size_t) y * s->pitch : NULL; /* the image's shadow keeps the same bytes */
        const uint8_t *src = rb->data + (size_t) (y - rb->y0) * w * im->hf->bytes;

        for (x = rb->x0; x < rb->x1; x++) {
            uint32_t at = s->offset + y * s->pitch + x * bpp;
            uint32_t v;
            size_t   i  = (size_t) (x - rb->x0);

            if ((at + (uint32_t) bpp > skip_lo) && (at < skip_hi))
                continue;
            switch (im->hf->conv) {
                case CONV_F16: {
                    float c[4];
                    int   k;

                    for (k = 0; k < 4; k++)
                        c[k] = h2f((uint16_t) rd16(src + i * 8 + 2 * k));
                    v = rgb_pack(im->format, c);
                    break;
                }
                case CONV_D24S8:
                    v = (rd32(src + i * 4) << 8) | ((im->format == VGPU98_FMT_D24S8) ? rb->stencil[(size_t) (y - rb->y0) * w + i] : 0);
                    break;
                case CONV_D24F32: {
                    float f;

                    memcpy(&f, src + i * 4, 4);
                    v = (depth_out(r, im, f, 24) << 8) |
                        ((im->format == VGPU98_FMT_D24S8) ? rb->stencil[(size_t) (y - rb->y0) * w + i] : 0);
                    break;
                }
                case CONV_D16F32: {
                    float f;

                    memcpy(&f, src + i * 4, 4);
                    v = depth_out(r, im, f, 16);
                    break;
                }
                default:
                    v = (bpp == 4) ? rd32(src + i * 4) : (bpp == 2) ? rd16(src + i * 2) : src[i];
                    /* Unused bits of the X formats are written as 0. */
                    if (im->format == VGPU98_FMT_X1R5G5B5)
                        v &= 0x7fff;
                    else if (im->format == VGPU98_FMT_X8R8G8B8)
                        v &= 0x00ffffff;
                    break;
            }
            if (bpp == 4)
                wr32(row + x * 4, v);
            else if (bpp == 2)
                wr16(row + x * 2, v);
            else
                row[x] = (uint8_t) v;
            if (shd && (bpp == 4))
                wr32(shd + x * 4, v);
            else if (shd)
                wr16(shd + x * 2, v);
        }
    }
    vgpu98_gpu_wrote(r->gpu, start, len);

    /* A copy back is not a change for the image it came from: the pages it
       wrote count one GPU write more (vgpu98_gpu_wrote), and a guest write
       meanwhile still shows. */
    for (p = 0; p < im->n_pages; p++) {
        const uint32_t page = im->first_page + p;

        if (was_current && was_current[p] && len && (page >= (start >> VGPU98_PAGE_SHIFT)) &&
            (page <= ((start + len - 1) >> VGPU98_PAGE_SHIFT)))
            im->versions[p]++;
    }
    for (j = 0; j < n_near; j++)
        near_im[j]->versions[near_at[j]]++;
    free(was_current);
    r->stats->copies_back++;
    r->stats->copy_back_bytes += (uint64_t) (rb->y1 - rb->y0) * w * bpp;
    vgpu98_gpu_note(r->gpu, VGPU98_NOTE_COPY_BACK, im->id, (rb->x0 << 16) | rb->y0, (rb->x1 << 16) | rb->y1, start, len);
    return 1;
}

/* Copies back the GPU-newer pixels of one level or face and writes them into
   video memory: the copy is recorded, submitted and waited for. */
static void
copy_back_sub(vgpu98_r3d_t *r, r3d_image_t *im, r3d_sub_t *s, uint32_t skip_lo, uint32_t skip_hi, int cause)
{
    const uint32_t     w          = s->dx1 - s->dx0;
    const uint32_t     h          = s->dy1 - s->dy0;
    const VkDeviceSize hb         = (VkDeviceSize) im->hf->bytes;
    const VkDeviceSize color      = (VkDeviceSize) w * h * hb;
    const int          sten       = (fmt_aspects(im->hf) & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    const uint32_t     generation = vgpu98_gpu_generation(r->gpu);
    VkBuffer           buf;
    VkDeviceSize       off;
    uint8_t           *p;
    VkBufferImageCopy  c[2];
    readback_t         rb;

    if (!s->dirty)
        return;
    pass_end(r);
    barrier(r);
    p = slot_alloc(r, 0, color + (sten ? (VkDeviceSize) w * h + 4 : 0), 16, &buf, &off);
    if (!p)
        return;
    copy_region(&c[0], off, w, (fmt_aspects(im->hf) & VK_IMAGE_ASPECT_COLOR_BIT) ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT,
                s, s->dx0, s->dy0, w, h);
    if (sten)
        copy_region(&c[1], off + ((color + 3) & ~(VkDeviceSize) 3), w, VK_IMAGE_ASPECT_STENCIL_BIT, s, s->dx0, s->dy0, w, h);
    r->vk->vkCmdCopyImageToBuffer(cmd(r), im->image, VK_IMAGE_LAYOUT_GENERAL, buf, sten ? 2 : 1, c);
    if (!submit(r, 1))
        return;
    rb.im      = im;
    rb.s       = s;
    rb.data    = p;
    rb.stencil = sten ? p + ((color + 3) & ~(VkDeviceSize) 3) : NULL;
    rb.x0      = s->dx0;
    rb.y0      = s->dy0;
    rb.x1      = s->dx1;
    rb.y1      = s->dy1;
    if (readback_write(r, &rb, skip_lo, skip_hi, generation))
        r->stats->copies_back_by[cause]++;
    sub_clean(r, im, s);
    s->aset = 0; /* video memory took the host's pixels: the texture audit hashes them anew */
}

/* Copies back the GPU-newer levels and faces of the listed images that
   overlap [lo, hi) (every one: 0, ~0). */
static void
copy_back(vgpu98_r3d_t *r, r3d_image_t **list, size_t n, uint32_t lo, uint32_t hi, uint32_t skip_lo, uint32_t skip_hi,
          int cause)
{
    const uint64_t t0   = vgpu98_diag_us();
    const int      step = vgpu98_gpu_step(r->gpu, VGPU98_STEP_COPY_BACK);
    size_t         i;
    uint32_t       k;

    r->timed++;
    for (i = 0; (i < n) && !r->broken; i++) {
        r3d_image_t *im = list[i];

        for (k = 0; (k < im->n_sub) && im->dirty && !r->broken; k++) {
            if (im->sub[k].dirty && sub_overlaps(&im->sub[k], lo, hi))
                copy_back_sub(r, im, &im->sub[k], skip_lo, skip_hi, cause);
        }
    }
    vgpu98_gpu_step(r->gpu, step);
    r->timed--;
    if (!r->timed)
        r->stats->copy_back_us += us_since(t0);
}

/* Brings video memory up to date over [lo, hi), but for the images marked
   `merging` (image_merge moves their pixels on the host GPU). */
static void
sync_range(vgpu98_r3d_t *r, uint32_t lo, uint32_t hi, const r3d_image_t *except, int cause)
{
    r3d_image_t  *few[16];
    r3d_image_t **list = few;
    size_t        n    = 0;
    size_t        i;

    /* A list of its own (copying back takes images off the dirty list), on
       the stack unless the overlaps are many. */
    for (i = 0; i < r->n_dirty; i++) {
        r3d_image_t *im = r->dirty[i];

        if ((im == except) || im->merging || !image_overlaps(im, lo, hi, 1))
            continue;
        if ((n == 16) && (list == few)) {
            list = malloc(r->n_dirty * sizeof(*list));
            if (!list)
                return;
            memcpy(list, few, sizeof(few));
        }
        list[n++] = im;
    }
    if (n)
        copy_back(r, list, n, lo, hi, 0, 0, cause);
    if (list != few)
        free(list);
}

/* Brings video memory up to date under the image's levels and faces (not the
   gaps between them, which are other surfaces'), but for the image itself and
   the images marked `merging`. */
static void
sync_image(vgpu98_r3d_t *r, r3d_image_t *im, int cause)
{
    uint32_t k;

    for (k = 0; (k < im->n_sub) && r->n_dirty; k++)
        sync_range(r, im->sub[k].lo, im->sub[k].hi, im, cause);
}

/* A new image takes over the images of its levels and faces made before it
   (a target drawn into before the texture it is a level of was first used),
   so a level or face has one image from then on. One with GPU-newer pixels
   and otherwise current moves its pixels into it on the host GPU, so nothing
   is copied back; the others go (a stale one after copying back what is
   GPU-newer in it; a current, clean one holds what the upload reads). An
   image the draw being prepared also binds stays: the two share the level
   for now. */
static int
image_merge(vgpu98_r3d_t *r, r3d_image_t *im, const uint32_t *palette, int pal_alpha)
{
    r3d_sub_t *from[MAX_SUBS];
    uint8_t    which[MAX_SUBS];
    uint32_t   k;
    int        n = 0;
    int        ok;

    for (k = 0; k < im->n_sub; k++) {
        const r3d_sub_t *s = &im->sub[k];
        r3d_sub_t       *o = sub_find(r, im->format, s->width, s->height, s->offset, s->pitch, im);
        r3d_image_t     *oi;

        from[k]  = NULL;
        which[k] = 1;
        if (!o || (o->im->n_sub != 1) || o->im->merging || (o->im->last_use == r->tick))
            continue;
        oi = o->im;
        /* The host GPU copies only between images of one format (a depth
           image moved to float has another): the others go through video
           memory. */
        if (!oi->dirty || !oi->valid || image_stale(r, oi) || (oi->hf->vk != im->hf->vk)) {
            if (oi->dirty)
                copy_back(r, &oi, 1, 0, 0xffffffffu, 0, 0, VGPU98_CB_IMAGE);
            image_remove_ptr(r, oi);
            continue;
        }
        oi->merging = 1;
        from[k]     = o;
        which[k]    = 0;
        n++;
    }
    if (!n)
        return 1;

    /* Video memory current but for the images moving in; then the image's
       other levels and faces from it, and their pixels in theirs. */
    sync_image(r, im, VGPU98_CB_IMAGE);
    ok = image_upload(r, im, palette, pal_alpha, which);
    if (ok) {
        pass_end(r);
        barrier(r);
    }
    for (k = 0; k < im->n_sub; k++) {
        r3d_sub_t   *o = from[k];
        r3d_sub_t   *s = &im->sub[k];
        r3d_image_t *oi;

        if (!o)
            continue;
        oi          = o->im;
        oi->merging = 0;
        if (ok) {
            VkImageCopy ic;

            memset(&ic, 0, sizeof(ic));
            ic.srcSubresource.aspectMask     = fmt_aspects(oi->hf);
            ic.srcSubresource.layerCount     = 1;
            ic.dstSubresource.aspectMask     = fmt_aspects(im->hf);
            ic.dstSubresource.mipLevel       = s->level;
            ic.dstSubresource.baseArrayLayer = s->face;
            ic.dstSubresource.layerCount     = 1;
            ic.extent.width                  = s->width;
            ic.extent.height                 = s->height;
            ic.extent.depth                  = 1;
            r->vk->vkCmdCopyImage(cmd(r), oi->image, VK_IMAGE_LAYOUT_GENERAL, im->image, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
            /* Its GPU-newer pixels are this level's now: the page map counts
               them for it before the old count goes. */
            image_add_dirty(r, s, o->dx0, o->dy0, o->dx1, o->dy1);
            sub_clean(r, oi, o);
            r->stats->merges++;
        } else if (oi->dirty)
            copy_back(r, &oi, 1, 0, 0xffffffffu, 0, 0, VGPU98_CB_IMAGE);
        image_remove_ptr(r, oi);
    }
    barrier(r);
    return ok;
}

/* A float depth image gone stale (a page under it changed): only the
   pixels on the pages that changed whose bytes differ from its shadow,
   which someone else wrote (a Lock, the guest's CPU; a 2D packet marks its
   rectangle in the shadow, vgpu98_r3d_written), go up again; the others
   keep what the image holds, which for W is more than video memory's form
   gives back (a page shared with the next surface changes without one of
   its pixels doing so). Each row's runs of changed pixels go up, rows with
   the same runs as one rectangle each, DEPTH_RECTS to a copy. Its own
   GPU-newer pixels go back to video memory first only when a run of
   changed pixels reaches their rectangle, so they win there as they do
   over an upload; otherwise neither image of a page shared with the next
   surface makes the other stale again. */
typedef struct depth_run_t {
    uint32_t x0, x1;
} depth_run_t;

typedef struct depth_rect_t {
    uint32_t x0, y0, x1, y1;
} depth_rect_t;

#define DEPTH_RECTS 4096 /* rectangles one copy takes */

/* The runs of changed pixels in row y of a shadowed image (`run` holds
   (width + 1) / 2, the most a row can have); none for a row on no page
   that changed (`changed`, by page from the image's first). */
static uint32_t
depth_row_runs(const vgpu98_r3d_t *r, const r3d_image_t *im, const uint8_t *changed, uint32_t y, depth_run_t *run)
{
    const r3d_sub_t *s   = &im->sub[0];
    const size_t     bpp = (size_t) vgpu98_fmt_bpp(im->format);
    const uint32_t   at  = s->offset + y * s->pitch;
    const uint8_t   *v   = r->vram->data + at;
    const uint8_t   *m   = im->shadow + (size_t) y * s->pitch;
    const uint32_t   w   = s->width;
    uint32_t         n   = 0;
    uint32_t         x   = 0;
    uint32_t         p;

    for (p = at >> VGPU98_PAGE_SHIFT; p <= ((at + w * bpp - 1) >> VGPU98_PAGE_SHIFT); p++) {
        if (changed[p - im->first_page])
            break;
    }
    if ((p > ((at + w * bpp - 1) >> VGPU98_PAGE_SHIFT)) || !memcmp(v, m, w * bpp))
        return 0;
    while (x < w) {
        while ((x < w) && !memcmp(v + x * bpp, m + x * bpp, bpp))
            x++;
        if (x == w)
            break;
        run[n].x0 = x;
        while ((x < w) && memcmp(v + x * bpp, m + x * bpp, bpp))
            x++;
        run[n++].x1 = x;
    }
    return n;
}

/* Uploads the rectangles in one copy from one staging block. Returns the
   bytes sent, or 0 when the staging buffer could not be had. */
static uint64_t
depth_upload_rects(vgpu98_r3d_t *r, r3d_image_t *im, const depth_rect_t *rect, size_t n)
{
    const r3d_sub_t   *s     = &im->sub[0];
    const int          sten  = (fmt_aspects(im->hf) & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    VkDeviceSize       color = 0;
    VkDeviceSize       total;
    VkDeviceSize       at;
    VkDeviceSize       sat;
    VkBuffer           buf;
    VkDeviceSize       off;
    VkBufferImageCopy *c;
    uint8_t           *p;
    size_t             i;
    uint32_t           k = 0;

    for (i = 0; i < n; i++)
        color += (VkDeviceSize) (rect[i].x1 - rect[i].x0) * (rect[i].y1 - rect[i].y0) * im->hf->bytes;
    total = color;
    if (sten) {
        for (i = 0; i < n; i++)
            total += ((VkDeviceSize) (rect[i].x1 - rect[i].x0) * (rect[i].y1 - rect[i].y0) + 3) & ~(VkDeviceSize) 3;
    }
    c = malloc(n * (sten ? 2 : 1) * sizeof(VkBufferImageCopy));
    p = c ? slot_alloc(r, 0, total + 4, 16, &buf, &off) : NULL;
    if (!p) {
        free(c);
        return 0;
    }
    at  = 0;
    sat = color; /* the stencil blocks follow the depth ones, each at a multiple of 4 */
    for (i = 0; i < n; i++) {
        const uint32_t w = rect[i].x1 - rect[i].x0;
        const uint32_t h = rect[i].y1 - rect[i].y0;

        convert_rect(r, im, sub_source(r, im, s, rect[i].x0, rect[i].y0, rect[i].x1, rect[i].y1), s->pitch, w, h, NULL, 0,
                     p + at, p + sat);
        copy_region(&c[k++], off + at, w, VK_IMAGE_ASPECT_DEPTH_BIT, s, rect[i].x0, rect[i].y0, w, h);
        if (sten) {
            copy_region(&c[k++], off + sat, w, VK_IMAGE_ASPECT_STENCIL_BIT, s, rect[i].x0, rect[i].y0, w, h);
            sat += ((VkDeviceSize) w * h + 3) & ~(VkDeviceSize) 3;
        }
        at += (VkDeviceSize) w * h * im->hf->bytes;
    }
    r->vk->vkCmdCopyBufferToImage(cmd(r), buf, im->image, VK_IMAGE_LAYOUT_GENERAL, k, c);
    free(c);
    return total ? total : 1;
}

static int
depth_resync(vgpu98_r3d_t *r, r3d_image_t *im)
{
    r3d_sub_t    *s      = &im->sub[0];
    const size_t  most   = (s->width + 1) / 2;
    depth_run_t  *base   = malloc(2 * most * sizeof(depth_run_t));
    depth_run_t  *cur    = base;
    depth_run_t  *prev   = base ? base + most : NULL;
    uint8_t      *changed = malloc(im->n_pages);
    depth_rect_t *rect   = NULL;
    size_t        n_rect = 0;
    size_t        cap    = 0;
    uint64_t      sent   = 0;
    uint32_t      n_prev = 0;
    uint32_t      y0     = 0;
    uint32_t      y;
    uint32_t      i;
    int           any = 0;
    int           ok  = 1;

    if (!base || !changed) {
        free(base);
        free(changed);
        return 0;
    }
    pass_end(r);
    barrier(r);
    /* Before video memory is read: a guest write meanwhile makes the image
       stale again, and the shadow shows it. Only rows on the pages that
       changed are compared. */
    image_snapshot_changed(r, im, changed);
    for (y = 0; y < s->height; y++) {
        const uint32_t n = depth_row_runs(r, im, changed, y, cur);

        any |= (n != 0);
        for (i = 0; s->dirty && (i < n) && (y >= s->dy0) && (y < s->dy1); i++) {
            if ((cur[i].x0 < s->dx1) && (s->dx0 < cur[i].x1)) {
                /* A changed pixel among the GPU-newer ones: they go back
                   first, and win there (the shadow takes them too). */
                copy_back(r, &im, 1, s->lo, s->hi, 0, 0, VGPU98_CB_IMAGE);
                break;
            }
        }
    }
    if (!any || r->broken) {
        free(base);
        free(changed);
        return !r->broken; /* only the page changed: the image stays as it is */
    }
    /* The rectangles: rows with the same runs are one rectangle per run. */
    for (y = 0; y <= s->height; y++) {
        const uint32_t n = (y < s->height) ? depth_row_runs(r, im, changed, y, cur) : 0;
        depth_run_t   *t;

        if (n_prev && ((n != n_prev) || memcmp(cur, prev, n * sizeof(depth_run_t)))) {
            if (n_rect + n_prev > cap) {
                depth_rect_t *g;

                cap = 2 * (n_rect + n_prev);
                g   = realloc(rect, cap * sizeof(depth_rect_t));
                if (!g) {
                    ok = 0;
                    break;
                }
                rect = g;
            }
            for (i = 0; i < n_prev; i++) {
                rect[n_rect].x0   = prev[i].x0;
                rect[n_rect].x1   = prev[i].x1;
                rect[n_rect].y0   = y0;
                rect[n_rect++].y1 = y;
            }
            n_prev = 0;
        }
        if (n && !n_prev) {
            t      = prev;
            prev   = cur;
            cur    = t;
            n_prev = n;
            y0     = y;
        }
    }
    /* In copies of DEPTH_RECTS rectangles at most: a change made of many
       small runs (a checkerboard) never asks one copy for millions. */
    for (i = 0; ok && (i < n_rect); i += DEPTH_RECTS) {
        const uint64_t k = depth_upload_rects(r, im, rect + i, ((n_rect - i) < DEPTH_RECTS) ? n_rect - i : DEPTH_RECTS);

        ok = (k != 0);
        sent += k;
    }
    free(rect);
    free(base);
    free(changed);
    if (!ok) {
        im->valid = 0; /* all of it goes up next time */
        return 0;
    }
    if (!n_rect)
        return 1; /* the changes were among the GPU-newer pixels, which won */
    barrier(r);
    r->stats->uploads++;
    r->stats->upload_bytes += sent;
    vgpu98_gpu_note(r->gpu, VGPU98_NOTE_UPLOAD, im->id, im->lo, im->hi, (uint32_t) sent, 0);
    return 1;
}

/* Makes an image ready for use: other images' GPU-newer data over its
   levels and faces goes back to video memory, then the levels and faces
   whose pages changed are uploaded again (all of them the first time, and
   for P8 when the palette changed; for a float depth image, the pixels
   that changed: depth_resync), unless the packet overwrites all of the
   image. */
static int
image_prepare(vgpu98_r3d_t *r, r3d_image_t *im, int overwrite_all, const uint32_t *palette, int pal_alpha)
{
    uint8_t  which[MAX_SUBS];
    uint8_t *pick = NULL;
    uint32_t k;
    int      need_upload;

    if (im->fresh) {
        im->fresh = 0;
        if (!image_merge(r, im, palette, pal_alpha))
            return 0;
    }
    sync_image(r, im, VGPU98_CB_IMAGE);
    if (!im->valid || ((im->format == VGPU98_FMT_P8) && (im->pal_sig != fnv(palette, 1024, pal_alpha ? 0x811c9dc5u : 0x01000193u))))
        need_upload = 1;
    else if ((need_upload = image_stale(r, im))) {
        for (k = 0; k < im->n_sub; k++)
            which[k] = (uint8_t) sub_stale(r, im, &im->sub[k]);
        pick = which;
    }
    if (need_upload) {
        const int resync = im->shadow && im->valid && !overwrite_all;

        im->mip_checked = 0; /* its content may be another's now (mip_check_image) */

        /* Their own GPU-newer pixels first, so the upload keeps them (a
           resync sees to its own; a packet overwriting all of the image
           supersedes them, and its dirty rectangle keeps the page map
           right). */
        for (k = 0; !resync && !overwrite_all && (k < im->n_sub) && im->dirty; k++) {
            if (im->sub[k].dirty && (!pick || pick[k]))
                copy_back(r, &im, 1, im->sub[k].lo, im->sub[k].hi, 0, 0, VGPU98_CB_IMAGE);
        }
        if (overwrite_all) {
            for (k = 0; k < im->n_sub; k++)
                im->sub[k].aset = 0; /* the packet's pixels, not video memory's: the texture audit hashes anew */
            image_snapshot(r, im);
            if (im->shadow) /* current as of the snapshot: later writes show against it */
                memcpy(im->shadow, r->vram->data + im->sub[0].offset, im->sub[0].hi - im->sub[0].lo);
            im->valid     = 1;
            im->key_valid = 0;
            im->kz_valid  = 0;
        } else {
            const uint64_t t0 = vgpu98_diag_us();
            int            ok;

            r->timed++;
            ok = resync ? depth_resync(r, im) : image_upload(r, im, palette, pal_alpha, pick);
            r->timed--;
            if (!r->timed)
                r->stats->upload_us += us_since(t0);
            if (!ok)
                return 0;
        }
    }
    im->last_use = r->tick;
    return 1;
}

/* Keeps host memory bounded: the least recently used images go, after
   copying back what is GPU-newer in them. */
static void
evict(vgpu98_r3d_t *r)
{
    while (((r->img_bytes > r->img_budget) || (r->n_img > IMG_MAX)) && r->n_img) {
        size_t i;
        size_t oldest = 0;

        for (i = 1; i < r->n_img; i++) {
            if (r->img[i]->last_use < r->img[oldest]->last_use)
                oldest = i;
        }
        if (r->img[oldest]->last_use >= r->tick)
            break; /* everything left is in use now */
        if (r->img[oldest]->dirty)
            copy_back(r, &r->img[oldest], 1, 0, 0xffffffffu, 0, 0, VGPU98_CB_EVICT);
        churn_evicted(r, r->img[oldest]);
        r->stats->evictions++;
        vgpu98_gpu_note(r->gpu, VGPU98_NOTE_EVICT, r->img[oldest]->id, r->img[oldest]->lo, r->img[oldest]->hi, 0, 0);
        image_remove(r, oldest);
    }
}

/* Once a second: clean images no draw used for STALE_SECONDS go (textures
   of scenes long past; video memory has their pixels), so the images the
   GPU keeps are the ones in use. */
static void
evict_stale(vgpu98_r3d_t *r)
{
    const uint64_t now = vgpu98_diag_us();
    uint64_t       before;
    size_t         i;

    if (now - r->stale_us < 1000000)
        return;
    r->stale_us = now;
    before      = r->tick_hist[0];
    memmove(r->tick_hist, r->tick_hist + 1, (STALE_SECONDS - 1) * sizeof(r->tick_hist[0]));
    r->tick_hist[STALE_SECONDS - 1] = r->tick;
    for (i = 0; i < r->n_img;) {
        r3d_image_t *im = r->img[i];

        if (im->dirty || (r->pass_color && (r->pass_color->im == im)) || (r->pass_depth && (r->pass_depth->im == im)) ||
            (im->last_use >= before)) {
            i++;
            continue;
        }
        churn_evicted(r, im);
        r->stats->evictions++;
        r->stats->stale_evictions++;
        vgpu98_gpu_note(r->gpu, VGPU98_NOTE_EVICT, im->id, im->lo, im->hi, 0, 0);
        {
            /* A depth buffer unused this long (its program may be gone
               without freeing it) leaves the records too: drawn into
               again, it is recorded again by the first packet that
               W-buffers. */
            const int      depth = vgpu98_fmt_is_depth(im->format);
            const uint32_t lo    = im->lo;
            const uint32_t hi    = im->hi;

            image_remove(r, i);
            if (depth)
                depth_rec_drop(r, lo, hi);
        }
    }
    /* So does a volume (2.1). */
    for (i = 0; i < r->n_vols;) {
        if (r->vols[i]->last_use < before) {
            r->stats->evictions++;
            r->stats->stale_evictions++;
            vol_remove(r, i);
        } else
            i++;
    }
    /* A record whose surface has no image (evicted for room) and that no
       packet W-buffered into for as long goes as well. */
    for (i = 0; i < DEPTH_RECS; i++) {
        depth_rec_t *d = &r->drec[i];

        if (d->used && (d->tick < before) && !sub_find(r, d->format, d->width, d->height, d->offset, d->pitch, NULL))
            d->used = 0;
    }
}

/* --- Render passes, framebuffers, samplers, pipelines ------------------------- */

/* A render pass for a colour and a depth format (VK_FORMAT_UNDEFINED:
   none): the renderer's, and a warm-up thread's, which match it. Any
   thread. A change here raises PCACHE_PIPE_VERSION. */
static VkResult
pass_make(const vgpu98_vk_t *vk, VkFormat color, VkFormat depth, VkRenderPass *pass)
{
    VkAttachmentDescription ad[2];
    VkAttachmentReference   cref = { 0, VK_IMAGE_LAYOUT_GENERAL };
    VkAttachmentReference   dref = { 1, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription    sd;
    VkSubpassDependency     dep[2];
    VkRenderPassCreateInfo  ci = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    int                     k;

    memset(ad, 0, sizeof(ad));
    for (k = 0; k < 2; k++) {
        ad[k].samples        = VK_SAMPLE_COUNT_1_BIT;
        ad[k].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        ad[k].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        ad[k].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
        ad[k].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
        ad[k].initialLayout  = VK_IMAGE_LAYOUT_GENERAL;
        ad[k].finalLayout    = VK_IMAGE_LAYOUT_GENERAL;
    }
    ad[0].format = color;
    ad[1].format = depth;
    memset(&sd, 0, sizeof(sd));
    sd.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount    = 1;
    sd.pColorAttachments       = &cref;
    sd.pDepthStencilAttachment = (depth != VK_FORMAT_UNDEFINED) ? &dref : NULL;
    memset(dep, 0, sizeof(dep));
    dep[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep[0].dstSubpass    = 0;
    dep[0].srcStageMask  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dep[0].dstStageMask  = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    dep[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    dep[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dep[1].srcSubpass    = 0;
    dep[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    dep[1].srcStageMask  = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    dep[1].dstStageMask  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dep[1].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    dep[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ci.attachmentCount = (depth != VK_FORMAT_UNDEFINED) ? 2 : 1;
    ci.pAttachments    = ad;
    ci.subpassCount    = 1;
    ci.pSubpasses      = &sd;
    ci.dependencyCount = 2;
    ci.pDependencies   = dep;
    return vk->vkCreateRenderPass(vk->dev, &ci, NULL, pass);
}

static VkRenderPass
pass_get(vgpu98_r3d_t *r, VkFormat color, VkFormat depth)
{
    VkRenderPass pass;
    VkResult     res;
    size_t       i;

    for (i = 0; i < r->n_passes; i++) {
        if ((r->passes[i].color == color) && (r->passes[i].depth == depth))
            return r->passes[i].pass;
    }
    if ((res = pass_make(r->vk, color, depth, &pass)) != VK_SUCCESS) {
        r3d_fail(r, "vkCreateRenderPass", res);
        return VK_NULL_HANDLE;
    }
    r->passes = realloc(r->passes, (r->n_passes + 1) * sizeof(pass_t));
    r->passes[r->n_passes].color = color;
    r->passes[r->n_passes].depth = depth;
    r->passes[r->n_passes].pass  = pass;
    r->n_passes++;
    return pass;
}

/* Begins (or continues) the render pass for the target and depth buffer:
   a level or face each. */
static int
pass_begin(vgpu98_r3d_t *r, r3d_sub_t *color, r3d_sub_t *depth)
{
    vgpu98_vk_t          *vk   = r->vk;
    VkFormat              df   = depth ? depth->im->hf->vk : VK_FORMAT_UNDEFINED;
    VkRenderPass          pass;
    VkFramebuffer         fb   = VK_NULL_HANDLE;
    VkRenderPassBeginInfo bi   = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    uint32_t              fw   = depth && (depth->width < color->width) ? depth->width : color->width;
    uint32_t              fh   = depth && (depth->height < color->height) ? depth->height : color->height;
    size_t                i;

    if (r->in_pass && (r->pass_color == color) && (r->pass_depth == depth))
        return 1;
    pass_end(r);
    pass = pass_get(r, color->im->hf->vk, df);
    if (!pass)
        return 0;
    for (i = 0; i < r->n_fbs; i++) {
        if ((r->fbs[i].color_id == color->im->id) && (r->fbs[i].color_sub == color->index) &&
            (r->fbs[i].depth_id == (depth ? depth->im->id : 0)) && (r->fbs[i].depth_sub == (depth ? depth->index : 0))) {
            fb = r->fbs[i].fb;
            break;
        }
    }
    if (!fb) {
        VkFramebufferCreateInfo ci = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        VkImageView             att[2];

        /* The framebuffer is the area both cover. */
        att[0]             = sub_att_view(r, color);
        att[1]             = depth ? sub_att_view(r, depth) : VK_NULL_HANDLE;
        if (!att[0] || (depth && !att[1]))
            return 0;
        ci.renderPass      = pass;
        ci.attachmentCount = depth ? 2 : 1;
        ci.pAttachments    = att;
        ci.width           = fw;
        ci.height          = fh;
        ci.layers          = 1;
        if (vk->vkCreateFramebuffer(vk->dev, &ci, NULL, &fb) != VK_SUCCESS) {
            r3d_fail(r, "vkCreateFramebuffer", VK_ERROR_INITIALIZATION_FAILED);
            return 0;
        }
        r->fbs = realloc(r->fbs, (r->n_fbs + 1) * sizeof(fb_t));
        r->fbs[r->n_fbs].color_id  = color->im->id;
        r->fbs[r->n_fbs].color_sub = color->index;
        r->fbs[r->n_fbs].depth_id  = depth ? depth->im->id : 0;
        r->fbs[r->n_fbs].depth_sub = depth ? depth->index : 0;
        r->fbs[r->n_fbs].fb        = fb;
        r->n_fbs++;
    }
    barrier(r);
    bi.renderPass               = pass;
    bi.framebuffer              = fb;
    bi.renderArea.extent.width  = fw;
    bi.renderArea.extent.height = fh;
    vk->vkCmdBeginRenderPass(cmd(r), &bi, VK_SUBPASS_CONTENTS_INLINE);
    r->in_pass    = 1;
    r->pass_color = color;
    r->pass_depth = depth;
    r->bound_pipe = VK_NULL_HANDLE;
    return 1;
}

/* --- Lookups by hash ------------------------------------------------------------ */

static uint32_t
key_hash(const void *key, size_t bytes)
{
    uint32_t x = fnv(key, bytes, 0x811c9dc5u);

    /* The tables take the low bits: mixed with the high ones. */
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return x;
}

/* The number (plus one) of the record whose key matches, from a table of
   `size` slots (a power of two); else 0, with the slot it would go in. */
static uint32_t
tab_find(const uint32_t *tab, uint32_t size, uint32_t h, const void *key, size_t bytes, const void *recs, size_t rec_size,
         uint32_t *slot)
{
    uint32_t i = h & (size - 1);

    while (tab[i]) {
        if (!memcmp((const uint8_t *) recs + (size_t) (tab[i] - 1) * rec_size, key, bytes))
            return tab[i];
        i = (i + 1) & (size - 1);
    }
    *slot = i;
    return 0;
}

/* Room for one more record: the table doubles (and is filled again) when
   it would be more than half full. 0 if there is no memory. */
static int
tab_grow(uint32_t **tab, uint32_t *size, uint32_t n, const void *recs, size_t rec_size, size_t key_bytes)
{
    uint32_t  ns;
    uint32_t *nt;
    uint32_t  k;

    if (*tab && (2 * (n + 1) <= *size))
        return 1;
    ns = *size ? 2 * *size : 256;
    nt = calloc(ns, sizeof(uint32_t));
    if (!nt)
        return 0;
    for (k = 0; k < n; k++) {
        const void *key = (const uint8_t *) recs + (size_t) k * rec_size;
        uint32_t    i   = key_hash(key, key_bytes) & (ns - 1);

        while (nt[i])
            i = (i + 1) & (ns - 1);
        nt[i] = k + 1;
    }
    free(*tab);
    *tab  = nt;
    *size = ns;
    return 1;
}

/* --- The pipeline cache ----------------------------------------------------------- */

/* The cache's files (two, saved in turn: vgpu98_gpu_cache_open; spec/
   pipeline-cache.md): a header (pcache_hdr_t), Vulkan's data, the key
   list. The newer file whose header checks out is read whole (its sizes
   within their limits, both hashes), the older when it does not. Its data
   is used only when its stamp is this 86Box's (the same shaders and the
   same code making the pipelines) and Vulkan's own header in it names the
   host GPU and driver; otherwise the cache starts empty, the reason
   logged. */
#define PCACHE_MAGIC     "VGPU98C4"
#define PCACHE_SEED      0x811c9dc5u
#define PCACHE_CAP       (64u << 20) /* Vulkan's data at most */
#define PCACHE_PERIOD_US 120000000u
#define PCACHE_JOIN_S    30 /* at close, a save still writing is waited for this long */
/* Raised whenever anything a pipeline is made from changes, but for the
   shaders' SPIR-V (already in the stamp): pipe_make's fixed and dynamic
   states and specialization constants, pass_make's render pass, and
   r3d_objects' pipeline layout (its sets, bindings and counts). Each of
   those says so. The stamp changes with it, as every pipeline's data
   does. */
#define PCACHE_PIPE_VERSION 1
/* Raised whenever pipe_key_t's layout or meaning changes: older keys are
   not used. */
#define PCACHE_KEY_VERSION 1
#define PCACHE_KEYS        4096
#define PCACHE_KEY_AGE     16 /* sessions a key is kept unused */
#define PCACHE_WARM_THREADS 2
#define PCACHE_WARM_PASSES  16 /* render passes a warm-up thread keeps */
#define PCACHE_WARM_JOIN_S  10 /* at close, the warm-up is waited for this long */
/* What a save named once a session (pcache_job_t.told): its failures, the
   cap, another 86Box's save merged. */
#define PCACHE_TOLD_DATA  1
#define PCACHE_TOLD_OPEN  2
#define PCACHE_TOLD_WRITE 4
#define PCACHE_TOLD_CAP   8
#define PCACHE_TOLD_MERGE 16
#define PCACHE_TOLD_MEM   32
/* The "Pipeline cache" setting (VGPU98_OPT_PCACHE_*): On, Start empty
   (nothing read or warmed, every pipeline made anew; still saved, merged
   with what is on disk), Off (neither read nor saved). */
#define PCACHE_ON    0
#define PCACHE_EMPTY 1
#define PCACHE_OFF   2

/* FNV-1a over dwords, then the last bytes, from `h`. */
static uint32_t
pcache_hash(uint32_t h, const uint8_t *p, size_t n)
{
    size_t i;

    for (i = 0; i + 4 <= n; i += 4) {
        uint32_t w;

        memcpy(&w, p + i, 4);
        h = (h ^ w) * 0x01000193u;
    }
    for (; i < n; i++)
        h = (h ^ p[i]) * 0x01000193u;
    return h;
}

/* This 86Box's stamp: its pipelines' shaders as built in, and the code that
   makes the pipelines (PCACHE_PIPE_VERSION). */
static uint32_t
pcache_stamp(void)
{
    uint32_t h = PCACHE_SEED ^ PCACHE_PIPE_VERSION;

    h = pcache_hash(h, (const uint8_t *) vgpu98_vert, sizeof(vgpu98_vert));
    h = pcache_hash(h, (const uint8_t *) vgpu98_frag, sizeof(vgpu98_frag));
    h = pcache_hash(h, (const uint8_t *) vgpu98_frag_depth, sizeof(vgpu98_frag_depth));
    return h;
}

/* Does Vulkan's header at the start of the data name the host GPU and its
   driver (VkPipelineCacheHeaderVersionOne: its size, version 1, the vendor,
   the device, the cache UUID)? The driver ignores data that does not;
   checked here so that the log says so. A header of another version is
   left to the driver. */
static int
pcache_vk_fits(const vgpu98_vk_t *vk, const uint8_t *data, uint32_t n)
{
    uint32_t w[4];

    if (n < 16 + VK_UUID_SIZE)
        return 0;
    memcpy(w, data, sizeof(w));
    if (w[1] != VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
        return 1;
    return (w[0] >= 16 + VK_UUID_SIZE) && (w[0] <= n) && (w[2] == vk->props.vendorID) && (w[3] == vk->props.deviceID) &&
           !memcmp(data + 16, vk->props.pipelineCacheUUID, VK_UUID_SIZE);
}

/* A header's own checks: NULL when it checks out, else why not. */
static const char *
pcache_hdr_why(const pcache_hdr_t *h)
{
    if (memcmp(h->magic, PCACHE_MAGIC, 8))
        return memcmp(h->magic, "VGPU98C", 7) ? "not a pipeline cache" : "another 86Box's format";
    if (h->header_bytes != sizeof(*h))
        return "a header of another size";
    /* Its own fields too: a flipped bit in its number or sessions would
       misorder the files or age every key out. */
    if (pcache_hash(PCACHE_SEED, (const uint8_t *) h, offsetof(pcache_hdr_t, header_hash)) != h->header_hash)
        return "damaged (its header's hash does not match)";
    if ((h->vk_bytes > PCACHE_CAP) || (h->key_count > PCACHE_KEYS) || (h->key_bytes > 256))
        return "its sizes past their limits";
    return NULL;
}

/* Why a file whose header checks out is not used: it is not as long as its
   header says (a save cut short, or two writing it at once). Such a file
   is named at start. */
static const char pcache_why_length[] = "not the length its header says";

/* File `which`'s header: NULL when it checks out and the file is as long as
   it says (a save cut short leaves a whole header and less), "" when there
   is no file, else why not. */
static const char *
pcache_header(const vgpu98_gpu_t *g, int which, pcache_hdr_t *h)
{
    FILE       *f = vgpu98_gpu_cache_open(g, which, "rb");
    const char *why;

    if (!f)
        return "";
    why = (fread(h, 1, sizeof(*h), f) == sizeof(*h)) ? pcache_hdr_why(h) : "cut short";
    if (!why && (fseek(f, 0, SEEK_END) ||
                 (ftell(f) != (long) (sizeof(*h) + (size_t) h->vk_bytes + (size_t) h->key_bytes * h->key_count))))
        why = pcache_why_length;
    fclose(f);
    return why;
}

/* The newer of the two files whose headers check out (by number, which
   wraps), with its header; -1 when neither does. `why`: each file's
   (pcache_header). */
static int
pcache_newest(const vgpu98_gpu_t *g, pcache_hdr_t *h, const char *why[2])
{
    pcache_hdr_t hh[2];
    int          which;

    why[0] = pcache_header(g, 0, &hh[0]);
    why[1] = pcache_header(g, 1, &hh[1]);
    if (why[0] && why[1])
        return -1;
    which = !why[1] && (why[0] || ((int32_t) (hh[1].number - hh[0].number) > 0));
    *h    = hh[which];
    return which;
}

/* File `which` whole: its header and its body (Vulkan's data, then the
   keys), or NULL and why not ("" when there is no file). */
static uint8_t *
pcache_read(const vgpu98_gpu_t *g, int which, pcache_hdr_t *h, const char **why)
{
    FILE    *f    = vgpu98_gpu_cache_open(g, which, "rb");
    uint8_t *data = NULL;
    size_t   body;

    *why = "";
    if (!f)
        return NULL;
    if (fread(h, 1, sizeof(*h), f) != sizeof(*h))
        *why = "cut short";
    else if (!(*why = pcache_hdr_why(h))) {
        body = (size_t) h->vk_bytes + (size_t) h->key_bytes * h->key_count;
        if (!(data = malloc(body ? body : 1)))
            *why = "too large to read";
        else if (fread(data, 1, body, f) != body)
            *why = "cut short";
        else if ((pcache_hash(PCACHE_SEED, data, h->vk_bytes) != h->vk_hash) ||
                 (pcache_hash(PCACHE_SEED, data + h->vk_bytes, body - h->vk_bytes) != h->key_hash))
            *why = "damaged (its hash does not match)";
    }
    fclose(f);
    if (*why) {
        free(data);
        return NULL;
    }
    return data;
}

/* Is the format one of the renderer's own render formats (formats_init:
   its colour targets blend, its depth buffers are attachments)? Only those
   are handed to Vulkan from a file. */
static int
pcache_fmt_known(const vgpu98_r3d_t *r, uint32_t format)
{
    int i;

    for (i = 0; i < N_FORMATS; i++) {
        if ((r->fmt[i].render && ((uint32_t) r->fmt[i].vk == format)) ||
            (r->wfmt[i].render && ((uint32_t) r->wfmt[i].vk == format)))
            return 1;
    }
    return 0;
}

/* A key from a file, checked before anything is made from it: each field
   within what pipe_get writes, its formats the renderer's own render formats
   (a list outlives a change of host GPU, and names the old one's), the
   colour one blendable and the depth one an attachment. */
static int
pcache_key_ok(const vgpu98_r3d_t *r, const pipe_key_t *k)
{
    const vgpu98_vk_t *vk = r->vk;

    if ((k->shape > SHAPE_FILLSPRITE) || (k->program > 2) || (k->topology > VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) ||
        (k->cull > VK_CULL_MODE_FRONT_AND_BACK) || (k->computed_depth > 1) || (k->blend > 1) ||
        (k->src > VK_BLEND_FACTOR_SRC_ALPHA_SATURATE) || (k->dst > VK_BLEND_FACTOR_SRC_ALPHA_SATURATE) ||
        (k->depth_test > 1) || (k->depth_write > 1) || (k->depth_func > VK_COMPARE_OP_ALWAYS) || (k->stencil > 1) ||
        (k->s_fail > VK_STENCIL_OP_DECREMENT_AND_WRAP) || (k->s_zfail > VK_STENCIL_OP_DECREMENT_AND_WRAP) ||
        (k->s_pass > VK_STENCIL_OP_DECREMENT_AND_WRAP) || (k->s_func > VK_COMPARE_OP_ALWAYS) ||
        (k->blend_op > VK_BLEND_OP_MAX) || (k->write_mask > 15) || (k->pixel_program > 1) || k->pad[0])
        return 0;
    if ((k->color_vk == VK_FORMAT_UNDEFINED) || !pcache_fmt_known(r, k->color_vk) ||
        !vgpu98_vk_format_has(vk, (VkFormat) k->color_vk,
                              VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT))
        return 0;
    return (k->depth_vk == VK_FORMAT_UNDEFINED) ||
           (pcache_fmt_known(r, k->depth_vk) &&
            vgpu98_vk_format_has(vk, (VkFormat) k->depth_vk, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT));
}

/* The most recently used first; equal ones by their bytes. */
static int
pcache_key_cmp(const void *a, const void *b)
{
    const pcache_key_t *x = (const pcache_key_t *) a;
    const pcache_key_t *y = (const pcache_key_t *) b;

    if (x->session != y->session)
        return (y->session > x->session) ? 1 : -1;
    return memcmp(&x->key, &y->key, sizeof(x->key));
}

/* The keys to save: this session's (`sess`, used now), the list loaded and
   the newest file's, each once with the latest session it was used in,
   those unused for PCACHE_KEY_AGE sessions dropped, the most recently used
   first, at most PCACHE_KEYS, halved with `halve` (the cap). The count;
   `*out` malloc'd (NULL with none). */
static uint32_t
pcache_keys_merge(pcache_key_t **out, const pipe_key_t *sess, uint32_t n_sess, uint32_t session, const pcache_key_t *a,
                  uint32_t na, const pcache_key_t *b, uint32_t nb, int halve)
{
    const size_t  total = (size_t) n_sess + na + nb;
    pcache_key_t *o;
    uint32_t     *tab;
    size_t        size = 64;
    size_t        i;
    uint32_t      n = 0;

    *out = NULL;
    if (!total)
        return 0;
    while (size < 2 * total)
        size <<= 1;
    o   = malloc(total * sizeof(*o));
    tab = calloc(size, sizeof(*tab));
    if (!o || !tab) {
        free(o);
        free(tab);
        return 0;
    }
    for (i = 0; i < total; i++) {
        const pipe_key_t *key = (i < n_sess) ? &sess[i] : (i < n_sess + na) ? &a[i - n_sess].key : &b[i - n_sess - na].key;
        uint32_t          s   = (i < n_sess) ? session : (i < n_sess + na) ? a[i - n_sess].session : b[i - n_sess - na].session;
        size_t            k;

        if ((int32_t) (s - session) > 0)
            s = session; /* ahead (another 86Box's count): this session's */
        if ((int32_t) (session - s) >= PCACHE_KEY_AGE)
            continue; /* unused too long */
        for (k = key_hash(key, sizeof(*key)) & (size - 1); tab[k]; k = (k + 1) & (size - 1)) {
            if (!memcmp(&o[tab[k] - 1].key, key, sizeof(*key)))
                break;
        }
        if (tab[k]) {
            if ((int32_t) (s - o[tab[k] - 1].session) > 0)
                o[tab[k] - 1].session = s;
        } else {
            o[n].key     = *key;
            o[n].session = s;
            tab[k]       = ++n;
        }
    }
    free(tab);
    qsort(o, n, sizeof(*o), pcache_key_cmp);
    if (n > PCACHE_KEYS)
        n = PCACHE_KEYS;
    if (halve)
        n = (n + 1) / 2;
    if (!n) {
        free(o);
        return 0;
    }
    *out = o;
    return n;
}

/* The file's key list, the keys pcache_key_ok passes, the most recently
   used first: what a warm-up makes, and every save keeps. */
static void
pcache_keys_take(vgpu98_r3d_t *r, const uint8_t *list, uint32_t count)
{
    pcache_key_t *keys = malloc((size_t) count * sizeof(*keys));
    uint32_t      i;
    uint32_t      n = 0;

    if (!keys)
        return;
    memcpy(keys, list, (size_t) count * sizeof(*keys)); /* the list follows Vulkan's data, at any byte */
    for (i = 0; i < count; i++) {
        if (pcache_key_ok(r, &keys[i].key))
            keys[n++] = keys[i];
    }
    qsort(keys, n, sizeof(*keys), pcache_key_cmp);
    r->pcache_keys   = keys;
    r->pcache_n_keys = n;
}

static void
pcache_load(vgpu98_r3d_t *r)
{
    vgpu98_vk_t              *vk = r->vk;
    VkPipelineCacheCreateInfo ci = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    pcache_hdr_t              h;
    const char               *hwhy[2];
    const char               *why  = NULL; /* why it starts empty */
    const char               *rwhy = NULL;
    uint8_t                  *data = NULL;
    char                      name[1024];
    char                      other[1024];
    char                      reason[1200];
    int                       which;

    r->pcache_stamp   = pcache_stamp();
    r->pcache_session = 1;
    r->pcache_mode    = (vgpu98_gpu_options(r->gpu) & VGPU98_OPT_PCACHE_OFF) ? PCACHE_OFF :
                        (vgpu98_gpu_options(r->gpu) & VGPU98_OPT_PCACHE_EMPTY) ? PCACHE_EMPTY : PCACHE_ON;
    reason[0]         = 0;
    if (r->pcache_mode != PCACHE_ON) {
        /* Nothing read; with Start empty, the newest header's number and
           sessions go on, so the saves merge with it and its keys age. */
        if ((r->pcache_mode == PCACHE_EMPTY) && (pcache_newest(r->gpu, &h, hwhy) >= 0)) {
            r->pcache_number  = h.number;
            r->pcache_session = h.session + 1;
        }
        if (vk->vkCreatePipelineCache(vk->dev, &ci, NULL, &r->pcache) != VK_SUCCESS)
            r->pcache = VK_NULL_HANDLE;
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: %s\n",
                (r->pcache_mode == PCACHE_OFF) ? "Off (neither read nor saved)" : "started empty (the setting is Start empty)");
        return;
    }
    which = pcache_newest(r->gpu, &h, hwhy);
    if (which < 0) {
        /* Neither header checks out: none saved yet, or why the first does
           not. */
        which = *hwhy[0] ? 0 : 1;
        if (*hwhy[which]) {
            vgpu98_gpu_cache_name(r->gpu, which, name, sizeof(name));
            snprintf(reason, sizeof(reason), "%s: %s", name, hwhy[which]);
        } else
            why = "none saved yet";
    } else {
        /* The newest header's number and sessions, whatever is read: a save
           tells another 86Box's from the number, the keys age by the
           sessions. The other file, if not the length its header says, is
           named. */
        r->pcache_number  = h.number;
        r->pcache_session = h.session + 1;
        if (hwhy[!which] == pcache_why_length) {
            vgpu98_gpu_cache_name(r->gpu, !which, other, sizeof(other));
            r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: %s failed its check (%s)\n", other, pcache_why_length);
        }
        data = pcache_read(r->gpu, which, &h, &rwhy);
        vgpu98_gpu_cache_name(r->gpu, which, name, sizeof(name));
        if (!data && !hwhy[!which]) {
            vgpu98_gpu_cache_name(r->gpu, !which, other, sizeof(other));
            r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: %s failed its check (%s)\n", name, *rwhy ? rwhy : "gone");
            which = !which;
            memcpy(name, other, sizeof(name));
            data = pcache_read(r->gpu, which, &h, &rwhy);
        }
        if (!data)
            snprintf(reason, sizeof(reason), "%s: %s", name, *rwhy ? rwhy : "gone");
    }
    if (data) {
        if (h.stamp != r->pcache_stamp)
            why = "the GPU's shaders changed";
        else if (!h.vk_bytes)
            why = "emptied at its limit";
        else if (!pcache_vk_fits(vk, data, h.vk_bytes))
            why = "the host GPU or its driver changed";
        else {
            ci.initialDataSize = h.vk_bytes;
            ci.pInitialData    = data;
        }
        /* The key list, whatever became of Vulkan's data (its version and
           record this 86Box's). */
        if ((h.key_version == PCACHE_KEY_VERSION) && (h.key_bytes == sizeof(pcache_key_t)) && h.key_count)
            pcache_keys_take(r, data + h.vk_bytes, h.key_count);
    }
    if (vk->vkCreatePipelineCache(vk->dev, &ci, NULL, &r->pcache) != VK_SUCCESS) {
        /* Data the driver refuses: an empty cache. */
        if (ci.initialDataSize)
            why = "the host's driver refused it";
        ci.initialDataSize = 0;
        ci.pInitialData    = NULL;
        if (vk->vkCreatePipelineCache(vk->dev, &ci, NULL, &r->pcache) != VK_SUCCESS)
            r->pcache = VK_NULL_HANDLE;
    }
    /* With a list, its pipelines are made in the background whatever became
       of the data (pcache_warm_start, once the renderer's objects are):
       those in the data take milliseconds, the rest (a warm-up cut short
       last time, data the driver ignores) are made then, not at first use. */
    if (r->pcache) {
        r->warm_n    = r->pcache_n_keys;
        r->warm_cold = !ci.initialDataSize;
    }
    if (ci.initialDataSize && r->warm_n)
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: %u KB from %s; its %u listed pipelines readied in the background\n",
                (unsigned) (ci.initialDataSize >> 10), name, r->warm_n);
    else if (ci.initialDataSize)
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: %u KB from %s\n", (unsigned) (ci.initialDataSize >> 10), name);
    else if (r->warm_n)
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: started empty (%s); its %u listed pipelines made again in the background\n",
                why ? why : reason, r->warm_n);
    else
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: started empty (%s)\n", why ? why : reason);
    free(data);
}

/* A cache's data (malloc'd), or NULL: none, or past PCACHE_CAP (`*bytes`
   then says how much). Pipelines made between the size and the data grow
   it (VK_INCOMPLETE): asked again, a few times. */
static uint8_t *
pcache_fetch(const vgpu98_vk_t *vk, VkPipelineCache c, size_t *bytes)
{
    uint8_t *data = NULL;
    int      tries;

    *bytes = 0;
    for (tries = 0; tries < 4; tries++) {
        size_t size = 0;

        free(data);
        data = NULL;
        if ((vk->vkGetPipelineCacheData(vk->dev, c, &size, NULL) != VK_SUCCESS) || !size)
            return NULL;
        *bytes = size;
        if ((size > PCACHE_CAP) || !(data = malloc(size)))
            return NULL;
        if (vk->vkGetPipelineCacheData(vk->dev, c, &size, data) == VK_SUCCESS) {
            *bytes = size;
            return data;
        }
    }
    free(data);
    *bytes = 0;
    return NULL;
}

/* A save's work, on its own thread or (at close) this one (the cache needs
   no external synchronization: a merge reads it as a source). The newest
   file on disk is read first: the save goes to the other file, numbered one
   past it, so the newest stays whole whatever becomes of this save, and it
   is merged in. Vulkan's data: this cache's, merged with the newest file's
   when that file's stamp and Vulkan's header are this 86Box's and the
   host's, so the pipelines another 86Box (or an earlier session) saved stay
   (a merge alone past the cap: this cache's alone); past the cap, none, and
   from then on the keys alone. The keys: this session's, the list loaded
   and the newest file's (pcache_keys_merge), halved at the cap. What fails
   is named once a session, as are the cap and the first save of another
   86Box merged. */
static void
pcache_write(void *priv)
{
    pcache_job_t             *j     = (pcache_job_t *) priv;
    vgpu98_vk_t              *vk    = j->vk;
    VkPipelineCacheCreateInfo ci    = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    VkPipelineCache           tmp   = VK_NULL_HANDLE;
    VkPipelineCache           src   = j->cache;
    size_t                    size  = 0;
    uint8_t                  *data  = NULL;
    uint8_t                  *ddata = NULL;
    pcache_key_t             *dkeys = NULL;
    pcache_key_t             *keys  = NULL;
    uint32_t                  n_dkeys = 0;
    uint32_t                  n_keys;
    uint32_t                  fail = 0;
    uint32_t                  session;
    int                       newest;
    int                       merged = 0; /* the newest file's data or keys taken in */
    pcache_hdr_t              h;
    pcache_hdr_t              nh;
    pcache_hdr_t              rh;
    const char               *nwhy[2];
    const char               *rwhy;
    char                      name[1024];
    FILE                     *f;

    j->ok     = 0;
    j->capped = 0;
    newest    = pcache_newest(j->gpu, &nh, nwhy);
    j->which  = (newest >= 0) ? !newest : 0;
    j->number = ((newest >= 0) ? nh.number : j->last) + 1;
    if (newest >= 0) {
        ddata = pcache_read(j->gpu, newest, &rh, &rwhy);
        /* The newest failing its read (damaged), the older read instead:
           the save replaces the damaged one, the good one stays. */
        if (!ddata && !nwhy[!newest] && (ddata = pcache_read(j->gpu, !newest, &rh, &rwhy)))
            j->which = newest;
    }
    /* A save's session is never behind the file it merges (one that read
       nothing at start, another 86Box saving since): its keys age as the
       file's. */
    session = (ddata && ((int32_t) (rh.session - j->session) > 0)) ? rh.session : j->session;
    if (ddata && (rh.key_version == PCACHE_KEY_VERSION) && (rh.key_bytes == sizeof(pcache_key_t)) && rh.key_count &&
        (dkeys = malloc((size_t) rh.key_count * sizeof(*dkeys)))) {
        memcpy(dkeys, ddata + rh.vk_bytes, (size_t) rh.key_count * sizeof(*dkeys));
        n_dkeys = rh.key_count;
        merged  = 1;
    }

    if (!j->keys_only) {
        if (ddata && rh.vk_bytes && (rh.stamp == j->stamp) && pcache_vk_fits(vk, ddata, rh.vk_bytes)) {
            ci.initialDataSize = rh.vk_bytes;
            ci.pInitialData    = ddata;
            if (vk->vkCreatePipelineCache(vk->dev, &ci, NULL, &tmp) != VK_SUCCESS)
                tmp = VK_NULL_HANDLE;
            else if (vk->vkMergePipelineCaches(vk->dev, tmp, 1, &j->cache) == VK_SUCCESS)
                src = tmp;
        }
        /* The file's copy is in the driver's now (or not wanted): gone
           before the fetch, so two copies of 64 MB are not held at once. */
        free(ddata);
        ddata = NULL;
        data  = pcache_fetch(vk, src, &size);
        if (data && (src != j->cache))
            merged = 1;
        else if (!data && (size > PCACHE_CAP) && (src != j->cache))
            data = pcache_fetch(vk, j->cache, &size); /* the merge past the cap: this cache's alone */
        if (tmp)
            vk->vkDestroyPipelineCache(vk->dev, tmp, NULL);
        if (!data) {
            if (size > PCACHE_CAP)
                j->capped = 1;
            else
                fail = size ? PCACHE_TOLD_MEM : PCACHE_TOLD_DATA;
            size = 0;
        }
    }

    /* Past the cap every save halves: the list stays at about half of what
       filled the cache, so the next start's warm-up fits. */
    n_keys = pcache_keys_merge(&keys, j->sess, j->n_sess, session, j->keys, j->n_keys, dkeys, n_dkeys,
                               j->capped || j->keys_only);
    if (j->capped && !(j->told & PCACHE_TOLD_CAP)) {
        j->told |= PCACHE_TOLD_CAP;
        vgpu98_gpu_log(j->gpu, VGPU98_LOG_EVENTS,
                       "VGPU98 3D: pipeline cache: past %u MB: from now on its pipelines' keys alone are saved; the next start "
                       "makes the %u most recently used again in the background\n",
                       PCACHE_CAP >> 20, n_keys);
    }

    if (!fail) {
        memset(&h, 0, sizeof(h));
        memcpy(h.magic, PCACHE_MAGIC, 8);
        h.header_bytes = sizeof(h);
        h.number       = j->number;
        h.session      = session;
        h.stamp        = j->stamp;
        h.vk_bytes     = (uint32_t) size;
        h.vk_hash      = pcache_hash(PCACHE_SEED, data, size);
        h.key_version  = PCACHE_KEY_VERSION;
        h.key_bytes    = sizeof(pcache_key_t);
        h.key_count    = n_keys;
        h.key_hash     = pcache_hash(PCACHE_SEED, (const uint8_t *) keys, (size_t) n_keys * sizeof(*keys));
        h.header_hash  = pcache_hash(PCACHE_SEED, (const uint8_t *) &h, offsetof(pcache_hdr_t, header_hash));
        if (!(f = vgpu98_gpu_cache_open(j->gpu, j->which, "wb")))
            fail = PCACHE_TOLD_OPEN;
        else {
            j->ok = (fwrite(&h, 1, sizeof(h), f) == sizeof(h)) && (fwrite(data, 1, size, f) == size) &&
                    (fwrite(keys, sizeof(*keys), n_keys, f) == n_keys);
            if (fclose(f))
                j->ok = 0;
            if (!j->ok)
                fail = PCACHE_TOLD_WRITE;
        }
    }
    /* Another 86Box's save (its number not the one loaded or last saved),
       taken into a save that worked: named once. */
    if (j->ok && merged && (nh.number != j->last) && !(j->told & PCACHE_TOLD_MERGE)) {
        j->told |= PCACHE_TOLD_MERGE;
        vgpu98_gpu_log(j->gpu, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: another 86Box's save merged into this one's\n");
    }
    if (fail && !(j->told & fail)) {
        j->told |= fail;
        if ((fail == PCACHE_TOLD_DATA) || (fail == PCACHE_TOLD_MEM))
            vgpu98_gpu_log(j->gpu, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: not saved (%s)\n",
                           (fail == PCACHE_TOLD_DATA) ? "the host's driver gave no data" : "no memory for its data");
        else {
            vgpu98_gpu_cache_name(j->gpu, j->which, name, sizeof(name));
            vgpu98_gpu_log(j->gpu, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: saving to %s failed (%s)\n", name,
                           (fail == PCACHE_TOLD_OPEN) ? "it could not be opened" : "it could not be written whole");
        }
    }
    free(data);
    free(ddata);
    free(dkeys);
    free(keys);
    free(j->sess);
    j->sess   = NULL;
    j->n_sess = 0;
    /* The event first: once `done` is set the thread touches nothing more,
       so the event may go at close. */
    if (j->ended)
        thread_set_event(j->ended);
    atomic_store(&j->done, 1);
}

static void pcache_done(vgpu98_r3d_t *r);

/* The last save, once its thread is done (with `wait`, waited for up to
   PCACHE_JOIN_S seconds): its file is the last saved. The thread itself is
   never waited for: a thread's handle may close as it ends (_beginthread's
   on Windows), and a wait on a closed handle, its number since given to
   another object, may never return (3DMark03's CPU test 2 hung that way,
   at a save two minutes after the last). */
static void
pcache_join(vgpu98_r3d_t *r, int wait)
{
    int i;

    if (!r->pcache_running)
        return;
    /* The event may be left set by an earlier save, and not every platform's
       resets itself: `done` decides, the event reset before each wait. */
    for (i = 0; wait && !atomic_load(&r->pcache_job.done) && (i < PCACHE_JOIN_S); i++) {
        thread_reset_event(r->pcache_ev);
        if (!atomic_load(&r->pcache_job.done))
            thread_wait_event(r->pcache_ev, 1000);
    }
    if (!atomic_load(&r->pcache_job.done))
        return; /* still writing */
    r->pcache_running = 0;
    pcache_done(r);
}

/* A save's results (on the GPU thread): its number is the last saved; not
   saved, the next time and at close; past the cap, the keys alone from now
   on (the cache in memory stays as it is). */
static void
pcache_done(vgpu98_r3d_t *r)
{
    const pcache_job_t *j = &r->pcache_job;

    if (j->ok)
        r->pcache_number = j->number;
    else
        r->pcache_new = 1;
    if (j->capped)
        r->pcache_capped = 1;
}

/* On a thread of its own (`sync`: on this one, at close); one save at a
   time. It carries a copy of this session's pipelines' keys. */
static void
pcache_save(vgpu98_r3d_t *r, int sync)
{
    vgpu98_vk_t  *vk = r->vk;
    pcache_job_t *j  = &r->pcache_job;
    size_t        i;

    r->pcache_us = vgpu98_diag_us();
    pcache_join(r, sync);
    if ((r->pcache_mode == PCACHE_OFF) || r->pcache_running || !r->pcache)
        return; /* Off, or the last save still writes: the next */
    r->pcache_new = 0;
    j->sess       = NULL;
    j->n_sess     = 0;
    if (r->n_pipes && (j->sess = malloc(r->n_pipes * sizeof(pipe_key_t)))) {
        for (i = 0; i < r->n_pipes; i++)
            j->sess[i] = r->pipes[i].key;
        j->n_sess = (uint32_t) r->n_pipes;
    }
    j->vk        = vk;
    j->cache     = r->pcache;
    j->gpu       = r->gpu;
    j->stamp     = r->pcache_stamp;
    j->session   = r->pcache_session;
    j->last      = r->pcache_number;
    j->keys      = r->pcache_keys;
    j->n_keys    = r->pcache_n_keys;
    j->keys_only = r->pcache_capped;
    j->ok        = 0;
    j->ended     = r->pcache_ev;
    atomic_store(&j->done, 0);
    if (!sync && r->pcache_ev) {
        /* A thread that could not be made is NULL, or -1 from _beginthread:
           the save goes on this thread instead. */
        const thread_t *t = thread_create_named(pcache_write, j, "VGPU98 pipeline cache");

        if (t && (t != (thread_t *) (intptr_t) -1)) {
            r->pcache_running = 1;
            return;
        }
    }
    j->ended = NULL;
    pcache_write(j);
    pcache_done(r);
}

static VkSampler
sampler_get(vgpu98_r3d_t *r, const samp_key_t *key)
{
    vgpu98_vk_t        *vk = r->vk;
    VkSamplerCreateInfo ci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    VkSamplerCustomBorderColorCreateInfoEXT border = { VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT };
    static const VkSamplerAddressMode addr[5] = { VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                  VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                                                  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                                  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER };
    VkSampler      s;
    const uint32_t h = key_hash(key, sizeof(*key));
    uint32_t       slot;
    uint32_t       found;

    if (r->samp_tab &&
        (found = tab_find(r->samp_tab, r->samp_tab_size, h, key, sizeof(*key), r->samps, sizeof(samp_t), &slot)))
        return r->samps[found - 1].sampler;
    ci.magFilter    = key->mag ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter    = key->min ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode   = key->mip ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = addr[key->addr_u];
    ci.addressModeV = addr[key->addr_v];
    ci.addressModeW = key->addr_w ? addr[key->addr_w] : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.anisotropyEnable = key->aniso > 1;
    ci.maxAnisotropy    = (float) key->aniso;
    ci.minLod           = key->min_lod;
    ci.maxLod           = key->max_lod;
    ci.borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (key->border) {
        ci.borderColor                  = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
        border.customBorderColor.float32[0] = ((key->border_color >> 16) & 255) / 255.0f;
        border.customBorderColor.float32[1] = ((key->border_color >> 8) & 255) / 255.0f;
        border.customBorderColor.float32[2] = (key->border_color & 255) / 255.0f;
        border.customBorderColor.float32[3] = ((key->border_color >> 24) & 255) / 255.0f;
        border.format                   = VK_FORMAT_UNDEFINED;
        ci.pNext                        = &border;
        r->n_custom_borders++;
    }
    if (vk->vkCreateSampler(vk->dev, &ci, NULL, &s) != VK_SUCCESS) {
        r3d_fail(r, "vkCreateSampler", VK_ERROR_INITIALIZATION_FAILED);
        return VK_NULL_HANDLE;
    }
    r->samps = realloc(r->samps, (r->n_samps + 1) * sizeof(samp_t));
    r->samps[r->n_samps].key     = *key;
    r->samps[r->n_samps].sampler = s;
    r->n_samps++;
    if (tab_grow(&r->samp_tab, &r->samp_tab_size, (uint32_t) r->n_samps - 1, r->samps, sizeof(samp_t), sizeof(*key)) &&
        !tab_find(r->samp_tab, r->samp_tab_size, h, key, sizeof(*key), r->samps, sizeof(samp_t), &slot))
        r->samp_tab[slot] = (uint32_t) r->n_samps;
    return s;
}

/* A pipeline for a key and a render pass that matches its formats, into
   the cache: what pipe_get makes, and a warm-up thread makes ahead. It
   reads nothing that changes after start (the shaders, the layout, the
   cache). Any thread. A change here (its fixed or dynamic states, its
   specialization constants) raises PCACHE_PIPE_VERSION. */
static VkResult
pipe_make(const vgpu98_r3d_t *r, const pipe_key_t *key, VkRenderPass pass, VkPipeline *p)
{
    vgpu98_vk_t                           *vk = r->vk;
    VkPipelineShaderStageCreateInfo        st[2];
    VkPipelineVertexInputStateCreateInfo   vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    /* The shaders' specialization constants: SHAPE (0), PROGRAM (1),
       LINE_RULE (2), PIXEL_PROGRAM (3, 2.2), one block for both stages. */
    const VkSpecializationMapEntry         sme[4] = { { 0, 0, 4 }, { 1, 4, 4 }, { 2, 8, 4 }, { 3, 12, 4 } };
    uint32_t                               spec[4];
    VkSpecializationInfo                   si;
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkPipelineViewportStateCreateInfo      vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineMultisampleStateCreateInfo   ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    VkPipelineDepthStencilStateCreateInfo  dss = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    VkPipelineColorBlendAttachmentState    cba;
    VkPipelineColorBlendStateCreateInfo    cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    VkDynamicState                         dyn[6] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                                      VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                                                      VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                                      VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo       ds = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    VkGraphicsPipelineCreateInfo           ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    VkStencilOpState                       so;

    spec[0]          = key->shape;
    spec[1]          = key->program;
    spec[2]          = ((key->shape == SHAPE_LINE) || (key->shape == SHAPE_WIRE)) ? VK_TRUE : VK_FALSE;
    spec[3]          = key->pixel_program ? VK_TRUE : VK_FALSE;
    si.mapEntryCount = 4;
    si.pMapEntries   = sme;
    si.dataSize      = sizeof(spec);
    si.pData         = spec;
    memset(st, 0, sizeof(st));
    st[0].sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[0].stage               = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module              = r->vs;
    st[0].pName               = "main";
    st[0].pSpecializationInfo = &si;
    st[1].sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[1].stage               = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module              = key->computed_depth ? r->fs_depth : r->fs;
    st[1].pName               = "main";
    st[1].pSpecializationInfo = &si;

    /* No vertex attributes: the vertex shader reads its storage buffers. */
    ia.topology      = (VkPrimitiveTopology) key->topology;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    /* Direct3D's clockwise (CULLMODE 2) is Vulkan's front face here: screen
       y points down in both. */
    rs.cullMode        = key->cull;
    rs.frontFace       = VK_FRONT_FACE_CLOCKWISE;
    rs.depthBiasEnable = VK_TRUE;
    rs.lineWidth       = 1.0f;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    dss.depthTestEnable  = key->depth_test;
    dss.depthWriteEnable = key->depth_write;
    dss.depthCompareOp   = (VkCompareOp) key->depth_func;
    dss.stencilTestEnable = key->stencil;
    memset(&so, 0, sizeof(so));
    so.failOp      = (VkStencilOp) key->s_fail;
    so.passOp      = (VkStencilOp) key->s_pass;
    so.depthFailOp = (VkStencilOp) key->s_zfail;
    so.compareOp   = (VkCompareOp) key->s_func;
    dss.front      = so;
    dss.back       = so;
    dss.maxDepthBounds = 1.0f;

    memset(&cba, 0, sizeof(cba));
    cba.blendEnable         = key->blend;
    cba.srcColorBlendFactor = (VkBlendFactor) key->src;
    cba.dstColorBlendFactor = (VkBlendFactor) key->dst;
    cba.colorBlendOp        = (VkBlendOp) key->blend_op;
    cba.srcAlphaBlendFactor = (VkBlendFactor) key->src;
    cba.dstAlphaBlendFactor = (VkBlendFactor) key->dst;
    cba.alphaBlendOp        = (VkBlendOp) key->blend_op;
    cba.colorWriteMask      = key->write_mask;
    cb.attachmentCount = 1;
    cb.pAttachments    = &cba;
    ds.dynamicStateCount = 6;
    ds.pDynamicStates    = dyn;

    ci.stageCount          = 2;
    ci.pStages             = st;
    ci.pVertexInputState   = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState   = &ms;
    ci.pDepthStencilState  = &dss;
    ci.pColorBlendState    = &cb;
    ci.pDynamicState       = &ds;
    ci.layout              = r->layout;
    ci.renderPass          = pass;
    return vk->vkCreateGraphicsPipelines(vk->dev, r->pcache, 1, &ci, NULL, p);
}

static VkPipeline
pipe_get(vgpu98_r3d_t *r, const pipe_key_t *key, VkRenderPass pass)
{
    VkPipeline     p;
    const uint32_t h = key_hash(key, sizeof(*key));
    uint32_t       slot;
    uint32_t       found;
    uint64_t       t0;
    VkResult       made;
    int            step;

    if (r->pipe_tab &&
        (found = tab_find(r->pipe_tab, r->pipe_tab_size, h, key, sizeof(*key), r->pipes, sizeof(pipe_t), &slot)))
        return r->pipes[found - 1].pipeline;
    t0   = vgpu98_diag_us();
    step = vgpu98_gpu_step(r->gpu, VGPU98_STEP_PIPELINE);
    made = pipe_make(r, key, pass, &p);
    vgpu98_gpu_step(r->gpu, step);
    if (made != VK_SUCCESS) {
        r3d_fail(r, "vkCreateGraphicsPipelines", made);
        return VK_NULL_HANDLE;
    }
    r->stats->pipeline_us += us_since(t0);
    r->pipes = realloc(r->pipes, (r->n_pipes + 1) * sizeof(pipe_t));
    r->pipes[r->n_pipes].key      = *key;
    r->pipes[r->n_pipes].pipeline = p;
    r->n_pipes++;
    if (tab_grow(&r->pipe_tab, &r->pipe_tab_size, (uint32_t) r->n_pipes - 1, r->pipes, sizeof(pipe_t), sizeof(*key)) &&
        !tab_find(r->pipe_tab, r->pipe_tab_size, h, key, sizeof(*key), r->pipes, sizeof(pipe_t), &slot))
        r->pipe_tab[slot] = (uint32_t) r->n_pipes;
    r->pcache_new++;
    r->stats->new_pipelines++;
    vgpu98_gpu_note(r->gpu, VGPU98_NOTE_PIPELINE, (uint32_t) r->n_pipes,
                    (key->shape | (key->program << 4)) | (key->blend << 8) | (key->src << 16) | ((uint32_t) key->dst << 24),
                    key->depth_test | (key->depth_write << 8) | (key->depth_func << 16) | ((uint32_t) key->stencil << 24), 0, 0);
    return p;
}

/* --- The pipeline cache's warm-up ---------- */

/* A warm-up thread: the next listed key's pipeline into the cache, until
   none is left, 86Box closes or the device is lost; its render passes its
   own, kept for the keys that follow. It lets each pipeline go: the GPU
   thread's own request then finds its work in the cache. */
static void
pcache_warm(void *priv)
{
    vgpu98_r3d_t *r  = (vgpu98_r3d_t *) priv;
    vgpu98_vk_t  *vk = r->vk;
    pass_t        passes[PCACHE_WARM_PASSES];
    uint32_t      n_passes = 0;
    uint32_t      i;
    uint32_t      k;

    while (!atomic_load(&r->warm_stop) && ((i = atomic_fetch_add(&r->warm_next, 1)) < r->warm_n)) {
        const pipe_key_t *key  = &r->pcache_keys[i].key;
        VkRenderPass      pass = VK_NULL_HANDLE;
        int               kept = 0;
        VkPipeline        p;
        VkResult          res = VK_SUCCESS;

        for (k = 0; (k < n_passes) && !pass; k++) {
            if ((passes[k].color == (VkFormat) key->color_vk) && (passes[k].depth == (VkFormat) key->depth_vk)) {
                pass = passes[k].pass;
                kept = 1;
            }
        }
        if (!pass) {
            if ((res = pass_make(vk, (VkFormat) key->color_vk, (VkFormat) key->depth_vk, &pass)) != VK_SUCCESS)
                pass = VK_NULL_HANDLE; /* not made: nothing to let go */
            else if (n_passes < PCACHE_WARM_PASSES) {
                passes[n_passes].color = (VkFormat) key->color_vk;
                passes[n_passes].depth = (VkFormat) key->depth_vk;
                passes[n_passes].pass  = pass;
                n_passes++;
                kept = 1;
            }
        }
        if (res == VK_SUCCESS)
            res = pipe_make(r, key, pass, &p);
        if (res == VK_SUCCESS) {
            vk->vkDestroyPipeline(vk->dev, p, NULL);
            atomic_fetch_add(&r->warm_made, 1);
            atomic_store(&r->warm_dirty, 1);
        } else
            atomic_fetch_add(&r->warm_failed, 1);
        if (pass && !kept)
            vk->vkDestroyRenderPass(vk->dev, pass, NULL);
        if (res == VK_ERROR_DEVICE_LOST)
            break;
    }
    for (k = 0; k < n_passes; k++)
        vk->vkDestroyRenderPass(vk->dev, passes[k].pass, NULL);
    atomic_store(&r->warm_end, (uint32_t) (vgpu98_diag_us() / 1000));
    thread_set_event(r->warm_ev);
    atomic_fetch_sub(&r->warm_live, 1); /* its last access to the renderer */
}

/* With a list to make again (pcache_load), its threads, once the
   renderer's shaders and layout are made. */
static void
pcache_warm_start(vgpu98_r3d_t *r)
{
    int i;

    if (!r->warm_n || !r->pcache)
        return;
    if (!r->warm_cold && (r->vk->vkGetPipelineCacheData(r->vk->dev, r->pcache, &r->warm_size0, NULL) != VK_SUCCESS))
        r->warm_size0 = 0;
    if ((r->warm_ev = thread_create_event())) {
        atomic_store(&r->warm_start, (uint32_t) (vgpu98_diag_us() / 1000));
        for (i = 0; i < PCACHE_WARM_THREADS; i++) {
            const thread_t *t;

            atomic_fetch_add(&r->warm_live, 1);
            t = thread_create_named(pcache_warm, r, "VGPU98 pipeline warm-up");
            if (!t || (t == (thread_t *) (intptr_t) -1)) {
                atomic_fetch_sub(&r->warm_live, 1);
                break;
            }
            r->warm_threads++;
        }
    }
    if (!r->warm_threads)
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: the background warm-up could not start\n");
}

/* On the GPU thread: what the warm-up made counts toward the next save,
   and its end is named once. */
static void
pcache_warm_check(vgpu98_r3d_t *r)
{
    uint32_t made;
    uint32_t failed;
    size_t   size = 0;
    char     why[64];

    /* Started cold, every pipeline it makes is new (saved as it goes);
       started from the data, most are found there: saved at its end only
       for what it added. */
    if (atomic_exchange(&r->warm_dirty, 0) && r->warm_cold)
        r->pcache_new++;
    if (!r->warm_threads || r->warm_told || atomic_load(&r->warm_live))
        return;
    r->warm_told = 1;
    if (!r->warm_cold && (r->vk->vkGetPipelineCacheData(r->vk->dev, r->pcache, &size, NULL) == VK_SUCCESS) &&
        (size > r->warm_size0))
        r->pcache_new++;
    made         = atomic_load(&r->warm_made);
    failed       = atomic_load(&r->warm_failed);
    why[0]       = 0;
    if (failed)
        snprintf(why, sizeof(why), "; %u could not be made", failed);
    if (made + failed < r->warm_n)
        snprintf(why + strlen(why), sizeof(why) - strlen(why), "; %s",
                 atomic_load(&r->warm_stop) ? "stopped at close" : "stopped: the host GPU failed");
    r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: pipeline cache: %u of %u listed pipelines ready in %.1f s%s\n", made, r->warm_n,
            (double) (atomic_load(&r->warm_end) - atomic_load(&r->warm_start)) / 1000.0, why);
}

/* At close: the warm-up stops after the pipelines in hand, waited for by its
   event up to PCACHE_WARM_JOIN_S seconds. 0 when a thread has not ended. */
static int
pcache_warm_stop(vgpu98_r3d_t *r)
{
    int i;

    atomic_store(&r->warm_stop, 1);
    for (i = 0; atomic_load(&r->warm_live) && (i < PCACHE_WARM_JOIN_S); i++) {
        thread_reset_event(r->warm_ev); /* as pcache_join's */
        if (atomic_load(&r->warm_live))
            thread_wait_event(r->warm_ev, 1000);
    }
    return !atomic_load(&r->warm_live);
}

/* --- State ------------------------------------------------------------------- */

static const uint32_t rs_defaults[][2] = {
    { VGPU98_RS_TEXTUREPERSPECTIVE, 1 }, { VGPU98_RS_ZENABLE, 0 },       { VGPU98_RS_FILLMODE, 3 },
    { VGPU98_RS_SHADEMODE, 2 },          { VGPU98_RS_ZWRITEENABLE, 1 },  { VGPU98_RS_ALPHATESTENABLE, 0 },
    { VGPU98_RS_LASTPIXEL, 1 },          { VGPU98_RS_SRCBLEND, 2 },      { VGPU98_RS_DESTBLEND, 1 },
    { VGPU98_RS_CULLMODE, 3 },           { VGPU98_RS_ZFUNC, 4 },         { VGPU98_RS_ALPHAREF, 0 },
    { VGPU98_RS_ALPHAFUNC, 8 },          { VGPU98_RS_DITHERENABLE, 0 },  { VGPU98_RS_ALPHABLENDENABLE, 0 },
    { VGPU98_RS_FOGENABLE, 0 },          { VGPU98_RS_SPECULARENABLE, 0 }, { VGPU98_RS_FOGCOLOR, 0 },
    { VGPU98_RS_FOGTABLEMODE, 0 },       { VGPU98_RS_FOGTABLESTART, 0x00000000 }, { VGPU98_RS_FOGTABLEEND, 0x3f800000 },
    { VGPU98_RS_FOGTABLEDENSITY, 0x3f800000 }, { VGPU98_RS_COLORKEYENABLE, 0 }, { VGPU98_RS_ZBIAS, 0 },
    { VGPU98_RS_STENCILENABLE, 0 },      { VGPU98_RS_STENCILFAIL, 1 },   { VGPU98_RS_STENCILZFAIL, 1 },
    { VGPU98_RS_STENCILPASS, 1 },        { VGPU98_RS_STENCILFUNC, 8 },   { VGPU98_RS_STENCILREF, 0 },
    { VGPU98_RS_STENCILMASK, 0xffffffff }, { VGPU98_RS_STENCILWRITEMASK, 0xffffffff },
    { VGPU98_RS_TEXTUREFACTOR, 0xffffffff },
    /* DirectX 7's, as its documentation gives them */
    { VGPU98_RS_CLIPPING, 1 },           { VGPU98_RS_LIGHTING, 1 },      { VGPU98_RS_COLORVERTEX, 1 },
    { VGPU98_RS_LOCALVIEWER, 1 },        { VGPU98_RS_DIFFUSEMATERIALSOURCE, VGPU98_MCS_COLOR1 },
    { VGPU98_RS_SPECULARMATERIALSOURCE, VGPU98_MCS_COLOR2 }, { VGPU98_RS_AMBIENTMATERIALSOURCE, VGPU98_MCS_COLOR2 },
    { VGPU98_RS_EMISSIVEMATERIALSOURCE, VGPU98_MCS_MATERIAL },
    /* Command set 2.1's, as DirectX 8 documents them */
    { VGPU98_RS_POINTSIZE, 0x3f800000 }, { VGPU98_RS_POINTSIZE_MAX, 0x42800000 /* 64.0 */ },
    { VGPU98_RS_POINTSCALE_A, 0x3f800000 }, { VGPU98_RS_MULTISAMPLEANTIALIAS, 1 },
    { VGPU98_RS_MULTISAMPLEMASK, 0xffffffff }, { VGPU98_RS_COLORWRITEENABLE, 0xf },
    { VGPU98_RS_BLENDOP, VGPU98_BLENDOP_ADD }, { VGPU98_RS_PATCHSEGMENTS, 0x3f800000 },
    { VGPU98_RS_POSITIONORDER, 3 /* cubic */ }, { VGPU98_RS_NORMALORDER, 1 /* linear */ },
};

/* Does the GPU take command set 2.1's values (PROFILE.FIXED8)? */
static int
fixed8(const vgpu98_r3d_t *r)
{
    return (vgpu98_gpu_profile(r->gpu) & VGPU98_PROFILE_FIXED8) != 0;
}

static void
state_defaults(vgpu98_r3d_t *r)
{
    int s;

    memset(r->rs, 0, sizeof(r->rs));
    for (s = 0; s < (int) (sizeof(rs_defaults) / sizeof(rs_defaults[0])); s++)
        r->rs[rs_defaults[s][0]] = rs_defaults[s][1];
    memset(r->tss, 0, sizeof(r->tss));
    for (s = 0; s < N_STAGES; s++) {
        r->tss[s][VGPU98_TSS_COLOROP]       = s ? VGPU98_TOP_DISABLE : VGPU98_TOP_MODULATE;
        r->tss[s][VGPU98_TSS_COLORARG1]     = VGPU98_TA_TEXTURE;
        r->tss[s][VGPU98_TSS_COLORARG2]     = VGPU98_TA_CURRENT;
        r->tss[s][VGPU98_TSS_ALPHAOP]       = s ? VGPU98_TOP_DISABLE : VGPU98_TOP_SELECTARG1;
        r->tss[s][VGPU98_TSS_ALPHAARG1]     = VGPU98_TA_TEXTURE;
        r->tss[s][VGPU98_TSS_ALPHAARG2]     = VGPU98_TA_CURRENT;
        r->tss[s][VGPU98_TSS_TEXCOORDINDEX] = (uint32_t) s;
        r->tss[s][VGPU98_TSS_ADDRESSU]      = 1;
        r->tss[s][VGPU98_TSS_ADDRESSV]      = 1;
        r->tss[s][VGPU98_TSS_MAGFILTER]     = 1;
        r->tss[s][VGPU98_TSS_MINFILTER]     = 1;
        r->tss[s][VGPU98_TSS_MIPFILTER]     = 1;
        r->tss[s][VGPU98_TSS_MAXANISOTROPY] = 1;
        /* 2.1's */
        r->tss[s][VGPU98_TSS_ADDRESSW]      = 1;
        r->tss[s][VGPU98_TSS_COLORARG0]     = VGPU98_TA_CURRENT;
        r->tss[s][VGPU98_TSS_ALPHAARG0]     = VGPU98_TA_CURRENT;
        r->tss[s][VGPU98_TSS_RESULTARG]     = VGPU98_TA_CURRENT;
    }
    memset(r->tex, 0, sizeof(r->tex));
    memset(r->pal, 0, sizeof(r->pal));
    r->have_target = 0;
    r->have_depth  = 0;
    memset(r->vp, 0, sizeof(r->vp));
    wrange_set(r, 1.0f, 1.0f);
    r->zrange[0] = 0.0f;
    r->zrange[1] = 1.0f;

    /* Every matrix the identity; every light slot a directional light with
       diffuse (1, 1, 1, 0) along +z, none enabled; the material and the
       clip planes 0 (spec/gpu-commands.md, "Fixed-function state"). */
    memset(r->xform, 0, sizeof(r->xform));
    for (s = 0; s < VGPU98_XFORM_COUNT; s++)
        r->xform[s][0] = r->xform[s][5] = r->xform[s][10] = r->xform[s][15] = 1.0f;
    /* The palette (2.1): every world matrix the identity. */
    memset(r->pal_world, 0, sizeof(r->pal_world));
    for (s = 0; s < VGPU98_PALETTE_SIZE; s++)
        r->pal_world[s][0] = r->pal_world[s][5] = r->pal_world[s][10] = r->pal_world[s][15] = 1.0f;
    r->pal_max   = 3;
    r->pal_dirty = 1;
    r->pal_all   = 1;
    r->pal_gen++;
    memset(r->light, 0, sizeof(r->light));
    for (s = 0; s < VGPU98_MAX_LIGHTS; s++) {
        r->light[s][0]  = VGPU98_LIGHT_DIRECTIONAL;
        r->light[s][1]  = 0x3f800000; /* diffuse r, g, b 1.0 */
        r->light[s][2]  = 0x3f800000;
        r->light[s][3]  = 0x3f800000;
        r->light[s][18] = 0x3f800000; /* direction z 1.0 */
    }
    r->light_mask = 0;
    memset(r->material, 0, sizeof(r->material));
    memset(r->plane, 0, sizeof(r->plane));
    r->proj_set = 0;
    r->ff_dirty = 1;

    /* 2.2: no program, the fixed function bound, every constant 0. */
    if (r->progs) {
        for (s = 1; s <= VGPU98_PROG_MAX_ID; s++) {
            free(r->progs[s].ins);
            memset(&r->progs[s], 0, sizeof(r->progs[s]));
        }
    }
    r->bind_v = 0;
    r->bind_p = 0;
    memset(r->vconst, 0, sizeof(r->vconst));
    memset(r->pconst, 0, sizeof(r->pconst));
    r->vconst_hi = 0;
    r->vconst_gen++;
}

static int
in_range(uint32_t v, uint32_t lo, uint32_t hi)
{
    return (v >= lo) && (v <= hi);
}

/* A texture stage argument: 2.0's selectors, and 2.1's TEMP with FIXED8. */
static int
arg_ok(const vgpu98_r3d_t *r, uint32_t v)
{
    const uint32_t sel = v & VGPU98_TA_SELECTMASK;

    if (v & ~0x3fu)
        return VGPU98_ERR_ARGUMENT;
    if (sel == VGPU98_TA_TEMP)
        return fixed8(r) ? 0 : VGPU98_ERR_PROFILE;
    return (sel > VGPU98_TA_SPECULAR) ? VGPU98_ERR_ARGUMENT : 0;
}

/* Command set 2.1's global states (spec/gpu-commands.md), with FIXED8. */
static int
set_state8(vgpu98_r3d_t *r, uint32_t id, uint32_t *v)
{
    switch (id) {
        case VGPU98_RS_POINTSIZE:
        case VGPU98_RS_POINTSIZE_MIN:
        case VGPU98_RS_POINTSIZE_MAX:
        case VGPU98_RS_POINTSCALE_A:
        case VGPU98_RS_POINTSCALE_B:
        case VGPU98_RS_POINTSCALE_C:
        case VGPU98_RS_TWEENFACTOR:
        case VGPU98_RS_MULTISAMPLEMASK:
        case VGPU98_RS_PATCHEDGESTYLE:
        case VGPU98_RS_PATCHSEGMENTS:
        case VGPU98_RS_POSITIONORDER:
        case VGPU98_RS_NORMALORDER:
            return 0; /* floats (reduced where they are used) and states with no effect */
        case VGPU98_RS_POINTSPRITEENABLE:
        case VGPU98_RS_POINTSCALEENABLE:
        case VGPU98_RS_MULTISAMPLEANTIALIAS:
        case VGPU98_RS_INDEXEDVERTEXBLENDENABLE:
            *v = *v ? 1 : 0;
            return 0;
        case VGPU98_RS_COLORWRITEENABLE:
            *v &= VGPU98_CWE_RED | VGPU98_CWE_GREEN | VGPU98_CWE_BLUE | VGPU98_CWE_ALPHA;
            return 0;
        case VGPU98_RS_BLENDOP:
            return in_range(*v, VGPU98_BLENDOP_ADD, VGPU98_BLENDOP_MAX) ? 0 : VGPU98_ERR_ARGUMENT;
        default:
            (void) r;
            return VGPU98_ERR_ARGUMENT; /* 153, 165, 169: no such state */
    }
}

/* One STATE pair: validates enumerated values and reduces numeric ones
   (spec/gpu-commands.md, "Render states"). */
static int
set_state(vgpu98_r3d_t *r, uint32_t id, uint32_t v)
{
    if ((id >= VGPU98_RS_GLOBAL_COUNT) && (id < VGPU98_RS_GLOBAL_COUNT8)) {
        int err;

        /* 153, 165 and 169 are no states of 2.1 either: refused as 2.0
           refuses any id it lacks. */
        if ((id == VGPU98_RS_GLOBAL_COUNT) || (id == 165) || (id == 169))
            return VGPU98_ERR_ARGUMENT;
        if (!fixed8(r))
            return VGPU98_ERR_PROFILE;
        if ((err = set_state8(r, id, &v)))
            return err;
        r->rs[id] = v;
        return 0;
    }
    if (id < VGPU98_RS_GLOBAL_COUNT) {
        switch (id) {
            case VGPU98_RS_TEXTUREPERSPECTIVE:
            case VGPU98_RS_ZWRITEENABLE:
            case VGPU98_RS_ALPHATESTENABLE:
            case VGPU98_RS_LASTPIXEL:
            case VGPU98_RS_DITHERENABLE:
            case VGPU98_RS_ALPHABLENDENABLE:
            case VGPU98_RS_FOGENABLE:
            case VGPU98_RS_SPECULARENABLE:
            case VGPU98_RS_COLORKEYENABLE:
            case VGPU98_RS_STENCILENABLE:
            case VGPU98_RS_RANGEFOGENABLE:
            case VGPU98_RS_CLIPPING:
            case VGPU98_RS_LIGHTING:
            case VGPU98_RS_EXTENTS:
            case VGPU98_RS_COLORVERTEX:
            case VGPU98_RS_LOCALVIEWER:
            case VGPU98_RS_NORMALIZENORMALS:
            case VGPU98_RS_COLORKEYBLENDENABLE:
                v = v ? 1 : 0;
                break;
            case VGPU98_RS_FOGVERTEXMODE:
                if (v > 3)
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_RS_VERTEXBLEND:
                /* 2.1 adds tweening and one indexed matrix. */
                if ((v == VGPU98_VBF_TWEENING) || (v == VGPU98_VBF_0WEIGHTS)) {
                    if (!fixed8(r))
                        return VGPU98_ERR_PROFILE;
                } else if (v > 3)
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_RS_DIFFUSEMATERIALSOURCE:
            case VGPU98_RS_SPECULARMATERIALSOURCE:
            case VGPU98_RS_AMBIENTMATERIALSOURCE:
            case VGPU98_RS_EMISSIVEMATERIALSOURCE:
                if (v > VGPU98_MCS_COLOR2)
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_RS_CLIPPLANEENABLE:
                v &= (1u << VGPU98_MAX_CLIPPLANES) - 1;
                break;
            case VGPU98_RS_AMBIENT:
                break;
            case VGPU98_RS_ZENABLE:
                if (v > 2)
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_RS_FILLMODE:
            case VGPU98_RS_SHADEMODE:
            case VGPU98_RS_CULLMODE:
                if (!in_range(v, 1, 3))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_RS_SRCBLEND:
            case VGPU98_RS_DESTBLEND:
                if (!in_range(v, 1, 13))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_RS_ZFUNC:
            case VGPU98_RS_ALPHAFUNC:
            case VGPU98_RS_STENCILFAIL:
            case VGPU98_RS_STENCILZFAIL:
            case VGPU98_RS_STENCILPASS:
            case VGPU98_RS_STENCILFUNC:
                if (!in_range(v, 1, 8))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_RS_ALPHAREF:
            case VGPU98_RS_STENCILREF:
                v &= 0xff;
                break;
            case VGPU98_RS_FOGTABLEMODE:
                if (v > 3)
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_RS_ZBIAS:
                if (v > 16)
                    v = 16;
                break;
            case VGPU98_RS_FOGCOLOR:
            case VGPU98_RS_FOGTABLESTART:
            case VGPU98_RS_FOGTABLEEND:
            case VGPU98_RS_FOGTABLEDENSITY:
            case VGPU98_RS_STENCILMASK:
            case VGPU98_RS_STENCILWRITEMASK:
            case VGPU98_RS_TEXTUREFACTOR:
                break;
            default:
                if ((id >= VGPU98_RS_WRAP0) && (id < VGPU98_RS_WRAP0 + 8)) {
                    v &= 15; /* WRAPCOORD_0-3 */
                    break;
                }
                return VGPU98_ERR_ARGUMENT;
        }
        r->rs[id] = v;
        return 0;
    }
    if ((id >= VGPU98_TSS_FIRST) && (id <= VGPU98_TSS_LAST)) {
        uint32_t s = (id - VGPU98_TSS_FIRST) / 0x20;
        uint32_t n = (id - VGPU98_TSS_FIRST) % 0x20;

        int err;

        /* 2.1's stage states and operations need FIXED8. */
        if ((n >= VGPU98_TSS_COUNT) && (n < VGPU98_TSS_COUNT8) && !fixed8(r))
            return VGPU98_ERR_PROFILE;
        if (((n == VGPU98_TSS_COLOROP) || (n == VGPU98_TSS_ALPHAOP)) && in_range(v, VGPU98_TOP_LAST + 1, VGPU98_TOP_LAST8) &&
            !fixed8(r))
            return VGPU98_ERR_PROFILE;
        switch (n) {
            case VGPU98_TSS_COLOROP:
                if (!in_range(v, 1, VGPU98_TOP_LAST8))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_TSS_ALPHAOP:
                if (!in_range(v, 1, VGPU98_TOP_LAST8) || in_range(v, 18, 21) || (v == VGPU98_TOP_DOTPRODUCT3))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_TSS_COLORARG1:
            case VGPU98_TSS_COLORARG2:
            case VGPU98_TSS_ALPHAARG1:
            case VGPU98_TSS_ALPHAARG2:
            case VGPU98_TSS_COLORARG0:
            case VGPU98_TSS_ALPHAARG0:
                if ((err = arg_ok(r, v)))
                    return err;
                break;
            case VGPU98_TSS_RESULTARG:
                if ((v != VGPU98_TA_CURRENT) && (v != VGPU98_TA_TEMP))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_TSS_ADDRESSW:
                if (!in_range(v, 1, 4))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_TSS_BUMPENVMAT00:
            case VGPU98_TSS_BUMPENVMAT01:
            case VGPU98_TSS_BUMPENVMAT10:
            case VGPU98_TSS_BUMPENVMAT11:
            case VGPU98_TSS_BORDERCOLOR:
            case VGPU98_TSS_MIPMAPLODBIAS:
            case VGPU98_TSS_BUMPENVLSCALE:
            case VGPU98_TSS_BUMPENVLOFFSET:
            case VGPU98_TSS_MAXMIPLEVEL:
                break;
            case VGPU98_TSS_TEXCOORDINDEX:
                v &= VGPU98_TCI_SET_MASK | VGPU98_TCI_GEN_MASK;
                break;
            case VGPU98_TSS_ADDRESSU:
            case VGPU98_TSS_ADDRESSV:
                if (!in_range(v, 1, 4))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_TSS_MAGFILTER:
                if ((v != 1) && (v != 2) && (v != 5))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_TSS_MINFILTER:
            case VGPU98_TSS_MIPFILTER:
                if (!in_range(v, 1, 3))
                    return VGPU98_ERR_ARGUMENT;
                break;
            case VGPU98_TSS_MAXANISOTROPY:
                if (v < 1)
                    v = 1;
                if (v > 16)
                    v = 16;
                break;
            case VGPU98_TSS_TEXTURETRANSFORMFLAGS:
                v &= VGPU98_TTFF_COUNT_MASK | VGPU98_TTFF_PROJECTED;
                if ((v & VGPU98_TTFF_COUNT_MASK) > 4)
                    v = (v & VGPU98_TTFF_PROJECTED) | 4;
                break;
            default:
                return VGPU98_ERR_ARGUMENT;
        }
        r->tss[s][n] = v;
        return 0;
    }
    return VGPU98_ERR_ARGUMENT;
}

/* --- Packets ----------------------------------------------------------------- */

static int
pkt_target(vgpu98_r3d_t *r, const uint32_t *p)
{
    vgpu98_surf_t c;
    vgpu98_surf_t d;
    int           err;

    if ((err = vgpu98_surf_parse(r->vram, p, &c)))
        return err;
    /* 2.1 adds A1R5G5B5 targets. */
    if ((c.format == VGPU98_FMT_A1R5G5B5) && !fixed8(r))
        return VGPU98_ERR_PROFILE;
    if ((c.format != VGPU98_FMT_R5G6B5) && (c.format != VGPU98_FMT_X1R5G5B5) && (c.format != VGPU98_FMT_X8R8G8B8) &&
        (c.format != VGPU98_FMT_A8R8G8B8) && (c.format != VGPU98_FMT_A1R5G5B5))
        return VGPU98_ERR_ARGUMENT;
    if (((p[5] >> 16) & 0xff) == VGPU98_FMT_NONE)
        r->have_depth = 0;
    else {
        if ((err = vgpu98_surf_parse(r->vram, p + 3, &d)))
            return err;
        if (!vgpu98_fmt_is_depth(d.format) || (d.width < c.width) || (d.height < c.height))
            return VGPU98_ERR_ARGUMENT;
        r->depth      = d;
        r->have_depth = 1;
    }
    r->target      = c;
    r->have_target = 1;
    return 0;
}

/* TEXTURE of a volume (2.1): 1-256 texels each way, 1-9 levels (no more
   than down to 1 x 1 x 1), each with its offset, pitch and slice pitch;
   any 2D texture format but P8 and DXT. */
static int
tex_volume(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len, uint32_t stage, uint32_t format, uint32_t levels)
{
    tex_t    t;
    uint32_t big;
    uint32_t full = 1;
    uint32_t i;
    int      err;

    if ((format == VGPU98_FMT_P8) || !vgpu98_fmt_bpp(format) || vgpu98_fmt_is_depth(format) ||
        !in_range(levels, 1, VGPU98_TEX_VOLUME_LEVELS))
        return VGPU98_ERR_ARGUMENT;
    if (len < 4 + 3 * levels)
        return VGPU98_ERR_LENGTH;
    memset(&t, 0, sizeof(t));
    t.format   = format;
    t.control  = p[0];
    t.width    = p[1] & 0xffff;
    t.height   = p[1] >> 16;
    t.depth    = p[2] >> VGPU98_TEX_DEPTH_SHIFT;
    t.levels   = levels;
    t.faces    = 1;
    t.colorkey = p[3];
    if (!in_range(t.width, 1, VGPU98_TEX_VOLUME_MAX) || !in_range(t.height, 1, VGPU98_TEX_VOLUME_MAX) ||
        !in_range(t.depth, 1, VGPU98_TEX_VOLUME_MAX))
        return VGPU98_ERR_ARGUMENT;
    big = (t.width > t.height) ? t.width : t.height;
    if (t.depth > big)
        big = t.depth;
    while (big >>= 1)
        full++;
    if (levels > full)
        return VGPU98_ERR_ARGUMENT;
    for (i = 0; i < levels; i++) {
        const uint32_t *lp = p + 4 + 3 * i;
        const uint32_t  w  = max1(t.width >> i);
        const uint32_t  h  = max1(t.height >> i);
        const uint32_t  d  = max1(t.depth >> i);
        uint32_t        dd[3];
        vgpu98_surf_t   s;

        dd[0] = lp[0];
        dd[1] = VGPU98_SURF_SIZE(w, h);
        dd[2] = VGPU98_SURF_PITCH(lp[1] & 0xffff, format);
        if (lp[1] >> 16)
            return VGPU98_ERR_ARGUMENT;
        if ((err = vgpu98_surf_parse(r->vram, dd, &s)))
            return err;
        /* The slices: each at least a level's rows apart, the last within
           video memory. */
        if (((uint64_t) lp[2] < (uint64_t) s.pitch * h) ||
            ((uint64_t) s.offset + (uint64_t) lp[2] * (d - 1) + s.bytes > r->vram->size))
            return VGPU98_ERR_ARGUMENT;
        t.offset[i] = s.offset;
        t.pitch[i]  = s.pitch;
        t.slice[i]  = lp[2];
    }
    r->tex[stage] = t;
    return 0;
}

static int
pkt_texture(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len)
{
    const uint32_t stage  = p[0] & VGPU98_TEX_STAGE_MASK;
    const uint32_t type   = (p[0] & VGPU98_TEX_TYPE_MASK) >> VGPU98_TEX_TYPE_SHIFT;
    const uint32_t format = p[2] & 0xff;
    const uint32_t levels = (p[2] >> 8) & 0xff;
    const uint32_t faces  = (type == VGPU98_TEX_TYPE_CUBE) ? VGPU98_CUBE_FACES : 1;
    tex_t          t;
    uint32_t       f;
    uint32_t       i;

    if ((p[0] & ~(VGPU98_TEX_STAGE_MASK | VGPU98_TEX_COLORKEY | VGPU98_TEX_PALETTE_ALPHA | VGPU98_TEX_TYPE_MASK)) ||
        (type > VGPU98_TEX_TYPE_VOLUME) || ((type != VGPU98_TEX_TYPE_VOLUME) && (p[2] >> 16)))
        return VGPU98_ERR_ARGUMENT;
    if ((type == VGPU98_TEX_TYPE_VOLUME) && !fixed8(r))
        return VGPU98_ERR_PROFILE;
    if (format == VGPU98_FMT_NONE) {
        memset(&r->tex[stage], 0, sizeof(tex_t));
        return 0;
    }
    if (vgpu98_fmt_programs(format) && !(vgpu98_gpu_profile(r->gpu) & VGPU98_PROFILE_PROGRAMS))
        return VGPU98_ERR_PROFILE; /* command set 2.2's formats */
    if (type == VGPU98_TEX_TYPE_VOLUME)
        return tex_volume(r, p, len, stage, format, levels);
    if ((!vgpu98_fmt_bpp(format) && !vgpu98_fmt_is_dxt(format)) || vgpu98_fmt_is_depth(format))
        return VGPU98_ERR_ARGUMENT;
    if (vgpu98_fmt_fixed8(format) && !fixed8(r))
        return VGPU98_ERR_PROFILE;
    if (!in_range(levels, 1, MAX_LEVELS))
        return VGPU98_ERR_ARGUMENT;
    if (len < 4 + 2 * levels * faces)
        return VGPU98_ERR_LENGTH;
    memset(&t, 0, sizeof(t));
    t.format   = format;
    t.control  = p[0];
    t.width    = p[1] & 0xffff;
    t.height   = p[1] >> 16;
    t.levels   = levels;
    t.faces    = faces;
    t.colorkey = p[3];
    if (!in_range(t.width, 1, VGPU98_TEX_MAX_SIZE) || !in_range(t.height, 1, VGPU98_TEX_MAX_SIZE))
        return VGPU98_ERR_ARGUMENT;
    if ((faces == VGPU98_CUBE_FACES) && (t.width != t.height))
        return VGPU98_ERR_ARGUMENT;
    /* Level dwords face by face, levels from 0 within each. */
    for (f = 0; f < faces; f++) {
        for (i = 0; i < levels; i++) {
            const uint32_t *lp = p + 4 + 2 * (f * levels + i);
            uint32_t        d[3];
            vgpu98_surf_t   s;
            uint32_t        w = max1(t.width >> i);
            uint32_t        h = max1(t.height >> i);
            int             err;

            d[0] = lp[0];
            d[1] = VGPU98_SURF_SIZE(w, h);
            d[2] = VGPU98_SURF_PITCH(lp[1] & 0xffff, format);
            if (lp[1] >> 16)
                return VGPU98_ERR_ARGUMENT;
            if ((err = vgpu98_surf_parse(r->vram, d, &s)))
                return err; /* ERROR_CODE 5 for a 2.1 format without FIXED8 */
            t.offset[f * MAX_LEVELS + i] = s.offset;
            t.pitch[f * MAX_LEVELS + i]  = s.pitch;
        }
    }
    r->tex[stage] = t;
    return 0;
}

/* Is the target's format without alpha (X1R5G5B5, X8R8G8B8)? */
static int
target_no_alpha(const vgpu98_r3d_t *r)
{
    return (r->target.format == VGPU98_FMT_X1R5G5B5) || (r->target.format == VGPU98_FMT_X8R8G8B8) ||
           (r->target.format == VGPU98_FMT_R5G6B5);
}

static VkBlendFactor
blend_factor(uint32_t f, int no_dst_alpha)
{
    switch (f) {
        case 1:
            return VK_BLEND_FACTOR_ZERO;
        case 2:
            return VK_BLEND_FACTOR_ONE;
        case 3:
            return VK_BLEND_FACTOR_SRC_COLOR;
        case 4:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5:
            return VK_BLEND_FACTOR_SRC_ALPHA;
        case 6:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 7: /* destination alpha reads 1 on a target without alpha */
            return no_dst_alpha ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_DST_ALPHA;
        case 8:
            return no_dst_alpha ? VK_BLEND_FACTOR_ZERO : VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9:
            return VK_BLEND_FACTOR_DST_COLOR;
        case 10:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 11:
            return no_dst_alpha ? VK_BLEND_FACTOR_ZERO : VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        default:
            return VK_BLEND_FACTOR_ONE;
    }
}

/* 2.1's BLENDOP: add, subtract (source - destination), reverse subtract,
   min, max; add on a 2.0 GPU, whose BLENDOP stays at its default. */
static VkBlendOp
blend_op(uint32_t op)
{
    switch (op) {
        case 2:
            return VK_BLEND_OP_SUBTRACT;
        case 3:
            return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 4:
            return VK_BLEND_OP_MIN;
        case 5:
            return VK_BLEND_OP_MAX;
        default:
            return VK_BLEND_OP_ADD;
    }
}

static VkStencilOp
stencil_op(uint32_t op)
{
    static const VkStencilOp ops[9] = { VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_ZERO,
                                        VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                                        VK_STENCIL_OP_DECREMENT_AND_CLAMP, VK_STENCIL_OP_INVERT,
                                        VK_STENCIL_OP_INCREMENT_AND_WRAP, VK_STENCIL_OP_DECREMENT_AND_WRAP };

    return ops[(op <= 8) ? op : 1];
}

static VkCompareOp
compare_op(uint32_t f)
{
    static const VkCompareOp ops[9] = { VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_NEVER, VK_COMPARE_OP_LESS,
                                        VK_COMPARE_OP_EQUAL, VK_COMPARE_OP_LESS_OR_EQUAL, VK_COMPARE_OP_GREATER,
                                        VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                        VK_COMPARE_OP_ALWAYS };

    return ops[(f <= 8) ? f : 8];
}

/* The stages the pixel pipeline will run: up to the first DISABLE. */
static int
active_stages(const vgpu98_r3d_t *r)
{
    int s;

    for (s = 0; s < N_STAGES; s++) {
        if (r->tss[s][VGPU98_TSS_COLOROP] == VGPU98_TOP_DISABLE)
            break;
    }
    return s;
}

static void
fill_ubo(vgpu98_r3d_t *r, r3d_ubo_t *u, const uint32_t *stage_flags, const r3d_sub_t *depth)
{
    const r3d_prog_t *pp = r->bind_p ? &r->progs[r->bind_p] : NULL;
    const int   stages = active_stages(r);
    const int   zmode  = depth ? (int) r->rs[VGPU98_RS_ZENABLE] : 0;
    const float span   = r->wfar - r->wnear;
    int         s;
    uint32_t    mf     = 0;

    memset(u, 0, sizeof(*u));
    for (s = 0; s < N_STAGES; s++) {
        const uint32_t *t = r->tss[s];
        float           bias = u2f(t[VGPU98_TSS_MIPMAPLODBIAS]);

        u->st[s].ops[0]  = (s < stages) ? t[VGPU98_TSS_COLOROP] : VGPU98_TOP_DISABLE;
        u->st[s].ops[1]  = t[VGPU98_TSS_COLORARG1];
        u->st[s].ops[2]  = t[VGPU98_TSS_COLORARG2];
        u->st[s].ops[3]  = t[VGPU98_TSS_ALPHAOP];
        u->st[s].ops2[0] = t[VGPU98_TSS_ALPHAARG1];
        u->st[s].ops2[1] = t[VGPU98_TSS_ALPHAARG2];
        u->st[s].ops2[2] = t[VGPU98_TSS_TEXTURETRANSFORMFLAGS];
        u->st[s].ops2[3] = stage_flags[s];
        u->st[s].bump[0] = u2f(t[VGPU98_TSS_BUMPENVMAT00]);
        u->st[s].bump[1] = u2f(t[VGPU98_TSS_BUMPENVMAT01]);
        u->st[s].bump[2] = u2f(t[VGPU98_TSS_BUMPENVMAT10]);
        u->st[s].bump[3] = u2f(t[VGPU98_TSS_BUMPENVMAT11]);
        u->st[s].misc[0] = u2f(t[VGPU98_TSS_BUMPENVLSCALE]);
        u->st[s].misc[1] = u2f(t[VGPU98_TSS_BUMPENVLOFFSET]);
        /* MIPMAPLODBIAS as a scale of the coordinates' gradients. */
        u->st[s].misc[2] = (isfinite(bias) && (bias > -16.0f) && (bias < 16.0f)) ? exp2f(bias) : 1.0f;
        u->st[s].border[0] = ((t[VGPU98_TSS_BORDERCOLOR] >> 16) & 255) / 255.0f;
        u->st[s].border[1] = ((t[VGPU98_TSS_BORDERCOLOR] >> 8) & 255) / 255.0f;
        u->st[s].border[2] = (t[VGPU98_TSS_BORDERCOLOR] & 255) / 255.0f;
        u->st[s].border[3] = ((t[VGPU98_TSS_BORDERCOLOR] >> 24) & 255) / 255.0f;
        /* 2.1: MULTIPLYADD's and LERP's third arguments, where the result
           goes (at their defaults on a 2.0 GPU). */
        u->st[s].ops3[0] = t[VGPU98_TSS_COLORARG0];
        u->st[s].ops3[1] = t[VGPU98_TSS_ALPHAARG0];
        u->st[s].ops3[2] = t[VGPU98_TSS_RESULTARG];
    }
    u->tfactor[0]  = ((r->rs[VGPU98_RS_TEXTUREFACTOR] >> 16) & 255) / 255.0f;
    u->tfactor[1]  = ((r->rs[VGPU98_RS_TEXTUREFACTOR] >> 8) & 255) / 255.0f;
    u->tfactor[2]  = (r->rs[VGPU98_RS_TEXTUREFACTOR] & 255) / 255.0f;
    u->tfactor[3]  = ((r->rs[VGPU98_RS_TEXTUREFACTOR] >> 24) & 255) / 255.0f;
    u->fogcolor[0] = ((r->rs[VGPU98_RS_FOGCOLOR] >> 16) & 255) / 255.0f;
    u->fogcolor[1] = ((r->rs[VGPU98_RS_FOGCOLOR] >> 8) & 255) / 255.0f;
    u->fogcolor[2] = (r->rs[VGPU98_RS_FOGCOLOR] & 255) / 255.0f;
    u->fog[0]      = u2f(r->rs[VGPU98_RS_FOGTABLESTART]);
    u->fog[1]      = u2f(r->rs[VGPU98_RS_FOGTABLEEND]);
    u->fog[2]      = u2f(r->rs[VGPU98_RS_FOGTABLEDENSITY]);
    /* With a vertex program (2.2) fog is its oFog's, whatever FOGTABLEMODE
       says: no table fog then. */
    u->mode[0]     = (r->bind_v ? 0u : (r->rs[VGPU98_RS_FOGTABLEMODE] & 3)) | (r->rs[VGPU98_RS_FOGENABLE] ? 0x10u : 0) |
                 (((r->wnear != 1.0f) || (r->wfar != 1.0f)) ? 0x20u : 0);
    u->mode[1]     = r->rs[VGPU98_RS_ALPHATESTENABLE] ? ((r->rs[VGPU98_RS_ALPHAREF] & 255) | (r->rs[VGPU98_RS_ALPHAFUNC] << 8)) : 0;
    if (r->rs[VGPU98_RS_SPECULARENABLE])
        mf |= MF_SPECULAR;
    if (r->rs[VGPU98_RS_COLORKEYENABLE])
        mf |= MF_COLORKEY;
    if (r->rs[VGPU98_RS_SHADEMODE] == 1)
        mf |= MF_FLAT;
    if (r->rs[VGPU98_RS_DITHERENABLE] && (vgpu98_fmt_bpp(r->target.format) == 2))
        mf |= MF_DITHER;
    if (!r->rs[VGPU98_RS_TEXTUREPERSPECTIVE])
        mf |= MF_LINEAR; /* transformed vertices only: vgpu98.frag */
    u->mode[2] = mf;
    /* The depth (vgpu98.frag): ZENABLE; 4, Z on the grid of a float image;
       8, W itself in a float image; ZBIAS; the guest buffer's bits. */
    u->mode[3] = (uint32_t) zmode | ((depth && depth_grid(r, depth->im)) ? 4u : 0) |
                 ((depth && depth_float(depth->im->hf)) ? 8u : 0) | (r->rs[VGPU98_RS_ZBIAS] << 8) |
                 ((uint32_t) depth_bits(depth ? depth->im->format : VGPU98_FMT_D24X8) << 16);
    /* wnear; the span W's ZBIAS steps divide (0: relative to W, for an
       infinite or empty range); wnear and the reference's normalization,
       for a fixed-point image. */
    u->wrange[0] = r->wnear;
    u->wrange[1] = (isfinite(span) && (span != 0.0f)) ? fabsf(span) : 0.0f;
    u->wrange[2] = r->wnear;
    u->wrange[3] = w_norm(r);
    if (r->target.format == VGPU98_FMT_R5G6B5) {
        u->dither[0] = 31.0f, u->dither[1] = 63.0f, u->dither[2] = 31.0f;
    } else {
        u->dither[0] = u->dither[1] = u->dither[2] = 31.0f;
    }
    u->zrange[0] = r->zrange[0];
    u->zrange[1] = r->zrange[1] - r->zrange[0];
    /* 2.2: the pixel program's header and the pixel constants. */
    if (pp) {
        u->pprog[0]  = pp->n;
        u->pprog[1]  = pp->info.stages;
        u->pprog[2]  = pp->info.flags;
        u->prange[0] = u2f(pp->range);
        memcpy(u->pconst, r->pconst, sizeof(u->pconst));
    }
}

/* The scissor: the VIEWPORT rectangle within the target. 0 if empty. */
static int
scissor(const vgpu98_r3d_t *r, uint32_t w, uint32_t h, VkRect2D *sc)
{
    uint32_t x0 = r->vp[0], y0 = r->vp[1];
    uint32_t x1 = r->vp[0] + r->vp[2], y1 = r->vp[1] + r->vp[3];

    if (x1 > w)
        x1 = w;
    if (y1 > h)
        y1 = h;
    if ((x0 >= x1) || (y0 >= y1))
        return 0;
    sc->offset.x      = (int32_t) x0;
    sc->offset.y      = (int32_t) y0;
    sc->extent.width  = x1 - x0;
    sc->extent.height = y1 - y0;
    return 1;
}

/* The level or face a surface drawn into is: of the image that has it, or
   of a new image of its own. */
static r3d_sub_t *
surface_sub(vgpu98_r3d_t *r, const vgpu98_surf_t *sf)
{
    r3d_sub_t   *s = sub_find(r, sf->format, sf->width, sf->height, sf->offset, sf->pitch, NULL);
    r3d_image_t *im;

    if (s)
        return s;
    im = image_create(r, sf->format, sf->width, sf->height, 1, 1, &sf->offset, &sf->pitch);
    return im ? &im->sub[0] : NULL;
}

/* The depth buffer's level: of the image that has it, or of a new one.
   A packet that W-buffers (`zuse` 2) into a depth surface makes it a
   surface W-buffered into (depth_rec_set): its images are made in a 32-bit
   float format, where W keeps its precision whatever the W range, and one
   in a fixed-point format moves there: its GPU-newer pixels go back to
   video memory (unless the packet overwrites the whole surface), it goes,
   and the float image takes its place from video memory, whose values it
   takes as W. Without the float format the image stays as it is. */
static r3d_sub_t *
depth_sub(vgpu98_r3d_t *r, int zuse, int overwrite)
{
    const vgpu98_surf_t *sf = &r->depth;
    r3d_sub_t           *s  = sub_find(r, sf->format, sf->width, sf->height, sf->offset, sf->pitch, NULL);
    r3d_image_t         *im;

    if (s) {
        /* W-buffered into: its record is used now. An image already float
           (a 24-bit one on a host without D24_UNORM_S8_UINT, D16 at full
           precision) without one is recorded now, as a move records it. */
        if ((zuse == 2) && (s->im->n_sub == 1)) {
            depth_rec_t *rec = depth_rec_find(r, sf->format, sf->width, sf->height, sf->offset, sf->pitch);

            if (rec)
                rec->tick = r->tick;
            else if (depth_float(s->im->hf))
                depth_rec_set(r, sf->format, sf->width, sf->height, sf->offset, sf->pitch, 1);
        }
        if ((zuse != 2) || depth_float(s->im->hf) || !r->has_wfmt[sf->format] || (s->im->n_sub != 1))
            return s;
        im = s->im;
        if (im->dirty && !overwrite)
            copy_back(r, &im, 1, 0, 0xffffffffu, 0, 0, VGPU98_CB_IMAGE);
        if (r->broken)
            return NULL;
        image_remove_ptr(r, im);
        r->stats->depth_moves++;
    }
    if ((zuse == 2) && r->has_wfmt[sf->format])
        depth_rec_set(r, sf->format, sf->width, sf->height, sf->offset, sf->pitch, 1);
    im = image_create(r, sf->format, sf->width, sf->height, 1, 1, &sf->offset, &sf->pitch);
    return im ? &im->sub[0] : NULL;
}

/* Prepares every active stage's texture, the target and the depth buffer,
   ending with the render pass begun. Textures first: a texture made now
   takes over the image of a target that is one of its levels or faces.
   `zuse`: the ZENABLE the packet uses the depth buffer with (2 W-buffers
   into it: depth_sub). */
static int
prepare_draw(vgpu98_r3d_t *r, r3d_sub_t **color, r3d_sub_t **depth, r3d_image_t **tex, uint32_t *flags,
             int overwrite_color, int overwrite_depth, uint32_t stages, int zuse)
{
    const int audit = tex_audit_on(r);
    uint32_t  zero[256];
    int       s;

    r->drop_why = VGPU98_DROP_IMAGES;
    memset(zero, 0, sizeof(zero));
    r->tick++;
    r->ta_draw_bytes = 0; /* the stale check's budget a draw */
    for (s = 0; s < N_STAGES; s++) {
        const tex_t *t = &r->tex[s];

        tex[s]          = NULL;
        flags[s]        = 0;
        r->vol_bound[s] = NULL;
        if (!((stages >> s) & 1u) || !t->format)
            continue;
        if (t->depth) {
            /* A volume (2.1): its own 3D image, current before the pass. */
            r->vol_bound[s] = vol_prepare(r, t);
            if (!r->vol_bound[s])
                return 0;
            flags[s] = SF_BOUND | SF_VOLUME;
            continue;
        }
        tex[s] = image_find(r, t->format, t->width, t->height, t->levels, t->faces, t->offset, t->pitch);
        if (!tex[s]) {
            tex[s] = image_create(r, t->format, t->width, t->height, t->levels, t->faces, t->offset, t->pitch);
            if (tex[s] && audit) {
                churn_t *c = churn_rec(r, tex[s]);

                if (c)
                    c->made++;
            }
        }
        if (!tex[s])
            return 0;
        /* In use by this draw: no other stage's new image takes it over
           (image_merge), so every stage's image lives until the draw. */
        tex[s]->last_use = r->tick;
        tex[s]->tex_use  = r->tick;
        flags[s]         = SF_BOUND | ((t->faces == VGPU98_CUBE_FACES) ? SF_CUBE : 0);
    }

    /* Everything up to date before the pass begins (uploads cannot happen
       inside it). */
    for (s = 0; s < N_STAGES; s++) {
        const tex_t *t         = &r->tex[s];
        const int    pal_alpha = (t->control & VGPU98_TEX_PALETTE_ALPHA) != 0;

        if (!tex[s])
            continue;
        if (!image_prepare(r, tex[s], 0, r->pal[s], pal_alpha))
            return 0;
        mip_check_image(r, tex[s], r->pal[s], pal_alpha); /* with captures on, once a texture */
        if (audit) {
            tex_check(r, tex[s]);
            /* An 8-bit texture the stage reads drawn through a palette of
               zeros (the DLL sends zeros for a stage with no palette
               bound): once an upload. */
            if ((tex[s]->format == VGPU98_FMT_P8) && !tex[s]->a_zero_told && pal_zero(tex[s]->pal_sig) &&
                stage_reads_texture(r, s)) {
                char d[96];

                tex[s]->a_zero_told = 1;
                tex_text(d, sizeof(d), tex[s]);
                tex_find(r, VGPU98_FIND_PALETTE, tex[s]->id,
                         "stage %d's 8-bit texture (%s) is drawn through a palette of zeros: black%s", s, d,
                         pal_alpha ? " and transparent" : "");
            }
        }
        /* Colour keys (DXT textures have none): COLORKEYBLENDENABLE samples
           the texture with matching texels zeroed; otherwise COLORKEYENABLE
           removes the pixels they reach. */
        if ((t->control & VGPU98_TEX_COLORKEY) && !vgpu98_fmt_is_dxt(t->format) &&
            (r->rs[VGPU98_RS_COLORKEYBLENDENABLE] || r->rs[VGPU98_RS_COLORKEYENABLE])) {
            const int      blend = r->rs[VGPU98_RS_COLORKEYBLENDENABLE] != 0;
            const uint64_t t0    = vgpu98_diag_us();
            int            ok;

            /* The key images are made from video memory: a texture drawn
               into (which also made its key images stale) puts its pixels
               there first. */
            if (tex[s]->dirty)
                copy_back(r, &tex[s], 1, 0, 0xffffffffu, 0, 0, VGPU98_CB_IMAGE);
            r->timed++;
            ok = image_key(r, tex[s], t->colorkey, blend, r->pal[s], pal_alpha);
            r->timed--;
            if (!r->timed)
                r->stats->upload_us += us_since(t0);
            if (!ok)
                return 0;
            flags[s] |= blend ? SF_KEYZ : SF_KEYED;
        }
    }

    *color = surface_sub(r, &r->target);
    if (!*color)
        return 0;
    *depth = NULL;
    if (r->have_depth) {
        *depth = depth_sub(r, zuse, overwrite_depth);
        if (!*depth)
            return 0;
    }
    /* A packet that overwrites a whole surface needs no upload of it first,
       when the surface is an image of its own. */
    if (!image_prepare(r, (*color)->im, overwrite_color && ((*color)->im->n_sub == 1), zero, 0))
        return 0;
    if (*depth && !image_prepare(r, (*depth)->im, overwrite_depth && ((*depth)->im->n_sub == 1), zero, 0))
        return 0;
    /* A texture that is also the target: Direct3D leaves the result
       undefined, and so does Vulkan (no fault, only undefined texels). */
    r->drop_why = VGPU98_DROP_HOST;
    return pass_begin(r, *color, *depth);
}

/* Does a stage the pixel pipeline runs read a coordinate set with WRAPn on? */
static int
wrap_used(const vgpu98_r3d_t *r)
{
    /* A pixel program's coordinate sets (2.2), or the fixed function's
       stages; a vertex program's stage n reads its oTn, set n. */
    const uint32_t mask = r->bind_p ? r->progs[r->bind_p].info.coords : ((1u << active_stages(r)) - 1u);
    int            s;

    for (s = 0; s < N_STAGES; s++) {
        const uint32_t set = r->bind_v ? (uint32_t) s : (r->tss[s][VGPU98_TSS_TEXCOORDINDEX] & VGPU98_TCI_SET_MASK);

        if (((mask >> s) & 1u) && (r->rs[VGPU98_RS_WRAP0 + set] & 15))
            return 1;
    }
    return 0;
}

/* POINTSIZE_MIN or POINTSIZE_MAX as the GPU uses it (2.1): 0-256, a
   negative or NaN value 0. */
static float
point_limit(uint32_t bits)
{
    const float f = u2f(bits);

    return !(f > 0.0f) ? 0.0f : (f > (float) VGPU98_POINT_SIZE_MAX) ? (float) VGPU98_POINT_SIZE_MAX : f;
}

/* How far beyond its vertex a point's square can reach, in whole pixels:
   half the largest size it can have (the smallest wins over the largest). */
static int64_t
point_reach(const vgpu98_r3d_t *r)
{
    const float lo = point_limit(r->rs[VGPU98_RS_POINTSIZE_MIN]);
    const float hi = point_limit(r->rs[VGPU98_RS_POINTSIZE_MAX]);

    return (int64_t) ceilf(((lo > hi) ? lo : hi) / 2.0f) + 1;
}

/* Are points drawn as squares (2.1): sized (a per-vertex size, POINTSIZE
   other than 1, or clamped away from 1), scaled, or sprites? Points that are
   single pixels draw as in 2.0. */
static int
points_square(const vgpu98_r3d_t *r)
{
    const float size = u2f(r->rs[VGPU98_RS_POINTSIZE]);
    const float lo   = u2f(r->rs[VGPU98_RS_POINTSIZE_MIN]);
    const float hi   = u2f(r->rs[VGPU98_RS_POINTSIZE_MAX]);

    /* A vertex program's points (2.2): its oPts, never scaled. */
    if (r->vin.program == 2)
        return r->rs[VGPU98_RS_POINTSPRITEENABLE] ||
               (r->bind_v && (r->progs[r->bind_v].info.outputs & (1u << VGPU98_PO_PTS))) || (size != 1.0f) ||
               (lo > 1.0f) || !(hi >= 1.0f);
    return r->rs[VGPU98_RS_POINTSPRITEENABLE] || r->rs[VGPU98_RS_POINTSCALEENABLE] || r->vin.slot[VGPU98_SLOT_PSIZE].on ||
           (size != 1.0f) || (lo > 1.0f) || !(hi >= 1.0f);
}

/* What the draw's vertex shader invocations draw: whole primitives only
   where the state needs them (spec/gpu-commands.md, "How a primitive is
   drawn"). */
static int
draw_shape(const vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d)
{
    if (d->prim == VGPU98_PRIM_POINTS)
        return points_square(r) ? SHAPE_SPRITE : SHAPE_PLAIN;
    if (d->prim <= VGPU98_PRIM_LINE_STRIP)
        return SHAPE_LINE;
    switch (r->rs[VGPU98_RS_FILLMODE]) {
        case 1:
            return points_square(r) ? SHAPE_FILLSPRITE : SHAPE_POINTS;
        case 2:
            return SHAPE_WIRE;
        default:
            return wrap_used(r) ? SHAPE_WRAP : SHAPE_PLAIN;
    }
}

static VkPrimitiveTopology
draw_topology(int shape, uint32_t prim)
{
    if (shape == SHAPE_POINTS)
        return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    if (shape != SHAPE_PLAIN)
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    /* The host GPU's strips and fans assemble as Direct3D's: (i, i+1, i+2)
       and (i, i+2, i+1), (i+1, i+2, 0), the first vertex providing flat
       shading's colour. */
    switch (prim) {
        case VGPU98_PRIM_POINTS:
            return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case VGPU98_PRIM_TRIANGLE_STRIP:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case VGPU98_PRIM_TRIANGLE_FAN:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        default:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

static void
mat_identity(float *m)
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

/* --- Fixed-function transform and lighting ------------------------------------ */

/* res = a * b, Direct3D's row-major matrices, summed as the reference T&L
   sums them. */
static void
mat_mul(float *res, const float *a, const float *b)
{
    float t[16];
    int   i;
    int   j;

    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++)
            t[4 * i + j] = a[4 * i] * b[j] + a[4 * i + 1] * b[4 + j] + a[4 * i + 2] * b[8 + j] + a[4 * i + 3] * b[12 + j];
    }
    memcpy(res, t, sizeof(t));
}

/* The inverse by cofactors in double precision, as the reference T&L's
   Cramer's rule; 0 when the matrix has none (determinant 0). */
static int
mat_inverse(const float *m, float *inv)
{
    double a[16];
    double c[16];
    double det;
    int    i;

    for (i = 0; i < 16; i++)
        a[i] = m[i];
    c[0]  = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    c[4]  = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    c[8]  = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    c[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    c[1]  = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    c[5]  = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    c[9]  = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    c[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    c[2]  = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    c[6]  = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    c[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    c[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    c[3]  = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    c[7]  = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    c[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    c[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
    det = a[0] * c[0] + a[1] * c[4] + a[2] * c[8] + a[3] * c[12];
    if (det == 0.0)
        return 0;
    for (i = 0; i < 16; i++)
        inv[i] = (float) (c[i] / det);
    return 1;
}

static void
vec_normalize(float *v)
{
    const float l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

    if (l2 > 0.0f) {
        const float k = 1.0f / sqrtf(l2);

        v[0] *= k;
        v[1] *= k;
        v[2] *= k;
    }
}

/* The fixed-function part of the vertex uniforms that depends only on the
   matrices, lights, material and clip planes (recomputed when one of them
   changes): composite and normal matrices, lights in camera space. */
static void
ff_update(vgpu98_r3d_t *r)
{
    r3d_vubo_t  *u    = &r->ff;
    const float *view = r->xform[VGPU98_XFORM_VIEW];
    float        vp[16];
    int          i;
    int          j;

    mat_mul(vp, view, r->xform[VGPU98_XFORM_PROJECTION]);
    for (j = 0; j < 4; j++) {
        mat_mul(u->wvp[j], r->xform[VGPU98_XFORM_WORLD + j], vp);
        mat_mul(u->wv[j], r->xform[VGPU98_XFORM_WORLD + j], view);
        if (!mat_inverse(u->wv[j], u->wvinv[j]))
            mat_identity(u->wvinv[j]); /* spec/gpu-commands.md: a singular World * View */
        memcpy(u->world[j], r->xform[VGPU98_XFORM_WORLD + j], sizeof(u->world[j]));
    }
    for (i = 0; i < N_STAGES; i++)
        memcpy(u->tex[i], r->xform[VGPU98_XFORM_TEXTURE + i], sizeof(u->tex[i]));
    memcpy(u->plane, r->plane, sizeof(u->plane));

    for (i = 0; i < VGPU98_MAX_LIGHTS; i++) {
        const uint32_t  *l  = r->light[i];
        r3d_light_ubo_t *o  = &u->light[i];
        const float      px = u2f(l[13]), py = u2f(l[14]), pz = u2f(l[15]);
        const float      dx = u2f(l[16]), dy = u2f(l[17]), dz = u2f(l[18]);
        const float      range = u2f(l[19]);
        float            ct;
        float            cp;
        float            inv;

        for (j = 0; j < 4; j++) {
            o->diffuse[j]  = u2f(l[1 + j]);
            o->specular[j] = u2f(l[5 + j]);
            o->ambient[j]  = u2f(l[9 + j]);
        }
        /* The position by the view matrix (4x3), the direction by its 3x3,
           normalized and reversed: the direction to the light. */
        o->position[0]  = px * view[0] + py * view[4] + pz * view[8] + view[12];
        o->position[1]  = px * view[1] + py * view[5] + pz * view[9] + view[13];
        o->position[2]  = px * view[2] + py * view[6] + pz * view[10] + view[14];
        o->position[3]  = (float) l[0];
        o->direction[0] = dx * view[0] + dy * view[4] + dz * view[8];
        o->direction[1] = dx * view[1] + dy * view[5] + dz * view[9];
        o->direction[2] = dx * view[2] + dy * view[6] + dz * view[10];
        vec_normalize(o->direction);
        o->direction[0] = -o->direction[0];
        o->direction[1] = -o->direction[1];
        o->direction[2] = -o->direction[2];
        o->direction[3] = 0.0f;
        o->atten[0]     = u2f(l[21]);
        o->atten[1]     = u2f(l[22]);
        o->atten[2]     = u2f(l[23]);
        o->atten[3]     = range * range;
        ct              = (float) cos(u2f(l[24]) / 2.0);
        cp              = (float) cos(u2f(l[25]) / 2.0);
        inv             = ct - cp;
        o->spot[0]      = ct;
        o->spot[1]      = cp;
        o->spot[2]      = (inv != 0.0f) ? (1.0f / inv) : 1.0f;
        o->spot[3]      = u2f(l[20]);
    }

    for (j = 0; j < 3; j++) {
        u->mat_diffuse[j]  = r->material[j] * 255.0f;
        u->mat_ambient[j]  = r->material[4 + j] * 255.0f;
        u->mat_specular[j] = r->material[8 + j] * 255.0f;
        u->mat_emissive[j] = r->material[12 + j] * 255.0f;
    }
    {
        const float power = r->material[16];
        const float da    = r->material[3] * 255.0f;
        const float sa    = r->material[11] * 255.0f;

        u->material[0] = power;
        u->material[1] = (power > 0.001f) ? (float) pow(0.001, 1.0 / power) : 0.0f;
        u->material[2] = !(da > 0.0f) ? 0.0f : (da >= 255.0f) ? 255.0f : truncf(da);
        u->material[3] = !(sa > 0.0f) ? 0.0f : (sa >= 255.0f) ? 255.0f : truncf(sa);
    }
    r->ff_dirty = 0;
}

/* The world matrix palette's entries (2.1): for matrices 0 to the highest
   set since the reset, as ff_update makes world matrices 0-3's, then an
   identity world matrix's, which the vertex shader uses for an index past
   them. Those of the matrices that changed, or all of them after the view or
   the projection did. */
static void
pal_update(vgpu98_r3d_t *r)
{
    const float *view = r->xform[VGPU98_XFORM_VIEW];
    float        vp[16];
    float        id[16];
    uint32_t     k;

    mat_mul(vp, view, r->xform[VGPU98_XFORM_PROJECTION]);
    mat_identity(id);
    for (k = 0; k <= r->pal_max + 1; k++) {
        const float *w = (k <= r->pal_max) ? r->pal_world[k] : id;
        r3d_pal_t   *e = &r->pal_ff[k];

        if (!r->pal_all && !((r->pal_bits[k / 32] >> (k % 32)) & 1u))
            continue;
        mat_mul(e->wvp, w, vp);
        mat_mul(e->wv, w, view);
        if (!mat_inverse(e->wv, e->wvinv))
            mat_identity(e->wvinv);
        memcpy(e->world, w, sizeof(e->world));
    }
    memset(r->pal_bits, 0, sizeof(r->pal_bits));
    r->pal_dirty = 0;
    r->pal_all   = 0;
    r->pal_gen++;
}

/* Palette entry k changed (2.1). */
static void
pal_mark(vgpu98_r3d_t *r, uint32_t k)
{
    r->pal_bits[k / 32] |= 1u << (k % 32);
    r->pal_dirty = 1;
}

/* A colour source for lighting: the vertex's diffuse (1) or specular (2)
   colour when COLORVERTEX is on, the state asks for it and the vertex has
   it; otherwise the material's (0). */
static uint32_t
color_source(const vgpu98_r3d_t *r, uint32_t state)
{
    const uint32_t s = r->rs[state];

    if (!r->rs[VGPU98_RS_COLORVERTEX])
        return VGPU98_MCS_MATERIAL;
    if ((s == VGPU98_MCS_COLOR1) && r->vin.slot[VGPU98_SLOT_DIFFUSE].on)
        return VGPU98_MCS_COLOR1;
    if ((s == VGPU98_MCS_COLOR2) && r->vin.slot[VGPU98_SLOT_SPECULAR].on)
        return VGPU98_MCS_COLOR2;
    return VGPU98_MCS_MATERIAL;
}

static int
popcount8(uint32_t v)
{
    int n = 0;

    for (; v; v &= v - 1)
        n++;
    return n;
}

/* The fixed-function part of the vertex uniforms: the matrices, lights,
   material and planes (ff_update, cached), then what follows the state of
   the moment and the declaration. */
static void
fill_vubo_fixed(vgpu98_r3d_t *r, r3d_vubo_t *u)
{
    const uint32_t amb = r->rs[VGPU98_RS_AMBIENT];
    const float    fs  = u2f(r->rs[VGPU98_RS_FOGTABLESTART]);
    const float    fe  = u2f(r->rs[VGPU98_RS_FOGTABLEEND]);
    const int      lit = r->rs[VGPU98_RS_LIGHTING] != 0;
    uint32_t       flags = 0;
    uint32_t       blend;

    if (r->ff_dirty)
        ff_update(r);
    memcpy(u->wvp, r->ff.wvp, offsetof(r3d_vubo_t, ambient) - offsetof(r3d_vubo_t, wvp));
    u->ambient[0] = ((amb >> 16) & 255) / 255.0f;
    u->ambient[1] = ((amb >> 8) & 255) / 255.0f;
    u->ambient[2] = (amb & 255) / 255.0f;
    u->fog[0]     = fs;
    u->fog[1]     = fe;
    u->fog[2]     = u2f(r->rs[VGPU98_RS_FOGTABLEDENSITY]);
    u->fog[3]     = (fe == fs) ? 0.0f : 255.0f / (fe - fs);
    if (lit)
        flags |= TNL_LIGHTING;
    if (r->rs[VGPU98_RS_SPECULARENABLE])
        flags |= TNL_SPECULAR;
    if (r->rs[VGPU98_RS_LOCALVIEWER])
        flags |= TNL_LOCAL;
    if (r->rs[VGPU98_RS_NORMALIZENORMALS])
        flags |= TNL_NORMALIZE;
    if (r->rs[VGPU98_RS_RANGEFOGENABLE])
        flags |= TNL_RANGEFOG;
    /* The weights used: VERTEXBLEND, as many as the vertex has; 2.1's
       tweening and one indexed matrix use none. */
    blend = r->rs[VGPU98_RS_VERTEXBLEND];
    if ((blend == VGPU98_VBF_TWEENING) || (blend == VGPU98_VBF_0WEIGHTS))
        blend = 0;
    if (blend > r->vin.weights)
        blend = r->vin.weights;
    u->tnl[0] = lit ? (r->light_mask & 0xffu) : 0;
    u->tnl[1] = flags;
    u->tnl[2] = blend;
    /* Indexed blending through the palette (not with tweening, which uses
       world matrix 0 alone). */
    if (r->rs[VGPU98_RS_VERTEXBLEND] == VGPU98_VBF_TWEENING)
        u->dx8[0] |= DX8_TWEEN;
    else if (r->rs[VGPU98_RS_INDEXEDVERTEXBLENDENABLE] && (r->rs[VGPU98_RS_VERTEXBLEND] != 0)) {
        if (r->pal_dirty)
            pal_update(r);
        u->dx8[0] |= DX8_INDEXED;
        u->dx8[1] = r->pal_max + 2;
    }
    /* Vertex fog: with FOGENABLE, a FOGVERTEXMODE and no table fog. */
    u->tnl[3] = (r->rs[VGPU98_RS_FOGENABLE] && !r->rs[VGPU98_RS_FOGTABLEMODE]) ? r->rs[VGPU98_RS_FOGVERTEXMODE] : 0;
    u->sources[0] = color_source(r, VGPU98_RS_DIFFUSEMATERIALSOURCE);
    u->sources[1] = color_source(r, VGPU98_RS_SPECULARMATERIALSOURCE);
    u->sources[2] = color_source(r, VGPU98_RS_AMBIENTMATERIALSOURCE);
    u->sources[3] = color_source(r, VGPU98_RS_EMISSIVEMATERIALSOURCE);
}

/* The summary's transform line, for a draw through the fixed-function
   transform: what of it the state and the declaration use. */
static void
count_fixed(vgpu98_r3d_t *r)
{
    const int stages = active_stages(r);
    int       s;

    if (r->rs[VGPU98_RS_LIGHTING]) {
        r->stats->draws_lit++;
        r->stats->lights_used += (uint32_t) popcount8(r->light_mask);
    }
    if ((r->rs[VGPU98_RS_VERTEXBLEND] >= 1) && (r->rs[VGPU98_RS_VERTEXBLEND] <= 3) && r->vin.weights)
        r->stats->draws_blend++;
    if (r->rs[VGPU98_RS_CLIPPING] && (r->rs[VGPU98_RS_CLIPPLANEENABLE] & 0x3f))
        r->stats->draws_planes++;
    for (s = 0; s < stages; s++) {
        if (r->tss[s][VGPU98_TSS_TEXCOORDINDEX] & VGPU98_TCI_GEN_MASK) {
            r->stats->draws_texgen++;
            break;
        }
    }
    for (s = 0; s < stages; s++) {
        if (r->tss[s][VGPU98_TSS_TEXTURETRANSFORMFLAGS] & VGPU98_TTFF_COUNT_MASK) {
            r->stats->draws_texxform++;
            break;
        }
    }
}

/* The vertex shader's uniform block, but for the slots (vgpu98_vin_copy). */
static void
fill_vubo(vgpu98_r3d_t *r, r3d_vubo_t *u, const vgpu98_vin_draw_t *d, uint32_t fw, uint32_t fh)
{
    uint32_t mode = 0;
    int      i;

    memset(u, 0, sizeof(*u));
    u->range[1]  = (int32_t) d->nverts;
    u->fb[0]     = 2.0f / (float) fw;
    u->fb[1]     = 2.0f / (float) fh;
    u->fb[2]     = r->rs[VGPU98_RS_TEXTUREPERSPECTIVE] ? 1.0f : 0.0f;
    u->vp[0]     = (float) r->vp[2] / 2.0f;
    u->vp[1]     = (float) r->vp[3] / 2.0f;
    u->vp[2]     = (float) r->vp[0] + (float) r->vp[2] / 2.0f;
    u->vp[3]     = (float) r->vp[1] + (float) r->vp[3] / 2.0f;
    u->zrange[0] = r->zrange[0];
    u->zrange[1] = r->zrange[1] - r->zrange[0];
    if (r->rs[VGPU98_RS_LASTPIXEL])
        mode |= VF_LASTPIXEL;
    if (r->rs[VGPU98_RS_CLIPPING])
        mode |= VF_CLIPPING;
    u->mode[0] = mode;
    for (i = 0; i < 8; i++)
        u->mode[1] |= (r->rs[VGPU98_RS_WRAP0 + i] & 15u) << (4 * i);
    u->mode[2] = r->rs[VGPU98_RS_CULLMODE];
    u->mode[3] = r->rs[VGPU98_RS_CLIPPLANEENABLE] & 0x3fu;
    for (i = 0; i < N_STAGES; i++) {
        u->tci[i / 4][i % 4]  = r->tss[i][VGPU98_TSS_TEXCOORDINDEX];
        u->ttff[i / 4][i % 4] = r->tss[i][VGPU98_TSS_TEXTURETRANSFORMFLAGS];
    }
    /* Points (2.1): POINTSIZE, POINTSIZE_MIN, the largest size and the
       viewport's height; the scale's factors; negative and NaN values as
       0, the smallest and largest sizes at most 256. */
    {
        int k;

        u->pt[0] = u2f(r->rs[VGPU98_RS_POINTSIZE]);
        u->pt[1] = point_limit(r->rs[VGPU98_RS_POINTSIZE_MIN]);
        u->pt[2] = point_limit(r->rs[VGPU98_RS_POINTSIZE_MAX]);
        u->pt[3] = (float) r->vp[3];
        u->ptscale[0] = u2f(r->rs[VGPU98_RS_POINTSCALE_A]);
        u->ptscale[1] = u2f(r->rs[VGPU98_RS_POINTSCALE_B]);
        u->ptscale[2] = u2f(r->rs[VGPU98_RS_POINTSCALE_C]);
        u->ptscale[3] = u2f(r->rs[VGPU98_RS_TWEENFACTOR]);
        for (k = 0; k < 3; k++) {
            if (!(u->pt[k] > 0.0f))
                u->pt[k] = 0.0f;
            if (!(u->ptscale[k] > 0.0f))
                u->ptscale[k] = 0.0f;
        }
        if (r->rs[VGPU98_RS_POINTSCALEENABLE])
            u->dx8[0] |= DX8_SCALE;
        if (r->rs[VGPU98_RS_POINTSPRITEENABLE])
            u->dx8[0] |= DX8_SPRITE;
    }
    if (r->vin.program == 1)
        fill_vubo_fixed(r, u);
    /* 2.2: the vertex program's header and the user clip planes as CLIPPLANE
       set them (clip space; the fixed function's block, which holds them
       otherwise, is not filled); a pixel program's coordinate padding. */
    if (r->bind_v) {
        const r3d_prog_t *vp = &r->progs[r->bind_v];

        u->vprog[0] = vp->n;
        u->vprog[1] = vconst_count(r, vp);
        u->vprog[2] = vp->info.outputs;
        memcpy(u->plane, r->plane, sizeof(u->plane));
    }
    if (r->bind_p)
        u->vprog[3] |= VPF_PIXEL;
}

/* The screen rectangle a draw of pre-transformed vertices can touch: the
   bounds of its window's positions, a pixel more on each side (points and
   line bands reach beyond their vertices) and `reach` more (the squares of
   sized points), within the scissor. */
static void
screen_bounds(const vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d, int64_t reach, uint32_t *x0, uint32_t *y0, uint32_t *x1,
              uint32_t *y1)
{
    float   b[4];
    int64_t ix0, iy0, ix1, iy1;

    if (!vgpu98_vin_bounds(&r->vin, d, b)) {
        *x1 = *x0;
        return;
    }
    ix0 = (int64_t) floorf(b[0]) - 1 - reach;
    iy0 = (int64_t) floorf(b[1]) - 1 - reach;
    ix1 = (int64_t) ceilf(b[2]) + 2 + reach;
    iy1 = (int64_t) ceilf(b[3]) + 2 + reach;
    if (ix0 > (int64_t) *x0)
        *x0 = (uint32_t) ((ix0 < (int64_t) *x1) ? ix0 : *x1);
    if (iy0 > (int64_t) *y0)
        *y0 = (uint32_t) ((iy0 < (int64_t) *y1) ? iy0 : *y1);
    if (ix1 < (int64_t) *x1)
        *x1 = (uint32_t) ((ix1 > (int64_t) *x0) ? ix1 : *x0);
    if (iy1 < (int64_t) *y1)
        *y1 = (uint32_t) ((iy1 > (int64_t) *y0) ? iy1 : *y0);
}

/* A draw's host buffers, all in the current slot: a submission between two
   of them would leave the draw reading a buffer its slot no longer owns. */
enum { MEM_VERTS, MEM_LIST, MEM_VUBO, MEM_UBO, MEM_PAL, MEM_VPROG, MEM_PPROG, MEM_VCONST, N_MEM };

typedef struct draw_mem_t {
    int          which[N_MEM]; /* slot_alloc's buffer */
    VkDeviceSize size[N_MEM];
    VkDeviceSize align[N_MEM];
    void        *p[N_MEM];
    VkBuffer     buf[N_MEM];
    VkDeviceSize off[N_MEM];
} draw_mem_t;

static int
draw_alloc(vgpu98_r3d_t *r, draw_mem_t *m)
{
    int tries;

    /* One block in each of the slot's buffers (the vertex buffer's for the
       copy and the list, the uniform buffer's for both uniform blocks). At
       most: a buffer is full inside the pass (end it, so the next try may
       submit); the first block's allocation submits and lands in the next
       slot, and the second fits there; or the second submits instead, and
       the third try puts both in the fresh slot. */
    for (tries = 0; tries < 3; tries++) {
        uint64_t serial = 0;
        int      ok     = 1;
        int      which;

        for (which = 1; (which <= 2) && ok; which++) {
            VkDeviceSize rel[N_MEM] = { 0 };
            VkDeviceSize total = 0;
            VkDeviceSize align = 1;
            VkBuffer     buf;
            VkDeviceSize off;
            uint8_t     *p;
            int          i;

            for (i = 0; i < N_MEM; i++) {
                if (m->which[i] != which)
                    continue;
                total  = (total + m->align[i] - 1) / m->align[i] * m->align[i];
                rel[i] = total;
                total += m->size[i];
                if (m->align[i] > align)
                    align = m->align[i];
            }
            p = slot_alloc(r, which, total, align, &buf, &off);
            if (!p) {
                ok = 0;
                break;
            }
            if (which == 1)
                serial = r->serial; /* the slot the first block landed in */
            else if (r->serial != serial)
                ok = 0; /* the second moved on to another slot: both again */
            for (i = 0; i < N_MEM; i++) {
                if (m->which[i] != which)
                    continue;
                m->p[i]   = p + rel[i];
                m->buf[i] = buf;
                m->off[i] = off + rel[i];
            }
        }
        if (ok)
            return 1;
        if (r->broken)
            return 0;
        pass_end(r);
    }
    return 0;
}

static VkDeviceSize
align_max(VkDeviceSize a, VkDeviceSize b)
{
    return (a > b) ? a : b;
}

/* A draw whose copy lacked vertices of its window (past the end of a
   stream as the driver sized it, or past the host's limits): counted, and
   named the first three times, and again after 10 seconds without one. */
#define PAST_LOGGED   3
#define PAST_QUIET_MS 10000

static void
note_past(vgpu98_r3d_t *r, const uint32_t *p, const vgpu98_vin_draw_t *d)
{
    static const char *const prims[] = { "?", "points", "lines", "a line strip", "triangles", "a triangle strip", "a triangle fan" };
    const vgpu98_vin_t      *v  = &r->vin;
    const uint32_t           s  = v->past_stream;
    const uint32_t           ms = (uint32_t) (vgpu98_diag_us() / 1000);
    char                     line[600];
    char                     ix[200];

    uint32_t                 key = 0x811c9dc5u;
    uint32_t                 i;
    int                      fresh = 1;

    r->stats->draws_past++;
    r->stats->verts_past += v->past;
    if (r->past_logged && ((uint32_t) (ms - r->past_ms) >= PAST_QUIET_MS))
        r->past_logged = 0;
    r->past_ms = ms;
    /* A kind of draw not named before is named whatever the rate (64 kinds
       a session: a character's draw comes after the scene's first three). */
    key = (key ^ v->stream[s].address) * 16777619u;
    key = (key ^ p[2]) * 16777619u;
    key = (key ^ p[3]) * 16777619u;
    key = (key ^ v->past) * 16777619u;
    for (i = 0; i < r->n_past_seen; i++) {
        if (r->past_seen[i] == key)
            fresh = 0;
    }
    if (fresh && (r->n_past_seen < 64))
        r->past_seen[r->n_past_seen++] = key;
    else if (r->past_logged >= PAST_LOGGED)
        return;
    r->past_logged++;
    /* Longer than r3d_log's lines. */
    if (!vgpu98_gpu_logging(r->gpu, VGPU98_LOG_EVENTS))
        return;
    ix[0] = 0;
    if (d->indexed && d->local) {
        /* The draw's indices that reach the missing vertices (window
           numbers from the first the copy lacks), the highest of them as
           the index buffer holds it and where, and where the indices came
           from. */
        const uint64_t stride = v->stream[s].stride;
        const uint64_t extent = v->stream[s].extent;
        const uint64_t have   = (uint64_t) v->copy[s].words * 4;
        uint64_t       fit    = (have < extent) ? 0 : !stride ? d->nverts : (have - extent) / stride + 1;
        uint32_t       reach  = 0;
        uint32_t       top    = 0;
        uint32_t       top_at = 0;
        uint32_t       i;

        if (fit > d->nverts)
            fit = d->nverts;
        for (i = 0; i < d->count; i++) {
            const uint32_t w = d->local[i];

            if ((w >= fit) && (w < d->nverts)) {
                if (!reach || (w > top)) {
                    top    = w;
                    top_at = i;
                }
                reach++;
            }
        }
        if (reach)
            snprintf(ix, sizeof(ix), "; %u of its %u indices reach them, the highest %lld (the %uth); indices %s", reach, d->count,
                     (long long) (d->first + top - d->base), top_at,
                     (v->index.source == VGPU98_SRC_INLINE) ? "inline" : v->index_res ? "resident" : "from guest RAM");
        else
            snprintf(ix, sizeof(ix), "; none of its %u indices reach them; indices %s", d->count,
                     (v->index.source == VGPU98_SRC_INLINE) ? "inline" : v->index_res ? "resident" : "from guest RAM");
    }
    snprintf(line, sizeof(line),
             "VGPU98 3D: a draw lacks %u of its %u vertices in stream %u (they read as 0): %s%s, start %u, count %u, base %d; "
             "window from %lld; the stream: %s %08X, %u bytes, stride %u, %u bytes a vertex%s\n",
             v->past, d->nverts, s, (d->prim <= VGPU98_PRIM_TRIANGLE_FAN) ? prims[d->prim] : "?", d->indexed ? ", indexed" : "",
             p[1], p[2], (int32_t) p[3], (long long) d->first,
             (v->stream[s].source == VGPU98_SRC_INLINE) ? "inline" : "GPU address", v->stream[s].address, v->stream[s].size,
             v->stream[s].stride, v->stream[s].extent, ix);
    vgpu98_gpu_log(r->gpu, VGPU98_LOG_EVENTS, "%s", line);
}

/* The coordinate checks, after a
   draw's layout: the fixed function's draws and pre-transformed ones (a
   vertex program's coordinates are its own), their first COORD_PRIMS
   primitives (a draw straight from the arena read there).
   - A stage that reads its texture with its coordinates from the vertices
     (not generated): the set must be there, with the components the
     texture reads (a cube map or a volume 3, a texture over one texel high
     2) when no transform makes them, each finite and within +-COORD_HUGE.
   - A pre-transformed draw: each RHW above 0 and finite, a triangle's
     within COORD_RHW_SPAN to 1, and no texel over more than COORD_STRETCH
     pixels when the texture is magnified past COORD_MAG pixels a texel both
     ways (a triangle of COORD_AREA pixels or more).
   Counted only: draws whose primitives all read one coordinate, triangles
   that WRAPn takes the short way round, pre-transformed triangles that
   smear a texel over more than COORD_SMEAR pixels one way only (bars and
   frames of a game's screens do), and pre-transformed draws with a texture
   transform set (the host applies no matrix to such vertices). */
#    define COORD_PRIMS    32
#    define COORD_HUGE     65536.0f
#    define COORD_RHW_SPAN 100000.0f
#    define COORD_STRETCH  4096.0f /* pixels a texel covers: 64 x 64 */
#    define COORD_MAG      64.0f   /* pixels a texel, both ways */
#    define COORD_SMEAR    256.0f  /* pixels a texel, one way */
#    define COORD_AREA     1024.0f
#    define COORD_KIND_MS    1000  /* a kind of draw read at most this often ... */
#    define COORD_PER_SECOND 2000  /* ... and draws read a second at most */

static uint32_t
mix32(uint32_t h, uint32_t v)
{
    return h ^ (v + 0x9e3779b9u + (h << 6) + (h >> 2));
}

/* A draw's set as the finding lines name it. */
static void
coord_set_text(char *buf, size_t size, const vgpu98_vin_t *v, uint32_t set)
{
    const uint32_t slot = VGPU98_SLOT_TEX0 + set;
    const uint32_t s    = v->slot[slot].stream;

    snprintf(buf, size, "set %u (type %u at byte %u of stream %u, stride %u)", set, v->slot[slot].type, v->slot[slot].offset, s,
             v->stream[s].stride);
}

/* After a named coordinate or RHW finding: the draw's numbers, the
   stream's, and the raw dwords of window vertex w and its two neighbours
   in stream s. */
static void
coord_dump(vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d, int direct, uint32_t slot, int32_t w)
{
    const vgpu98_vin_t *v = &r->vin;
    const uint32_t      s = v->slot[slot].stream;
    char                line[480];
    uint32_t            from;
    int32_t             k;

    if (s >= VGPU98_MAX_STREAMS)
        return;
    /* A vertex past 64 bytes: its 64 from the flagged element's dword. */
    from = (v->stream[s].stride > 64) ? (v->slot[slot].offset & ~3u) : 0;
    r3d_log(r, VGPU98_LOG_EVENTS,
            "VGPU98 3D: texture audit: the draw: %s, start %u, count %u, base %d, window from %lld (%u vertices)%s; stream %u: %s "
            "%08X, %u bytes, stride %u, version %08X, the draw's copy from its byte %llu\n",
            d->indexed ? "indexed" : "not indexed", d->start, d->count, d->base, (long long) d->first, d->nverts,
            direct ? ", straight from the arena" : "", s, (v->stream[s].source == VGPU98_SRC_GPUVM) ? "at GPU address" : "inline at",
            v->stream[s].address, v->stream[s].size, v->stream[s].stride, v->stream[s].version,
            (unsigned long long) v->copy[s].lo);
    for (k = w - 1; k <= w + 1; k++) {
        uint8_t        b[64];
        const uint32_t got = vgpu98_vin_raw(v, s, k, from, b, sizeof(b));
        size_t         n;
        uint32_t       i;

        if (!got)
            continue;
        n = (size_t) snprintf(line, sizeof(line), "VGPU98 3D: texture audit: window vertex %d%s, %u bytes from its byte %u:", k,
                              (k == w) ? " (the finding's)" : "", got, from);
        for (i = 0; (i + 4 <= got) && (n + 10 < sizeof(line)); i += 4)
            n += (size_t) snprintf(line + n, sizeof(line) - n, " %08X", rd32(b + i));
        r3d_log(r, VGPU98_LOG_EVENTS, "%s\n", line);
    }
}

/* Does a fixed-function stage read its texture: an argument of a colour or
   alpha operation (not DISABLE) that the operation uses selects it, or the
   operation reads it itself (a bump map's, BLENDTEXTUREALPHA(PM)), or the
   stage before premodulates with it? A pixel program's stages are its own. */
static int
stage_reads_texture(const vgpu98_r3d_t *r, int s)
{
    const uint32_t *t = r->tss[s];
    int             k;

    if (r->bind_p)
        return 1;
    if ((s > 0) && ((r->tss[s - 1][VGPU98_TSS_COLOROP] == VGPU98_TOP_PREMODULATE) ||
                    (r->tss[s - 1][VGPU98_TSS_ALPHAOP] == VGPU98_TOP_PREMODULATE)))
        return 1;
    for (k = 0; k < 2; k++) {
        const uint32_t op = t[k ? VGPU98_TSS_ALPHAOP : VGPU98_TSS_COLOROP];
        const uint32_t a1 = t[k ? VGPU98_TSS_ALPHAARG1 : VGPU98_TSS_COLORARG1] & VGPU98_TA_SELECTMASK;
        const uint32_t a2 = t[k ? VGPU98_TSS_ALPHAARG2 : VGPU98_TSS_COLORARG2] & VGPU98_TA_SELECTMASK;
        const uint32_t a0 = t[k ? VGPU98_TSS_ALPHAARG0 : VGPU98_TSS_COLORARG0] & VGPU98_TA_SELECTMASK;

        if (op == VGPU98_TOP_DISABLE)
            continue;
        if ((op == VGPU98_TOP_BUMPENVMAP) || (op == VGPU98_TOP_BUMPENVMAPLUMINANCE) || (op == VGPU98_TOP_BLENDTEXTUREALPHA) ||
            (op == VGPU98_TOP_BLENDTEXTUREALPHAPM))
            return 1;
        if (((op != VGPU98_TOP_SELECTARG2) && (a1 == VGPU98_TA_TEXTURE)) ||
            ((op != VGPU98_TOP_SELECTARG1) && (a2 == VGPU98_TA_TEXTURE)) ||
            (((op == VGPU98_TOP_MULTIPLYADD) || (op == VGPU98_TOP_LERP)) && (a0 == VGPU98_TA_TEXTURE)))
            return 1;
    }
    return 0;
}

/* The window vertex the draw's element e reads (vgpu98_vin_element): a
   draw straight from the arena numbers its vertices from 0, its indices
   read there; -1 for an index past its buffer. */
static int32_t
coord_vertex(const vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d, int direct, uint32_t e)
{
    const vgpu98_vin_t *v = &r->vin;
    uint64_t            at;
    int64_t             x;

    if (!direct)
        return d->indexed ? (int32_t) d->local[e] : (int32_t) e;
    if (!d->indexed)
        return (int32_t) (d->start + e);
    at = (uint64_t) d->start + e;
    if (!v->index_res || !v->index.isize || (at >= v->index.size / v->index.isize))
        return -1;
    x = (int64_t) ((v->index.isize == 2) ? rd16(v->index_res + at * 2) : rd32(v->index_res + at * 4)) + d->base;
    return ((x < 0) || (x > 0x7fffffff)) ? -1 : (int32_t) x;
}

/* The texture-matrix checks, for a
   draw through the transform (World 0, no vertex blending) the coordinate
   checks read: its first stage that reads a 2D texture with coordinates
   generated from camera-space position, or from the vertices through a
   texture transform. On the CPU, as DirectX's rules have them: camera space
   (World x View), clip space (x Projection), the viewport, the generated
   position (x, y, z, 1) or the vertices' set padded (its first missing
   component 1), the texture matrix, PROJECTED's divide. Then each triangle's
   texels per pixel (as for pre-transformed draws):
   magnified both ways past COORD_MAG, a texel over more than COORD_STRETCH
   pixels, is a finding; a one-way smear is counted, edge-on to the mapping
   (the triangle's normal within MX_EDGE of the mapping's plane, as a
   top-down projection draws a cliff) or not. A non-finite matrix is a
   finding; one whose u and v follow one direction, and a kind's matrix
   that changed while the view did not, are counted. */
#    define MX_EDGE     0.1045f /* sin(6 degrees) */
#    define MX_COLLAPSE 1e-4f

static float
vec3_dot(const float *a, const float *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void
vec3_cross(const float *a, const float *b, float *c)
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

/* A row vector times a D3D matrix (mat_mul's order). */
static void
vec4_mul(const float *v, const float *m, float *out)
{
    int j;

    for (j = 0; j < 4; j++)
        out[j] = v[0] * m[j] + v[1] * m[4 + j] + v[2] * m[8 + j] + v[3] * m[12 + j];
}

static void
matrix_check(vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d, r3d_image_t *const *tex, int direct, coord_kind_t *kind)
{
    const vgpu98_vin_t *v       = &r->vin;
    const uint32_t      mask    = (1u << active_stages(r)) - 1u;
    const uint32_t      nprims  = vgpu98_vin_nprims(d);
    const uint32_t      n       = (nprims < COORD_PRIMS) ? nprims : COORD_PRIMS;
    const float        *T       = NULL;
    uint32_t            gen     = 0;
    uint32_t            ttf     = 0;
    uint32_t            set     = 0;
    uint32_t            thash   = 0x811c9dc5u;
    int                 stage   = -1;
    int                 told    = 0;
    int                 smeared = 0;
    int                 edge    = 0;
    float               wv[16];
    float               wvp[16];
    float               gu[3];
    float               gvv[3];
    float               m[3] = { 0.0f, 0.0f, 0.0f };
    float               mlen = 0.0f;
    uint32_t            i;
    uint32_t            k;
    int                 s;

    if ((v->program != 1) || r->rs[VGPU98_RS_VERTEXBLEND] || (d->prim < VGPU98_PRIM_TRIANGLES) || !n)
        return;
    for (s = 0; s < N_STAGES; s++) {
        const tex_t   *t   = &r->tex[s];
        const uint32_t tci = r->tss[s][VGPU98_TSS_TEXCOORDINDEX];
        const uint32_t tf  = r->tss[s][VGPU98_TSS_TEXTURETRANSFORMFLAGS];
        const uint32_t g   = (tci >> VGPU98_TCI_GEN_SHIFT) & 3u;
        uint32_t       e;

        if (!((mask >> s) & 1u) || !t->format || !stage_reads_texture(r, s) || (!g && !(tf & VGPU98_TTFF_COUNT_MASK)))
            continue;
        if ((tf & VGPU98_TTFF_COUNT_MASK)) {
            const float   *x    = r->xform[VGPU98_XFORM_TEXTURE + s];
            const uint32_t cols = tf & VGPU98_TTFF_COUNT_MASK; /* the outputs used: columns 1 to COUNT */

            for (e = 0; e < 16; e++) {
                if (((e & 3u) < cols) && !isfinite(x[e])) {
                    tex_find(r, VGPU98_FIND_MATRIX, tex[s] ? tex[s]->id : 0,
                             "possible bad texture matrix: stage %d's element %u (row %u, column %u) is %g, coordinates %s, "
                             "flags %X",
                             s, e, e / 4 + 1, e % 4 + 1, x[e], g ? "generated" : "from the vertices", tf);
                    return;
                }
            }
            thash = fnv(x, 64, thash);
        }
        if ((g == 1) || (g == 3)) {
            r->ta_mx_unfollowed++; /* normals and reflections: not followed */
            continue;
        }
        if ((stage < 0) && (t->width > 1) && (t->height > 1) && (t->faces != VGPU98_CUBE_FACES) && !t->depth) {
            stage = s;
            gen   = g;
            ttf   = tf;
            set   = tci & VGPU98_TCI_SET_MASK;
            T     = (tf & VGPU98_TTFF_COUNT_MASK) ? r->xform[VGPU98_XFORM_TEXTURE + s] : NULL;
        }
    }
    /* A kind's texture matrices that moved while its view did not. */
    {
        const uint32_t vh = fnv(r->xform[VGPU98_XFORM_VIEW], 64, 0x811c9dc5u);

        if (kind->mx_set && (kind->view == vh) && (kind->tex != thash))
            r->ta_mx_moving++;
        kind->mx_set = 1;
        kind->view   = vh;
        kind->tex    = thash;
    }
    if (stage < 0)
        return;
    r->ta_mx_draws++;
    mat_mul(wv, r->xform[VGPU98_XFORM_WORLD], r->xform[VGPU98_XFORM_VIEW]);
    mat_mul(wvp, wv, r->xform[VGPU98_XFORM_PROJECTION]);
    /* The mapping's gradients in camera space (generated position), and
       the plane they span. */
    if (T && (gen == 2)) {
        gu[0] = T[0], gu[1] = T[4], gu[2] = T[8];
        gvv[0] = T[1], gvv[1] = T[5], gvv[2] = T[9];
        vec3_cross(gu, gvv, m);
        mlen = sqrtf(vec3_dot(m, m));
        if (mlen <= MX_COLLAPSE * sqrtf(vec3_dot(gu, gu)) * sqrtf(vec3_dot(gvv, gvv)))
            r->ta_mx_collapsed++;
    }
    for (i = 0; i < n; i++) {
        float scr[3][2];
        float uv[3][2];
        float eye[3][4];
        int   ok = 1;

        for (k = 0; (k < 3) && ok; k++) {
            const int32_t w = coord_vertex(r, d, direct, vgpu98_vin_element(d, i, k));
            float         p[4];
            float         clip[4];
            float         c[4];
            float         o[4];

            if (!vgpu98_vin_fetch(v, VGPU98_SLOT_POSITION, w, p)) {
                ok = 0;
                break;
            }
            p[3] = 1.0f;
            vec4_mul(p, wv, eye[k]);
            vec4_mul(p, wvp, clip);
            /* Within the near and far planes and a guard band of twice the
               screen: a triangle past them draws as the clipper's pieces,
               which these numbers are not. */
            if (!(clip[3] > 0.0f) || !(clip[2] >= 0.0f) || !(clip[2] <= clip[3]) || !(fabsf(clip[0]) <= 2.0f * clip[3]) ||
                !(fabsf(clip[1]) <= 2.0f * clip[3])) {
                ok = 0;
                break;
            }
            scr[k][0] = (float) r->vp[0] + (1.0f + clip[0] / clip[3]) * (float) r->vp[2] / 2.0f;
            scr[k][1] = (float) r->vp[1] + (1.0f - clip[1] / clip[3]) * (float) r->vp[3] / 2.0f;
            if (gen == 2) {
                c[0] = eye[k][0], c[1] = eye[k][1], c[2] = eye[k][2], c[3] = 1.0f;
            } else {
                const int got = vgpu98_vin_fetch(v, VGPU98_SLOT_TEX0 + set, w, c);

                c[0] = got > 0 ? c[0] : 0.0f;
                if ((got > 0) && (got < 4)) {
                    int j;

                    for (j = got; j < 4; j++)
                        c[j] = (j == got) ? 1.0f : 0.0f;
                }
            }
            if (T)
                vec4_mul(c, T, o);
            else
                memcpy(o, c, sizeof(o));
            if ((ttf & VGPU98_TTFF_PROJECTED) && ((ttf & VGPU98_TTFF_COUNT_MASK) >= 2)) {
                const float q = o[(ttf & VGPU98_TTFF_COUNT_MASK) - 1];

                if (q == 0.0f) {
                    ok = 0;
                    break;
                }
                o[0] /= q;
                o[1] /= q;
            }
            uv[k][0] = o[0];
            uv[k][1] = o[1];
            ok       = isfinite(o[0]) && isfinite(o[1]);
        }
        if (!ok)
            continue;
        {
            const tex_t *t    = &r->tex[stage];
            const float  W    = (float) t->width;
            const float  H    = (float) t->height;
            const float  p00  = scr[1][0] - scr[0][0], p10 = scr[1][1] - scr[0][1];
            const float  p01  = scr[2][0] - scr[0][0], p11 = scr[2][1] - scr[0][1];
            const float  det  = p00 * p11 - p01 * p10;
            const float  area = fabsf(det) / 2.0f;
            float        t00, t10, t01, t11, j00, j01, j10, j11, sq, dj, big;

            if ((area < COORD_AREA) || !isfinite(det))
                continue;
            t00 = (uv[1][0] - uv[0][0]) * W, t10 = (uv[1][1] - uv[0][1]) * H;
            t01 = (uv[2][0] - uv[0][0]) * W, t11 = (uv[2][1] - uv[0][1]) * H;
            j00 = (t00 * p11 - t01 * p10) / det, j01 = (t01 * p00 - t00 * p01) / det;
            j10 = (t10 * p11 - t11 * p10) / det, j11 = (t11 * p00 - t10 * p01) / det;
            sq  = j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11;
            dj  = fabsf(j00 * j11 - j01 * j10);
            big = sqrtf((sq + sqrtf(fmaxf(sq * sq - 4.0f * dj * dj, 0.0f))) / 2.0f);
            if (!(big > 0.0f) || !isfinite(big))
                continue;
            if ((dj > 0.0f) && (big * COORD_MAG < 1.0f) && (1.0f / dj > COORD_STRETCH)) {
                if (!told) {
                    told = 1;
                    tex_find(r, VGPU98_FIND_STRETCH, tex[stage] ? tex[stage]->id : 0,
                             "possible stretching: a transformed triangle of %.0f pixels spreads one texel of stage %d's %ux%u %s "
                             "texture over %.0f pixels (coordinates %s%s, flags %X; texture coordinates (%g, %g), (%g, %g), (%g, "
                             "%g))",
                             area, stage, t->width, t->height, fmt_name(t->format), 1.0f / dj,
                             (gen == 2) ? "from camera-space position" : "from the vertices", T ? " through its matrix" : "", ttf,
                             uv[0][0], uv[0][1], uv[1][0], uv[1][1], uv[2][0], uv[2][1]);
                }
            } else if (dj / big * COORD_SMEAR < 1.0f) {
                /* One way: edge-on to a planar mapping, or not. */
                if ((gen == 2) && (mlen > 0.0f)) {
                    float e1[3], e2[3], nn[3];
                    int   j;

                    for (j = 0; j < 3; j++) {
                        e1[j] = eye[1][j] - eye[0][j];
                        e2[j] = eye[2][j] - eye[0][j];
                    }
                    vec3_cross(e1, e2, nn);
                    if (fabsf(vec3_dot(nn, m)) <= MX_EDGE * sqrtf(vec3_dot(nn, nn)) * mlen)
                        edge = 1;
                    else
                        smeared = 1;
                } else
                    smeared = 1;
            }
        }
    }
    r->ta_mx_edge += (uint32_t) edge;
    r->ta_mx_smeared += (uint32_t) smeared;
}

static void
coord_check(vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d, r3d_image_t *const *tex, int direct)
{
    static const char *const prims[] = { "?", "point list", "line list", "line strip", "triangle list", "triangle strip",
                                         "triangle fan" };
    const vgpu98_vin_t *v       = &r->vin;
    const int           pre     = (v->program == 0);
    const uint32_t      mask    = r->bind_p ? r->progs[r->bind_p].info.coords : ((1u << active_stages(r)) - 1u);
    const uint32_t      nprims  = vgpu98_vin_nprims(d);
    const uint32_t      n       = (nprims < COORD_PRIMS) ? nprims : COORD_PRIMS;
    const uint32_t      corners = (d->prim >= VGPU98_PRIM_TRIANGLES) ? 3 : (d->prim == VGPU98_PRIM_POINTS) ? 1 : 2;
    const char         *pname   = (d->prim < sizeof(prims) / sizeof(prims[0])) ? prims[d->prim] : "?";
    int                 stage_of[VGPU98_MAX_TEXSETS];
    coord_kind_t       *kind = NULL;
    uint32_t            sets         = 0;
    uint32_t            first_set    = VGPU98_MAX_TEXSETS;
    int                 told_coord   = 0;
    int                 told_rhw     = 0;
    int                 told_stretch = 0;
    int                 smeared      = 0;
    int                 xform        = 0;
    int                 same         = 1;
    int                 have_first   = 0;
    float               first_uv[2]  = { 0.0f, 0.0f };
    uint32_t            corners_read = 0;
    uint32_t            i;
    uint32_t            k;
    uint32_t            set;
    int                 s;

    /* Point sprites make their own coordinates. */
    if ((v->program == 2) || !n || ((d->prim == VGPU98_PRIM_POINTS) && r->rs[VGPU98_RS_POINTSPRITEENABLE]))
        return;
    /* A kind of draw (its declaration, primitive, stages' textures and
       their coordinate states, WRAPn) is read at most once per
       COORD_KIND_MS, and COORD_PER_SECOND draws a second at most: the
       checks cost the GPU thread a bounded share. */
    {
        const uint32_t ms  = (uint32_t) (vgpu98_diag_us() / 1000);
        uint32_t       key = fnv(v->decl, (size_t) v->n_elements * 8, mix32(d->prim, (uint32_t) v->program));
        coord_kind_t  *e;

        for (s = 0; s < N_STAGES; s++) {
            if (r->tex[s].format)
                key = mix32(mix32(mix32(key, r->tex[s].offset[0]), r->tex[s].format | ((uint32_t) s << 8)),
                            mix32(r->tss[s][VGPU98_TSS_TEXCOORDINDEX], r->tss[s][VGPU98_TSS_TEXTURETRANSFORMFLAGS]));
        }
        for (set = 0; set < VGPU98_MAX_TEXSETS; set++)
            key = mix32(key, r->rs[VGPU98_RS_WRAP0 + set]);
        if ((uint32_t) (ms - r->coord_second) >= 1000) {
            r->coord_second = ms;
            r->coord_read   = 0;
        }
        if (!r->coord_kinds && !(r->coord_kinds = calloc(COORD_KINDS, sizeof(coord_kind_t))))
            return;
        e = &r->coord_kinds[key & (COORD_KINDS - 1)];
        if ((e->used && (e->key == key) && ((uint32_t) (ms - e->ms) < COORD_KIND_MS)) || (r->coord_read >= COORD_PER_SECOND)) {
            r->ta_coord_skipped++;
            return;
        }
        if (e->key != key)
            e->mx_set = 0; /* another kind: its matrices are not this one's */
        e->used = 1;
        e->key  = key;
        e->ms   = ms;
        r->coord_read++;
        kind = e;
    }
    r->ta_coord_draws++;
    for (set = 0; set < VGPU98_MAX_TEXSETS; set++)
        stage_of[set] = -1;
    for (s = 0; s < N_STAGES; s++) {
        const tex_t   *t   = &r->tex[s];
        const uint32_t tci = r->tss[s][VGPU98_TSS_TEXCOORDINDEX];
        const uint32_t ttf = r->tss[s][VGPU98_TSS_TEXTURETRANSFORMFLAGS];
        const uint32_t id  = tex[s] ? tex[s]->id : 0;
        uint32_t       need;
        uint32_t       type;
        char           st[96];

        set = tci & VGPU98_TCI_SET_MASK;
        if (!((mask >> s) & 1u) || !t->format || (tci & VGPU98_TCI_GEN_MASK) || !stage_reads_texture(r, s))
            continue;
        if (pre && (ttf & VGPU98_TTFF_COUNT_MASK))
            xform = 1;
        if (!v->slot[VGPU98_SLOT_TEX0 + set].on) {
            uint32_t have = 0;
            uint32_t j;

            for (j = 0; j < VGPU98_MAX_TEXSETS; j++)
                have += v->slot[VGPU98_SLOT_TEX0 + j].on;
            tex_find(r, VGPU98_FIND_SET, id,
                     "stage %d samples a %ux%u %s texture with coordinate set %u, which the vertices of this %s of %u vertices lack "
                     "(they have %u set%s): it reads (0, 0)",
                     s, t->width, t->height, fmt_name(t->format), set, pname, d->count, have, (have == 1) ? "" : "s");
            continue;
        }
        type = v->slot[VGPU98_SLOT_TEX0 + set].type;
        need = ((t->faces == VGPU98_CUBE_FACES) || t->depth) ? 3 : (t->height > 1) ? 2 : 1;
        if (!(ttf & VGPU98_TTFF_COUNT_MASK) && (type <= VGPU98_DT_FLOAT4) && (type < need)) {
            coord_set_text(st, sizeof(st), v, set);
            tex_find(r, VGPU98_FIND_SET, id,
                     "stage %d samples a %s%ux%u %s texture with coordinate %s, which has %u component%s of the %u it reads: the rest "
                     "read 0",
                     s, (t->faces == VGPU98_CUBE_FACES) ? "cube map " : t->depth ? "volume " : "", t->width, t->height,
                     fmt_name(t->format), st, type, (type == 1) ? "" : "s", need);
        }
        sets |= 1u << set;
        if (stage_of[set] < 0)
            stage_of[set] = s;
        if (set < first_set)
            first_set = set;
    }
    r->ta_pre_xform += (uint32_t) xform;
    if (!pre && kind)
        matrix_check(r, d, tex, direct, kind);
    if (!sets && !pre)
        return;

    for (i = 0; i < n; i++) {
        float    pos[3][4];
        float    uv[VGPU98_MAX_TEXSETS][3][4];
        int32_t  wv[3];
        uint32_t have_pos = 0;

        for (k = 0; k < corners; k++) {
            wv[k] = coord_vertex(r, d, direct, vgpu98_vin_element(d, i, k));
            if (pre && vgpu98_vin_fetch(v, VGPU98_SLOT_POSITION, wv[k], pos[k])) {
                const float rhw = pos[k][3];

                have_pos++;
                if (!told_rhw && (!isfinite(rhw) || !(rhw > 0.0f))) {
                    told_rhw = 1;
                    if (tex_find(r, VGPU98_FIND_RHW,
                                 (first_set < VGPU98_MAX_TEXSETS) && tex[stage_of[first_set]] ? tex[stage_of[first_set]]->id : 0,
                                 "a pre-transformed vertex's RHW is %g (window vertex %d of a %s of %u vertices, at (%g, %g)): it "
                                 "should be above 0 and finite",
                                 rhw, wv[k], pname, d->count, pos[k][0], pos[k][1]))
                        coord_dump(r, d, direct, VGPU98_SLOT_POSITION, wv[k]);
                }
            }
            for (set = 0; set < VGPU98_MAX_TEXSETS; set++) {
                int c;
                int j;

                if (!((sets >> set) & 1u))
                    continue;
                c = vgpu98_vin_fetch(v, VGPU98_SLOT_TEX0 + set, wv[k], uv[set][k]);
                for (j = 0; (j < c) && !told_coord; j++) {
                    const float f = uv[set][k][j];

                    if (!isfinite(f) || (fabsf(f) > COORD_HUGE)) {
                        char st[96];

                        told_coord = 1;
                        coord_set_text(st, sizeof(st), v, set);
                        if (tex_find(r, VGPU98_FIND_COORD, tex[stage_of[set]] ? tex[stage_of[set]]->id : 0,
                                     "a coordinate %s: %s, component %d, is %g at window vertex %d of a %s of %u vertices (stage "
                                     "%d samples it)",
                                     isfinite(f) ? "beyond +-65536" : "not finite", st, j, f, wv[k], pname, d->count,
                                     stage_of[set]))
                            coord_dump(r, d, direct, VGPU98_SLOT_TEX0 + set, wv[k]);
                    }
                }
                if (set == first_set) {
                    corners_read++;
                    if (!have_first) {
                        first_uv[0] = uv[set][k][0];
                        first_uv[1] = uv[set][k][1];
                        have_first  = 1;
                    } else if ((uv[set][k][0] != first_uv[0]) || (uv[set][k][1] != first_uv[1]))
                        same = 0;
                }
            }
        }
        if (corners != 3)
            continue;

        /* A triangle's RHW span. */
        if (pre && (have_pos == 3) && !told_rhw) {
            float lo = pos[0][3];
            float hi = pos[0][3];

            for (k = 1; k < 3; k++) {
                lo = (pos[k][3] < lo) ? pos[k][3] : lo;
                hi = (pos[k][3] > hi) ? pos[k][3] : hi;
            }
            if ((lo > 0.0f) && (hi > lo * COORD_RHW_SPAN)) {
                told_rhw = 1;
                tex_find(r, VGPU98_FIND_RHW, 0,
                         "a pre-transformed triangle's RHW spans %g to %g (over %g to 1; window vertices %d, %d, %d of a %s of %u "
                         "vertices)",
                         lo, hi, COORD_RHW_SPAN, wv[0], wv[1], wv[2], pname, d->count);
            }
        }

        /* WRAPn's short way round. */
        for (set = 0; set < VGPU98_MAX_TEXSETS; set++) {
            const uint32_t wrap = r->rs[VGPU98_RS_WRAP0 + set] & 3;
            uint32_t       a;
            int            hit = 0;

            if (!((sets >> set) & 1u) || !wrap)
                continue;
            for (a = 0; (a < 2) && !hit; a++) {
                if (!((wrap >> a) & 1))
                    continue;
                for (k = 0; k < 3; k++) {
                    if (fabsf(uv[set][k][a] - uv[set][(k + 1) % 3][a]) > 0.5f)
                        hit = 1;
                }
            }
            r->ta_wrap_short += (uint32_t) hit;
        }

        /* The first set's texture (2D) over the screen: its texels per pixel
           (J, from screen to texels; its singular values, the most and the
           fewest texels a pixel along any direction). */
        if (pre && (have_pos == 3) && !told_stretch && (first_set < VGPU98_MAX_TEXSETS)) {
            const tex_t *t    = &r->tex[stage_of[first_set]];
            const float  W    = (float) t->width;
            const float  H    = (float) t->height;
            const float  p00  = pos[1][0] - pos[0][0], p10 = pos[1][1] - pos[0][1];
            const float  p01  = pos[2][0] - pos[0][0], p11 = pos[2][1] - pos[0][1];
            const float  det  = p00 * p11 - p01 * p10;
            const float  area = fabsf(det) / 2.0f;

            if ((t->width > 1) && (t->height > 1) && (t->faces != VGPU98_CUBE_FACES) && !t->depth && (area >= COORD_AREA) &&
                isfinite(det)) {
                const float *u0  = uv[first_set][0], *u1 = uv[first_set][1], *u2 = uv[first_set][2];
                const float  t00 = (u1[0] - u0[0]) * W, t10 = (u1[1] - u0[1]) * H;
                const float  t01 = (u2[0] - u0[0]) * W, t11 = (u2[1] - u0[1]) * H;
                const float  j00 = (t00 * p11 - t01 * p10) / det, j01 = (t01 * p00 - t00 * p01) / det;
                const float  j10 = (t10 * p11 - t11 * p10) / det, j11 = (t11 * p00 - t10 * p01) / det;
                const float  sq  = j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11;
                const float  dj  = fabsf(j00 * j11 - j01 * j10);
                const float  big = sqrtf((sq + sqrtf(fmaxf(sq * sq - 4.0f * dj * dj, 0.0f))) / 2.0f);

                if ((big > 0.0f) && isfinite(big)) {
                    /* Magnified both ways, past one texel over COORD_STRETCH
                       pixels: a finding. One way only: counted. */
                    if ((dj > 0.0f) && (big * COORD_MAG < 1.0f) && (1.0f / dj > COORD_STRETCH)) {
                        told_stretch = 1;
                        tex_find(r, VGPU98_FIND_STRETCH, tex[stage_of[first_set]] ? tex[stage_of[first_set]]->id : 0,
                                 "possible stretching: a pre-transformed triangle of %.0f pixels spreads one texel of stage %d's %ux%u "
                                 "%s texture over %.0f pixels (coordinates (%g, %g), (%g, %g), (%g, %g); window vertices %d, %d, %d "
                                 "of a %s of %u vertices)",
                                 area, stage_of[first_set], t->width, t->height, fmt_name(t->format), 1.0f / dj, u0[0], u0[1],
                                 u1[0], u1[1], u2[0], u2[1], wv[0], wv[1], wv[2], pname, d->count);
                    } else if (!smeared && (dj / big * COORD_SMEAR < 1.0f))
                        smeared = 1;
                }
            }
        }
    }
    r->ta_smeared += (uint32_t) smeared;
    if (same && have_first && (corners_read >= 3))
        r->ta_one_coord++;
}

/* The display check: a draw, counted for its
   frame, and as on display when its colour target is the surface the last
   FLIP showed. */
static void
shown_draw(vgpu98_r3d_t *r)
{
    r->shown_draws++;
    if (r->shown_set && (r->target.offset == r->shown_start))
        r->shown_on++;
}

static void vsum_resolve(vgpu98_r3d_t *r, int have, uint64_t key, uint32_t lock, uint64_t at);

/* A FLIP to `start`: the frame drawn since the last one judged. With draws
   it is looked at, and drawn on display when more than half its draws
   were (a cursor or an overlay drawn there is not enough); SHOWN_FRAMES
   such frames in a row are named, once a run. A frame without draws is
   passed over, and a game that never flips is never looked at. */
void
vgpu98_r3d_frame(vgpu98_r3d_t *r, uint32_t start)
{
    if (!r)
        return;
    if (r->shown_set && r->shown_draws) {
        r->ta_frames++;
        if (2 * (uint64_t) r->shown_on > r->shown_draws) {
            r->ta_shown++;
            if (++r->shown_run == SHOWN_FRAMES)
                tex_find(r, VGPU98_FIND_SHOWN, 0,
                         "possible drawing into the surface on display: %u of %u draws a frame went to the surface the last "
                         "FLIP showed (%08X), %u frames running",
                         r->shown_on, r->shown_draws, r->shown_start, SHOWN_FRAMES);
        } else
            r->shown_run = 0;
    }
    r->flip_frame++;
    vsum_resolve(r, 0, 0, 0, 0); /* the vertex race detector's races held too long */
    r->shown_start = start;
    r->shown_set   = 1;
    r->shown_draws = 0;
    r->shown_on    = 0;
}

/* --- The vertex race detector ----------- */

_Static_assert(VGPU98_LEN_NOTE_VSUM == 7 + VGPU98_NOTE_VSUM_PARTS + 1, "VSUM: seven words, the sums, the check");

/* A lock word as the lines name it. */
static void
vsum_locks(uint32_t lock, char *buf, size_t size)
{
    const uint32_t n = lock >> VGPU98_NOTE_LOCK_COUNT_SHIFT;

    if (!(lock & VGPU98_NOTE_LOCK_SEEN)) {
        snprintf(buf, size, "none%s", (lock & VGPU98_NOTE_LOCK_OPEN) ? " (one open at a draw)" : "");
        return;
    }
    snprintf(buf, size, "%u%s, flags %X%s%s%s%s%s", n, (n >= 255) ? " or more" : "", lock & VGPU98_NOTE_LOCK_FLAGS,
             !(lock & VGPU98_NOTE_LOCK_UNWAITED) ? ""
             : (lock & VGPU98_NOTE_LOCK_NOOW_BUSY) ? ", NOOVERWRITE given with a draw pending"
                                                   : ", given unwaited with a draw pending",
             (lock & VGPU98_NOTE_LOCK_IDLE) ? ", given with none pending" : "",
             (lock & VGPU98_NOTE_LOCK_WAITED) ? ", given after a wait" : "",
             (lock & VGPU98_NOTE_LOCK_RENAMED) ? ", given new memory" : "",
             (lock & VGPU98_NOTE_LOCK_OPEN) ? ", one open at a draw" : "");
}

/* The Locks after a race (its later notes' words together; `before`, the
   race's own) by kind: the most suspect the word holds. */
static int
vsum_after(uint32_t lock, uint32_t before)
{
    if (lock & VGPU98_NOTE_LOCK_UNWAITED)
        return (lock & VGPU98_NOTE_LOCK_NOOW_BUSY) ? VSUM_AFTER_NOOW : VSUM_AFTER_UNWAITED;
    if (lock & VGPU98_NOTE_LOCK_IDLE)
        return VSUM_AFTER_IDLE;
    if (lock & VGPU98_NOTE_LOCK_WAITED)
        return VSUM_AFTER_WAITED;
    if (lock & VGPU98_NOTE_LOCK_RENAMED)
        return VSUM_AFTER_RENAMED;
    return ((lock | before) & VGPU98_NOTE_LOCK_OPEN) ? VSUM_AFTER_OPEN : VSUM_AFTER_NONE;
}

/* A race held told: counted by the Locks after it, and their line written
   when its own was (named by the process that drew it). `early`: told
   before they were all in. */
static void
vsum_tell(vgpu98_r3d_t *r, vsum_pend_t *q, int early)
{
    const uint32_t word = (q->after & ~VGPU98_NOTE_LOCK_COUNT) |
                          (((q->n_after < 255) ? q->n_after : 255) << VGPU98_NOTE_LOCK_COUNT_SHIFT);
    char           locks[160];

    r->ta_vsum_after[q->later ? vsum_after(word, q->lock) : VSUM_AFTER_UNSEEN]++;
    if (q->named) {
        if (q->later)
            vsum_locks(word, locks, sizeof(locks));
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: texture audit: %s%s%spossible vertex race %u, the buffer's Locks after the draw%s: %s\n",
                q->proc[0] ? "[" : "", q->proc, q->proc[0] ? "] " : "", q->seq,
                !q->later ? "" : early ? " (its draws past the GPU's read not seen)" : ", to its first draw past the GPU's read",
                q->later ? locks : "no later draw of the buffer seen");
    }
    q->used = 0;
}

/* The races held: the buffer's (`key`), when a later draw's note of it came
   (`lock` its word, `at` its place in the ring), take its Locks, and are
   told once a note written after the GPU's read came; those held too long
   are told with what came. */
static void
vsum_resolve(vgpu98_r3d_t *r, int have, uint64_t key, uint32_t lock, uint64_t at)
{
    uint32_t i;

    for (i = 0; i < VSUM_PENDING; i++) {
        vsum_pend_t *q = &r->vsum_pend[i];

        if (!q->used)
            continue;
        if (have && (q->key == key) && (q->draw != r->vsum_draw)) {
            q->later = 1;
            q->after |= lock & ~VGPU98_NOTE_LOCK_COUNT;
            q->n_after += lock >> VGPU98_NOTE_LOCK_COUNT_SHIFT;
            if (at >= q->until)
                vsum_tell(r, q, 0);
        } else if (((uint32_t) (r->flip_frame - q->frame) >= VSUM_PEND_FRAMES) ||
                   ((uint32_t) (r->note_count - q->notes) >= VSUM_PEND_NOTES))
            vsum_tell(r, q, 1);
    }
}

static uint32_t
note_hash(const uint8_t *p, uint32_t n, uint32_t h)
{
    uint32_t i;

    for (i = 0; i + 4 <= n; i += 4)
        h = (h ^ rd32(p + i)) * VGPU98_NOTE_HASH_PRIME;
    for (; i < n; i++)
        h = (h ^ p[i]) * VGPU98_NOTE_HASH_PRIME;
    return h;
}

/* The parts of a window whose sums differ, as "1, 3, 8", and how many (0
   when none). */
static int
vsum_parts(const uint32_t *want, const uint8_t *bytes, uint32_t len, uint32_t stride, char *buf, size_t size)
{
    const uint32_t nv  = len / stride;
    const uint32_t per = (nv + VGPU98_NOTE_VSUM_PARTS - 1) / VGPU98_NOTE_VSUM_PARTS;
    size_t         n   = 0;
    int            any = 0;
    uint32_t       k;

    buf[0] = 0;
    for (k = 0; k < VGPU98_NOTE_VSUM_PARTS; k++) {
        const uint32_t v0 = k * per;
        const uint32_t v1 = ((k + 1) * per < nv) ? (k + 1) * per : nv;
        const uint32_t h  = (v0 < v1) ? note_hash(bytes + (size_t) v0 * stride, (v1 - v0) * stride, VGPU98_NOTE_HASH_SEED)
                                      : VGPU98_NOTE_HASH_SEED;

        if (h != want[k]) {
            if (n + 8 < size)
                n += (size_t) snprintf(buf + n, size - n, "%s%u", any ? ", " : "", k + 1);
            any++;
        }
    }
    return any;
}

/* The buffer `id`'s note at the memory ending at `end`: recorded, and the
   buffer that held that memory before it, lately (0: none, or no id). */
static uint32_t
vsum_owner(vgpu98_r3d_t *r, uint32_t end, uint32_t id)
{
    vsum_owner_t *o   = NULL;
    vsum_owner_t *old = NULL;
    uint32_t      i;

    if (!id || (!r->vsum_owners && !(r->vsum_owners = calloc(VSUM_OWNERS, sizeof(vsum_owner_t)))))
        return 0;
    for (i = 0; (i < VSUM_OWNERS) && !o; i++) {
        vsum_owner_t *e = &r->vsum_owners[i];

        if (!e->used) {
            if (!old || old->used)
                old = e;
        } else if (e->end == end)
            o = e;
        else if (!old || (old->used && ((int32_t) (e->seen - old->seen) < 0)))
            old = e;
    }
    if (!o) {
        o = old;
        memset(o, 0, sizeof(*o));
        o->used = 1;
        o->end  = end;
        o->id   = id;
    } else if (o->id != id) {
        o->prev  = o->id;
        o->moved = r->note_count;
        o->id    = id;
    }
    o->seen = r->note_count;
    return (o->prev && (o->prev != id) && ((uint32_t) (r->note_count - o->moved) < VSUM_OWNER_NOTES)) ? o->prev : 0;
}

/* A VSUM note at `at` in the ring (`known`: the bytes the driver had
   written then): stream s's window (or the index buffer's) as the DirectX
   driver drew it, against the bytes the draw that follows reads: the
   arena's copy when it holds the buffer, else guest RAM (through the page
   table). Its lock word first goes to the races held for the buffer (by its
   id, even when the note is refused). A race found is told and held for
   the Locks after it; a window the arena's copy alone differs in (guest RAM
   matches) is a stale copy. */
static void
vsum_check(vgpu98_r3d_t *r, const uint32_t *p, uint64_t at, uint64_t known)
{
    const vgpu98_vin_t *v      = &r->vin;
    const uint32_t      s      = p[2] & VGPU98_NOTE_VSUM_STREAM;
    const uint32_t      id     = p[2] >> VGPU98_NOTE_VSUM_BUF_SHIFT;
    const uint32_t      off    = p[3];
    const uint32_t      len    = p[4];
    const uint32_t      stride = p[5];
    const uint32_t      lock   = p[6];
    const uint32_t     *want   = p + 7;
    char                pram[40];
    char                parena[40];
    char                before[160];
    char                what[24];
    char                owner[64];
    uint32_t            base    = 0; /* the buffer as the GPU has it bound: GPU address, bytes, version, stride */
    uint32_t            size    = 0;
    uint32_t            version = 0;
    uint32_t            bstride = 0;
    int                 bound   = 0;
    uint32_t            nv;
    uint32_t            per;
    uint32_t            parts;
    uint32_t            prev;
    uint64_t            key;
    uint32_t            delta;
    uint32_t            i;
    res_t              *e;
    int                 ram;
    int                 arena    = 0;
    int                 resident = 0;
    vsum_pend_t        *q        = NULL;

    if (s == VGPU98_NOTE_VSUM_INDEX) {
        bound   = v->index.set && (v->index.source == VGPU98_SRC_GPUVM);
        base    = v->index.address;
        size    = v->index.size;
        version = v->index.version;
        bstride = v->index.isize;
        snprintf(what, sizeof(what), "the index buffer");
    } else if (s < VGPU98_MAX_STREAMS) {
        bound   = v->stream[s].set && (v->stream[s].source == VGPU98_SRC_GPUVM);
        base    = v->stream[s].address;
        size    = v->stream[s].size;
        version = v->stream[s].version;
        bstride = v->stream[s].stride;
        snprintf(what, sizeof(what), "stream %u", s);
    }
    if (!bound || !len || !stride || (len > VGPU98_NOTE_VSUM_MAX) || (len % stride) || ((uint64_t) off + len > size) ||
        (stride != bstride)) {
        r->ta_vsum_refused++;
        vsum_resolve(r, id != 0, id, lock, at);
        return;
    }
    key = id ? id : ((1ull << 32) | (uint32_t) (base + size));
    vsum_resolve(r, 1, key, lock, at);
    prev = vsum_owner(r, base + size, id);
    if (!r->vsum_buf && !(r->vsum_buf = malloc(VGPU98_NOTE_VSUM_MAX)))
        return;
    if (vgpu98_gpu_gpuvm_read(r->gpu, base + off, len, r->vsum_buf, -1)) {
        r->ta_vsum_refused++;
        return;
    }
    r->ta_vsum_checked++;
    r->ta_vsum_bytes += len;
    nv    = len / stride;
    per   = (nv + VGPU98_NOTE_VSUM_PARTS - 1) / VGPU98_NOTE_VSUM_PARTS;
    parts = (nv + per - 1) / per; /* the parts not empty */
    ram   = vsum_parts(want, r->vsum_buf, len, stride, pram, sizeof(pram));
    /* The arena's copy of the same buffer and version, when it holds one:
       what the draw reads. */
    e = res_find(r, base, (uint64_t) base + size, version, &delta);
    if (e && r->res_buf.map && ((uint64_t) delta + off + len <= e->size)) {
        resident = 1;
        arena = vsum_parts(want, (const uint8_t *) r->res_buf.map + e->at + delta + off, len, stride, parena, sizeof(parena));
    }
    if (resident && arena && !ram) {
        r->ta_vsum_arena++;
        tex_find(r, VGPU98_FIND_ARENA, 0,
                 "possible stale arena copy: %s's resident copy (GPU address %08X, version %08X) differs from what the "
                 "DirectX driver drew in parts %s of 8 (%u bytes from byte %u), guest RAM does not; draws read the copy",
                 what, base, version, parena, len, off);
        return;
    }
    if (resident ? !arena : !ram) {
        r->ta_vsum_ram += ram != 0; /* changed in guest RAM after the copy the draw reads was made */
        return;
    }
    /* What the draw reads changed: told now, and held for the Locks after
       it (the oldest held told early to make room). */
    r->ta_vsum_races++;
    r->ta_vsum_whole += (uint32_t) (resident ? arena : ram) >= parts;
    r->ta_vsum_reused += prev != 0;
    owner[0] = 0;
    if (prev)
        snprintf(owner, sizeof(owner), "; until lately, buffer %06X's memory", prev);
    for (i = 0; (i < VSUM_PENDING) && !q; i++) {
        if (!r->vsum_pend[i].used)
            q = &r->vsum_pend[i];
    }
    if (!q) {
        q = &r->vsum_pend[0];
        for (i = 1; i < VSUM_PENDING; i++) {
            if ((int32_t) (r->vsum_pend[i].notes - q->notes) < 0)
                q = &r->vsum_pend[i];
        }
        r->ta_vsum_early++;
        vsum_tell(r, q, 1);
    }
    vsum_locks(lock, before, sizeof(before));
    memset(q, 0, sizeof(*q));
    q->used  = 1;
    q->key   = key;
    q->until = known;
    q->seq   = p[1];
    q->draw  = r->vsum_draw;
    q->frame = r->flip_frame;
    q->notes = r->note_count;
    q->lock  = lock;
    memcpy(q->proc, r->proc_name, sizeof(q->proc));
    q->named = tex_find(r, VGPU98_FIND_RACE, 0,
                        "possible vertex race %u: %s changed between the driver's call and the GPU's read in parts %s of 8 (%u "
                        "bytes from byte %u, GPU address %08X, stride %u, version %08X, buffer %06X%s; drawn from %s); Locks "
                        "before the draw: %s; those after it: a later line",
                        p[1], what, resident ? parena : pram, len, off, base, stride, version, id, owner,
                        !resident ? "guest RAM" : ram ? "the arena's copy, guest RAM too" : "the arena's copy", before);
}

void
vgpu98_r3d_note(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len, uint64_t at, uint64_t known)
{
    uint32_t h = VGPU98_NOTE_HASH_SEED;
    uint32_t want;
    uint32_t i;

    if (!r || (len < 3))
        return;
    if (p[0] == VGPU98_NOTE_VSUM)
        want = VGPU98_LEN_NOTE_VSUM;
    else if (p[0] == VGPU98_NOTE_PROC)
        want = VGPU98_LEN_NOTE_PROC;
    else
        return;
    if (len != want)
        return;
    for (i = 0; i < len - 1; i++)
        h = (h ^ p[i]) * VGPU98_NOTE_HASH_PRIME;
    /* A note checks out, and is newer than the last (a NOP padding the ring
       may hold an old one); those that do not are counted. */
    if ((h != p[len - 1]) || !p[1] || (r->note_seq && ((int32_t) (p[1] - r->note_seq) <= 0))) {
        r->ta_note_refused++;
        return;
    }
    r->note_seq = p[1];
    r->note_count++;
    if (p[0] == VGPU98_NOTE_VSUM)
        vsum_check(r, p, at, known);
    else {
        /* PROC: the drawing process's executable, 16 characters at most,
           as printable characters (the findings' lines name it). */
        for (i = 0; i < 16; i++) {
            const char c = (char) (p[2 + i / 4] >> (8 * (i % 4)));

            if (!c)
                break;
            r->proc_name[i] = ((c >= 0x20) && (c < 0x7f)) ? c : '?';
        }
        r->proc_name[i] = 0;
    }
}

/* A draw dropped without an error, counted by reason. */
#define DROP(why)                     \
    do {                              \
        r->stats->drops[(why)]++;     \
        return 0;                     \
    } while (0)

static int
pkt_draw(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len)
{
    vgpu98_vk_t                  *vk  = r->vk;
    const VkPhysicalDeviceLimits *lim = &vk->props.limits;
    vgpu98_vin_draw_t             d;
    r3d_sub_t                    *color;
    r3d_sub_t                    *depth;
    r3d_image_t                  *tex[N_STAGES];
    uint32_t                      flags[N_STAGES];
    pipe_key_t                    key;
    VkPipeline                    pipe;
    VkRect2D                      sc;
    VkViewport                    vp;
    draw_mem_t                    m;
    r3d_vubo_t                   *vu;
    VkDescriptorSet               sets[2];  /* 0: the textures' and uniforms' (perhaps reused), 1: the draw's buffers */
    int                           new0 = 0; /* set 0 is new: written in full ... */
    set_cache_t                  *sc0  = NULL; /* ... and published there */
    VkDescriptorSetAllocateInfo   ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorImageInfo         ii[5 * N_STAGES];
    VkDescriptorBufferInfo        bi[N_MEM];
    VkDescriptorBufferInfo        res_bi;
    VkWriteDescriptorSet          wr[14];
    uint32_t                      dyn[2];
    r3d_prog_t                   *vprog; /* 2.2: the bound programs, or NULL (their copies recorded) */
    r3d_prog_t                   *pprog;
    r3d_up_t                     *up[3];    /* their copies' records: the vertex program, the pixel program, the vertex constants */
    int                           up_mem[3];
    int                           up_need[3];
    int                           up_reuse[3];
    uint64_t                      up_gen[3];
    const void                   *up_src[3];
    VkDeviceSize                  up_size[3];
    uint32_t                      words;
    uint32_t                      nprims;
    uint32_t                      nvx;
    uint32_t                      fw, fh;
    int                           shape;
    int                           program;
    int                           indexed;
    int                           direct; /* straight from the arena of resident buffers */
    res_t                        *ibe;    /* the index buffer's record there, or NULL */
    int                           pal_need;  /* indexed vertex blending (2.1): the palette ... */
    int                           pal_reuse; /* ... copied for an earlier draw already */
    int                           no_alpha;
    int                           zmode;
    int                           zuse;
    int                           err;
    int                           s;

    r->vsum_draw++; /* the vertex race detector: the notes from here are the next draw's */
    if ((err = vgpu98_vin_check(&r->vin, p, len, &d)))
        return err;
    if (!r->have_target)
        return VGPU98_ERR_ARGUMENT;
    /* 2.2: a vertex program reads a declaration of INPUT elements, the
       fixed function one without them. */
    if ((r->bind_v != 0) != (r->vin.program == 2))
        return VGPU98_ERR_ARGUMENT;
    vprog   = r->bind_v ? &r->progs[r->bind_v] : NULL;
    pprog   = r->bind_p ? &r->progs[r->bind_p] : NULL;
    program = r->vin.program;
    shape   = draw_shape(r, &d);
    /* The buffers the arena holds; a draw straight from it reads no index
       on the CPU and copies no vertex. */
    direct = res_draw(r, &d, program, shape, &ibe);
    if (!direct && (err = vgpu98_vin_window(&r->vin, &d)))
        return err;
    nprims = vgpu98_vin_nprims(&d);
    r->stats->draws++;
    if (program == 1) {
        r->stats->draws_fixed++;
        count_fixed(r);
    } else if (program == 2)
        r->stats->draws_fixed++; /* transformed, by its program (2.2) */
    else
        r->stats->draws_pre++;
    if (d.prim >= VGPU98_PRIM_TRIANGLES)
        r->stats->triangles += d.ntri;
    else if (d.prim == VGPU98_PRIM_POINTS)
        r->stats->points += d.count;
    else
        r->stats->lines += nprims;
    r->stats->vertices += d.nverts;
    if (d.indexed)
        r->stats->indices += d.count;
    if (r->broken)
        DROP(VGPU98_DROP_STOPPED);

    indexed = (shape == SHAPE_PLAIN) && d.indexed;
    switch (shape) {
        case SHAPE_WRAP:
            nvx = 3 * nprims;
            r->stats->shape_wrap++;
            break;
        case SHAPE_POINTS:
            nvx = 3 * nprims;
            r->stats->shape_points++;
            break;
        case SHAPE_WIRE:
            nvx = 3 * BAND_VERTICES * nprims;
            r->stats->shape_wire++;
            break;
        case SHAPE_LINE:
            nvx = BAND_VERTICES * nprims;
            r->stats->shape_lines++;
            break;
        case SHAPE_SPRITE:
            nvx = SPRITE_VERTICES * nprims;
            r->stats->shape_sprites++;
            break;
        case SHAPE_FILLSPRITE:
            nvx = 3 * SPRITE_VERTICES * nprims;
            r->stats->shape_sprites++;
            break;
        default:
            nvx = d.count;
            break;
    }
    /* The summary's command set 2.1 counts: streams read, the palette,
       tweening, per-vertex point sizes. */
    {
        uint32_t n = 0;
        uint32_t used;

        for (used = r->vin.streams_used; used; used &= used - 1)
            n++;
        r->stats->draws_streams[(n <= 2) ? ((n == 2) ? 1 : 0) : ((n <= 4) ? 2 : 3)]++;
        /* summary streams: their sources, strides and elements, the index
           size, how many of those at GPU addresses the arena holds. */
        {
            uint32_t gpuvm = 0;
            uint32_t k;

            for (k = 0; k < VGPU98_MAX_STREAMS; k++) {
                const uint32_t st = r->vin.stream[k].stride;

                if (!((r->vin.streams_used >> k) & 1))
                    continue;
                if (r->vin.stream[k].source == VGPU98_SRC_GPUVM) {
                    r->stats->streams_gpuvm++;
                    gpuvm |= 1u << k;
                } else
                    r->stats->streams_inline++;
                r->stats->stream_strides[(st <= 16) ? 0 : (st <= 32) ? 1 : (st <= 64) ? 2 : 3]++;
            }
            for (k = 0; k < r->vin.n_elements; k++) {
                const uint32_t type = VGPU98_DECL_TYPE(r->vin.decl[2 * k]);

                if (type < 9)
                    r->stats->elements[type]++;
            }
            if (d.indexed) {
                const int wide = r->vin.index.isize == 4;

                r->stats->draws_idx[wide]++;
                if (r->vin.index.source == VGPU98_SRC_INLINE)
                    r->stats->draws_idx_inline[wide]++;
            }
            if (gpuvm)
                r->stats->draws_res[((r->vin.res_mask & gpuvm) == gpuvm) ? 0 : (r->vin.res_mask & gpuvm) ? 1 : 2]++;
        }
        if ((program == 1) && r->rs[VGPU98_RS_INDEXEDVERTEXBLENDENABLE] && r->rs[VGPU98_RS_VERTEXBLEND] &&
            (r->rs[VGPU98_RS_VERTEXBLEND] != VGPU98_VBF_TWEENING)) {
            r->stats->draws_palette++;
            /* Without indices, a draw that blends (weights, or one indexed
               matrix): those without weights blend nothing. */
            if (!r->vin.slot[VGPU98_SLOT_INDICES].on &&
                (r->vin.weights || (r->rs[VGPU98_RS_VERTEXBLEND] == VGPU98_VBF_0WEIGHTS)))
                r->stats->draws_palette_noidx++;
        }
        if ((program == 1) && (r->rs[VGPU98_RS_VERTEXBLEND] == VGPU98_VBF_TWEENING))
            r->stats->draws_tween++;
        if ((program != 2) && r->vin.slot[VGPU98_SLOT_PSIZE].on)
            r->stats->draws_psize++;
    }
    if (r->rs[VGPU98_RS_ALPHABLENDENABLE] && (r->rs[VGPU98_RS_BLENDOP] != VGPU98_BLENDOP_ADD))
        r->stats->draws_blendop++;
    if ((r->rs[VGPU98_RS_COLORWRITEENABLE] & 0xfu) != 0xfu)
        r->stats->draws_masked++;
    if (!nvx)
        return 0;

    zuse = r->have_depth ? (int) r->rs[VGPU98_RS_ZENABLE] : 0;
    if (!prepare_draw(r, &color, &depth, tex, flags, 0, 0, stage_mask(r), zuse))
        DROP(r->drop_why);
    if (r->slot[r->cur].sets + 2 > 2 * SETS_PER_SLOT) {
        /* The descriptor pool is used up: carry on in the next slot. */
        if (!submit(r, 0))
            DROP(VGPU98_DROP_HOST);
        if (!prepare_draw(r, &color, &depth, tex, flags, 0, 0, stage_mask(r), zuse))
            DROP(r->drop_why);
    }
    /* The texture audit's stages drawn, by format and conversion. */
    if (tex_audit_on(r)) {
        for (s = 0; s < N_STAGES; s++) {
            if (tex[s]) {
                r->ta_fmt[tex[s]->format]++;
                r->ta_conv[tex[s]->hf->conv]++;
            } else if (r->vol_bound[s]) {
                r->ta_fmt[r->vol_bound[s]->format]++;
                r->ta_conv[r->vol_bound[s]->hf->conv]++;
            }
        }
    }
    fw = depth && (depth->width < color->width) ? depth->width : color->width;
    fh = depth && (depth->height < color->height) ? depth->height : color->height;
    if (!scissor(r, fw, fh, &sc))
        DROP(VGPU98_DROP_SCISSOR);

    zmode    = depth ? (int) r->rs[VGPU98_RS_ZENABLE] : 0;
    no_alpha = target_no_alpha(r);
    if (zmode == 2) {
        r->stats->draws_w++;
        if (depth_float(depth->im->hf))
            r->stats->draws_w_float++;
    }
    memset(&key, 0, sizeof(key));
    key.color_vk = color->im->hf->vk;
    key.depth_vk = depth ? depth->im->hf->vk : VK_FORMAT_UNDEFINED;
    key.shape    = (uint8_t) shape;
    key.program  = (uint8_t) program;
    key.pixel_program = (uint8_t) (pprog != NULL);
    key.topology = (uint8_t) draw_topology(shape, d.prim);
    key.cull     = VK_CULL_MODE_NONE;
    if (((shape == SHAPE_PLAIN) || (shape == SHAPE_WRAP)) && (d.prim >= VGPU98_PRIM_TRIANGLES)) {
        if (r->rs[VGPU98_RS_CULLMODE] == 2)
            key.cull = VK_CULL_MODE_FRONT_BIT;
        else if (r->rs[VGPU98_RS_CULLMODE] == 3)
            key.cull = VK_CULL_MODE_BACK_BIT;
    }
    /* The fragment shader computes the depth for W-buffering, for Z on the
       guest buffer's grid in a float image, and with a ZBIAS for points
       (the host GPU biases only polygons) and for a float image (whose bias
       steps are not the guest buffer's). */
    key.computed_depth = (uint8_t) (zmode && ((zmode == 2) || depth_grid(r, depth->im) ||
                                              (r->rs[VGPU98_RS_ZBIAS] &&
                                               ((key.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST) ||
                                                depth_float(depth->im->hf))) ||
                                              /* 2.2: a pixel program's own depth */
                                              (pprog && (pprog->info.flags & VGPU98_PROGF_DEPTH))));
    key.blend          = (uint8_t) (r->rs[VGPU98_RS_ALPHABLENDENABLE] != 0);
    if (key.blend) {
        uint32_t src = r->rs[VGPU98_RS_SRCBLEND];
        uint32_t dst = r->rs[VGPU98_RS_DESTBLEND];

        if (src == 12) { /* both source alpha */
            src = 5;
            dst = 6;
        } else if (src == 13) { /* both inverse source alpha */
            src = 6;
            dst = 5;
        }
        key.src = (uint8_t) blend_factor(src, no_alpha);
        key.dst = (uint8_t) blend_factor(dst, no_alpha);
        /* 2.1's BLENDOP (add on a 2.0 GPU); min and max take no factors. */
        key.blend_op = (uint8_t) blend_op(r->rs[VGPU98_RS_BLENDOP]);
        if ((key.blend_op == VK_BLEND_OP_MIN) || (key.blend_op == VK_BLEND_OP_MAX))
            key.src = key.dst = VK_BLEND_FACTOR_ONE;
    } else {
        key.src = VK_BLEND_FACTOR_ONE;
        key.dst = VK_BLEND_FACTOR_ZERO;
    }
    /* 2.1's COLORWRITEENABLE (all four on a 2.0 GPU): its bits are
       Vulkan's. */
    key.write_mask = (uint8_t) (r->rs[VGPU98_RS_COLORWRITEENABLE] & 0xfu);
    key.depth_test  = (uint8_t) (zmode != 0);
    key.depth_write = (uint8_t) (zmode && r->rs[VGPU98_RS_ZWRITEENABLE]);
    key.depth_func  = (uint8_t) compare_op(r->rs[VGPU98_RS_ZFUNC]);
    key.stencil     = (uint8_t) (depth && fmt_has_stencil(depth->im->format) && r->rs[VGPU98_RS_STENCILENABLE]);
    if (key.stencil) {
        key.s_fail  = (uint8_t) stencil_op(r->rs[VGPU98_RS_STENCILFAIL]);
        key.s_zfail = (uint8_t) stencil_op(r->rs[VGPU98_RS_STENCILZFAIL]);
        key.s_pass  = (uint8_t) stencil_op(r->rs[VGPU98_RS_STENCILPASS]);
        key.s_func  = (uint8_t) compare_op(r->rs[VGPU98_RS_STENCILFUNC]);
    }
    pipe = pipe_get(r, &key, pass_get(r, color->im->hf->vk, depth ? depth->im->hf->vk : VK_FORMAT_UNDEFINED));
    if (!pipe)
        DROP(VGPU98_DROP_PIPELINE);

    /* The vertices' bytes (those in main memory read now, but those the
       arena holds), the primitive or index list, the uniforms. A draw
       straight from the arena numbers its vertices as the GPU does: from 0,
       every stream whole. */
    if (direct) {
        d.first  = 0;
        d.nverts = VGPU98_VERTEX_LIMIT;
    }
    if ((err = vgpu98_vin_layout(&r->vin, &d, lim->maxStorageBufferRange / 4, &words)))
        return err;
    if ((r->vin.res_mask || r->res_ib) && vgpu98_gpu_logging(r->gpu, VGPU98_LOG_AUDIT))
        res_check_draw(r, &d);
    if (tex_audit_on(r)) {
        coord_check(r, &d, tex, direct);
        shown_draw(r);
    }
    if (r->vin.past && !direct)
        note_past(r, p, &d);
    if (!direct) {
        uint32_t k;

        for (k = 0; k < VGPU98_MAX_STREAMS; k++) {
            if ((r->vin.res_mask >> k) & 1)
                r->stats->res_saved_bytes += (uint64_t) r->vin.copy[k].words * 4;
        }
    }
    memset(&m, 0, sizeof(m));
    m.which[MEM_VERTS] = 1;
    m.size[MEM_VERTS]  = (VkDeviceSize) (words ? words : 4) * 4;
    m.align[MEM_VERTS] = align_max(lim->minStorageBufferOffsetAlignment, 16);
    m.which[MEM_LIST]  = 1;
    m.size[MEM_LIST]   = (shape != SHAPE_PLAIN) ? (VkDeviceSize) nprims * 16 : (indexed && !direct) ? (VkDeviceSize) d.count * 4 : 16;
    m.align[MEM_LIST]  = align_max(lim->minStorageBufferOffsetAlignment, 16);
    m.which[MEM_VUBO]  = 2;
    m.size[MEM_VUBO]   = sizeof(r3d_vubo_t);
    m.align[MEM_VUBO]  = lim->minUniformBufferOffsetAlignment;
    m.which[MEM_UBO]   = 2;
    m.size[MEM_UBO]    = sizeof(r3d_ubo_t);
    m.align[MEM_UBO]   = lim->minUniformBufferOffsetAlignment;
    /* Indexed vertex blending (2.1): the palette's entries 0 to pal_max + 1,
       copied once per submission while they stay the same. */
    pal_need = (program == 1) && r->rs[VGPU98_RS_INDEXEDVERTEXBLENDENABLE] && r->rs[VGPU98_RS_VERTEXBLEND] &&
               (r->rs[VGPU98_RS_VERTEXBLEND] != VGPU98_VBF_TWEENING);
    if (pal_need && r->pal_dirty)
        pal_update(r);
    pal_reuse = pal_need && r->pal_up_buf && (r->pal_up_gen == r->pal_gen) && (r->pal_up_serial == r->serial);
    if (pal_need && !pal_reuse) {
        m.which[MEM_PAL] = 1;
        m.size[MEM_PAL]  = (VkDeviceSize) (r->pal_max + 2) * sizeof(r3d_pal_t);
        m.align[MEM_PAL] = align_max(lim->minStorageBufferOffsetAlignment, 16);
    }
    /* 2.2: the bound programs' instructions (each program's own record, so
       draws alternating between programs reuse both) and the vertex
       constants the vertex program reads, copied once per submission while
       they stay the same, as the palette. */
    up[0]      = vprog ? &vprog->up : &r->up_vprog;
    up_mem[0]  = MEM_VPROG;
    up_need[0] = vprog != NULL;
    up_gen[0]  = vprog ? vprog->gen : 0;
    up_src[0]  = vprog ? (const void *) vprog->ins : NULL;
    up_size[0] = vprog ? (VkDeviceSize) vprog->n * VGPU98_PROG_INS_DWORDS * 4 : 0;
    up[1]      = pprog ? &pprog->up : &r->up_pprog;
    up_mem[1]  = MEM_PPROG;
    up_need[1] = pprog != NULL;
    up_gen[1]  = pprog ? pprog->gen : 0;
    up_src[1]  = pprog ? (const void *) pprog->ins : NULL;
    up_size[1] = pprog ? (VkDeviceSize) pprog->n * VGPU98_PROG_INS_DWORDS * 4 : 0;
    up[2]      = &r->up_vconst;
    up_mem[2]  = MEM_VCONST;
    up_need[2] = vprog != NULL;
    up_gen[2]  = r->vconst_gen;
    up_src[2]  = r->vconst;
    up_size[2] = (VkDeviceSize) vconst_count(r, vprog) * 16;
    for (s = 0; s < 3; s++) {
        up_reuse[s] = up_need[s] && up[s]->buf && (up[s]->gen == up_gen[s]) && (up[s]->serial == r->serial) &&
                      (up[s]->size >= up_size[s]);
        if (up_need[s] && !up_reuse[s]) {
            m.which[up_mem[s]] = 1;
            m.size[up_mem[s]]  = up_size[s];
            m.align[up_mem[s]] = align_max(lim->minStorageBufferOffsetAlignment, 16);
        }
    }
    if (!draw_alloc(r, &m))
        DROP(VGPU98_DROP_MEMORY);
    if ((pal_reuse && (r->serial != r->pal_up_serial)) || (up_reuse[0] && (r->serial != up[0]->serial)) ||
        (up_reuse[1] && (r->serial != up[1]->serial)) || (up_reuse[2] && (r->serial != up[2]->serial))) {
        /* The draw's blocks moved on to the next submission: what it would
           have reused goes there too, all of it, so another move on this
           try leaves nothing behind. */
        if (pal_need) {
            pal_reuse        = 0;
            m.which[MEM_PAL] = 1;
            m.size[MEM_PAL]  = (VkDeviceSize) (r->pal_max + 2) * sizeof(r3d_pal_t);
            m.align[MEM_PAL] = align_max(lim->minStorageBufferOffsetAlignment, 16);
        }
        for (s = 0; s < 3; s++) {
            if (!up_need[s])
                continue;
            up_reuse[s]        = 0;
            m.which[up_mem[s]] = 1;
            m.size[up_mem[s]]  = up_size[s];
            m.align[up_mem[s]] = align_max(lim->minStorageBufferOffsetAlignment, 16);
        }
        if (!draw_alloc(r, &m))
            DROP(VGPU98_DROP_MEMORY);
    }
    for (s = 0; s < 3; s++) {
        if (!m.which[up_mem[s]])
            continue;
        memcpy(m.p[up_mem[s]], up_src[s], (size_t) up_size[s]);
        up[s]->gen    = up_gen[s];
        up[s]->serial = r->serial;
        up[s]->buf    = m.buf[up_mem[s]];
        up[s]->off    = m.off[up_mem[s]];
        up[s]->size   = up_size[s];
    }
    if (m.which[MEM_PAL]) {
        memcpy(m.p[MEM_PAL], r->pal_ff, (size_t) m.size[MEM_PAL]);
        r->pal_up_gen    = r->pal_gen;
        r->pal_up_serial = r->serial;
        r->pal_up_buf    = m.buf[MEM_PAL];
        r->pal_up_off    = m.off[MEM_PAL];
        r->pal_up_size   = m.size[MEM_PAL];
    }
    if (!r->in_pass && !pass_begin(r, color, depth))
        DROP(VGPU98_DROP_HOST);
    vu = m.p[MEM_VUBO];
    fill_vubo(r, vu, &d, fw, fh);
    vgpu98_vin_copy(&r->vin, m.p[MEM_VERTS], vu->slot);
    r->stats->vertex_copy_bytes += (uint64_t) words * 4;
    if (shape != SHAPE_PLAIN)
        vgpu98_vin_prims(&d, r->rs[VGPU98_RS_LASTPIXEL] != 0, m.p[MEM_LIST]);
    else if (indexed && !direct)
        memcpy(m.p[MEM_LIST], d.local, (size_t) d.count * 4);
    else
        memset(m.p[MEM_LIST], 0, 16);

    /* Border colours: the sampler's when the host has custom ones, else
       the shader's. */
    {
        samp_key_t sk[N_STAGES];
        const int  stages_on = active_stages(r);

        for (s = 0; s < N_STAGES; s++) {
            const uint32_t *t      = r->tss[s];
            uint32_t        maxlev = t[VGPU98_TSS_MAXMIPLEVEL];
            uint32_t        lev    = tex[s] ? tex[s]->levels : r->vol_bound[s] ? r->vol_bound[s]->levels : 1;
            int             aniso;

            memset(&sk[s], 0, sizeof(samp_key_t));
            aniso = ((t[VGPU98_TSS_MAGFILTER] == 5) || (t[VGPU98_TSS_MINFILTER] == 3)) && vk->anisotropy;
            sk[s].mag    = (uint8_t) (t[VGPU98_TSS_MAGFILTER] != 1);
            sk[s].min    = (uint8_t) (t[VGPU98_TSS_MINFILTER] != 1);
            sk[s].mip    = (uint8_t) (t[VGPU98_TSS_MIPFILTER] == 3);
            sk[s].aniso  = (uint8_t) (aniso ? ((t[VGPU98_TSS_MAXANISOTROPY] < lim->maxSamplerAnisotropy) ?
                                                   t[VGPU98_TSS_MAXANISOTROPY] : (uint32_t) lim->maxSamplerAnisotropy) : 1);
            if (maxlev >= lev)
                maxlev = lev - 1;
            /* The summary's textures: what the game asks of each stage (the
               bump stages among them; tex[] is set for active stages only). */
            if (tex[s] || r->vol_bound[s]) {
                const uint32_t gen   = program ? ((t[VGPU98_TSS_TEXCOORDINDEX] >> VGPU98_TCI_GEN_SHIFT) & 3u) : 0;
                const uint32_t count = t[VGPU98_TSS_TEXTURETRANSFORMFLAGS] & VGPU98_TTFF_COUNT_MASK;
                const uint32_t cop   = t[VGPU98_TSS_COLOROP];

                r->stats->tex_uses++;
                r->stats->tex_src[gen]++;
                if (program && count)
                    r->stats->tex_xform++;
                if ((t[VGPU98_TSS_TEXTURETRANSFORMFLAGS] & VGPU98_TTFF_PROJECTED) && (count >= 2))
                    r->stats->tex_projected++;
                if (cop == VGPU98_TOP_DOTPRODUCT3)
                    r->stats->tex_dot3++;
                else if ((cop == VGPU98_TOP_BUMPENVMAP) || (cop == VGPU98_TOP_BUMPENVMAPLUMINANCE))
                    r->stats->tex_bumpenv++;
                if (((t[VGPU98_TSS_MAGFILTER] == 5) || (t[VGPU98_TSS_MINFILTER] == 3)) && (t[VGPU98_TSS_MAXANISOTROPY] > 1))
                    r->stats->tex_aniso++;
                if ((t[VGPU98_TSS_MIPFILTER] == 1) || (lev == 1))
                    r->stats->tex_nomip++;
                else if (t[VGPU98_TSS_MIPFILTER] == 3)
                    r->stats->tex_trilinear++;
                if (u2f(t[VGPU98_TSS_MIPMAPLODBIAS]) != 0.0f)
                    r->stats->tex_lodbias++;
            }
            sk[s].min_lod = (float) maxlev;
            sk[s].max_lod = (t[VGPU98_TSS_MIPFILTER] == 1) ? (float) maxlev : (float) (lev - 1);
            sk[s].addr_u  = (uint8_t) t[VGPU98_TSS_ADDRESSU];
            sk[s].addr_v  = (uint8_t) t[VGPU98_TSS_ADDRESSV];
            if (flags[s] & SF_VOLUME)
                sk[s].addr_w = (uint8_t) t[VGPU98_TSS_ADDRESSW]; /* 2.1 */
            if ((sk[s].addr_u == 4) || (sk[s].addr_v == 4) || (sk[s].addr_w == 4)) {
                if (vk->border_custom && (r->n_custom_borders < MAX_CUSTOM_BORDERS)) {
                    sk[s].border       = 1;
                    sk[s].border_color = t[VGPU98_TSS_BORDERCOLOR];
                } else {
                    if (sk[s].addr_u == 4)
                        flags[s] |= SF_BORDER_U;
                    if (sk[s].addr_v == 4)
                        flags[s] |= SF_BORDER_V;
                    if (sk[s].addr_w == 4)
                        flags[s] |= SF_BORDER_W;
                    if (sk[s].addr_u == 4)
                        sk[s].addr_u = 3;
                    if (sk[s].addr_v == 4)
                        sk[s].addr_v = 3;
                    if (sk[s].addr_w == 4)
                        sk[s].addr_w = 3;
                }
            }
            /* Bindings 1 and 2 hold 2D textures and their key companions,
               6 and 7 cube maps and theirs, 9 volumes (2.1); a stage's
               others hold the dummies. With COLORKEYBLENDENABLE the zeroed
               texture stands in for the texture. */
            {
                const int   cube = (flags[s] & SF_CUBE) != 0;
                VkImageView view = !tex[s] ? VK_NULL_HANDLE : (flags[s] & SF_KEYZ) ? tex[s]->kz_view : tex[s]->view;
                VkImageView kv   = (flags[s] & SF_KEYED) ? tex[s]->key_view : VK_NULL_HANDLE;
                int         k;

                VkSampler smp = sampler_get(r, &sk[s]);

                if (!smp)
                    DROP(VGPU98_DROP_HOST);
                for (k = 0; k < 5; k++) {
                    ii[k * N_STAGES + s].sampler     = smp;
                    ii[k * N_STAGES + s].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                }
                ii[s].imageView                = (!cube && view) ? view : r->dummy_view;
                ii[N_STAGES + s].imageView     = (!cube && kv) ? kv : r->dummy_view;
                ii[2 * N_STAGES + s].imageView = (cube && view) ? view : r->dummy_cube_view;
                ii[3 * N_STAGES + s].imageView = (cube && kv) ? kv : r->dummy_cube_view;
                ii[4 * N_STAGES + s].imageView = r->vol_bound[s] ? r->vol_bound[s]->view : r->dummy_vol_view;
            }
            if (r->vol_bound[s])
                r->stats->tex_volume++;
            if ((s < stages_on) &&
                ((t[VGPU98_TSS_COLOROP] == VGPU98_TOP_MULTIPLYADD) || (t[VGPU98_TSS_COLOROP] == VGPU98_TOP_LERP) ||
                 (t[VGPU98_TSS_ALPHAOP] == VGPU98_TOP_MULTIPLYADD) || (t[VGPU98_TSS_ALPHAOP] == VGPU98_TOP_LERP)))
                r->stats->tex_madd_lerp++;
            if ((s < stages_on) && (t[VGPU98_TSS_RESULTARG] == VGPU98_TA_TEMP))
                r->stats->tex_temp++;
        }
    }
    fill_ubo(r, m.p[MEM_UBO], flags, depth);
    /* A sprite's coordinates span its square: a stage's PROJECTED does not
       divide them (DirectX 8 leaves the pair undefined; a set the vertex
       lacks would divide by 0). */
    if (((shape == SHAPE_SPRITE) || (shape == SHAPE_FILLSPRITE)) && r->rs[VGPU98_RS_POINTSPRITEENABLE]) {
        r3d_ubo_t *fu = m.p[MEM_UBO];

        for (s = 0; s < N_STAGES; s++)
            fu->st[s].ops2[2] &= ~(uint32_t) VGPU98_TTFF_PROJECTED;
    }

    for (s = 0; s < N_MEM; s++) {
        bi[s].buffer = m.buf[s];
        bi[s].offset = ((s == MEM_VUBO) || (s == MEM_UBO)) ? 0 : m.off[s]; /* the uniforms' offsets are dynamic */
        bi[s].range  = m.size[s];
    }
    /* Without indexed blending the shader never reads the palette: the
       primitive list stands in for it. */
    if (pal_reuse) {
        bi[MEM_PAL].buffer = r->pal_up_buf;
        bi[MEM_PAL].offset = r->pal_up_off;
        bi[MEM_PAL].range  = r->pal_up_size;
    } else if (!m.which[MEM_PAL])
        bi[MEM_PAL] = bi[MEM_LIST];
    /* 2.2: the programs' copies; without one the primitive list stands in. */
    for (s = 0; s < 3; s++) {
        if (up_reuse[s]) {
            bi[up_mem[s]].buffer = up[s]->buf;
            bi[up_mem[s]].offset = up[s]->off;
            bi[up_mem[s]].range  = up[s]->size;
        } else if (!m.which[up_mem[s]])
            bi[up_mem[s]] = bi[MEM_LIST];
    }
    /* Set 0, an earlier draw's when it binds the same. */
    {
        set_key_t    sk0;
        set_cache_t *sc;

        memset(&sk0, 0, sizeof(sk0));
        for (s = 0; s < 5 * N_STAGES; s++)
            sk0.view[s] = ii[s].imageView;
        for (s = 0; s < N_STAGES; s++)
            sk0.samp[s] = ii[s].sampler;
        sk0.ubuf  = m.buf[MEM_UBO];
        sk0.arena = r->res_buf.buffer;
        sc        = &r->set_cache[key_hash(&sk0, sizeof(sk0)) % SET_CACHE];
        if (sc->valid && (sc->slot == r->cur) && (sc->gen == r->slot[r->cur].gen) && !memcmp(&sc->key, &sk0, sizeof(sk0))) {
            sets[0] = sc->set;
            r->stats->sets_reused++;
        } else {
            ai.descriptorPool     = r->slot[r->cur].pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &r->dsl;
            if (vk->vkAllocateDescriptorSets(vk->dev, &ai, &sets[0]) != VK_SUCCESS) {
                r3d_fail(r, "vkAllocateDescriptorSets", VK_ERROR_OUT_OF_POOL_MEMORY);
                DROP(VGPU98_DROP_MEMORY);
            }
            new0     = 1;
            sc0      = sc;
            sc->valid = 0; /* published once written */
            sc->key  = sk0;
            sc->set  = sets[0];
            sc->slot = r->cur;
            sc->gen  = r->slot[r->cur].gen;
        }
        ai.descriptorPool     = r->slot[r->cur].pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &r->dsl_draw;
        if (vk->vkAllocateDescriptorSets(vk->dev, &ai, &sets[1]) != VK_SUCCESS) {
            r3d_fail(r, "vkAllocateDescriptorSets", VK_ERROR_OUT_OF_POOL_MEMORY);
            DROP(VGPU98_DROP_MEMORY);
        }
    }
    memset(wr, 0, sizeof(wr));
    for (s = 0; s < 14; s++) {
        wr[s].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[s].dstSet          = ((s == 4) || (s == 5) || (s == 8) || (s >= 11)) ? sets[1] : sets[0];
        wr[s].dstBinding      = (uint32_t) s;
        wr[s].descriptorCount = 1;
    }
    wr[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    wr[0].pBufferInfo     = &bi[MEM_UBO];
    wr[1].descriptorCount = N_STAGES;
    wr[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[1].pImageInfo      = ii;
    wr[2].descriptorCount = N_STAGES;
    wr[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[2].pImageInfo      = ii + N_STAGES;
    wr[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    wr[3].pBufferInfo     = &bi[MEM_VUBO];
    wr[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[4].pBufferInfo     = &bi[MEM_VERTS];
    wr[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[5].pBufferInfo     = &bi[MEM_LIST];
    wr[6].descriptorCount = N_STAGES;
    wr[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[6].pImageInfo      = ii + 2 * N_STAGES;
    wr[7].descriptorCount = N_STAGES;
    wr[7].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[7].pImageInfo      = ii + 3 * N_STAGES;
    wr[8].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[8].pBufferInfo     = &bi[MEM_PAL];
    wr[9].descriptorCount = N_STAGES;
    wr[9].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[9].pImageInfo      = ii + 4 * N_STAGES;
    /* The arena (the primitive list stands in for it until there is one). */
    if (r->res_buf.buffer) {
        res_bi.buffer = r->res_buf.buffer;
        res_bi.offset = 0;
        res_bi.range  = VK_WHOLE_SIZE;
    } else
        res_bi = bi[MEM_LIST];
    wr[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[10].pBufferInfo    = &res_bi;
    /* 2.2: the vertex program, the pixel program, the vertex constants. */
    wr[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[11].pBufferInfo    = &bi[MEM_VPROG];
    wr[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[12].pBufferInfo    = &bi[MEM_PPROG];
    wr[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[13].pBufferInfo    = &bi[MEM_VCONST];
    r->slot[r->cur].sets += new0 ? 2 : 1;
    if (new0) {
        vk->vkUpdateDescriptorSets(vk->dev, 14, wr, 0, NULL);
        sc0->valid = 1;
    } else {
        /* Set 0 holds its writes already: only set 1's. */
        VkWriteDescriptorSet w1[6];

        w1[0] = wr[4];
        w1[1] = wr[5];
        w1[2] = wr[8];
        w1[3] = wr[11];
        w1[4] = wr[12];
        w1[5] = wr[13];
        vk->vkUpdateDescriptorSets(vk->dev, 6, w1, 0, NULL);
    }

    if (r->bound_pipe != pipe) {
        vk->vkCmdBindPipeline(cmd(r), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        r->bound_pipe = pipe;
    }
    /* Dynamic offsets in binding order: the fragment uniforms (0), the
       vertex uniforms (3). */
    dyn[0] = (uint32_t) m.off[MEM_UBO];
    dyn[1] = (uint32_t) m.off[MEM_VUBO];
    vk->vkCmdBindDescriptorSets(cmd(r), VK_PIPELINE_BIND_POINT_GRAPHICS, r->layout, 0, 2, sets, 2, dyn);
    /* Transformed vertices take ZRANGE as the depth range; pre-transformed
       ones carry their screen z. A computed depth (W itself, Z on a grid)
       may fall outside ZRANGE, and Vulkan leaves a fragment's depth
       outside the viewport's range undefined without depth clamping: its
       pipelines keep 0-1, and the fragment shader applies ZRANGE to z. */
    vp.x        = 0.0f;
    vp.y        = 0.0f;
    vp.width    = (float) fw;
    vp.height   = (float) fh;
    vp.minDepth = (program && !key.computed_depth) ? r->zrange[0] : 0.0f;
    vp.maxDepth = (program && !key.computed_depth) ? r->zrange[1] : 1.0f;
    vk->vkCmdSetViewport(cmd(r), 0, 1, &vp);
    vk->vkCmdSetScissor(cmd(r), 0, 1, &sc);
    vk->vkCmdSetStencilCompareMask(cmd(r), VK_STENCIL_FACE_FRONT_AND_BACK, r->rs[VGPU98_RS_STENCILMASK] & 0xff);
    vk->vkCmdSetStencilWriteMask(cmd(r), VK_STENCIL_FACE_FRONT_AND_BACK, r->rs[VGPU98_RS_STENCILWRITEMASK] & 0xff);
    vk->vkCmdSetStencilReference(cmd(r), VK_STENCIL_FACE_FRONT_AND_BACK, r->rs[VGPU98_RS_STENCILREF] & 0xff);
    /* ZBIAS: n steps of a fixed-point depth buffer toward the viewer (with
       a computed depth, the fragment shader applies it). */
    vk->vkCmdSetDepthBias(cmd(r), key.computed_depth ? 0.0f : -(float) r->rs[VGPU98_RS_ZBIAS], 0.0f, 0.0f);
    res_stamp(r);
    if (direct)
        r->stats->draws_direct++;
    if (direct && indexed) {
        /* The arena's indices, the draw's base added by the host. */
        vk->vkCmdBindIndexBuffer(cmd(r), r->res_buf.buffer, r->res_ib_at,
                                 (r->vin.index.isize == 2) ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        vk->vkCmdDrawIndexed(cmd(r), d.count, 1, d.start, d.base, 0);
    } else if (direct)
        vk->vkCmdDraw(cmd(r), d.count, 1, d.start, 0);
    else if (indexed) {
        vk->vkCmdBindIndexBuffer(cmd(r), m.buf[MEM_LIST], m.off[MEM_LIST], VK_INDEX_TYPE_UINT32);
        vk->vkCmdDrawIndexed(cmd(r), d.count, 1, 0, 0, 0);
    } else
        vk->vkCmdDraw(cmd(r), nvx, 1, 0, 0);

    /* What changed: the scissor, or for pre-transformed vertices, their
       bounds within it. */
    {
        uint32_t x0 = (uint32_t) sc.offset.x, y0 = (uint32_t) sc.offset.y;
        uint32_t x1 = x0 + sc.extent.width, y1 = y0 + sc.extent.height;

        if (!program)
            screen_bounds(r, &d, ((shape == SHAPE_SPRITE) || (shape == SHAPE_FILLSPRITE)) ? point_reach(r) : 0, &x0, &y0,
                          &x1, &y1);
        if ((x0 < x1) && (y0 < y1)) {
            image_add_dirty(r, color, x0, y0, x1, y1);
            if (depth && (key.depth_write || key.stencil))
                image_add_dirty(r, depth, x0, y0, x1, y1);
        }
    }
    /* A float image holds what the last draw writing its depth wrote. */
    if (key.depth_write)
        depth->im->depth_w = (zmode == 2);
    if (r->census_on)
        census_draw(r, &d);
    return 0;
}

#undef DROP

static int
pkt_clear(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len)
{
    vgpu98_vk_t      *vk    = r->vk;
    uint32_t          flags = p[0];
    const uint32_t    count = p[4];
    r3d_sub_t        *color;
    r3d_sub_t        *depth;
    r3d_image_t      *tex[N_STAGES];
    uint32_t          tflags[N_STAGES];
    VkClearAttachment ca[2];
    VkClearRect      *cr;
    uint32_t          ncr = 0;
    uint32_t          nca = 0;
    uint32_t          fw, fh;
    uint32_t          i;
    int               full_color = 0;
    int               full_depth = 0;
    int               zuse;

    if (flags & ~7u)
        return VGPU98_ERR_ARGUMENT;
    if ((uint64_t) len < 5 + 2 * (uint64_t) count)
        return VGPU98_ERR_LENGTH;
    if (!r->have_target)
        return VGPU98_ERR_ARGUMENT;
    if (!r->have_depth)
        flags &= ~(VGPU98_CLEAR_DEPTH | VGPU98_CLEAR_STENCIL);
    else if (!fmt_has_stencil(r->depth.format))
        flags &= ~VGPU98_CLEAR_STENCIL;
    r->stats->clears++;
    if (!flags || !count || r->broken)
        return 0;

    fw = r->have_depth && (r->depth.width < r->target.width) ? r->depth.width : r->target.width;
    fh = r->have_depth && (r->depth.height < r->target.height) ? r->depth.height : r->target.height;
    cr = calloc(count, sizeof(VkClearRect));
    for (i = 0; i < count; i++) {
        uint32_t x0 = p[5 + 2 * i] & 0xffff, y0 = p[5 + 2 * i] >> 16;
        uint32_t x1 = p[6 + 2 * i] & 0xffff, y1 = p[6 + 2 * i] >> 16;

        if (x1 > fw)
            x1 = fw;
        if (y1 > fh)
            y1 = fh;
        if ((x0 >= x1) || (y0 >= y1))
            continue;
        if ((x0 == 0) && (y0 == 0) && (x1 == r->target.width) && (y1 == r->target.height))
            full_color = 1;
        if (r->have_depth && (x0 == 0) && (y0 == 0) && (x1 == r->depth.width) && (y1 == r->depth.height))
            full_depth = 1;
        cr[ncr].rect.offset.x      = (int32_t) x0;
        cr[ncr].rect.offset.y      = (int32_t) y0;
        cr[ncr].rect.extent.width  = x1 - x0;
        cr[ncr].rect.extent.height = y1 - y0;
        cr[ncr].layerCount         = 1;
        ncr++;
    }
    if (!ncr) {
        free(cr);
        return 0;
    }

    /* A clear touches no texture; a clear of a whole surface needs no upload
       first (for depth with stencil, both cleared). */
    zuse = (flags & (VGPU98_CLEAR_DEPTH | VGPU98_CLEAR_STENCIL)) ? (int) r->rs[VGPU98_RS_ZENABLE] : 0;
    if (!prepare_draw(r, &color, &depth, tex, tflags, full_color && (flags & VGPU98_CLEAR_COLOR),
                      full_depth && (flags & VGPU98_CLEAR_DEPTH) &&
                          (!fmt_has_stencil(r->depth.format) || (flags & VGPU98_CLEAR_STENCIL)),
                      0, zuse)) {
        free(cr);
        return 0;
    }

    memset(ca, 0, sizeof(ca));
    if (flags & VGPU98_CLEAR_COLOR) {
        uint32_t c = p[1];

        ca[nca].aspectMask                  = VK_IMAGE_ASPECT_COLOR_BIT;
        ca[nca].colorAttachment             = 0;
        ca[nca].clearValue.color.float32[0] = ((c >> 16) & 255) / 255.0f;
        ca[nca].clearValue.color.float32[1] = ((c >> 8) & 255) / 255.0f;
        ca[nca].clearValue.color.float32[2] = (c & 255) / 255.0f;
        ca[nca].clearValue.color.float32[3] = ((c >> 24) & 255) / 255.0f;
        nca++;
    }
    if (depth && (flags & (VGPU98_CLEAR_DEPTH | VGPU98_CLEAR_STENCIL))) {
        float d = u2f(p[2]);

        if (!(d >= 0.0f))
            d = 0.0f;
        if (d > 1.0f)
            d = 1.0f;
        ca[nca].aspectMask = 0;
        if (flags & VGPU98_CLEAR_DEPTH) {
            ca[nca].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
            /* A clear while depth is on says what the image holds from now:
               W with ZENABLE 2, Z with 1; with 0 it holds what it did. */
            if (zuse)
                depth->im->depth_w = (zuse == 2);
        }
        if (flags & VGPU98_CLEAR_STENCIL)
            ca[nca].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        ca[nca].clearValue.depthStencil.depth   = depth_clear(r, depth->im, d);
        ca[nca].clearValue.depthStencil.stencil = p[3] & 0xff;
        nca++;
    }
    vk->vkCmdClearAttachments(cmd(r), nca, ca, ncr, cr);
    for (i = 0; i < ncr; i++) {
        uint32_t x0 = (uint32_t) cr[i].rect.offset.x, y0 = (uint32_t) cr[i].rect.offset.y;
        uint32_t x1 = x0 + cr[i].rect.extent.width, y1 = y0 + cr[i].rect.extent.height;

        if (flags & VGPU98_CLEAR_COLOR)
            image_add_dirty(r, color, x0, y0, x1, y1);
        if (depth && (flags & (VGPU98_CLEAR_DEPTH | VGPU98_CLEAR_STENCIL)))
            image_add_dirty(r, depth, x0, y0, x1, y1);
    }
    free(cr);
    return 0;
}

int
vgpu98_r3d_packet(vgpu98_r3d_t *r, uint32_t op, const uint32_t *p, uint32_t len)
{
    uint32_t i;
    int      err;

    switch (op) {
        case VGPU98_OP_TARGET:
            return pkt_target(r, p);
        case VGPU98_OP_VIEWPORT:
            if ((p[0] | p[1] | p[2] | p[3]) > 0xffff)
                return VGPU98_ERR_ARGUMENT;
            memcpy(r->vp, p, sizeof(r->vp));
            return 0;
        case VGPU98_OP_WRANGE:
            /* wfar may be +infinity (an infinite far plane: float images
               hold W whatever the range); anything else not finite is
               refused, the old range kept. */
            if (!isfinite(u2f(p[0])) || isnan(u2f(p[1])) || (u2f(p[1]) == -INFINITY))
                return VGPU98_ERR_ARGUMENT;
            wrange_set(r, u2f(p[0]), u2f(p[1]));
            if (isinf(r->wfar))
                r->stats->wrange_infinite++;
            return 0;
        case VGPU98_OP_STATE:
            if (len & 1)
                return VGPU98_ERR_LENGTH;
            for (i = 0; i < len; i += 2) {
                if ((err = set_state(r, p[i], p[i + 1]))) {
                    vgpu98_gpu_fail_item(r->gpu, i);
                    return err;
                }
            }
            return 0;
        case VGPU98_OP_TEXTURE:
            return pkt_texture(r, p, len);
        case VGPU98_OP_PALETTE: {
            uint32_t stage = p[0];
            uint32_t first = p[1] & 0xffff;
            uint32_t count = p[1] >> 16;

            if ((stage >= N_STAGES) || (first + count > 256))
                return VGPU98_ERR_ARGUMENT;
            if (len < 2 + count)
                return VGPU98_ERR_LENGTH;
            memcpy(&r->pal[stage][first], p + 2, count * sizeof(uint32_t));
            return 0;
        }
        case VGPU98_OP_DECLARE:
        case VGPU98_OP_STREAM:
        case VGPU98_OP_INDICES:
        case VGPU98_OP_INLINE:
            return vgpu98_vin_packet(&r->vin, op, p, len, vgpu98_gpu_profile(r->gpu));
        case VGPU98_OP_DRAW:
            return pkt_draw(r, p, len);
        case VGPU98_OP_ZRANGE:
            for (i = 0; i < 2; i++) {
                float z = u2f(p[i]);

                /* Vulkan's depth range is 0-1: clamped, NaN as 0. */
                r->zrange[i] = !(z > 0.0f) ? 0.0f : (z > 1.0f) ? 1.0f : z;
            }
            return 0;
        case VGPU98_OP_TRANSFORM:
            if (VGPU98_XFORM_IN_PALETTE(p[0])) {
                /* 2.1: a world matrix of the palette; 0-3 are also the
                   world matrices 0-3. */
                const uint32_t k = p[0] - VGPU98_XFORM_PALETTE;

                if (!fixed8(r))
                    return VGPU98_ERR_PROFILE;
                memcpy(r->pal_world[k], p + 1, sizeof(r->pal_world[0]));
                if (k < 4) {
                    memcpy(r->xform[VGPU98_XFORM_WORLD + k], p + 1, sizeof(r->xform[0]));
                    r->ff_dirty = 1;
                }
                /* A higher matrix: the entries up to it, and the identity
                   past them, are new. */
                while (r->pal_max < k)
                    pal_mark(r, ++r->pal_max);
                pal_mark(r, k);
                pal_mark(r, r->pal_max + 1);
                return 0;
            }
            if (!VGPU98_XFORM_VALID(p[0]))
                return VGPU98_ERR_ARGUMENT;
            memcpy(r->xform[p[0]], p + 1, sizeof(r->xform[0]));
            if (p[0] < 4) {
                memcpy(r->pal_world[p[0]], p + 1, sizeof(r->pal_world[0]));
                pal_mark(r, p[0]);
            }
            if (p[0] == VGPU98_XFORM_PROJECTION)
                r->proj_set = 1;
            r->ff_dirty = 1;
            if ((p[0] == VGPU98_XFORM_VIEW) || (p[0] == VGPU98_XFORM_PROJECTION)) {
                /* Every entry's composites follow the view and projection. */
                r->pal_all   = 1;
                r->pal_dirty = 1;
            }
            return 0;
        case VGPU98_OP_LIGHT:
            if ((p[0] >= VGPU98_MAX_LIGHTS) || (p[1] < VGPU98_LIGHT_POINT) || (p[1] > VGPU98_LIGHT_DIRECTIONAL))
                return VGPU98_ERR_ARGUMENT;
            memcpy(r->light[p[0]], p + 1, sizeof(r->light[0]));
            r->ff_dirty = 1;
            return 0;
        case VGPU98_OP_LIGHTENABLE:
            r->light_mask = p[0] & ((1u << VGPU98_MAX_LIGHTS) - 1);
            return 0;
        case VGPU98_OP_MATERIAL:
            memcpy(r->material, p, sizeof(r->material));
            r->ff_dirty = 1;
            return 0;
        case VGPU98_OP_CLIPPLANE:
            if (p[0] >= VGPU98_MAX_CLIPPLANES)
                return VGPU98_ERR_ARGUMENT;
            memcpy(r->plane[p[0]], p + 1, sizeof(r->plane[0]));
            r->ff_dirty = 1;
            return 0;
        case VGPU98_OP_CLEAR:
            return pkt_clear(r, p, len);
        case VGPU98_OP_PROGRAM:
            return pkt_program(r, p, len);
        case VGPU98_OP_BIND:
            return pkt_bind(r, p);
        case VGPU98_OP_CONSTANTS:
            return pkt_constants(r, p, len);
        default:
            return VGPU98_ERR_OPCODE;
    }
}

/* --- Programs (command set 2.2) ---------------------------------------------- */

/* PROGRAM: a program made, replaced or freed (spec/gpu-commands.md). */
static int
pkt_program(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len)
{
    const uint32_t     id = p[0];
    uint32_t           n;
    vgpu98_prog_info_t info;
    r3d_prog_t        *pg;
    uint32_t          *ins;

    if ((len - VGPU98_LEN_PROGRAM) % VGPU98_PROG_INS_DWORDS)
        return VGPU98_ERR_LENGTH;
    n = (len - VGPU98_LEN_PROGRAM) / VGPU98_PROG_INS_DWORDS;
    if ((id < 1) || (id > VGPU98_PROG_MAX_ID)) {
        vgpu98_gpu_fail_item(r->gpu, 0);
        return VGPU98_ERR_ARGUMENT;
    }
    pg = &r->progs[id];
    if (!n) {
        /* A free: (id, 0, 0); a bound program is unbound. */
        if (p[1] || p[2]) {
            vgpu98_gpu_fail_item(r->gpu, p[1] ? 1 : 2);
            return VGPU98_ERR_ARGUMENT;
        }
        if (r->bind_v == id)
            r->bind_v = 0;
        if (r->bind_p == id)
            r->bind_p = 0;
        free(pg->ins);
        memset(pg, 0, sizeof(*pg));
        return 0;
    }
    if (vgpu98_prog_check(p[1], p[2], p + VGPU98_LEN_PROGRAM, n, &info)) {
        vgpu98_gpu_fail_item(r->gpu, info.bad);
        return VGPU98_ERR_ARGUMENT;
    }
    ins = malloc((size_t) n * VGPU98_PROG_INS_DWORDS * sizeof(uint32_t));
    if (!ins) {
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: no memory for program %u\n", id);
        return VGPU98_ERR_ARGUMENT;
    }
    memcpy(ins, p + VGPU98_LEN_PROGRAM, (size_t) n * VGPU98_PROG_INS_DWORDS * sizeof(uint32_t));
    /* A bound program replaced by one of the other kind is unbound. */
    if ((r->bind_v == id) && (p[1] != VGPU98_PROG_VERTEX))
        r->bind_v = 0;
    if ((r->bind_p == id) && (p[1] != VGPU98_PROG_PIXEL))
        r->bind_p = 0;
    free(pg->ins);
    memset(pg, 0, sizeof(*pg));
    pg->kind  = p[1];
    pg->range = p[2];
    pg->n     = n;
    pg->info  = info;
    pg->ins   = ins;
    pg->gen   = ++r->prog_gen;
    /* The vertex constants it reads: a draw copies those alone. */
    if (pg->kind == VGPU98_PROG_VERTEX) {
        uint32_t k;
        uint32_t i;

        for (k = 0; k < n; k++) {
            for (i = 1; i < VGPU98_PROG_INS_DWORDS; i++) {
                const uint32_t s = ins[VGPU98_PROG_INS_DWORDS * k + i];

                if (VGPU98_SRC_FILE(s) != VGPU98_PF_CONST)
                    continue;
                if (s & VGPU98_SRC_RELATIVE)
                    pg->cneed = VGPU98_PROG_VCONSTS;
                else if (VGPU98_SRC_REG(s) + 1 > pg->cneed)
                    pg->cneed = VGPU98_SRC_REG(s) + 1;
            }
        }
    }
    return 0;
}

/* The vertex constants a draw with program `vp` reads, as copied for it:
   those CONSTANTS set, up to past the highest the program reads; one at
   least (a binding is never empty). */
static uint32_t
vconst_count(const vgpu98_r3d_t *r, const r3d_prog_t *vp)
{
    uint32_t n = r->vconst_hi;

    if (vp && (vp->cneed < n))
        n = vp->cneed;
    return n ? n : 1u;
}

/* BIND: the programs the draws that follow run (0: the fixed function). */
static int
pkt_bind(vgpu98_r3d_t *r, const uint32_t *p)
{
    if (p[0] && ((p[0] > VGPU98_PROG_MAX_ID) || (r->progs[p[0]].kind != VGPU98_PROG_VERTEX))) {
        vgpu98_gpu_fail_item(r->gpu, 0);
        return VGPU98_ERR_ARGUMENT;
    }
    if (p[1] && ((p[1] > VGPU98_PROG_MAX_ID) || (r->progs[p[1]].kind != VGPU98_PROG_PIXEL))) {
        vgpu98_gpu_fail_item(r->gpu, 1);
        return VGPU98_ERR_ARGUMENT;
    }
    r->bind_v = p[0];
    r->bind_p = p[1];
    return 0;
}

/* CONSTANTS: vertex or pixel constant registers; pixel ones are kept
   clamped to -1..1 (NaN as 0), the range DirectX 8 gives them. */
static int
pkt_constants(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len)
{
    const uint32_t kind  = p[0];
    const uint32_t first = p[1];
    uint32_t       n;
    uint32_t       i;

    if ((len < 6) || ((len - 2) % 4))
        return VGPU98_ERR_LENGTH; /* one register at least */
    n = (len - 2) / 4;
    if (kind == VGPU98_PROG_VERTEX) {
        if ((first >= VGPU98_PROG_VCONSTS) || (n > VGPU98_PROG_VCONSTS - first)) {
            vgpu98_gpu_fail_item(r->gpu, 1);
            return VGPU98_ERR_ARGUMENT;
        }
        memcpy(r->vconst[first], p + 2, (size_t) n * 4 * sizeof(uint32_t));
        if (first + n > r->vconst_hi)
            r->vconst_hi = first + n;
        r->vconst_gen++;
        return 0;
    }
    if (kind != VGPU98_PROG_PIXEL) {
        vgpu98_gpu_fail_item(r->gpu, 0);
        return VGPU98_ERR_ARGUMENT;
    }
    if ((first >= VGPU98_PROG_PCONSTS) || (n > VGPU98_PROG_PCONSTS - first)) {
        vgpu98_gpu_fail_item(r->gpu, 1);
        return VGPU98_ERR_ARGUMENT;
    }
    for (i = 0; i < 4 * n; i++) {
        float f = u2f(p[2 + i]);

        r->pconst[first + i / 4][i % 4] = isnan(f) ? 0.0f : (f < -1.0f) ? -1.0f : (f > 1.0f) ? 1.0f : f;
    }
    return 0;
}

/* The stages whose textures a draw binds: the pixel program's, or the
   fixed function's from stage 0 to the first COLOROP DISABLE. */
static uint32_t
stage_mask(const vgpu98_r3d_t *r)
{
    if (r->bind_p)
        return r->progs[r->bind_p].info.stages;
    return (1u << active_stages(r)) - 1u;
}

/* --- Coherency entry points ------------------------------------------------ */

/* After a host GPU failure the GPU-newer pixels cannot come back: the
   images stop holding them, so no page waits for them and the display
   shows video memory. (No lock is held at the entry points that call this.) */
static void
broken_forget(vgpu98_r3d_t *r)
{
    while (r->n_dirty) {
        r3d_image_t *im = r->dirty[r->n_dirty - 1];

        image_clean(r, im);
    }
}

void
vgpu98_r3d_set_disp(vgpu98_r3d_t *r, vgpu98_disp_t *d)
{
    if (r)
        r->disp = d;
}

void
vgpu98_r3d_closing(vgpu98_r3d_t *r)
{
    if (r)
        atomic_store(&r->warm_stop, 1);
}

/* Can the display show the image's pixels as the host GPU holds them? The
   host format must hold the guest's bytes as they are (a copy back would
   copy them unchanged), the size of the mode's pixels. */
static int
publishable(const r3d_image_t *im, uint32_t bpp)
{
    const int bytes = vgpu98_fmt_bpp(im->format);

    return (im->hf->conv == CONV_COPY) && (fmt_aspects(im->hf) == VK_IMAGE_ASPECT_COLOR_BIT) && (im->n_sub == 1) &&
           (bytes == im->hf->bytes) && (bytes == (int) ((bpp + 7) / 8)) && (bytes >= 2) && !(im->sub[0].pitch % 4) &&
           !(im->sub[0].offset % 4);
}

void
vgpu98_r3d_publish(vgpu98_r3d_t *r, vgpu98_disp_t *d, uint32_t lo, uint32_t hi, uint32_t bpp)
{
    vgpu98_vk_t *vk;
    r3d_image_t *pub[16];
    r3d_image_t *back[16];
    size_t       n_pub  = 0;
    size_t       n_back = 0;
    size_t       i;
    VkBuffer     layer;
    int          step;

    if (!r || (lo >= hi))
        return;
    if (r->broken) {
        broken_forget(r);
        return;
    }
    if (!r->n_dirty)
        return;
    vk = r->vk;
    if (!vgpu98_disp_showing(d)) {
        /* 86Box's own display reads video memory. */
        sync_range(r, lo, hi, NULL, VGPU98_CB_FRAME);
        return;
    }
    for (i = 0; i < r->n_dirty; i++) {
        r3d_image_t *im = r->dirty[i];

        if (!image_overlaps(im, lo, hi, 1))
            continue;
        if (!im->pub_new && publishable(im, bpp))
            continue; /* the layer holds its pixels as they are */
        if ((n_pub == 16) || (n_back == 16)) {
            /* More images over the frame than this keeps track of: all of
               them go back to video memory. */
            sync_range(r, lo, hi, NULL, VGPU98_CB_FRAME);
            return;
        }
        if (publishable(im, bpp))
            pub[n_pub++] = im;
        else
            back[n_back++] = im;
    }
    if (n_back)
        copy_back(r, back, n_back, lo, hi, 0, 0, VGPU98_CB_FRAME);
    if (!n_pub || r->broken)
        return;

    /* Straight from the image to its place in the layer: the bytes video
       memory would hold after a copy back. */
    layer = (VkBuffer) (uintptr_t) vgpu98_disp_layer_buffer(d);
    pass_end(r);
    barrier(r);
    for (i = 0; i < n_pub; i++) {
        r3d_image_t      *im    = pub[i];
        const uint32_t    bytes = (uint32_t) vgpu98_fmt_bpp(im->format);
        /* From an even pixel at 2 bytes, so the buffer offset is a multiple
           of 4 (Vulkan 1.1); the extra pixel lies outside the rectangle the
           display reads. */
        const r3d_sub_t  *s     = &im->sub[0];
        const uint32_t    x0    = (bytes == 2) ? (s->dx0 & ~1u) : s->dx0;
        VkBufferImageCopy c;

        memset(&c, 0, sizeof(c));
        c.bufferOffset                = (VkDeviceSize) s->offset + (VkDeviceSize) s->dy0 * s->pitch + (VkDeviceSize) x0 * bytes;
        c.bufferRowLength             = s->pitch / bytes;
        c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        c.imageSubresource.layerCount = 1;
        c.imageOffset.x               = (int32_t) x0;
        c.imageOffset.y               = (int32_t) s->dy0;
        c.imageExtent.width           = s->dx1 - x0;
        c.imageExtent.height          = s->dy1 - s->dy0;
        c.imageExtent.depth           = 1;
        vk->vkCmdCopyImageToBuffer(cmd(r), im->image, VK_IMAGE_LAYOUT_GENERAL, layer, 1, &c);
    }
    barrier(r);
    /* The display composes from the layer only while it is not being
       written. */
    step = vgpu98_gpu_step(r->gpu, VGPU98_STEP_LAYER);
    vgpu98_disp_layer_lock(d);
    vgpu98_gpu_step(r->gpu, step);
    if (!submit(r, 1)) {
        vgpu98_disp_layer_unlock(d);
        return;
    }
    n_back = 0;
    for (i = 0; i < n_pub; i++) {
        r3d_image_t        *im = pub[i];
        vgpu98_layer_rect_t rect;

        rect.id     = im->id;
        rect.offset = im->sub[0].offset;
        rect.pitch  = im->sub[0].pitch;
        rect.bytes  = (uint32_t) vgpu98_fmt_bpp(im->format);
        rect.x0     = im->sub[0].dx0;
        rect.y0     = im->sub[0].dy0;
        rect.x1     = im->sub[0].dx1;
        rect.y1     = im->sub[0].dy1;
        if (vgpu98_disp_layer_set(d, &rect)) {
            im->pub_new = 0;
            r->stats->publishes++;
            r->stats->publish_bytes += (uint64_t) (rect.x1 - rect.x0) * (rect.y1 - rect.y0) * rect.bytes;
        } else
            back[n_back++] = im;
    }
    vgpu98_disp_layer_unlock(d);
    /* The layer holds as many rectangles as it can: the rest go back to
       video memory, which the display reads where the layer has nothing. */
    if (n_back)
        copy_back(r, back, n_back, lo, hi, 0, 0, VGPU98_CB_FRAME);
}

void
vgpu98_r3d_uploaded(vgpu98_r3d_t *r, const vgpu98_surf_t *sf, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                    uint32_t start, uint32_t len)
{
    const uint32_t p0 = start >> VGPU98_PAGE_SHIFT;
    const uint32_t p1 = (start + len - 1) >> VGPU98_PAGE_SHIFT;
    r3d_sub_t     *s;

    if (!r || r->broken || !len)
        return;
    for (s = r->sub_hash[image_bucket(sf->format, sf->width, sf->height, sf->offset, sf->pitch)]; s; s = s->hash_next) {
        r3d_image_t *im      = s->im;
        int          current = 1;
        uint32_t     p;

        if ((im->format != sf->format) || (s->width != sf->width) || (s->height != sf->height) || (s->offset != sf->offset) ||
            (s->pitch != sf->pitch))
            continue;
        /* P8 goes through a palette the image does not keep: it uploads
           again when next used. */
        if (!im->valid || s->dirty || (im->format == VGPU98_FMT_P8))
            continue;
        /* Current but for this write: every page of its levels and faces as
           of its last upload, the written ones with one GPU write more (a
           guest write meanwhile shows, and the image uploads again). */
        for (p = 0; current && (p < im->n_pages); p++) {
            const uint32_t page = im->first_page + p;

            current = !im->covered[p] ||
                      (vgpu98_page_version(r->vram, page) == im->versions[p] + (((page >= p0) && (page <= p1)) ? 1u : 0u));
        }
        if (!current)
            continue;
        pass_end(r);
        barrier(r);
        if (!sub_upload(r, im, s, x0, y0, x1, y1, NULL, 0))
            continue;
        barrier(r);
        for (p = 0; p < im->n_pages; p++) {
            const uint32_t page = im->first_page + p;

            if ((page >= p0) && (page <= p1))
                im->versions[p]++;
        }
        s->aset       = 0; /* the texture audit hashes the level anew */
        im->key_valid = 0;
        im->kz_valid  = 0;
        r->stats->upload_rects++;
    }
}

void
vgpu98_r3d_written(vgpu98_r3d_t *r, const vgpu98_surf_t *sf, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
{
    uint32_t row_lo;
    uint32_t row_hi;
    uint32_t rows_from;
    uint32_t rows_to;
    uint32_t lo;
    uint32_t hi;
    size_t   i;

    if (!r || r->broken || !r->n_shadowed || (x0 >= x1) || (y0 >= y1))
        return;
    /* The bytes of each row written (rows of blocks for DXT). */
    if (vgpu98_fmt_is_dxt(sf->format)) {
        const uint32_t block = (sf->format == VGPU98_FMT_DXT1) ? 8 : 16;

        row_lo    = (x0 / 4) * block;
        row_hi    = ((x1 + 3) / 4) * block;
        rows_from = y0 / 4;
        rows_to   = (y1 + 3) / 4;
    } else {
        const uint32_t bpp = (uint32_t) vgpu98_fmt_bpp(sf->format);

        row_lo    = x0 * bpp;
        row_hi    = x1 * bpp;
        rows_from = y0;
        rows_to   = y1;
    }
    lo = sf->offset + rows_from * sf->pitch + row_lo;
    hi = sf->offset + (rows_to - 1) * sf->pitch + row_hi;
    for (i = 0; i < r->n_img; i++) {
        r3d_image_t *im = r->img[i];
        uint32_t     y;

        if (!im->shadow || (im->hi <= lo) || (hi <= im->lo))
            continue;
        /* Its shadow no longer matches there, so a resync takes them. */
        for (y = rows_from; y < rows_to; y++) {
            uint32_t a = sf->offset + y * sf->pitch + row_lo;
            uint32_t b = sf->offset + y * sf->pitch + row_hi;
            uint32_t k;

            if (a < im->lo)
                a = im->lo;
            if (b > im->hi)
                b = im->hi;
            for (k = a; k < b; k++)
                im->shadow[k - im->lo] = (uint8_t) ~r->vram->data[k];
        }
    }
}

void
vgpu98_r3d_sync(vgpu98_r3d_t *r, uint32_t offset, uint32_t len, int cause)
{
    if (!r || !len)
        return;
    if (r->broken) {
        broken_forget(r);
        return;
    }
    sync_range(r, offset, offset + len, NULL, cause);
}

void
vgpu98_r3d_discard(vgpu98_r3d_t *r, uint32_t offset, uint32_t len)
{
    size_t i;

    if (!r || !len)
        return;
    /* The program freed the surfaces there: the mip check measures one
       made anew (with 3D stopped too, as its records outlast a RESET). */
    mip_recs_drop(r, offset, offset + len);
    if (r->broken)
        return;
    for (i = 0; i < r->n_img;) {
        r3d_image_t *im = r->img[i];

        if (!image_overlaps(im, offset, offset + len, 0)) {
            i++;
            continue;
        }
        /* GPU-newer data outside the range still belongs to someone. */
        if (im->dirty && ((im->lo < offset) || (im->hi > offset + len)))
            copy_back(r, &im, 1, 0, 0xffffffffu, offset, offset + len, VGPU98_CB_DISCARD);
        r->stats->discards++;
        vgpu98_gpu_note(r->gpu, VGPU98_NOTE_DISCARD, im->id, offset, len, 0, 0);
        image_remove(r, i);
    }
    /* Volumes there (2.1) go too. */
    for (i = 0; i < r->n_vols;) {
        if (vol_overlaps(r->vols[i], offset, offset + len)) {
            r->stats->discards++;
            vol_remove(r, i);
        } else
            i++;
    }
    /* The program freed the surfaces there: one made anew starts afresh. */
    depth_rec_drop(r, offset, offset + len);
}

void
vgpu98_r3d_flush(vgpu98_r3d_t *r)
{
    if (!r)
        return;
    if (r->broken) {
        broken_forget(r);
        return;
    }
    if (r->recording)
        submit(r, 0);
    evict_stale(r);
    evict(r);
    /* New pipelines (the warm-up's too) go to disk now and then (86Box may
       not end cleanly), but rarely: a save writes the whole cache (on a
       thread of its own). */
    pcache_warm_check(r);
    if (r->pcache_new && (vgpu98_diag_us() - r->pcache_us > PCACHE_PERIOD_US)) {
        const int step = vgpu98_gpu_step(r->gpu, VGPU98_STEP_PCACHE);

        pcache_save(r, 0);
        vgpu98_gpu_step(r->gpu, step);
    }
}

/* Everything goes once the host GPU is idle. 0 when it could not go idle
   (3D stops: a reset does not start it again). */
static int
drop_all(vgpu98_r3d_t *r)
{
    vgpu98_vk_t *vk = r->vk;
    int          i;
    int          step;
    VkResult     res;

    if (r->recording) {
        pass_end(r);
        submit(r, 0);
    }
    step = vgpu98_gpu_step(r->gpu, VGPU98_STEP_IDLE);
    res  = vgpu98_vk_wait_idle(vk);
    vgpu98_gpu_step(r->gpu, step);
    if (res != VK_SUCCESS)
        r3d_fail(r, "vkDeviceWaitIdle", res);
    if (r->sink) {
        vk->vkFreeCommandBuffers(vk->dev, r->cmdpool, 1, &r->sink);
        r->sink = VK_NULL_HANDLE;
    }
    r->idle = 1;
    while (r->n_img)
        image_remove(r, r->n_img - 1);
    while (r->n_vols)
        vol_remove(r, r->n_vols - 1);
    res_forget(r);
    r->idle = 0;
    for (i = 0; i < SLOTS; i++) {
        r->slot[i].pending = 0;
        dead_free(r, &r->slot[i]);
    }
    r->img_bytes        = 0;
    r->recording        = 0;
    r->in_pass          = 0;
    r->pass_color       = NULL;
    r->pass_depth       = NULL;
    return res == VK_SUCCESS;
}

void
vgpu98_r3d_reset(vgpu98_r3d_t *r, uint64_t deadline_us, int forget)
{
    int idle;

    if (!r)
        return;
    /* Video memory keeps all the GPU drew: what it held newer goes back
       before the images go, while whoever reset the GPU still waits for it
       (until the deadline); what is not back by then is dropped, so no
       copy back lands after the guest moved on. */
    if (!r->broken && r->n_dirty) {
        r3d_image_t **list = malloc(r->n_dirty * sizeof(*list));

        if (list) {
            const size_t n = r->n_dirty;
            size_t       i;

            memcpy(list, r->dirty, n * sizeof(*list));
            for (i = 0; (i < n) && !r->broken && (vgpu98_diag_us() < deadline_us); i++)
                copy_back(r, &list[i], 1, 0, 0xffffffffu, 0, 0, VGPU98_CB_RESET);
            free(list);
        }
    }
    idle = drop_all(r);
    memset(r->drec, 0, sizeof(r->drec)); /* no surface W-buffered into: the depth formats are the host's own again */
    /* The mip check's records outlast a RESET that frees nothing (the
       program frees its surfaces with DISCARDs); a new session (power on, a
       machine reset, the driver's start) starts them afresh. */
    if (forget) {
        r->n_mip_recs   = 0;
        r->mip_rec_next = 0;
        r->mip_logged   = 0;
        r->note_seq     = 0; /* a new display driver's shared block numbers the audit notes from 1 */
        r->proc_name[0] = 0;
        {
            uint32_t q;

            for (q = 0; q < VSUM_PENDING; q++)
                r->ta_vsum_dropped += (uint32_t) r->vsum_pend[q].used; /* the races held: counted, not told */
        }
        memset(r->vsum_pend, 0, sizeof(r->vsum_pend));
        if (r->vsum_owners)
            memset(r->vsum_owners, 0, VSUM_OWNERS * sizeof(vsum_owner_t));
        r->shown_set = 0; /* the display check: no FLIP of this session yet */
        r->shown_run = 0;
    }
    state_defaults(r);
    vgpu98_vin_reset(&r->vin);
    if (!idle)
        return; /* the host GPU could not go idle (its device lost): 3D stays stopped */
    if (r->broken)
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: reset; drawing again\n");
    r->broken = 0;
}

vgpu98_vk_t *
vgpu98_r3d_vk(vgpu98_r3d_t *r)
{
    return r ? r->vk : NULL;
}

/* --- The Glide device's use of the renderer (vid_vgpu98_vk.h) -------------- */

void
vgpu98_r3d_glide_hook(vgpu98_r3d_t *r, void (*end)(void *priv), void *priv)
{
    r->glide_end  = end;
    r->glide_priv = priv;
}

/* A device's buffer to record into: the other device's render pass ended
   first (dev 0 Glide, 1 OpenGL). */
static VkCommandBuffer
dev_cmd(vgpu98_r3d_t *r, int dev)
{
    VkCommandBuffer c;

    if ((dev == 0) && r->ogl_end)
        r->ogl_end(r->ogl_priv);
    if ((dev == 1) && r->glide_end)
        r->glide_end(r->glide_priv);
    r->in_glide = 1;
    pass_end(r);
    c             = cmd(r);
    r->bound_pipe = VK_NULL_HANDLE; /* the device binds its own */
    r->in_glide   = 0;
    return c;
}

VkCommandBuffer
vgpu98_r3d_glide_cmd(vgpu98_r3d_t *r)
{
    return dev_cmd(r, 0);
}

void
vgpu98_r3d_ogl_hook(vgpu98_r3d_t *r, void (*end)(void *priv), void *priv)
{
    r->ogl_end  = end;
    r->ogl_priv = priv;
}

VkCommandBuffer
vgpu98_r3d_ogl_cmd(vgpu98_r3d_t *r)
{
    return dev_cmd(r, 1);
}

VkCommandBuffer
vgpu98_r3d_glide_rec(vgpu98_r3d_t *r)
{
    return rec_cmd(r);
}

void *
vgpu98_r3d_glide_alloc(vgpu98_r3d_t *r, int which, VkDeviceSize size, VkDeviceSize align, VkBuffer *buf,
                       VkDeviceSize *off)
{
    void *p;

    r->in_glide = 1;
    pass_end(r);
    p           = slot_alloc(r, which, size, align, buf, off);
    r->in_glide = 0;
    return p;
}

uint32_t
vgpu98_r3d_glide_slot(vgpu98_r3d_t *r, uint32_t *gen)
{
    r->in_glide = 1;
    cmd(r);
    r->in_glide = 0;
    *gen        = r->slot[r->cur].gen;
    return (uint32_t) r->cur;
}

void
vgpu98_r3d_glide_dead(vgpu98_r3d_t *r, int kind, uint64_t handle)
{
    static const int kinds[] = { DEAD_IMAGE, DEAD_VIEW, DEAD_MEMORY, DEAD_FRAMEBUFFER, DEAD_BUFFER };

    if ((kind < 0) || (kind >= (int) (sizeof(kinds) / sizeof(kinds[0]))))
        return;
    r->in_glide = 1;
    dead_add(r, kinds[kind], handle);
    r->in_glide = 0;
}

int
vgpu98_r3d_glide_submit(vgpu98_r3d_t *r, int wait)
{
    if (!r->recording && wait)
        return !r->broken;
    return submit(r, wait);
}

int
vgpu98_r3d_glide_image_memory(vgpu98_r3d_t *r, VkImage image, VkDeviceMemory *mem, VkDeviceSize *bytes)
{
    return image_memory(r, image, mem, bytes);
}

VkPipelineCache
vgpu98_r3d_glide_pcache(vgpu98_r3d_t *r)
{
    return r->pcache;
}

void
vgpu98_r3d_glide_new_pipeline(vgpu98_r3d_t *r)
{
    r->pcache_new++; /* saved with the renderer's in time */
}

int
vgpu98_r3d_broken(const vgpu98_r3d_t *r)
{
    return !r || r->broken;
}

void
vgpu98_r3d_glide_fail(vgpu98_r3d_t *r, const char *what, int res)
{
    r3d_fail(r, what, (VkResult) res);
}

/* --- Start-up ---------------------------------------------------------------- */

static VkShaderModule
shader(vgpu98_vk_t *vk, const uint32_t *code, size_t bytes)
{
    VkShaderModuleCreateInfo ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkShaderModule           m;

    ci.codeSize = bytes;
    ci.pCode    = code;
    return (vk->vkCreateShaderModule(vk->dev, &ci, NULL, &m) == VK_SUCCESS) ? m : VK_NULL_HANDLE;
}

static int
r3d_objects(vgpu98_r3d_t *r)
{
    vgpu98_vk_t                    *vk = r->vk;
    VkDescriptorSetLayoutBinding    b[14];
    VkDescriptorSetLayoutCreateInfo dl = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPipelineLayoutCreateInfo      pl = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkCommandPoolCreateInfo         cp = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    VkCommandBufferAllocateInfo     ca = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    VkFenceCreateInfo               fc = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkDescriptorPoolSize            ps[3];
    VkDescriptorPoolCreateInfo      dp = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    VkImageCreateInfo               ic = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    VkImageViewCreateInfo           vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    int                             i;

    pcache_load(r);
    r->pcache_us = vgpu98_diag_us();
    r->pcache_ev = thread_create_event(); /* without it, saves go on the GPU thread */
    r->vs        = shader(vk, vgpu98_vert, sizeof(vgpu98_vert));
    r->fs       = shader(vk, vgpu98_frag, sizeof(vgpu98_frag));
    r->fs_depth = shader(vk, vgpu98_frag_depth, sizeof(vgpu98_frag_depth));
    if (!r->vs || !r->fs || !r->fs_depth)
        return 0;

    /* 0: the fragment uniforms; 1, 2: the stages' 2D textures and their
       colour-key companions; 3: the vertex uniforms; 4: the draw's copy of
       its vertices; 5: its primitive list; 6, 7: the stages' cube maps and
       their companions; 8: the world matrix palette (2.1); 9: the stages'
       volumes (2.1); 10: the arena of resident buffers (vgpu98.vert,
       vgpu98.frag); 11, 12: the vertex and pixel programs' instructions,
       13: the vertex constants (2.2). */
    memset(b, 0, sizeof(b));
    for (i = 0; i < 14; i++) {
        b[i].binding         = (uint32_t) i;
        b[i].descriptorCount = 1;
    }
    b[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    b[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].descriptorCount = N_STAGES;
    b[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[2].descriptorCount = N_STAGES;
    b[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    b[3].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
    b[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[4].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
    b[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[5].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
    b[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[6].descriptorCount = N_STAGES;
    b[6].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[7].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[7].descriptorCount = N_STAGES;
    b[7].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[8].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[8].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
    b[9].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[9].descriptorCount = N_STAGES;
    b[9].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[10].stageFlags     = VK_SHADER_STAGE_VERTEX_BIT;
    b[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[11].stageFlags     = VK_SHADER_STAGE_VERTEX_BIT;
    b[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[12].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[13].stageFlags     = VK_SHADER_STAGE_VERTEX_BIT;
    /* Set 0 is everything but 4, 5, 8 and 11-13, a draw's own buffers: set 1. */
    {
        VkDescriptorSetLayoutBinding b0[8];
        VkDescriptorSetLayoutBinding b1[6];
        VkDescriptorSetLayout        sl[2];
        static const int             in0[8] = { 0, 1, 2, 3, 6, 7, 9, 10 };
        static const int             in1[6] = { 4, 5, 8, 11, 12, 13 };

        for (i = 0; i < 8; i++)
            b0[i] = b[in0[i]];
        for (i = 0; i < 6; i++)
            b1[i] = b[in1[i]];
        dl.bindingCount = 8;
        dl.pBindings    = b0;
        if (vk->vkCreateDescriptorSetLayout(vk->dev, &dl, NULL, &r->dsl) != VK_SUCCESS)
            return 0;
        dl.bindingCount = 6;
        dl.pBindings    = b1;
        if (vk->vkCreateDescriptorSetLayout(vk->dev, &dl, NULL, &r->dsl_draw) != VK_SUCCESS)
            return 0;
        /* The pipelines' layout: a change to it (its sets, bindings and
           counts) raises PCACHE_PIPE_VERSION. */
        sl[0]             = r->dsl;
        sl[1]             = r->dsl_draw;
        pl.setLayoutCount = 2;
        pl.pSetLayouts    = sl;
        if (vk->vkCreatePipelineLayout(vk->dev, &pl, NULL, &r->layout) != VK_SUCCESS)
            return 0;
    }

    cp.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = vk->queue_family;
    if (vk->vkCreateCommandPool(vk->dev, &cp, NULL, &r->cmdpool) != VK_SUCCESS)
        return 0;
    ps[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    ps[0].descriptorCount = SETS_PER_SLOT * 2;
    ps[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[1].descriptorCount = SETS_PER_SLOT * 5 * N_STAGES;
    ps[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[2].descriptorCount = SETS_PER_SLOT * 13; /* up to 2 * SETS_PER_SLOT draws, 6 each, when set 0 is reused, and set 0's arena */
    dp.maxSets            = 2 * SETS_PER_SLOT; /* a draw's two sets */
    dp.poolSizeCount      = 3;
    dp.pPoolSizes         = ps;
    for (i = 0; i < SLOTS; i++) {
        ca.commandPool        = r->cmdpool;
        ca.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        if (vk->vkAllocateCommandBuffers(vk->dev, &ca, &r->slot[i].cmd) != VK_SUCCESS)
            return 0;
        if (vk->vkCreateFence(vk->dev, &fc, NULL, &r->slot[i].fence) != VK_SUCCESS)
            return 0;
        if (vk->vkCreateDescriptorPool(vk->dev, &dp, NULL, &r->slot[i].pool) != VK_SUCCESS)
            return 0;
    }

    /* 1x1 transparent black textures, 2D and cube, for stages without one
       (the reference rasterizer reads 0 there) and for keyless textures. */
    for (i = 0; i < 2; i++) {
        const uint32_t  layers = i ? VGPU98_CUBE_FACES : 1;
        VkImage        *image  = i ? &r->dummy_cube : &r->dummy;
        VkDeviceMemory *mem    = i ? &r->dummy_cube_mem : &r->dummy_mem;
        VkImageView    *view   = i ? &r->dummy_cube_view : &r->dummy_view;
        VkBuffer        buf;
        VkDeviceSize    off;
        uint8_t        *p;
        VkBufferImageCopy c;

        ic.flags         = i ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
        ic.imageType     = VK_IMAGE_TYPE_2D;
        ic.format        = VK_FORMAT_R8G8B8A8_UNORM;
        ic.extent.width  = 1;
        ic.extent.height = 1;
        ic.extent.depth  = 1;
        ic.mipLevels     = 1;
        ic.arrayLayers   = layers;
        ic.samples       = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ic.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (vk->vkCreateImage(vk->dev, &ic, NULL, image) != VK_SUCCESS)
            return 0;
        if (!image_memory(r, *image, mem, NULL))
            return 0;
        vi.image                       = *image;
        vi.viewType                    = i ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        vi.format                      = VK_FORMAT_R8G8B8A8_UNORM;
        vi.components                  = SWZ_ID;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = layers;
        if (vk->vkCreateImageView(vk->dev, &vi, NULL, view) != VK_SUCCESS)
            return 0;
        image_init_layout(r, *image, VK_IMAGE_ASPECT_COLOR_BIT, 1, layers);
        p = slot_alloc(r, 0, 4 * layers, 4, &buf, &off);
        if (!p)
            return 0;
        memset(p, 0, 4 * layers);
        copy_region(&c, off, 1, VK_IMAGE_ASPECT_COLOR_BIT, NULL, 0, 0, 1, 1);
        c.imageSubresource.layerCount = layers;
        vk->vkCmdCopyBufferToImage(cmd(r), buf, *image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    /* And a 1x1x1 volume (2.1), for stages without one. */
    {
        VkBuffer          buf;
        VkDeviceSize      off;
        uint8_t          *p;
        VkBufferImageCopy c;

        ic.flags         = 0;
        ic.imageType     = VK_IMAGE_TYPE_3D;
        ic.format        = VK_FORMAT_R8G8B8A8_UNORM;
        ic.extent.width  = 1;
        ic.extent.height = 1;
        ic.extent.depth  = 1;
        ic.mipLevels     = 1;
        ic.arrayLayers   = 1;
        if (vk->vkCreateImage(vk->dev, &ic, NULL, &r->dummy_vol) != VK_SUCCESS)
            return 0;
        if (!image_memory(r, r->dummy_vol, &r->dummy_vol_mem, NULL))
            return 0;
        vi.image                       = r->dummy_vol;
        vi.viewType                    = VK_IMAGE_VIEW_TYPE_3D;
        vi.subresourceRange.layerCount = 1;
        if (vk->vkCreateImageView(vk->dev, &vi, NULL, &r->dummy_vol_view) != VK_SUCCESS)
            return 0;
        image_init_layout(r, r->dummy_vol, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1);
        p = slot_alloc(r, 0, 4, 4, &buf, &off);
        if (!p)
            return 0;
        memset(p, 0, 4);
        copy_region(&c, off, 1, VK_IMAGE_ASPECT_COLOR_BIT, NULL, 0, 0, 1, 1);
        vk->vkCmdCopyBufferToImage(cmd(r), buf, r->dummy_vol, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    barrier(r);
    return submit(r, 1);
}

vgpu98_r3d_t *
vgpu98_r3d_init(vgpu98_gpu_t *g, const vgpu98_vram_t *vram)
{
    vgpu98_r3d_t *r = calloc(1, sizeof(vgpu98_r3d_t));
    VkDeviceSize  local = 0;
    uint32_t      i;

    r->gpu    = g;
    r->stats  = vgpu98_gpu_stats(g);
    r->vram   = vram;
    r->serial = 1; /* a record's serial 0 would read as finished */
    r->vk    = vgpu98_vk_create(vk_log_cb, g);
    if (!r->vk) {
        free(r);
        return NULL;
    }
    if (!r->vk->clip_distance || !r->vk->int64 || !r->vk->sampler_indexing) {
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS,
                       "VGPU98 3D: the host GPU lacks the shaders' features (clip distances, 64-bit shader integers, "
                       "indexed texture arrays); no 3D\n");
        vgpu98_vk_destroy(r->vk);
        free(r);
        return NULL;
    }
    {
        /* Every draw binds five image samplers a stage (2D, its key, cube,
           its key, volume) to the fragment stage. */
        const VkPhysicalDeviceLimits *lim  = &r->vk->props.limits;
        const uint32_t                need = 5 * N_STAGES;

        if ((lim->maxPerStageDescriptorSamplers < need) || (lim->maxPerStageDescriptorSampledImages < need) ||
            (lim->maxDescriptorSetSamplers < need) || (lim->maxDescriptorSetSampledImages < need) ||
            (lim->maxPerStageResources < need + 2)) {
            vgpu98_gpu_log(g, VGPU98_LOG_EVENTS,
                           "VGPU98 3D: the host GPU binds fewer than %u textures to a shader (%u samplers, %u images); no 3D\n",
                           need, lim->maxPerStageDescriptorSamplers, lim->maxPerStageDescriptorSampledImages);
            vgpu98_vk_destroy(r->vk);
            free(r);
            return NULL;
        }
    }
    /* The vertex stage reads six storage buffers (its vertices, primitive
       list, palette and the arena; 2.2's program and constants), more than
       Vulkan guarantees (four). */
    if ((r->vk->props.limits.maxPerStageDescriptorStorageBuffers < 6) ||
        (r->vk->props.limits.maxDescriptorSetStorageBuffers < 7)) {
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 3D: the host GPU binds fewer than 6 storage buffers to a shader (%u); no 3D\n",
                       r->vk->props.limits.maxPerStageDescriptorStorageBuffers);
        vgpu98_vk_destroy(r->vk);
        free(r);
        return NULL;
    }
    formats_init(r);
    if (!r->has_fmt[VGPU98_FMT_D16] || !r->has_fmt[VGPU98_FMT_D24S8]) {
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 3D: the host GPU lacks the depth formats; no 3D\n");
        vgpu98_vk_destroy(r->vk);
        free(r);
        return NULL;
    }
    for (i = 0; i < r->vk->mem.memoryHeapCount; i++) {
        if ((r->vk->mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && (r->vk->mem.memoryHeaps[i].size > local))
            local = r->vk->mem.memoryHeaps[i].size;
    }
    /* Host images take at most half the host GPU's memory, and 1 GB. */
    r->img_budget = local / 2;
    if (!r->img_budget || (r->img_budget > ((VkDeviceSize) 1 << 30)))
        r->img_budget = (VkDeviceSize) 1 << 30;
    r->progs = calloc(VGPU98_PROG_MAX_ID + 1, sizeof(r3d_prog_t));
    state_defaults(r);
    r->pal_ff = calloc(VGPU98_PALETTE_SIZE + 1, sizeof(r3d_pal_t));
    if (!r->progs || !r->pal_ff || !vgpu98_vin_init(&r->vin, g) || !r3d_objects(r)) {
        vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 3D: setting up the host GPU failed; no 3D\n");
        vgpu98_r3d_close(r);
        return NULL;
    }
    pcache_warm_start(r);
    vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 3D: ready; host image budget %u MB%s\n", (unsigned) (r->img_budget >> 20),
                   r->d16_full                                          ? "; 16-bit depth at full precision" :
                   (vgpu98_gpu_options(g) & VGPU98_OPT_DEPTH16_FULL) ? "; 16-bit depth as the guest's (the host has no 32-bit float depth format)" :
                                                                          "");
    return r;
}

int
vgpu98_r3d_close(vgpu98_r3d_t *r)
{
    vgpu98_vk_t *vk;
    size_t       i;
    int          warm_ended = 1;

    if (!r)
        return 1;
    /* The texture dumps' thread (which owns its job, and needs nothing
       here): a while to finish its files. */
    for (i = 0; atomic_load(&dump_busy) && (i < 50); i++)
        plat_delay_ms(100);
    if (atomic_load(&dump_busy))
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: the texture dumps' thread had not ended after 5 s; its last pictures may be cut short\n");
    vk = r->vk;
    if (vk && vk->dev) {
        /* The warm-up stops first. A thread that has not ended keeps what
           it uses (the device, the shaders, the layout and its sets, the
           cache, the key list: left rather than freed under it); everything
           else goes, images and buffers first. */
        warm_ended = pcache_warm_stop(r);
        pcache_warm_check(r);
        vgpu98_vk_wait_idle(vk);
        r->idle = 1;
        while (r->n_img)
            image_remove(r, r->n_img - 1);
        while (r->n_vols)
            vol_remove(r, r->n_vols - 1);
        vgpu98_vk_buffer_destroy(vk, &r->res_buf);
        free(r->res);
        free(r->res_order);
        free(r->res_scratch);
        for (i = 0; i < SLOTS; i++) {
            slot_t *s = &r->slot[i];

            dead_free(r, s);
            free(s->dead);
            vgpu98_vk_buffer_destroy(vk, &s->staging);
            vgpu98_vk_buffer_destroy(vk, &s->vbuf);
            vgpu98_vk_buffer_destroy(vk, &s->ubuf);
            if (s->fence)
                vk->vkDestroyFence(vk->dev, s->fence, NULL);
            if (s->pool)
                vk->vkDestroyDescriptorPool(vk->dev, s->pool, NULL);
        }
        for (i = 0; i < r->n_fbs; i++)
            vk->vkDestroyFramebuffer(vk->dev, r->fbs[i].fb, NULL);
        /* A save still writing ends first; the last goes on this thread.
           One that does not end in time keeps its cache and event (86Box is
           closing: they are left, not freed under it). */
        pcache_join(r, 1);
        if (r->pcache_new)
            pcache_save(r, 1);
        if (!r->pcache_running) {
            if (r->pcache && warm_ended)
                vk->vkDestroyPipelineCache(vk->dev, r->pcache, NULL);
            if (r->pcache_ev)
                thread_destroy_event(r->pcache_ev);
        }
        if (r->warm_ev && warm_ended)
            thread_destroy_event(r->warm_ev);
        free(r->pipe_tab);
        free(r->samp_tab);
        for (i = 0; i < r->n_pipes; i++)
            vk->vkDestroyPipeline(vk->dev, r->pipes[i].pipeline, NULL);
        for (i = 0; i < r->n_passes; i++)
            vk->vkDestroyRenderPass(vk->dev, r->passes[i].pass, NULL);
        for (i = 0; i < r->n_samps; i++)
            vk->vkDestroySampler(vk->dev, r->samps[i].sampler, NULL);
        if (r->dummy_view)
            vk->vkDestroyImageView(vk->dev, r->dummy_view, NULL);
        if (r->dummy)
            vk->vkDestroyImage(vk->dev, r->dummy, NULL);
        if (r->dummy_mem)
            vk->vkFreeMemory(vk->dev, r->dummy_mem, NULL);
        if (r->dummy_cube_view)
            vk->vkDestroyImageView(vk->dev, r->dummy_cube_view, NULL);
        if (r->dummy_cube)
            vk->vkDestroyImage(vk->dev, r->dummy_cube, NULL);
        if (r->dummy_cube_mem)
            vk->vkFreeMemory(vk->dev, r->dummy_cube_mem, NULL);
        if (r->dummy_vol_view)
            vk->vkDestroyImageView(vk->dev, r->dummy_vol_view, NULL);
        if (r->dummy_vol)
            vk->vkDestroyImage(vk->dev, r->dummy_vol, NULL);
        if (r->dummy_vol_mem)
            vk->vkFreeMemory(vk->dev, r->dummy_vol_mem, NULL);
        if (r->cmdpool)
            vk->vkDestroyCommandPool(vk->dev, r->cmdpool, NULL);
        if (warm_ended) {
            if (r->layout)
                vk->vkDestroyPipelineLayout(vk->dev, r->layout, NULL);
            if (r->dsl)
                vk->vkDestroyDescriptorSetLayout(vk->dev, r->dsl, NULL);
            if (r->dsl_draw)
                vk->vkDestroyDescriptorSetLayout(vk->dev, r->dsl_draw, NULL);
            if (r->vs)
                vk->vkDestroyShaderModule(vk->dev, r->vs, NULL);
            if (r->fs)
                vk->vkDestroyShaderModule(vk->dev, r->fs, NULL);
            if (r->fs_depth)
                vk->vkDestroyShaderModule(vk->dev, r->fs_depth, NULL);
        }
    }
    /* A save still writing keeps its device, its cache and the renderer
       that holds its record, a warm-up thread still making a pipeline what
       it uses (86Box is closing: left, not freed under them). */
    if (r->pcache_running || !warm_ended) {
        r3d_log(r, VGPU98_LOG_EVENTS, "VGPU98 3D: at close, the pipeline cache's %s had not ended: left as it is\n",
                r->pcache_running ? "save" : "warm-up");
        return 0;
    }
    vgpu98_vk_destroy(vk);
    free(r->passes);
    free(r->fbs);
    free(r->pipes);
    free(r->samps);
    free(r->img);
    free(r->dirty);
    free(r->census_text);
    free(r->mip_recs);
    free(r->churn);
    free(r->pal_kept);
    free(r->pal_sigs);
    free(r->coord_kinds);
    free(r->vsum_buf);
    free(r->vsum_owners);
    free(r->pal_ff);
    if (r->progs) {
        for (i = 1; i <= VGPU98_PROG_MAX_ID; i++)
            free(r->progs[i].ins);
        free(r->progs);
    }
    vgpu98_vin_close(&r->vin);
    free(r->pcache_keys);
    free(r);
    return 1;
}

/* --- Diagnostics ------------------------------------------------------------- */

void
vgpu98_r3d_usage(const vgpu98_r3d_t *r, uint32_t *images, uint64_t *bytes, uint64_t *budget)
{
    *images = (uint32_t) r->n_img;
    *bytes  = r->img_bytes;
    *budget = r->img_budget;
}

void
vgpu98_r3d_vol_usage(const vgpu98_r3d_t *r, uint32_t *volumes, uint64_t *bytes)
{
    *volumes = (uint32_t) r->n_vols;
    *bytes   = r->vol_bytes;
}

void
vgpu98_r3d_wrange(const vgpu98_r3d_t *r, float *wnear, float *wfar)
{
    *wnear = r->wnear;
    *wfar  = r->wfar;
}

static void
app(char *buf, size_t size, size_t *n, const char *fmt, ...)
{
    va_list ap;
    int     k;

    if (*n + 1 >= size)
        return;
    va_start(ap, fmt);
    k = vsnprintf(buf + *n, size - *n, fmt, ap);
    va_end(ap);
    if (k > 0)
        *n += ((size_t) k < size - *n) ? (size_t) k : size - *n - 1;
}

static void
app_surf(char *buf, size_t size, size_t *n, const vgpu98_surf_t *s)
{
    app(buf, size, n, "%06X %ux%u pitch %u format %02X", s->offset, s->width, s->height, s->pitch, s->format);
}

static const char *
depth_vk_name(VkFormat f)
{
    switch (f) {
        case VK_FORMAT_D16_UNORM:
            return "D16_UNORM";
        case VK_FORMAT_D24_UNORM_S8_UINT:
            return "D24_UNORM_S8_UINT";
        case VK_FORMAT_D32_SFLOAT:
            return "D32_SFLOAT";
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return "D32_SFLOAT_S8_UINT";
        default:
            return "?";
    }
}

/* --- Textures in the diagnostics --------------------------------------------- */

static void app_mean(char *buf, size_t size, size_t *n, const vgpu98_r3d_t *r, uint32_t s, const char *lead);

/* A stage's operations, coordinates and transform as the GPU uses them (a
   pre-transformed vertex's stage takes its set as it is, divided in the
   pixel stage when PROJECTED), sampling, and texture with its levels (face
   0 of a cube map), as text (app_mean adds its content's line). */
static void
app_stage(char *buf, size_t size, size_t *n, const vgpu98_r3d_t *r, uint32_t s, int program)
{
    static const char *const src[4] = { "set", "camera normal", "camera position", "reflection" };
    const uint32_t          *t      = r->tss[s];
    const uint32_t           tci    = t[VGPU98_TSS_TEXCOORDINDEX];
    const uint32_t           gen    = program ? ((tci >> VGPU98_TCI_GEN_SHIFT) & 3u) : 0;
    const uint32_t           ttff   = t[VGPU98_TSS_TEXTURETRANSFORMFLAGS];
    const uint32_t           count  = ttff & VGPU98_TTFF_COUNT_MASK;
    const tex_t             *tx     = &r->tex[s];
    uint32_t                 k;

    app(buf, size, n, "colour op %u (%02X, %02X), alpha op %u (%02X, %02X); coordinates ", t[VGPU98_TSS_COLOROP],
        t[VGPU98_TSS_COLORARG1], t[VGPU98_TSS_COLORARG2], t[VGPU98_TSS_ALPHAOP], t[VGPU98_TSS_ALPHAARG1],
        t[VGPU98_TSS_ALPHAARG2]);
    if (gen)
        app(buf, size, n, "%s", src[gen]);
    else
        app(buf, size, n, "set %u", tci & VGPU98_TCI_SET_MASK);
    if (program && count)
        app(buf, size, n, ", transform count %u", count);
    if ((ttff & VGPU98_TTFF_PROJECTED) && (count >= 2))
        app(buf, size, n, ", projected");
    app(buf, size, n, "; address %u/%u, filters %u/%u/%u, anisotropy %u, mip bias %g, largest level %u; ", t[VGPU98_TSS_ADDRESSU],
        t[VGPU98_TSS_ADDRESSV], t[VGPU98_TSS_MAGFILTER], t[VGPU98_TSS_MINFILTER], t[VGPU98_TSS_MIPFILTER],
        t[VGPU98_TSS_MAXANISOTROPY], u2f(t[VGPU98_TSS_MIPMAPLODBIAS]), t[VGPU98_TSS_MAXMIPLEVEL]);
    if (!tx->format) {
        app(buf, size, n, "no texture");
        return;
    }
    if (tx->depth) {
        /* A volume (2.1): offset, pitch and slice pitch of each level. */
        app(buf, size, n, "volume format %02X %ux%ux%u, %u levels at", tx->format, tx->width, tx->height, tx->depth, tx->levels);
        for (k = 0; k < tx->levels; k++)
            app(buf, size, n, " %06X/%u/%u", tx->offset[k], tx->pitch[k], tx->slice[k]);
        return;
    }
    app(buf, size, n, "%s format %02X %ux%u, %u levels at", (tx->faces == VGPU98_CUBE_FACES) ? "cube map" : "texture", tx->format,
        tx->width, tx->height, tx->levels);
    for (k = 0; k < tx->levels; k++)
        app(buf, size, n, " %06X/%u", tx->offset[k], tx->pitch[k]);
    if (tx->faces == VGPU98_CUBE_FACES)
        app(buf, size, n, " (face 0)");
    if (tx->control & VGPU98_TEX_COLORKEY)
        app(buf, size, n, ", colour key");
}

/* A matrix, row by row (Direct3D's order). */
static void
app_matrix(char *buf, size_t size, size_t *n, const float *m)
{
    app(buf, size, n, "%g %g %g %g / %g %g %g %g / %g %g %g %g / %g %g %g %g", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
}

/* The mip check (with captures on): a texture's levels compared with what
   a 2x2 box downsample of the level above gives. Bump formats hold signed
   values a downsample does not compare: only their layout is checked. */
static int
mip_readable(uint32_t format)
{
    return (format != VGPU98_FMT_V8U8) && (format != VGPU98_FMT_L6V5U5) && (format != VGPU98_FMT_X8L8V8U8) &&
           (format != VGPU98_FMT_Q8W8V8U8) && (format != VGPU98_FMT_V16U16) && (format != VGPU98_FMT_A2W10V10U10);
}

/* The DXT block texel_at decoded last, so a downsample's four texels, which
   share one, decode it once. */
typedef struct blk_cache_t {
    size_t  at; /* its offset in video memory, + 1 (0: none) */
    uint8_t blk[64]; /* the 4x4 block, B8G8R8A8 */
} blk_cache_t;

/* A texel of a level in video memory as RGBA floats. */
static void
texel_at(const vgpu98_r3d_t *r, blk_cache_t *bc, uint32_t format, uint32_t offset, uint32_t pitch, uint32_t x, uint32_t y,
         const uint32_t *palette, int pal_alpha, float *c)
{
    const uint8_t *base = r->vram->data + offset;

    if (vgpu98_fmt_is_dxt(format)) {
        const uint32_t bsize = (format == VGPU98_FMT_DXT1) ? 8 : 16;
        const size_t   at    = (size_t) offset + (size_t) (y / 4) * pitch + (size_t) (x / 4) * bsize;
        const uint8_t *t;

        if (bc->at != at + 1) {
            dxt_decode(format, r->vram->data + at, pitch, 4, 4, bc->blk);
            bc->at = at + 1;
        }
        t    = bc->blk + ((y & 3) * 4 + (x & 3)) * 4;
        c[0] = t[2] / 255.0f;
        c[1] = t[1] / 255.0f;
        c[2] = t[0] / 255.0f;
        c[3] = t[3] / 255.0f;
    } else if (format == VGPU98_FMT_P8) {
        const uint32_t v = palette ? palette[base[(size_t) y * pitch + x]] : 0;

        c[0] = ((v >> 16) & 255) / 255.0f;
        c[1] = ((v >> 8) & 255) / 255.0f;
        c[2] = (v & 255) / 255.0f;
        c[3] = pal_alpha ? ((v >> 24) & 255) / 255.0f : 1.0f;
    } else
        texel_rgba(format, base + (size_t) y * pitch + (size_t) x * vgpu98_fmt_bpp(format), c);
}

/* A stage's texture (level 0, face 0) in a capture's text, a line of its
   own after `lead` (none without a 2D texture): its mean colour and the
   share of its texels that are black, from video memory over a grid of at
   most 32x32 (each cell's centre), and where its content comes from: the
   host image the draw samples (image_find, as the draw finds it) uploaded
   from video memory, or drawn into by the GPU (at some time), whether
   that image or another drawn into over the same memory holds texels
   newer than video memory's (the mean is then older). Bump formats hold
   signed values: no mean. */
static void
app_mean(char *buf, size_t size, size_t *n, const vgpu98_r3d_t *r, uint32_t s, const char *lead)
{
    const tex_t       *tx = &r->tex[s];
    const uint32_t     gw = (tx->width < 32) ? tx->width : 32;
    const uint32_t     gh = (tx->height < 32) ? tx->height : 32;
    const r3d_image_t *im;
    blk_cache_t        bc;
    double             sum[4] = { 0.0, 0.0, 0.0, 0.0 };
    uint32_t           black  = 0;
    uint32_t           drawn  = 0;
    uint32_t           newer  = 0;
    uint32_t           row;
    uint32_t           rows;
    uint32_t           x;
    uint32_t           y;
    uint32_t           k;

    if (!tx->format || tx->depth || !gw || !gh)
        return;
    app(buf, size, n, "%sstage %u content:", lead, s);
    if (mip_readable(tx->format)) {
        memset(&bc, 0, sizeof(bc));
        for (y = 0; y < gh; y++) {
            for (x = 0; x < gw; x++) {
                float c[4];

                texel_at(r, &bc, tx->format, tx->offset[0], tx->pitch[0], (uint32_t) (((2ull * x + 1) * tx->width) / (2ull * gw)),
                         (uint32_t) (((2ull * y + 1) * tx->height) / (2ull * gh)), r->pal[s],
                         (tx->control & VGPU98_TEX_PALETTE_ALPHA) != 0, c);
                sum[0] += c[0];
                sum[1] += c[1];
                sum[2] += c[2];
                sum[3] += c[3];
                black += (c[0] + c[1] + c[2]) < (1.5f / 255.0f);
            }
        }
        app(buf, size, n, " mean colour %.3f %.3f %.3f, alpha %.3f, black %.1f%% (level 0, video memory)%s", sum[0] / (gw * gh),
            sum[1] / (gw * gh), sum[2] / (gw * gh), sum[3] / (gw * gh), 100.0 * black / (gw * gh),
            (tx->format == VGPU98_FMT_P8) ? ", through the stage's palette" : "");
    } else
        app(buf, size, n, " a bump format (no mean)");
    /* The image the draw samples (image_find only reads). */
    im = image_find((vgpu98_r3d_t *) r, tx->format, tx->width, tx->height, tx->levels, tx->faces, tx->offset, tx->pitch);
    if (!im) {
        app(buf, size, n, "; not on the host GPU yet\n");
        return;
    }
    for (k = 0; k < im->n_sub; k++)
        drawn |= im->sub[k].att_view != VK_NULL_HANDLE;
    /* Another image over level 0's memory with newer texels (a target
       drawn into apart from the texture). */
    level_layout(tx->format, tx->width, tx->height, &row, &rows);
    for (k = 0; (k < IMG_HASH) && !newer; k++) {
        const r3d_image_t *o;

        for (o = r->img_hash[k]; o && !newer; o = o->hash_next) {
            if ((o != im) && image_overlaps(o, tx->offset[0], tx->offset[0] + (rows - 1) * tx->pitch[0] + row, 1))
                newer = 1;
        }
    }
    app(buf, size, n, "; its image %s%s%s\n", drawn ? "drawn into by the GPU" : "uploaded",
        im->dirty ? ", holding texels newer than video memory's (the mean is older)" : "",
        newer ? "; another image drawn into over its memory holds newer texels" : "");
}

/* The 2x2 box downsample of a parent level at child texel (x, y), and the
   four texels it averages (t: the point samples a chain made by point
   sampling takes one of). */
static void
mip_down(const vgpu98_r3d_t *r, blk_cache_t *bc, uint32_t format, uint32_t pw, uint32_t ph, uint32_t poff, uint32_t ppitch, uint32_t x,
         uint32_t y, const uint32_t *palette, int pal_alpha, float *c, float t[4][4])
{
    const uint32_t x0 = (2 * x < pw) ? 2 * x : pw - 1;
    const uint32_t x1 = (2 * x + 1 < pw) ? 2 * x + 1 : pw - 1;
    const uint32_t y0 = (2 * y < ph) ? 2 * y : ph - 1;
    const uint32_t y1 = (2 * y + 1 < ph) ? 2 * y + 1 : ph - 1;
    int            k;

    texel_at(r, bc, format, poff, ppitch, x0, y0, palette, pal_alpha, t[0]);
    texel_at(r, bc, format, poff, ppitch, x1, y0, palette, pal_alpha, t[1]);
    texel_at(r, bc, format, poff, ppitch, x0, y1, palette, pal_alpha, t[2]);
    texel_at(r, bc, format, poff, ppitch, x1, y1, palette, pal_alpha, t[3]);
    for (k = 0; k < 4; k++)
        c[k] = (t[0][k] + t[1][k] + t[2][k] + t[3][k]) * 0.25f;
}

/* One level against the level above, over at most MIP_SAMPLES x MIP_SAMPLES
   texels spread across it, each at its own place in its cell (so odd rows
   and columns, where a wrong pitch shows, and every texel of a DXT block
   are read):
   - how far it is from the downsample (*score);
   - how far the downsample is from itself moved by an odd number of
     texels, across and down and across only, the larger of the two (no
     power-of-two repeat lines up with an odd move): what unrelated content
     of this texture would score (*base);
   - the downsample's spread, the mean of its channels' standard
     deviations (*spread): under MIP_FLAT the texture is flat;
   - and how far it is from the nearest of the four point samples of the
     level above, the same one at every texel (*point): a chain made by
     point sampling is near one.
   All over the four channels, 0-1. */
#define MIP_SAMPLES 32

static void
mip_measure(const vgpu98_r3d_t *r, uint32_t format, uint32_t pw, uint32_t ph, uint32_t poff, uint32_t ppitch, uint32_t cw,
            uint32_t ch, uint32_t coff, uint32_t cpitch, const uint32_t *palette, int pal_alpha, float *score, float *base,
            float *spread, float *point)
{
    const uint32_t sx = (cw < MIP_SAMPLES) ? cw : MIP_SAMPLES;
    const uint32_t sy = (ch < MIP_SAMPLES) ? ch : MIP_SAMPLES;
    const uint32_t mx = (cw / 3) | 1; /* the odd moves */
    const uint32_t my = (ch / 3) | 1;
    const double   ns = (double) sx * sy;
    blk_cache_t    bc_child;
    blk_cache_t    bc_parent;
    double         ds = 0.0;
    double         da = 0.0;
    double         db = 0.0;
    double         sum[4] = { 0.0, 0.0, 0.0, 0.0 };
    double         sq[4]  = { 0.0, 0.0, 0.0, 0.0 };
    double         sd     = 0.0;
    double         dp[4]  = { 0.0, 0.0, 0.0, 0.0 };
    double         best;
    float          quad[4][4];
    float          other[4][4];
    uint32_t       i;
    uint32_t       j;
    int            k;
    int            q;

    bc_child.at  = 0;
    bc_parent.at = 0;
    for (j = 0; j < sy; j++) {
        const uint32_t y0 = (uint32_t) (((uint64_t) j * ch) / sy);
        const uint32_t hy = (uint32_t) (((uint64_t) (j + 1) * ch) / sy) - y0; /* the cell's height, 1 or more */

        for (i = 0; i < sx; i++) {
            const uint32_t x0 = (uint32_t) (((uint64_t) i * cw) / sx);
            const uint32_t hx = (uint32_t) (((uint64_t) (i + 1) * cw) / sx) - x0;
            uint32_t       h  = (i * 0x9e3779b1u) ^ (j * 0x85ebca77u);
            uint32_t       x;
            uint32_t       y;
            float          child[4];
            float          down[4];
            float          away[4];
            float          side[4];

            h ^= h >> 15;
            h *= 0x2c1b3c6du;
            h ^= h >> 12;
            x = x0 + (h & 0xffffu) % hx;
            y = y0 + (h >> 16) % hy;
            texel_at(r, &bc_child, format, coff, cpitch, x, y, palette, pal_alpha, child);
            mip_down(r, &bc_parent, format, pw, ph, poff, ppitch, x, y, palette, pal_alpha, down, quad);
            mip_down(r, &bc_parent, format, pw, ph, poff, ppitch, (x + mx) % cw, (y + my) % ch, palette, pal_alpha, away, other);
            mip_down(r, &bc_parent, format, pw, ph, poff, ppitch, (x + mx) % cw, y, palette, pal_alpha, side, other);
            for (k = 0; k < 4; k++) {
                ds += fabs((double) child[k] - down[k]);
                da += fabs((double) down[k] - away[k]);
                db += fabs((double) down[k] - side[k]);
                sum[k] += down[k];
                sq[k] += (double) down[k] * down[k];
                for (q = 0; q < 4; q++)
                    dp[q] += fabs((double) child[k] - quad[q][k]);
            }
        }
    }
    for (k = 0; k < 4; k++) {
        const double m = sum[k] / ns;
        const double v = sq[k] / ns - m * m;

        sd += (v > 0.0) ? sqrt(v) : 0.0;
    }
    *score  = (float) (ds / (ns * 4.0));
    *base   = (float) (((da > db) ? da : db) / (ns * 4.0));
    *spread = (float) (sd / 4.0);
    best    = dp[0];
    for (q = 1; q < 4; q++) {
        if (dp[q] < best)
            best = dp[q];
    }
    *point = (float) (best / (ns * 4.0));
}

/* A level is named when it is at least MIP_FLOOR off the downsample and at
   least half as far from it as unrelated content of the texture is (or the
   texture is flat: anything off it shows). */
#define MIP_FLOOR 0.02f
#define MIP_FLAT  0.005f

static int
mip_named(float score, float base, float spread)
{
    return (score >= MIP_FLOOR) && ((spread < MIP_FLAT) || (score >= 0.5f * base));
}

/* A level made by point sampling the one above: one the downsample's test
   would name, at most half as far from a point sample as from the
   downsample, and not named against the point sample. */
static int
mip_point(const mip_rec_t *m, uint32_t l)
{
    return (m->point[l] >= 0.0f) && mip_named(m->score[l], m->base[l], m->spread[l]) && (m->point[l] <= 0.5f * m->score[l]) &&
           !mip_named(m->point[l], m->base[l], m->spread[l]);
}

/* A level that looks wrong: off the downsample, and not a point sample. */
static int
mip_off(const mip_rec_t *m, uint32_t l)
{
    return (m->score[l] >= 0.0f) && mip_named(m->score[l], m->base[l], m->spread[l]) && !mip_point(m, l);
}

/* A texture's levels (face 0) measured, or found measured before: the
   session's records, keyed by the texture's description, so a texture made
   again (after an eviction) is not measured or named twice, and by roughly
   what it holds, so one rewritten in place (or a P8 one given another
   palette) is measured again, MIP_AGAIN times at most. Returns 1 when
   found (out->stale when its content changed since), 0 when measured now,
   2 when measured again. */
#define MIP_AGAIN 3

static uint32_t
mip_key(uint32_t format, uint32_t w, uint32_t h, uint32_t levels, const uint32_t *offset, const uint32_t *pitch)
{
    const uint32_t d[4] = { format, w, h, levels };
    uint32_t       k    = fnv(d, sizeof(d), 0x811c9dc5u);

    k = fnv(offset, levels * sizeof(uint32_t), k);
    return fnv(pitch, levels * sizeof(uint32_t), k);
}

/* Roughly what a texture's levels hold: 8 rows of each, 8 spans of 8 bytes
   along each row (the whole row when it is 64 bytes or less), and a P8
   texture's palette. */
static uint32_t
mip_sig(const vgpu98_r3d_t *r, uint32_t format, uint32_t w, uint32_t h, uint32_t levels, const uint32_t *offset,
        const uint32_t *pitch, const uint32_t *palette, int pal_alpha)
{
    uint32_t k = 0x811c9dc5u;
    uint32_t l;

    for (l = 0; l < levels; l++) {
        uint32_t row;
        uint32_t rows;
        uint32_t i;
        uint32_t j;

        level_layout(format, max1(w >> l), max1(h >> l), &row, &rows);
        for (j = 0; j < 8; j++) {
            const uint8_t *p = r->vram->data + offset[l] + (size_t) (((uint64_t) j * rows) / 8) * pitch[l];

            if (row <= 64)
                k = fnv(p, row, k);
            else {
                for (i = 0; i < 8; i++)
                    k = fnv(p + (size_t) (((uint64_t) i * (row - 8)) / 7), 8, k);
            }
        }
    }
    if (format == VGPU98_FMT_P8)
        k = palette ? fnv(palette, 1024, k ^ (uint32_t) (pal_alpha != 0)) : (k ^ 0x5a5a5a5au);
    return k;
}

static int
mip_results(vgpu98_r3d_t *r, uint32_t format, uint32_t w, uint32_t h, uint32_t levels, const uint32_t *offset,
            const uint32_t *pitch, const uint32_t *palette, int pal_alpha, mip_rec_t *out)
{
    const uint32_t key   = mip_key(format, w, h, levels, offset, pitch);
    const uint32_t sig   = mip_sig(r, format, w, h, levels, offset, pitch, palette, pal_alpha);
    uint32_t       slot  = 0xffffffffu; /* the record measured again */
    uint8_t        again = 0;
    uint32_t       l;
    uint32_t       k;

    for (k = 0; k < r->n_mip_recs; k++) {
        if ((r->mip_recs[k].key == key) && (r->mip_recs[k].levels == levels)) {
            if ((r->mip_recs[k].sig == sig) || (r->mip_recs[k].again >= MIP_AGAIN)) {
                *out       = r->mip_recs[k];
                out->stale = (r->mip_recs[k].sig != sig);
                return 1;
            }
            slot  = k;
            again = (uint8_t) (r->mip_recs[k].again + 1);
            break;
        }
    }
    memset(out, 0, sizeof(*out));
    out->key    = key;
    out->sig    = sig;
    out->again  = again;
    out->levels = (uint8_t) levels;
    out->lo     = 0xffffffffu;
    for (l = 0; l < levels; l++) {
        const uint32_t lw = max1(w >> l);
        const uint32_t lh = max1(h >> l);
        uint32_t       row;
        uint32_t       rows;
        uint32_t       lo;
        uint32_t       hi;

        out->score[l]  = -1.0f;
        out->base[l]   = -1.0f;
        out->spread[l] = -1.0f;
        out->point[l]  = -1.0f;
        level_layout(format, lw, lh, &row, &rows);
        lo             = offset[l];
        hi             = offset[l] + (rows - 1) * pitch[l] + row;
        out->lev_lo[l] = lo;
        out->lev_hi[l] = hi;
        if (lo < out->lo)
            out->lo = lo;
        if (hi > out->hi)
            out->hi = hi;
        /* Spans (pitch gaps included) that overlap another level's. */
        for (k = 0; k < levels; k++) {
            uint32_t krow;
            uint32_t krows;

            if (k == l)
                continue;
            level_layout(format, max1(w >> k), max1(h >> k), &krow, &krows);
            if ((offset[k] < hi) && (lo < offset[k] + (krows - 1) * pitch[k] + krow))
                out->overlap[l] = 1;
        }
        if (l && mip_readable(format) && (lw >= 4) && (lh >= 4))
            mip_measure(r, format, max1(w >> (l - 1)), max1(h >> (l - 1)), offset[l - 1], pitch[l - 1], lw, lh, offset[l], pitch[l],
                        palette, pal_alpha, &out->score[l], &out->base[l], &out->spread[l], &out->point[l]);
    }
    if (slot != 0xffffffffu) {
        r->mip_recs[slot] = *out;
        return 2;
    }
    if (!r->mip_recs)
        r->mip_recs = calloc(MIP_RECS, sizeof(mip_rec_t));
    if (r->mip_recs) {
        /* Full: one replaced in turn, so a texture measured long ago may be
           measured, and counted, again. */
        if (r->n_mip_recs < MIP_RECS)
            r->mip_recs[r->n_mip_recs++] = *out;
        else {
            r->mip_recs[r->mip_rec_next] = *out;
            r->mip_rec_next              = (r->mip_rec_next + 1) % MIP_RECS;
        }
    }
    return 0;
}

/* A DISCARD: the records of textures with a level in the range go (the
   memory is another's now). */
static void
mip_recs_drop(vgpu98_r3d_t *r, uint32_t lo, uint32_t hi)
{
    uint32_t k = 0;

    while (k < r->n_mip_recs) {
        const mip_rec_t *m   = &r->mip_recs[k];
        int              hit = 0;
        uint32_t         l;

        if ((m->lo < hi) && (lo < m->hi)) {
            for (l = 0; (l < m->levels) && !hit; l++)
                hit = (m->lev_lo[l] < hi) && (lo < m->lev_hi[l]);
        }
        if (hit) {
            r->mip_recs[k] = r->mip_recs[--r->n_mip_recs];
            if (r->mip_rec_next > r->n_mip_recs)
                r->mip_rec_next = 0;
        } else
            k++;
    }
}

/* A texture with mip levels at its first use, with captures on: its levels
   are checked once a session, and those that look wrong named (the first
   MIP_LOGGED in a session) and counted. A pointer to look, not a verdict:
   a game may draw its levels as it likes. */
static void
mip_check_image(vgpu98_r3d_t *r, r3d_image_t *im, const uint32_t *palette, int pal_alpha)
{
    uint32_t  off[MAX_LEVELS];
    uint32_t  pit[MAX_LEVELS];
    mip_rec_t rec;
    uint64_t  t0;
    uint32_t  l;
    int       again;

    if (im->mip_checked)
        return;
    im->mip_checked = 1;
    if ((im->levels < 2) || vgpu98_fmt_is_depth(im->format) || !vgpu98_gpu_logging(r->gpu, VGPU98_LOG_CAPTURE))
        return;
    t0 = vgpu98_diag_us();
    for (l = 0; l < im->levels; l++) {
        off[l] = im->sub[l].offset;
        pit[l] = im->sub[l].pitch;
    }
    switch ((again = mip_results(r, im->format, im->width, im->height, im->levels, off, pit, palette, pal_alpha, &rec))) {
        case 0:
            r->stats->mip_textures++;
            break;
        case 2:
            r->stats->mip_again++;
            break;
        default: /* measured before */
            r->stats->mip_us += us_since(t0);
            return;
    }
    {
        for (l = 1; l < im->levels; l++) {
            const int off_it = mip_off(&rec, l);

            if (!off_it && !rec.overlap[l])
                continue;
            r->stats->mip_named++;
            if (r->mip_logged >= MIP_LOGGED)
                continue;
            r->mip_logged++;
            /* Under r3d_log's 256 bytes at worst. */
            if (off_it)
                r3d_log(r, VGPU98_LOG_CAPTURE,
                        "VGPU98 mip check: texture at %06X, format %02X %ux%u, %u levels: level %u (%ux%u at %06X, pitch %u) is %.1f%% "
                        "off a box downsample, %.1f%% off a point sample, unrelated %.1f%%%s%s\n",
                        im->sub[0].offset, im->format, im->width, im->height, im->levels, l, im->sub[l].width, im->sub[l].height,
                        im->sub[l].offset, im->sub[l].pitch, rec.score[l] * 100.0f, rec.point[l] * 100.0f, rec.base[l] * 100.0f,
                        rec.overlap[l] ? "; span overlaps another level's" : "", (again == 2) ? ", measured again" : "");
            else
                r3d_log(r, VGPU98_LOG_CAPTURE,
                        "VGPU98 mip check: texture at %06X, format %02X %ux%u, %u levels: level %u (%ux%u at %06X, pitch %u): its span "
                        "overlaps another level's%s\n",
                        im->sub[0].offset, im->format, im->width, im->height, im->levels, l, im->sub[l].width, im->sub[l].height,
                        im->sub[l].offset, im->sub[l].pitch, (again == 2) ? ", measured again" : "");
        }
    }
    r->stats->mip_us += us_since(t0);
}

/* Command set 2.1's states, for the census and captures: points and
   sprites, the palette and tweening, the blend operation and write mask. */
static void
app_states8(char *buf, size_t size, size_t *n, const vgpu98_r3d_t *r, const char *lead)
{
    app(buf, size, n,
        "%s2.1 states: POINTSIZE %g (%g to %g), POINTSPRITEENABLE %u, POINTSCALEENABLE %u (%g, %g, %g); "
        "INDEXEDVERTEXBLENDENABLE %u, VERTEXBLEND %u (palette to %u), TWEENFACTOR %g; BLENDOP %u, COLORWRITEENABLE %X\n",
        lead, u2f(r->rs[VGPU98_RS_POINTSIZE]), u2f(r->rs[VGPU98_RS_POINTSIZE_MIN]), u2f(r->rs[VGPU98_RS_POINTSIZE_MAX]),
        r->rs[VGPU98_RS_POINTSPRITEENABLE], r->rs[VGPU98_RS_POINTSCALEENABLE], u2f(r->rs[VGPU98_RS_POINTSCALE_A]),
        u2f(r->rs[VGPU98_RS_POINTSCALE_B]), u2f(r->rs[VGPU98_RS_POINTSCALE_C]), r->rs[VGPU98_RS_INDEXEDVERTEXBLENDENABLE],
        r->rs[VGPU98_RS_VERTEXBLEND], r->pal_max, u2f(r->rs[VGPU98_RS_TWEENFACTOR]), r->rs[VGPU98_RS_BLENDOP],
        r->rs[VGPU98_RS_COLORWRITEENABLE]);
}

/* The draw census (vgpu98_r3d_census_start): the first draw of each kind
   (the declaration, and every active stage's operations, coordinates,
   transform, sampling and texture) described in full, CENSUS_KINDS with
   generated coordinates or a texture transform and as many others; the
   kinds past those, or past the text's room, are counted. */
static void
census_draw(vgpu98_r3d_t *r, const vgpu98_vin_draw_t *d)
{
    const uint32_t stages  = (uint32_t) active_stages(r);
    const int      program = r->vin.program;
    const size_t   size    = 16384;
    uint32_t       h       = fnv(&program, sizeof(program), 0x811c9dc5u);
    int            gen     = 0;
    char          *text;
    size_t         n = 0;
    uint32_t       s;
    uint32_t       i;

    h = fnv(r->vin.decl, 2 * r->vin.n_elements * sizeof(uint32_t), h);
    {
        /* The states its line names (a decal pass differs by its ZBIAS). */
        const uint32_t rs[13] = { r->have_depth ? r->rs[VGPU98_RS_ZENABLE] : 0, r->rs[VGPU98_RS_ZWRITEENABLE], r->rs[VGPU98_RS_ZFUNC],
                                  r->rs[VGPU98_RS_ZBIAS], r->rs[VGPU98_RS_ALPHABLENDENABLE], r->rs[VGPU98_RS_SRCBLEND],
                                  r->rs[VGPU98_RS_DESTBLEND], r->rs[VGPU98_RS_ALPHATESTENABLE], r->rs[VGPU98_RS_FOGENABLE],
                                  r->rs[VGPU98_RS_LIGHTING], r->rs[VGPU98_RS_FOGTABLEMODE], r->rs[VGPU98_RS_FOGVERTEXMODE],
                                  r->rs[VGPU98_RS_DITHERENABLE] };

        h = fnv(rs, sizeof(rs), h);
    }
    if (fixed8(r)) {
        /* Command set 2.1's: the streams' strides, sprites, blending,
           masks, the palette (not the sizes: particles vary them per
           draw). */
        uint32_t rs8[11] = { r->rs[VGPU98_RS_POINTSPRITEENABLE], r->rs[VGPU98_RS_POINTSCALEENABLE],
                             r->rs[VGPU98_RS_INDEXEDVERTEXBLENDENABLE], r->rs[VGPU98_RS_VERTEXBLEND], r->rs[VGPU98_RS_BLENDOP],
                             r->rs[VGPU98_RS_COLORWRITEENABLE], r->vin.streams_used, 0, 0, 0, 0 };

        for (s = 0; s < VGPU98_MAX_STREAMS; s++) {
            if ((r->vin.streams_used >> s) & 1)
                rs8[7 + (s & 3)] = rs8[7 + (s & 3)] * 31 + r->vin.stream[s].stride;
        }
        h = fnv(rs8, sizeof(rs8), h);
    }
    for (s = 0; s < stages; s++) {
        const uint32_t *t       = r->tss[s];
        const tex_t    *tx      = &r->tex[s];
        const uint32_t  key[21] = { t[VGPU98_TSS_COLOROP],      t[VGPU98_TSS_ALPHAOP],    t[VGPU98_TSS_TEXCOORDINDEX],
                                    t[VGPU98_TSS_TEXTURETRANSFORMFLAGS], t[VGPU98_TSS_MAGFILTER], t[VGPU98_TSS_MINFILTER],
                                    t[VGPU98_TSS_MIPFILTER],    t[VGPU98_TSS_MAXANISOTROPY], t[VGPU98_TSS_MIPMAPLODBIAS],
                                    t[VGPU98_TSS_MAXMIPLEVEL],  t[VGPU98_TSS_ADDRESSU],   t[VGPU98_TSS_ADDRESSV],
                                    tx->format,                 tx->width,                tx->height,
                                    tx->levels,                 tx->offset[0],            t[VGPU98_TSS_COLORARG1],
                                    t[VGPU98_TSS_COLORARG2],    t[VGPU98_TSS_ALPHAARG1],  t[VGPU98_TSS_ALPHAARG2] };

        h = fnv(key, sizeof(key), h);
        if (program && (((t[VGPU98_TSS_TEXCOORDINDEX] >> VGPU98_TCI_GEN_SHIFT) & 3u) ||
                        (t[VGPU98_TSS_TEXTURETRANSFORMFLAGS] & VGPU98_TTFF_COUNT_MASK)))
            gen = 1;
    }
    for (i = 0; i < r->n_census_seen; i++) {
        if (r->census_seen[i] == h)
            return;
    }
    if (r->n_census_seen < CENSUS_SEEN)
        r->census_seen[r->n_census_seen++] = h;
    else {
        /* Past the list: a bit each, so a kind counts once (two that share
           a bit count as one: the count is at least). */
        const uint32_t b = h % CENSUS_BITS;

        if (r->census_bits[b / 32] & (1u << (b % 32)))
            return;
        r->census_bits[b / 32] |= 1u << (b % 32);
        r->census_lost = 1;
    }
    if (((gen ? r->census_gen : r->census_other) >= CENSUS_KINDS) || (r->census_len >= CENSUS_MAX)) {
        r->census_untold++;
        return;
    }
    text = malloc(size);
    if (!text) {
        r->census_untold++;
        return;
    }
    if (gen)
        r->census_gen++;
    else
        r->census_other++;
    text[0] = 0;
    app(text, size, &n, "  Draw kind %u, %+d ms: primitive %u, %u %s, window from %lld (%u vertices), %s\n",
        r->census_gen + r->census_other, (int) ((vgpu98_diag_us() - r->census_us) / 1000), d->prim, d->count,
        d->indexed ? "indices" : "vertices", (long long) d->first, d->nverts, program ? "transformed" : "pre-transformed");
    app(text, size, &n, "    declaration:");
    for (i = 0; i < r->vin.n_elements; i++) {
        const uint32_t d0 = r->vin.decl[2 * i];

        app(text, size, &n, " s%u+%u type %u use %u.%u", VGPU98_DECL_STREAM(d0), r->vin.decl[2 * i + 1], VGPU98_DECL_TYPE(d0),
            VGPU98_DECL_USE(d0), VGPU98_DECL_INDEX(d0));
    }
    {
        const char *sep = "\n    streams (source, stride):";

        for (i = 0; i < VGPU98_MAX_STREAMS; i++) {
            if (!((r->vin.streams_used >> i) & 1))
                continue;
            /* Short: 16 of them stay within the log's 500 characters. */
            app(text, size, &n, "%s s%u %s %u%s", sep, i, (r->vin.stream[i].source == VGPU98_SRC_INLINE) ? "inline" : "GPU",
                r->vin.stream[i].stride, ((r->vin.res_mask >> i) & 1) ? " resident" : "");
            sep = ",";
        }
    }
    if (d->indexed)
        app(text, size, &n, "; %u-bit indices %s", 8 * r->vin.index.isize,
            (r->vin.index.source == VGPU98_SRC_INLINE) ? "inline" : "GPU");
    app(text, size, &n,
        "\n    states: ZENABLE %u ZWRITEENABLE %u ZFUNC %u ZBIAS %u, ALPHABLENDENABLE %u (%u, %u), ALPHATESTENABLE %u, FOGENABLE %u "
        "(table %u, vertex %u), LIGHTING %u, DITHERENABLE %u\n",
        r->have_depth ? r->rs[VGPU98_RS_ZENABLE] : 0, r->rs[VGPU98_RS_ZWRITEENABLE], r->rs[VGPU98_RS_ZFUNC], r->rs[VGPU98_RS_ZBIAS],
        r->rs[VGPU98_RS_ALPHABLENDENABLE], r->rs[VGPU98_RS_SRCBLEND], r->rs[VGPU98_RS_DESTBLEND], r->rs[VGPU98_RS_ALPHATESTENABLE],
        r->rs[VGPU98_RS_FOGENABLE], r->rs[VGPU98_RS_FOGTABLEMODE], r->rs[VGPU98_RS_FOGVERTEXMODE], r->rs[VGPU98_RS_LIGHTING],
        r->rs[VGPU98_RS_DITHERENABLE]);
    if (fixed8(r))
        app_states8(text, size, &n, r, "    ");
    for (s = 0; s < stages; s++) {
        const tex_t *tx = &r->tex[s];

        app(text, size, &n, "    stage %u: ", s);
        app_stage(text, size, &n, r, s, program);
        app(text, size, &n, "\n");
        app_mean(text, size, &n, r, s, "    ");
        if (program && (r->tss[s][VGPU98_TSS_TEXTURETRANSFORMFLAGS] & VGPU98_TTFF_COUNT_MASK)) {
            app(text, size, &n, "    stage %u matrix: ", s);
            app_matrix(text, size, &n, r->xform[VGPU98_XFORM_TEXTURE + s]);
            app(text, size, &n, "\n");
        }
        if (tx->format && !tx->depth && (tx->levels > 1)) {
            mip_rec_t rec;
            uint32_t  l;

            mip_results(r, tx->format, tx->width, tx->height, tx->levels, tx->offset, tx->pitch, r->pal[s],
                        (tx->control & VGPU98_TEX_PALETTE_ALPHA) != 0, &rec);
            /* Four levels a line, so no line passes the log's 500 characters. */
            for (l = 1; l < tx->levels; l++) {
                if ((l % 4) == 1)
                    app(text, size, &n, "    stage %u levels (off a box downsample of the one above, off a point sample; unrelated content)%s:",
                        s, rec.stale ? ", measured before its content last changed" : "");
                if (rec.score[l] >= 0.0f)
                    app(text, size, &n, " %u: %.1f%%, %.1f%%; %.1f%%%s", l, rec.score[l] * 100.0f, rec.point[l] * 100.0f,
                        rec.base[l] * 100.0f, mip_off(&rec, l) ? " (off)" : mip_point(&rec, l) ? " (point-sampled)" : "");
                else
                    app(text, size, &n, " %u: -", l);
                if (rec.overlap[l])
                    app(text, size, &n, " (span overlaps another level's)");
                if (((l % 4) == 0) || (l + 1 == tx->levels))
                    app(text, size, &n, "\n");
            }
        }
    }
    /* Into the census's text. */
    if (r->census_len + n + 1 > r->census_cap) {
        size_t cap = r->census_cap ? 2 * r->census_cap : 32768;
        char  *t;

        while (cap < r->census_len + n + 1)
            cap *= 2;
        t = realloc(r->census_text, cap);
        if (!t) {
            free(text);
            r->census_untold++;
            return;
        }
        r->census_text = t;
        r->census_cap  = cap;
    }
    memcpy(r->census_text + r->census_len, text, n + 1);
    r->census_len += n;
    free(text);
}

/* --- The texture dumps ------------------------ */

#    define DUMP_MAX       32          /* pictures a capture: a texture's level, or a cube map's face */
#    define DUMP_SESSION   256         /* and a session */
#    define DUMP_SIDE      1024        /* the longest side a level may have unshrunk; a picture past it is downscaled */
#    define DUMP_LEVEL_MAX (8u << 20)  /* host bytes a level read back may have */
#    define DUMP_BYTES     (64u << 20) /* host and guest bytes a capture's pictures hold together */

/* A picture as the job writes it: the host's copy as read back and the
   guest's bytes, decoded and compared on the job's thread. */
typedef struct dump_pic_t {
    char               name[24]; /* its files' names, before "-host.png" and "-guest.png" */
    char               what[176]; /* what it is, for its comparison line */
    uint32_t           format;
    uint32_t           w;
    uint32_t           h;
    VkFormat           vk; /* the host image's format, swizzle and conversion */
    VkComponentMapping swz;
    int                conv;
    uint32_t           hbytes; /* host bytes a texel */
    uint8_t           *host;   /* w x h host texels */
    uint8_t           *guest;  /* the level's rows in video memory, `grow` bytes each (rows of blocks for DXT) */
    uint32_t           grow;
    uint32_t           palette[256];
    int                pal;    /* P8: 1 with the palette its texels went through, 0 without one kept */
    int                pal_alpha;
    int                newer;  /* the host's copy holds pixels drawn on the GPU, newer than video memory */
    int                changed; /* video memory changed since its last upload (1) or while it was read (2): not compared */
} dump_pic_t;

typedef struct dump_job_t {
    char       folder[1024];
    uint32_t   capture;
    uint32_t   n;
    dump_pic_t pic[DUMP_MAX];
} dump_job_t;

/* dump_busy (declared above): a job's thread is writing, one at a time for
   every renderer; the job owns its memory and frees it, and
   vgpu98_r3d_close waits for it a while. */

/* The components of a format a picture shows signed, from -1 to 1 (a bit
   each, R first): bump maps' du and dv, Q8W8V8U8's four, 2.2's signed ones. */
static uint32_t
dump_signed(uint32_t format)
{
    switch (format) {
        case VGPU98_FMT_V8U8:
        case VGPU98_FMT_L6V5U5:
        case VGPU98_FMT_X8L8V8U8:
        case VGPU98_FMT_V16U16:
            return 0x3;
        case VGPU98_FMT_A2W10V10U10:
            return 0x7;
        case VGPU98_FMT_Q8W8V8U8:
            return 0xf;
        default:
            return 0;
    }
}

/* Does the guest's picture of a format come through the upload's own
   conversion (texel_rgba, which only the converted formats go through)? Its
   match then shows the host's copy is current, not that the conversion is
   right. */
static int
dump_by_upload(const dump_pic_t *t)
{
    return (t->conv == CONV_F16) || (t->conv == CONV_BUMP) || (t->conv == CONV_WIDE);
}

/* A palette kept by its signature (the image's pal_sig), so a dump decodes
   an 8-bit texture with the palette its texels went through. */
static void
pal_keep(vgpu98_r3d_t *r, uint32_t sig, const uint32_t *palette, int alpha)
{
    uint32_t i;

    if (!r->pal_sigs && !(r->pal_sigs = calloc(PAL_KEPT, sizeof(uint32_t))))
        return;
    if (!r->pal_kept && !(r->pal_kept = calloc(PAL_KEPT, sizeof(pal_kept_t))))
        return;
    for (i = 0; i < ((r->pal_next < PAL_KEPT) ? r->pal_next : PAL_KEPT); i++) {
        if (r->pal_sigs[i] == sig)
            return;
    }
    i                        = r->pal_next++ % PAL_KEPT;
    r->pal_sigs[i]           = sig;
    r->pal_kept[i].used      = 1;
    r->pal_kept[i].sig       = sig;
    r->pal_kept[i].alpha     = alpha;
    memcpy(r->pal_kept[i].c, palette, sizeof(r->pal_kept[i].c));
}

static const pal_kept_t *
pal_find(const vgpu98_r3d_t *r, uint32_t sig)
{
    uint32_t i;

    for (i = 0; r->pal_kept && r->pal_sigs && (i < ((r->pal_next < PAL_KEPT) ? r->pal_next : PAL_KEPT)); i++) {
        if (r->pal_sigs[i] == sig)
            return &r->pal_kept[i];
    }
    return NULL;
}

static float
unorm(uint32_t v, uint32_t bits)
{
    return (float) v / (float) ((1u << bits) - 1u);
}

/* A component in 8 bits; a signed one (bump maps, 2.2's signed formats)
   shown from -1 to 1. */
static uint8_t
to8(float f, int sgn)
{
    if (sgn)
        f = f * 0.5f + 0.5f;
    if (!(f > 0.0f))
        return 0;
    if (f >= 1.0f)
        return 255;
    return (uint8_t) (f * 255.0f + 0.5f);
}

/* A texel of the host's copy: its Vulkan format's R, G, B and A, as the
   format defines them (c / (2^n - 1) for UNORM). */
static void
host_texel(VkFormat vk, const uint8_t *p, float *c)
{
    uint32_t v;
    int      k;

    c[0] = c[1] = c[2] = 0.0f;
    c[3]               = 1.0f;
    switch (vk) {
        case VK_FORMAT_R5G6B5_UNORM_PACK16:
            v    = rd16(p);
            c[0] = unorm((v >> 11) & 31, 5);
            c[1] = unorm((v >> 5) & 63, 6);
            c[2] = unorm(v & 31, 5);
            break;
        case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
            v    = rd16(p);
            c[0] = unorm((v >> 10) & 31, 5);
            c[1] = unorm((v >> 5) & 31, 5);
            c[2] = unorm(v & 31, 5);
            c[3] = (float) (v >> 15);
            break;
        case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
            v    = rd16(p);
            c[0] = unorm((v >> 4) & 15, 4);
            c[1] = unorm((v >> 8) & 15, 4);
            c[2] = unorm(v >> 12, 4);
            c[3] = unorm(v & 15, 4);
            break;
        case VK_FORMAT_B8G8R8A8_UNORM:
            c[0] = unorm(p[2], 8);
            c[1] = unorm(p[1], 8);
            c[2] = unorm(p[0], 8);
            c[3] = unorm(p[3], 8);
            break;
        case VK_FORMAT_R8_UNORM:
            c[0] = unorm(p[0], 8);
            break;
        case VK_FORMAT_R8G8_UNORM:
            c[0] = unorm(p[0], 8);
            c[1] = unorm(p[1], 8);
            break;
        case VK_FORMAT_R4G4_UNORM_PACK8:
            c[0] = unorm(p[0] >> 4, 4);
            c[1] = unorm(p[0] & 15, 4);
            break;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            for (k = 0; k < 4; k++)
                c[k] = h2f(rd16(p + 2 * k));
            break;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            memcpy(c, p, 16);
            break;
        default:
            break;
    }
}

static float
swz_pick(VkComponentSwizzle s, const float *c, int self)
{
    switch (s) {
        case VK_COMPONENT_SWIZZLE_ZERO:
            return 0.0f;
        case VK_COMPONENT_SWIZZLE_ONE:
            return 1.0f;
        case VK_COMPONENT_SWIZZLE_R:
            return c[0];
        case VK_COMPONENT_SWIZZLE_G:
            return c[1];
        case VK_COMPONENT_SWIZZLE_B:
            return c[2];
        case VK_COMPONENT_SWIZZLE_A:
            return c[3];
        default:
            return c[self];
    }
}

/* The host's copy as the GPU samples it (its format, then its swizzle),
   8 bits a component, RGBA. */
static void
dump_host_rgba(const dump_pic_t *t, uint8_t *out)
{
    const uint32_t sgn = dump_signed(t->format);
    size_t         i;

    for (i = 0; i < (size_t) t->w * t->h; i++) {
        float c[4];

        host_texel(t->vk, t->host + i * t->hbytes, c);
        out[4 * i + 0] = to8(swz_pick(t->swz.r, c, 0), sgn & 1);
        out[4 * i + 1] = to8(swz_pick(t->swz.g, c, 1), (sgn >> 1) & 1);
        out[4 * i + 2] = to8(swz_pick(t->swz.b, c, 2), (sgn >> 2) & 1);
        out[4 * i + 3] = to8(swz_pick(t->swz.a, c, 3), (sgn >> 3) & 1);
    }
}

/* A DXT surface decoded on its own (the DirectX documents' S3TC rules, not
   the upload's dxt_decode): RGBA, 8 bits a component. */
static void
dump_dxt(const dump_pic_t *t, uint8_t *out)
{
    const int dxt1  = (t->format == VGPU98_FMT_DXT1);
    const int bsize = dxt1 ? 8 : 16;
    uint32_t  bx;
    uint32_t  by;

    for (by = 0; by < (t->h + 3) / 4; by++) {
        for (bx = 0; bx < (t->w + 3) / 4; bx++) {
            const uint8_t *b  = t->guest + (size_t) by * t->grow + (size_t) bx * bsize;
            const uint8_t *cb = dxt1 ? b : b + 8;
            const uint32_t c0 = rd16(cb);
            const uint32_t c1 = rd16(cb + 2);
            const uint32_t ix = rd32(cb + 4);
            float          e[4][4];
            uint64_t       abits = 0;
            int            i;
            int            k;

            e[0][0] = unorm((c0 >> 11) & 31, 5), e[0][1] = unorm((c0 >> 5) & 63, 6), e[0][2] = unorm(c0 & 31, 5);
            e[1][0] = unorm((c1 >> 11) & 31, 5), e[1][1] = unorm((c1 >> 5) & 63, 6), e[1][2] = unorm(c1 & 31, 5);
            e[0][3] = e[1][3] = e[2][3] = e[3][3] = 1.0f;
            for (k = 0; k < 3; k++) {
                if (!dxt1 || (c0 > c1)) {
                    e[2][k] = (2.0f * e[0][k] + e[1][k]) / 3.0f;
                    e[3][k] = (e[0][k] + 2.0f * e[1][k]) / 3.0f;
                } else {
                    e[2][k] = (e[0][k] + e[1][k]) / 2.0f;
                    e[3][k] = 0.0f;
                }
            }
            if (dxt1 && (c0 <= c1))
                e[3][3] = 0.0f; /* transparent black */
            for (k = 0; k < 6; k++)
                abits |= (uint64_t) b[2 + k] << (8 * k);
            for (i = 0; i < 16; i++) {
                const uint32_t x  = bx * 4 + (i & 3);
                const uint32_t y  = by * 4 + (i >> 2);
                const float   *cc = e[(ix >> (2 * i)) & 3];
                float          a  = cc[3];
                uint8_t       *o;

                if ((x >= t->w) || (y >= t->h))
                    continue;
                if ((t->format == VGPU98_FMT_DXT2) || (t->format == VGPU98_FMT_DXT3))
                    a = unorm((b[i / 2] >> ((i & 1) * 4)) & 15, 4);
                else if ((t->format == VGPU98_FMT_DXT4) || (t->format == VGPU98_FMT_DXT5)) {
                    const float    a0   = b[0];
                    const float    a1   = b[1];
                    const uint32_t code = (uint32_t) (abits >> (3 * i)) & 7;

                    if (code == 0)
                        a = a0;
                    else if (code == 1)
                        a = a1;
                    else if (b[0] > b[1])
                        a = ((float) (8 - code) * a0 + (float) (code - 1) * a1) / 7.0f;
                    else if (code < 6)
                        a = ((float) (6 - code) * a0 + (float) (code - 1) * a1) / 5.0f;
                    else
                        a = (code == 6) ? 0.0f : 255.0f;
                    a /= 255.0f;
                }
                o    = out + ((size_t) y * t->w + x) * 4;
                o[0] = to8(cc[0], 0);
                o[1] = to8(cc[1], 0);
                o[2] = to8(cc[2], 0);
                o[3] = to8(a, 0);
            }
        }
    }
}

/* The guest's bytes as the GPU should sample them: DXT by dump_dxt, P8
   through its palette, the others by texel_rgba (the upload's own
   conversion only for those it converts: bump maps, 2.2's formats, a
   format the host GPU lacks). Without a palette kept, an 8-bit texture's
   indices show as grey. */
static void
dump_guest_rgba(const dump_pic_t *t, uint8_t *out)
{
    const int      bpp = vgpu98_fmt_bpp(t->format);
    const uint32_t sgn = dump_signed(t->format);
    uint32_t       x;
    uint32_t       y;

    if (vgpu98_fmt_is_dxt(t->format)) {
        dump_dxt(t, out);
        return;
    }
    for (y = 0; y < t->h; y++) {
        for (x = 0; x < t->w; x++) {
            const uint8_t *s = t->guest + (size_t) y * t->grow + (size_t) x * bpp;
            uint8_t       *o = out + ((size_t) y * t->w + x) * 4;

            if (t->format == VGPU98_FMT_P8) {
                uint32_t c = t->pal ? t->palette[*s] : (0xff000000u | (*s * 0x010101u));

                if (t->pal && !t->pal_alpha)
                    c |= 0xff000000u;
                o[0] = (uint8_t) (c >> 16);
                o[1] = (uint8_t) (c >> 8);
                o[2] = (uint8_t) c;
                o[3] = (uint8_t) (c >> 24);
            } else {
                float c[4];

                texel_rgba(t->format, s, c);
                o[0] = to8(c[0], sgn & 1);
                o[1] = to8(c[1], (sgn >> 1) & 1);
                o[2] = to8(c[2], (sgn >> 2) & 1);
                o[3] = to8(c[3], (sgn >> 3) & 1);
            }
        }
    }
}

/* A picture past DUMP_SIDE, shrunk by a whole factor (box filter). */
static uint8_t *
dump_shrink(const uint8_t *src, uint32_t w, uint32_t h, uint32_t f, uint32_t *ow, uint32_t *oh)
{
    const uint32_t nw = (w + f - 1) / f;
    const uint32_t nh = (h + f - 1) / f;
    uint8_t       *d  = malloc((size_t) nw * nh * 4);
    uint32_t       x;
    uint32_t       y;

    if (!d)
        return NULL;
    for (y = 0; y < nh; y++) {
        for (x = 0; x < nw; x++) {
            uint32_t sum[4] = { 0, 0, 0, 0 };
            uint32_t n      = 0;
            uint32_t i;
            uint32_t j;
            int      k;

            for (j = y * f; (j < y * f + f) && (j < h); j++) {
                for (i = x * f; (i < x * f + f) && (i < w); i++, n++) {
                    for (k = 0; k < 4; k++)
                        sum[k] += src[((size_t) j * w + i) * 4 + k];
                }
            }
            for (k = 0; k < 4; k++)
                d[((size_t) y * nw + x) * 4 + k] = (uint8_t) ((sum[k] + n / 2) / n);
        }
    }
    *ow = nw;
    *oh = nh;
    return d;
}

static int
dump_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h)
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

/* The job's thread: each picture decoded both ways, compared texel by
   texel (within 1 a component, 4 for DXT, whose two decoders round apart),
   shrunk past DUMP_SIDE, written; a line each, then one for the job. */
static void
dump_thread(void *priv)
{
    dump_job_t    *j     = (dump_job_t *) priv;
    const uint64_t t0    = vgpu98_diag_us();
    uint32_t       files = 0;
    uint32_t       i;
    char           stamp[24];

    for (i = 0; i < j->n; i++) {
        dump_pic_t    *t   = &j->pic[i];
        const size_t   px  = (size_t) t->w * t->h;
        const int      tol = vgpu98_fmt_is_dxt(t->format) ? 4 : 1;
        uint8_t       *hb  = malloc(px * 4);
        uint8_t       *gb  = malloc(px * 4);
        const uint8_t *hw  = hb;
        const uint8_t *gw  = gb;
        uint8_t       *hs  = NULL;
        uint8_t       *gs  = NULL;
        uint32_t       ow  = t->w;
        uint32_t       oh  = t->h;
        char           res[224];
        char           path[1100];

        if (!hb || !gb) {
            snprintf(res, sizeof(res), "no memory to decode it");
            goto told;
        }
        dump_host_rgba(t, hb);
        dump_guest_rgba(t, gb);
        if (t->newer)
            snprintf(res, sizeof(res), "drawn into on the host GPU (newer than video memory): not compared");
        else if (t->changed == 1)
            snprintf(res, sizeof(res), "video memory changed since its last upload (it uploads again when next drawn): not compared");
        else if (t->changed == 2)
            snprintf(res, sizeof(res), "video memory changed while it was read: not compared");
        else if ((t->format == VGPU98_FMT_P8) && !t->pal)
            snprintf(res, sizeof(res), "the palette its texels went through was not kept (the guest's picture shows its indices as grey): not compared");
        else {
            size_t   k;
            size_t   first  = px;
            uint32_t differ = 0;

            for (k = 0; k < px; k++) {
                int c;

                for (c = 0; c < 4; c++) {
                    const int d = (int) hb[4 * k + c] - (int) gb[4 * k + c];

                    if ((d > tol) || (d < -tol))
                        break;
                }
                if (c < 4) {
                    if (first == px)
                        first = k;
                    differ++;
                }
            }
            if (!differ)
                snprintf(res, sizeof(res), "the host's copy matches the guest's bytes%s",
                         dump_by_upload(t) ? " (decoded by the upload's own conversion: the copy is current; the conversion is not checked)"
                                           : "");
            else
                snprintf(res, sizeof(res),
                         "the host's copy differs from the guest's bytes in %u of %u texels; the first at (%u, %u): host %02X%02X%02X%02X, "
                         "guest %02X%02X%02X%02X (RGBA)",
                         differ, (uint32_t) px, (uint32_t) (first % t->w), (uint32_t) (first / t->w), hb[4 * first],
                         hb[4 * first + 1], hb[4 * first + 2], hb[4 * first + 3], gb[4 * first], gb[4 * first + 1],
                         gb[4 * first + 2], gb[4 * first + 3]);
        }
        if ((t->w > DUMP_SIDE) || (t->h > DUMP_SIDE)) {
            const uint32_t f = (((t->w > t->h) ? t->w : t->h) + DUMP_SIDE - 1) / DUMP_SIDE;

            hs = dump_shrink(hb, t->w, t->h, f, &ow, &oh);
            gs = dump_shrink(gb, t->w, t->h, f, &ow, &oh);
            if (hs && gs) {
                hw = hs;
                gw = gs;
            } else
                ow = t->w, oh = t->h;
        }
        snprintf(path, sizeof(path), "%s%c%s-host.png", j->folder, '/', t->name);
        files += (uint32_t) dump_png(path, hw, ow, oh);
        snprintf(path, sizeof(path), "%s%c%s-guest.png", j->folder, '/', t->name);
        files += (uint32_t) dump_png(path, gw, ow, oh);
told:
        vgpu98_log_stamp(stamp, sizeof(stamp));
        pclog("%s VGPU98 capture %u textures: %s, %s: %s\n", stamp, j->capture, t->name, t->what, res);
        free(hs);
        free(gs);
        free(hb);
        free(gb);
        free(t->host);
        free(t->guest);
    }
    vgpu98_log_stamp(stamp, sizeof(stamp));
    pclog("%s VGPU98 capture %u textures: %u files written to %s in %.1f s\n", stamp, j->capture, files, j->folder,
          (vgpu98_diag_us() - t0) / 1e6);
    free(j);
    atomic_store(&dump_busy, 0);
}

/* The level a texture's pictures show: the first DUMP_SIDE or less on its
   long side whose host bytes are DUMP_LEVEL_MAX or less and whose pictures
   (every face) fit the capture's bytes left, else the first that fits
   (shrunk on the job's thread); -1 for none. */
static int
dump_level(const r3d_image_t *im, uint64_t left)
{
    int      fit = -1;
    uint32_t l;

    for (l = 0; l < im->levels; l++) {
        const r3d_sub_t *s     = &im->sub[l];
        const uint64_t   hostb = (uint64_t) s->width * s->height * im->hf->bytes;
        uint32_t         row;
        uint32_t         rows;

        level_layout(im->format, s->width, s->height, &row, &rows);
        if ((hostb > DUMP_LEVEL_MAX) || ((hostb + (uint64_t) row * rows) * im->faces > left))
            continue;
        if ((s->width <= DUMP_SIDE) && (s->height <= DUMP_SIDE))
            return (int) l;
        if (fit < 0)
            fit = (int) l;
    }
    return fit;
}

/* The current versions of the pages under [lo, hi), as one signature. */
static uint32_t
pages_vsig(const vgpu98_r3d_t *r, uint32_t lo, uint32_t hi)
{
    uint32_t h = 0x811c9dc5u;
    uint32_t p;

    for (p = lo >> VGPU98_PAGE_SHIFT; p <= ((hi - 1) >> VGPU98_PAGE_SHIFT); p++)
        h = (h ^ vgpu98_page_version(r->vram, p)) * 0x01000193u;
    return h;
}

/* A level or face of an image (or, with `vol`, a volume's level's first
   slice) read back from the host GPU into a picture, with its guest bytes.
   Returns the bytes the picture holds, 0 when it could not be had. */
static uint64_t
dump_read(vgpu98_r3d_t *r, dump_pic_t *t, r3d_image_t *im, const r3d_sub_t *s, r3d_vol_t *vol, uint32_t level)
{
    const hfmt_t      *hf     = vol ? vol->hf : im->hf;
    const uint32_t     format = vol ? vol->format : im->format;
    const uint32_t     w      = vol ? max1(vol->width >> level) : s->width;
    const uint32_t     h      = vol ? max1(vol->height >> level) : s->height;
    const uint32_t     offset = vol ? vol->offset[level] : s->offset;
    const uint32_t     pitch  = vol ? vol->pitch[level] : s->pitch;
    const VkDeviceSize bytes  = (VkDeviceSize) w * h * hf->bytes;
    VkBufferImageCopy  c;
    VkBuffer           buf;
    VkDeviceSize       off;
    uint8_t           *p;
    uint32_t           row;
    uint32_t           rows;
    uint32_t           y;

    level_layout(format, w, h, &row, &rows);
    t->format = format;
    t->w      = w;
    t->h      = h;
    t->vk     = hf->vk;
    t->swz    = hf->swz;
    t->conv   = hf->conv;
    t->hbytes = (uint32_t) hf->bytes;
    t->grow   = row;
    t->host   = malloc((size_t) bytes);
    t->guest  = malloc((size_t) row * rows);
    if (!t->host || !t->guest)
        goto fail;
    pass_end(r);
    barrier(r);
    p = slot_alloc(r, 0, bytes, 16, &buf, &off);
    if (!p)
        goto fail;
    if (vol) {
        memset(&c, 0, sizeof(c));
        c.bufferOffset                = off;
        c.bufferRowLength             = w;
        c.bufferImageHeight           = h;
        c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        c.imageSubresource.mipLevel   = level;
        c.imageSubresource.layerCount = 1;
        c.imageExtent.width           = w;
        c.imageExtent.height          = h;
        c.imageExtent.depth           = 1;
    } else
        copy_region(&c, off, w, VK_IMAGE_ASPECT_COLOR_BIT, s, 0, 0, w, h);
    r->vk->vkCmdCopyImageToBuffer(cmd(r), vol ? vol->image : im->image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &c);
    if (!submit(r, 1))
        goto fail;
    memcpy(t->host, p, (size_t) bytes);
    /* The guest's bytes, between two readings of their pages' versions
       (writers store a byte before its page's version moves): a write
       meanwhile shows, as does one since the image last uploaded. */
    {
        const uint32_t lo     = offset;
        const uint32_t hi     = offset + (rows - 1) * pitch + row;
        uint32_t       before = pages_vsig(r, lo, hi);
        int            stale  = 0;
        uint32_t       k;

        atomic_thread_fence(memory_order_acquire);
        for (y = 0; y < rows; y++)
            memcpy(t->guest + (size_t) y * row, r->vram->data + offset + (size_t) y * pitch, row);
        atomic_thread_fence(memory_order_acquire);
        if (vol) {
            for (k = 0; (k < vol->n_pages) && !stale; k++)
                stale = vol->versions[k] != vgpu98_page_version(r->vram, vol->first_page + k);
        } else
            stale = sub_stale(r, im, s);
        t->changed = stale ? 1 : (pages_vsig(r, lo, hi) != before) ? 2 : 0;
    }
    t->newer = s && s->dirty;
    return (uint64_t) bytes + (uint64_t) row * rows;
fail:
    free(t->host);
    free(t->guest);
    t->host  = NULL;
    t->guest = NULL;
    return 0;
}

/* The capture's folder, made: the machine's vgpu98-textures, then
   "<date> <time> capture N" in it. 0 when it could not be made, with buf
   naming what could not (86Box's paths hold 260 characters at most). */
static int
dump_folder(const vgpu98_r3d_t *r, uint32_t capture, char *buf, size_t size)
{
    const char *base = vgpu98_gpu_capture_dir(r->gpu);
    time_t      now  = time(NULL);
    struct tm  *tm   = localtime(&now);
    char        dir[1024];
    char        name[64];

    snprintf(buf, size, "%s", base);
    if (!base[0])
        return 0;
    if (tm)
        snprintf(name, sizeof(name), "%04d-%02d-%02d %02d-%02d-%02d capture %u", tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                 tm->tm_hour, tm->tm_min, tm->tm_sec, capture);
    else
        snprintf(name, sizeof(name), "capture %u", capture);
    if (strlen(base) + 1 + strlen(name) + sizeof("t32-face5-guest.png") >= 260) {
        snprintf(buf, size, "%s (the path would pass 260 characters)", base);
        return 0;
    }
    snprintf(dir, sizeof(dir), "%s", base);
    if (!plat_dir_check(dir) && plat_dir_create(dir))
        return 0;
    path_append_filename(buf, dir, name);
    return plat_dir_check(buf) || !plat_dir_create(buf);
}

/* A picture's own fields once read: its name, what it is, its palette. */
static void
dump_name(const vgpu98_r3d_t *r, dump_pic_t *t, const r3d_image_t *im, uint32_t num, const char *why, const char *desc, int l,
          uint32_t face)
{
    char fs[24];

    fs[0] = 0;
    if (im->faces > 1) {
        snprintf(t->name, sizeof(t->name), "t%02u-face%u", num, face);
        snprintf(fs, sizeof(fs), ", face %u", face);
    } else
        snprintf(t->name, sizeof(t->name), "t%02u", num);
    snprintf(t->what, sizeof(t->what), "%s, %s, level %d (%ux%u)%s", why, desc, l, t->w, t->h, fs);
    if (im->format == VGPU98_FMT_P8) {
        const pal_kept_t *pk = pal_find(r, im->pal_sig);

        if (pk) {
            memcpy(t->palette, pk->c, sizeof(t->palette));
            t->pal       = 1;
            t->pal_alpha = pk->alpha;
        }
    }
}

size_t
vgpu98_r3d_dump(vgpu98_r3d_t *r, uint32_t capture, uint32_t image, uint32_t sub, char *buf, size_t size)
{
    static const char *const stage_names[N_STAGES] = { "stage 0", "stage 1", "stage 2", "stage 3",
                                                       "stage 4", "stage 5", "stage 6", "stage 7" };
    r3d_image_t     *pick[DUMP_MAX];
    const char      *why[DUMP_MAX];
    r3d_vol_t       *vols[N_STAGES];
    int              vstage[N_STAGES];
    const r3d_sub_t *fsub   = NULL; /* the level or face the finding names */
    uint32_t         n_pick = 0;
    uint32_t         n_vols = 0;
    dump_job_t      *j;
    uint64_t         left = DUMP_BYTES;
    size_t           n    = 0;
    uint32_t         i;
    uint32_t         k;
    int              s;

#    define DAPP(...)                                                                      \
        do {                                                                               \
            if (n + 1 < size) {                                                            \
                const int w_ = snprintf(buf + n, size - n, __VA_ARGS__);                   \
                n += (w_ < 0) ? 0 : ((size_t) w_ >= size - n) ? size - n - 1 : (size_t) w_; \
            }                                                                              \
        } while (0)

    if (!r || (size < 2))
        return 0;
    if (r->broken) {
        DAPP("  Textures: not dumped: 3D stopped after a host GPU failure\n");
        return n;
    }
    if (atomic_load(&dump_busy)) {
        DAPP("  Textures: not dumped: the last capture's pictures are still being written\n");
        return n;
    }
    if (r->dump_count >= DUMP_SESSION) {
        DAPP("  Textures: not dumped: %u pictures written this session, the limit\n", DUMP_SESSION);
        return n;
    }

    /* The images: the finding's, the stages', then the most recently drawn. */
    for (i = 0; image && (i < r->n_img); i++) {
        if (r->img[i]->id == image) {
            pick[n_pick]  = r->img[i];
            why[n_pick++] = "the finding's";
            if (sub && (sub <= r->img[i]->n_sub))
                fsub = &r->img[i]->sub[sub - 1];
            break;
        }
    }
    for (s = 0; s < N_STAGES; s++) {
        const tex_t *t = &r->tex[s];
        r3d_image_t *im;

        if (!t->format)
            continue;
        if (t->depth) {
            r3d_vol_t *v = vol_find(r, t);

            if (v) {
                vstage[n_vols] = s;
                vols[n_vols++] = v;
            }
            continue;
        }
        im = image_find(r, t->format, t->width, t->height, t->levels, t->faces, t->offset, t->pitch);
        for (k = 0; im && (k < n_pick); k++) {
            if (pick[k] == im)
                im = NULL;
        }
        if (im && (n_pick < DUMP_MAX)) {
            pick[n_pick]  = im;
            why[n_pick++] = stage_names[s];
        }
    }
    while (n_pick < DUMP_MAX) {
        r3d_image_t *best = NULL;

        for (i = 0; i < r->n_img; i++) {
            r3d_image_t *im = r->img[i];

            if (!im->tex_use || !im->valid || (best && (im->tex_use <= best->tex_use)))
                continue;
            for (k = 0; (k < n_pick) && (pick[k] != im); k++)
                ;
            if (k == n_pick)
                best = im;
        }
        if (!best)
            break;
        pick[n_pick]  = best;
        why[n_pick++] = "drawn recently";
    }
    if (!n_pick && !n_vols) {
        DAPP("  Textures: none to dump\n");
        return n;
    }

    j = calloc(1, sizeof(dump_job_t));
    if (!j) {
        DAPP("  Textures: not dumped: no memory\n");
        return n;
    }
    j->capture = capture;
    if (!dump_folder(r, capture, j->folder, sizeof(j->folder))) {
        DAPP("  Textures: not dumped: the folder %s could not be made\n", j->folder);
        free(j);
        return n;
    }
    DAPP("  Textures, as pictures in %s; each one's comparison is a \"VGPU98 capture %u textures\" line:\n",
         j->folder, capture);
    if (image && (!n_pick || (pick[0]->id != image)))
        DAPP("  The finding's texture (host image %u) is no longer held\n", image);

    /* Each image's pictures: its level, each face of a cube map; for the
       finding's, the level and face it names when the limits take it. */
    for (i = 0; (i < n_pick) && (j->n < DUMP_MAX) && (r->dump_count + j->n < DUMP_SESSION) && !r->broken; i++) {
        r3d_image_t   *im    = pick[i];
        const int      named = fsub && (fsub->im == im);
        uint32_t       f0    = 0;
        uint32_t       f1    = im->faces;
        uint64_t       got   = 0;
        int            l     = -1;
        char           desc[96];
        const char    *by;
        uint32_t       f;

        if (!vgpu98_fmt_is_depth(im->format) && im->valid) {
            if (named && ((uint64_t) fsub->width * fsub->height * im->hf->bytes <= DUMP_LEVEL_MAX) &&
                ((uint64_t) fsub->width * fsub->height * im->hf->bytes + (fsub->hi - fsub->lo) <= left)) {
                l  = (int) fsub->level;
                f0 = fsub->face;
                f1 = f0 + 1;
            } else
                l = dump_level(im, left);
        }
        tex_text(desc, sizeof(desc), im);
        if (l < 0) {
            DAPP("  t%02u, %s: %s: not dumped (%s)\n", i + 1, why[i], desc,
                 vgpu98_fmt_is_depth(im->format) ? "a depth format"
                 : !im->valid                    ? "never uploaded"
                                                 : "no level within the limits");
            continue;
        }
        if (j->n + (f1 - f0) > DUMP_MAX) {
            DAPP("  t%02u, %s: %s: not dumped (no room left for its faces)\n", i + 1, why[i], desc);
            continue;
        }
        for (f = f0; f < f1; f++) {
            dump_pic_t    *t = &j->pic[j->n];
            const uint64_t b = dump_read(r, t, im, &im->sub[f * im->levels + (uint32_t) l], NULL, (uint32_t) l);

            if (!b)
                break;
            dump_name(r, t, im, i + 1, why[i], desc, l, f);
            got += b;
            left = (b < left) ? left - b : 0;
            j->n++;
        }
        if (f < f1) {
            DAPP("  t%02u, %s: %s: not dumped (the host GPU's copy could not be read back)\n", i + 1, why[i], desc);
            /* Its faces read so far go, and give their bytes back. */
            j->n -= f - f0;
            for (k = 0; k < f - f0; k++) {
                free(j->pic[j->n + k].host);
                free(j->pic[j->n + k].guest);
            }
            memset(&j->pic[j->n], 0, (f - f0) * sizeof(dump_pic_t));
            left += got;
            continue;
        }
        by = dump_by_upload(&j->pic[j->n - 1]) ? "; decoded by the upload's own conversion" : "";
        if (f1 - f0 > 1)
            DAPP("  t%02u, %s: %s: level %d (%ux%u), its six faces, files t%02u-face0-host.png and t%02u-face0-guest.png to "
                 "t%02u-face5-*%s%s\n",
                 i + 1, why[i], desc, l, max1(im->width >> l), max1(im->height >> l), i + 1, i + 1, i + 1,
                 ((max1(im->width >> l) > DUMP_SIDE) || (max1(im->height >> l) > DUMP_SIDE)) ? " (shrunk)" : "", by);
        else
            DAPP("  t%02u, %s: %s: level %d (%ux%u)%s, files %s-host.png and %s-guest.png%s%s\n", i + 1, why[i], desc, l,
                 max1(im->width >> l), max1(im->height >> l), named ? ", the one the finding names" : "", j->pic[j->n - 1].name,
                 j->pic[j->n - 1].name,
                 ((max1(im->width >> l) > DUMP_SIDE) || (max1(im->height >> l) > DUMP_SIDE)) ? " (shrunk)" : "", by);
    }
    for (k = 0; (k < n_vols) && (j->n < DUMP_MAX) && (r->dump_count + j->n < DUMP_SESSION) && !r->broken; k++) {
        dump_pic_t    *t = &j->pic[j->n];
        r3d_vol_t     *v = vols[k];
        const uint64_t b = (v->valid && ((uint64_t) v->width * v->height * v->hf->bytes <= DUMP_LEVEL_MAX) &&
                            (2 * (uint64_t) v->width * v->height * v->hf->bytes <= left))
                               ? dump_read(r, t, NULL, NULL, v, 0)
                               : 0;

        if (!b) {
            DAPP("  v%u, stage %d: volume %ux%ux%u %s at %08X: not dumped\n", k + 1, vstage[k], v->width, v->height, v->depth,
                 fmt_name(v->format), v->offset[0]);
            continue;
        }
        left = (b < left) ? left - b : 0;
        snprintf(t->name, sizeof(t->name), "v%u", k + 1);
        snprintf(t->what, sizeof(t->what), "stage %d, volume %ux%ux%u %s at %08X, level 0's first slice", vstage[k], v->width,
                 v->height, v->depth, fmt_name(v->format), v->offset[0]);
        j->n++;
        DAPP("  v%u, stage %d: volume %ux%ux%u %s at %08X: level 0's first slice, files v%u-host.png and v%u-guest.png%s\n", k + 1,
             vstage[k], v->width, v->height, v->depth, fmt_name(v->format), v->offset[0], k + 1, k + 1,
             dump_by_upload(t) ? "; decoded by the upload's own conversion" : "");
    }
#    undef DAPP
    if (!j->n) {
        free(j);
        return n;
    }
    r->dump_count += j->n;
    atomic_store(&dump_busy, 1);
    {
        /* 86Box's Windows threads answer -1, not NULL, when none starts. */
        const thread_t *th = thread_create_named(dump_thread, j, "VGPU98 texture dumps");

        if (!th || (th == (const thread_t *) (intptr_t) -1))
            dump_thread(j); /* no thread: on this one, then */
    }
    return n;
}

/* The churn table's three records with the most of a count (uploads or
   images made), most first; `top` holds NULL past those there are. */
static void
churn_top(const vgpu98_r3d_t *r, int made, const churn_t *top[3])
{
    uint32_t i;
    int      k;

    top[0] = top[1] = top[2] = NULL;
    for (i = 0; r->churn && (i < CHURN_RECS); i++) {
        const churn_t *c = &r->churn[i];
        const uint32_t v = made ? c->made : c->uploads;

        if (!v)
            continue;
        for (k = 0; k < 3; k++) {
            if (!top[k] || (v > (made ? top[k]->made : top[k]->uploads))) {
                memmove(&top[k + 1], &top[k], (size_t) (2 - k) * sizeof(top[0]));
                top[k] = c;
                break;
            }
        }
    }
}

void
vgpu98_r3d_tex_summary(vgpu98_r3d_t *r)
{
    static const char *const kinds[VGPU98_FINDS] = { "stale", "coordinate sets", "coordinates", "RHW", "stretching",
                                                     "zero palettes", "drawing on display", "vertex races", "arena copies",
                                                     "texture matrices" };
    char     line[1024];
    size_t   n = 0;
    uint32_t total = 0;
    uint32_t uploads = 0;
    uint32_t made    = 0;
    uint32_t evicted = 0;
    uint64_t bytes   = 0;
    uint32_t finds   = 0;
    uint32_t i;
    int      k;

    if (!r)
        return;
#    define APP(...)                                                                         \
        do {                                                                                 \
            if (n < sizeof(line)) {                                                          \
                const int w_ = snprintf(line + n, sizeof(line) - n, __VA_ARGS__);            \
                n += (w_ < 0) ? 0 : ((size_t) w_ >= sizeof(line) - n) ? sizeof(line) - n : (size_t) w_; \
            }                                                                                \
        } while (0)

    /* The stages drawn, by format and by conversion. */
    for (i = 0; i < N_FORMATS; i++)
        total += r->ta_fmt[i];
    if (total) {
        APP("VGPU98 summary textures: stages drawn %u, by format:", total);
        for (i = 0, k = 0; i < N_FORMATS; i++) {
            if (r->ta_fmt[i])
                APP("%s %s %u", k++ ? "," : "", fmt_name(i), r->ta_fmt[i]);
        }
        APP("; by the host's conversion:");
        for (i = 0, k = 0; i < CONV_COUNT; i++) {
            if (r->ta_conv[i])
                APP("%s %s %u", k++ ? "," : "", conv_name((int) i), r->ta_conv[i]);
        }
        r3d_log(r, VGPU98_LOG_SUMMARY, "%s\n", line);
    }

    /* Churn. */
    for (i = 0; r->churn && (i < CHURN_RECS); i++) {
        uploads += r->churn[i].uploads;
        bytes += r->churn[i].bytes;
        made += r->churn[i].made;
        evicted += r->churn[i].evicted;
    }
    if (uploads || made || evicted) {
        const churn_t *top[3];

        n = 0;
        APP("VGPU98 summary texture churn: %u textures, %u uploads (%.1f MB), %u host images made, %u evicted", r->n_churn, uploads,
            bytes / 1048576.0, made, evicted);
        if (r->churn_lost)
            APP(" (and %u events of textures the full table had no room for)", r->churn_lost);
        APP("; uploaded the most:");
        churn_top(r, 0, top);
        for (k = 0; (k < 3) && top[k]; k++)
            APP("%s %08X %ux%u %s %u (%.1f MB)", k ? "," : "", top[k]->offset, top[k]->width, top[k]->height,
                fmt_name(top[k]->format), top[k]->uploads, top[k]->bytes / 1048576.0);
        if (!top[0])
            APP(" none");
        APP("; made the most:");
        churn_top(r, 1, top);
        for (k = 0; (k < 3) && top[k]; k++)
            APP("%s %08X %ux%u %s %u (evicted %u)", k ? "," : "", top[k]->offset, top[k]->width, top[k]->height,
                fmt_name(top[k]->format), top[k]->made, top[k]->evicted);
        if (!top[0])
            APP(" none");
        r3d_log(r, VGPU98_LOG_SUMMARY, "%s\n", line);
    }

    /* The checks and the findings. */
    for (k = 0; k < VGPU98_FINDS; k++)
        finds += r->ta_finds[k];
    if (r->ta_checks || r->ta_hash_up || finds) {
        n = 0;
        APP("VGPU98 summary texture audit: %u checks (%.1f MB hashed; %.1f MB more at uploads), a texture checked every %u s at "
            "most, %u past %u MB not checked, %u found changed once; findings %u",
            r->ta_checks, r->ta_hash_check / 1048576.0, r->ta_hash_up / 1048576.0,
            (uint32_t) ((r->ta_every ? r->ta_every : 1000000) / 1000000), r->ta_big, TEX_AUDIT_BIG >> 20, r->ta_suspects, finds);
        for (k = 0, i = 0; k < VGPU98_FINDS; k++) {
            if (r->ta_finds[k])
                APP("%s %s %u", i++ ? "," : ":", kinds[k], r->ta_finds[k]);
        }
        r3d_log(r, VGPU98_LOG_SUMMARY, "%s\n", line);
    }
    /* The vertex race detector's. */
    if (r->ta_vsum_checked || r->ta_vsum_refused || r->ta_note_refused || r->ta_vsum_dropped) {
        const uint32_t *c = r->ta_vsum_after;

        n = 0;
        APP("VGPU98 summary texture races: %u vertex and index windows checked (%.1f MB), %u changed before the GPU read them (%u in "
            "every part, %u in memory another buffer held lately); races told "
            "by the buffer's Locks between the draw and its next: NOOVERWRITE given while a draw was pending %u, another given "
            "so %u, one given with none pending %u, one that waited %u, new memory only %u, none but one open %u, none %u, no "
            "later draw %u (told early for room %u, dropped at a reset %u); changed in guest RAM only (the arena's copy drawn) "
            "%u; arena copies differing %u; notes refused: %u naming no stream or not readable, %u damaged or out of sequence",
            r->ta_vsum_checked, r->ta_vsum_bytes / 1048576.0, r->ta_vsum_races, r->ta_vsum_whole, r->ta_vsum_reused,
            c[VSUM_AFTER_NOOW], c[VSUM_AFTER_UNWAITED],
            c[VSUM_AFTER_IDLE], c[VSUM_AFTER_WAITED], c[VSUM_AFTER_RENAMED], c[VSUM_AFTER_OPEN], c[VSUM_AFTER_NONE],
            c[VSUM_AFTER_UNSEEN], r->ta_vsum_early, r->ta_vsum_dropped, r->ta_vsum_ram, r->ta_vsum_arena, r->ta_vsum_refused,
            r->ta_note_refused);
        r3d_log(r, VGPU98_LOG_SUMMARY, "%s\n", line);
    }
    r->ta_vsum_checked = 0;
    r->ta_vsum_bytes   = 0;
    r->ta_vsum_races   = 0;
    memset(r->ta_vsum_after, 0, sizeof(r->ta_vsum_after));
    r->ta_vsum_ram     = 0;
    r->ta_vsum_arena   = 0;
    r->ta_vsum_refused = 0;
    r->ta_note_refused = 0;
    r->ta_vsum_early   = 0;
    r->ta_vsum_dropped = 0;
    r->ta_vsum_whole   = 0;
    r->ta_vsum_reused  = 0;
    /* The texture-matrix checks'. */
    if (r->ta_mx_draws || r->ta_mx_unfollowed) {
        n = 0;
        APP("VGPU98 summary texture matrices: %u transformed draws read, %u matrices mapping u and v to one direction, %u draws "
            "smearing a texel one way edge-on to their planar mapping (the mapping's own), %u smearing otherwise, %u kinds whose "
            "texture matrix moved while the view did not; %u stages with normal or reflection coordinates not followed",
            r->ta_mx_draws, r->ta_mx_collapsed, r->ta_mx_edge, r->ta_mx_smeared, r->ta_mx_moving, r->ta_mx_unfollowed);
        r3d_log(r, VGPU98_LOG_SUMMARY, "%s\n", line);
    }
    r->ta_mx_draws      = 0;
    r->ta_mx_collapsed  = 0;
    r->ta_mx_edge       = 0;
    r->ta_mx_smeared    = 0;
    r->ta_mx_moving     = 0;
    r->ta_mx_unfollowed = 0;
    /* The coordinate checks' and the display check's counted-only figures. */
    if (r->ta_coord_draws || r->ta_frames) {
        n = 0;
        APP("VGPU98 summary texture coordinates: %u draws read (%u others not: a kind a second), %u whose primitives all read one "
            "coordinate, %u triangles WRAPn takes the short way round, %u pre-transformed draws smearing a texel one way, %u with "
            "a texture transform set (no matrix applied); frames looked at for drawing on display %u, drawn mostly into the "
            "surface on display %u",
            r->ta_coord_draws, r->ta_coord_skipped, r->ta_one_coord, r->ta_wrap_short, r->ta_smeared, r->ta_pre_xform,
            r->ta_frames, r->ta_shown);
        r3d_log(r, VGPU98_LOG_SUMMARY, "%s\n", line);
    }
#    undef APP

    memset(r->ta_fmt, 0, sizeof(r->ta_fmt));
    memset(r->ta_conv, 0, sizeof(r->ta_conv));
    memset(r->ta_finds, 0, sizeof(r->ta_finds));
    r->ta_checks      = 0;
    r->ta_frames      = 0;
    r->ta_shown       = 0;
    r->ta_coord_draws   = 0;
    r->ta_coord_skipped = 0;
    r->ta_one_coord     = 0;
    r->ta_wrap_short  = 0;
    r->ta_smeared     = 0;
    r->ta_pre_xform   = 0;
    r->ta_big         = 0;
    r->ta_hash_up    = 0;
    r->ta_hash_check = 0;
    r->ta_suspects   = 0;
    if (r->churn)
        memset(r->churn, 0, CHURN_RECS * sizeof(churn_t));
    r->n_churn    = 0;
    r->churn_lost = 0;
}

void
vgpu98_r3d_census_start(vgpu98_r3d_t *r)
{
    if (!r)
        return;
    free(r->census_text);
    r->census_text   = NULL;
    r->census_len    = 0;
    r->census_cap    = 0;
    r->census_on     = 1;
    r->census_us     = vgpu98_diag_us();
    r->n_census_seen = 0;
    memset(r->census_bits, 0, sizeof(r->census_bits));
    r->census_gen    = 0;
    r->census_other  = 0;
    r->census_untold = 0;
    r->census_lost   = 0;
}

char *
vgpu98_r3d_census_end(vgpu98_r3d_t *r)
{
    const size_t size = r ? r->census_len + 512 : 0;
    char        *t;
    size_t       n = 0;

    if (!r)
        return NULL;
    r->census_on = 0;
    t            = malloc(size);
    if (t) {
        t[0] = 0;
        app(t, size, &n, "  Draws after the trigger, the first of each kind (%u with generated coordinates or a texture transform and %u others at most):\n",
            CENSUS_KINDS, CENSUS_KINDS);
        if (r->census_text)
            app(t, size, &n, "%s", r->census_text);
        else
            app(t, size, &n, "  (no draws)\n");
        if (r->census_untold)
            app(t, size, &n, "  %u kinds more%s, not described\n", r->census_untold, r->census_lost ? " at least" : "");
    }
    free(r->census_text);
    r->census_text = NULL;
    r->census_len  = 0;
    r->census_cap  = 0;
    return t;
}

size_t
vgpu98_r3d_describe(const vgpu98_r3d_t *r, char *buf, size_t size)
{
    static const struct {
        uint8_t     id;
        const char *name;
    } names[] = {
        { VGPU98_RS_ZENABLE, "ZENABLE" },          { VGPU98_RS_ZWRITEENABLE, "ZWRITEENABLE" },
        { VGPU98_RS_ZFUNC, "ZFUNC" },              { VGPU98_RS_ZBIAS, "ZBIAS" },
        { VGPU98_RS_FILLMODE, "FILLMODE" },        { VGPU98_RS_SHADEMODE, "SHADEMODE" },
        { VGPU98_RS_CULLMODE, "CULLMODE" },        { VGPU98_RS_ALPHABLENDENABLE, "ALPHABLENDENABLE" },
        { VGPU98_RS_SRCBLEND, "SRCBLEND" },        { VGPU98_RS_DESTBLEND, "DESTBLEND" },
        { VGPU98_RS_ALPHATESTENABLE, "ALPHATESTENABLE" }, { VGPU98_RS_ALPHAFUNC, "ALPHAFUNC" },
        { VGPU98_RS_ALPHAREF, "ALPHAREF" },        { VGPU98_RS_COLORKEYENABLE, "COLORKEYENABLE" },
        { VGPU98_RS_FOGENABLE, "FOGENABLE" },      { VGPU98_RS_FOGTABLEMODE, "FOGTABLEMODE" },
        { VGPU98_RS_SPECULARENABLE, "SPECULARENABLE" }, { VGPU98_RS_STENCILENABLE, "STENCILENABLE" },
        { VGPU98_RS_TEXTUREPERSPECTIVE, "TEXTUREPERSPECTIVE" }, { VGPU98_RS_TEXTUREFACTOR, "TEXTUREFACTOR" },
        { VGPU98_RS_WRAP0, "WRAP0" },
        /* The dithering and the fog */
        { VGPU98_RS_DITHERENABLE, "DITHERENABLE" }, { VGPU98_RS_FOGVERTEXMODE, "FOGVERTEXMODE" },
        { VGPU98_RS_FOGCOLOR, "FOGCOLOR" },        { VGPU98_RS_FOGTABLESTART, "FOGSTART" },
        { VGPU98_RS_FOGTABLEEND, "FOGEND" },       { VGPU98_RS_FOGTABLEDENSITY, "FOGDENSITY" },
        { VGPU98_RS_RANGEFOGENABLE, "RANGEFOGENABLE" },
    };
    size_t   n = 0;
    uint32_t s;
    size_t   i;

    if (!size)
        return 0;
    buf[0] = 0;
    app(buf, size, &n, "  3D target ");
    if (r->have_target)
        app_surf(buf, size, &n, &r->target);
    else
        app(buf, size, &n, "none");
    app(buf, size, &n, "; depth ");
    if (r->have_depth) {
        const r3d_sub_t *ds = sub_find(r, r->depth.format, r->depth.width, r->depth.height, r->depth.offset, r->depth.pitch, NULL);

        app_surf(buf, size, &n, &r->depth);
        /* Its host image: the format, and what it holds (gpu-formats.md). */
        if (!ds)
            app(buf, size, &n, " (no host image)");
        else
            app(buf, size, &n, " (host %s holding %s%s)", depth_vk_name(ds->im->hf->vk),
                !depth_float(ds->im->hf) ? "Z, or the reference's form of W" :
                ds->im->depth_w          ? "W" :
                depth_grid(r, ds->im)    ? "Z on the guest's grid" :
                                           "Z at full precision",
                depth_rec_find((vgpu98_r3d_t *) r, r->depth.format, r->depth.width, r->depth.height, r->depth.offset,
                               r->depth.pitch) ? "; W-buffered into" : "");
    } else
        app(buf, size, &n, "none");
    app(buf, size, &n, "; viewport %u,%u %ux%u; W %g to %g; Z %g to %g\n", r->vp[0], r->vp[1], r->vp[2], r->vp[3], r->wnear,
        r->wfar, r->zrange[0], r->zrange[1]);
    if (r->vin.have_decl) {
        app(buf, size, &n, "  3D declaration (%s):", r->vin.program ? "transformed" : "pre-transformed");
        for (i = 0; i < r->vin.n_elements; i++) {
            const uint32_t d0 = r->vin.decl[2 * i];

            app(buf, size, &n, " s%u+%u type %u use %u.%u", VGPU98_DECL_STREAM(d0), r->vin.decl[2 * i + 1],
                VGPU98_DECL_TYPE(d0), VGPU98_DECL_USE(d0), VGPU98_DECL_INDEX(d0));
        }
        app(buf, size, &n, "\n");
        for (s = 0; s < VGPU98_MAX_STREAMS; s++) {
            if (((r->vin.streams_used >> s) & 1) && r->vin.stream[s].set)
                app(buf, size, &n, "  3D stream %u: %s %08X, %u bytes, stride %u, version %u%s\n", s,
                    (r->vin.stream[s].source == VGPU98_SRC_INLINE) ? "inline" : "GPU address", r->vin.stream[s].address,
                    r->vin.stream[s].size, r->vin.stream[s].stride, r->vin.stream[s].version,
                    ((r->vin.res_mask >> s) & 1) ? " (resident at the last draw)" : "");
        }
    } else
        app(buf, size, &n, "  3D declaration: none\n");
    if (r->vin.program) {
        app(buf, size, &n, "  3D transform: LIGHTING %u, lights enabled %02X (types", r->rs[VGPU98_RS_LIGHTING], r->light_mask);
        for (s = 0; s < VGPU98_MAX_LIGHTS; s++) {
            if ((r->light_mask >> s) & 1)
                app(buf, size, &n, " %u", r->light[s][0]);
        }
        app(buf, size, &n, "), VERTEXBLEND %u, clip planes %02X, ZRANGE %g to %g; world0 row 4 %g %g %g; view row 4 %g %g %g\n",
            r->rs[VGPU98_RS_VERTEXBLEND], r->rs[VGPU98_RS_CLIPPING] ? r->rs[VGPU98_RS_CLIPPLANEENABLE] : 0, r->zrange[0],
            r->zrange[1], r->xform[VGPU98_XFORM_WORLD][12], r->xform[VGPU98_XFORM_WORLD][13], r->xform[VGPU98_XFORM_WORLD][14],
            r->xform[VGPU98_XFORM_VIEW][12], r->xform[VGPU98_XFORM_VIEW][13], r->xform[VGPU98_XFORM_VIEW][14]);
    }
    if (r->proj_set) {
        /* The projection's depth terms, and W at its near and far planes
           (where z/w is 0 and 1): the W range the runtime gives WINFO. */
        const float *m  = r->xform[VGPU98_XFORM_PROJECTION];
        const double wn = (double) m[15] - (double) m[14] * m[11] / m[10];
        const double wf = (double) m[11] * ((double) m[15] - m[14]) / ((double) m[10] - m[11]) + m[15];

        app(buf, size, &n,
            "  3D projection (the last a DirectX 7 context sent): _33 %g, _34 %g, _43 %g, _44 %g; W at its near and far planes %g and %g\n",
            m[10], m[11], m[14], m[15], wn, wf);
    } else
        app(buf, size, &n, "  3D projection: none since the reset (a DirectX 6 program's transform is its own)\n");
    if (r->vin.index.set)
        app(buf, size, &n, "  3D indices: %u-byte, %s %08X, %u bytes, version %u\n", r->vin.index.isize,
            (r->vin.index.source == VGPU98_SRC_INLINE) ? "inline" : "GPU address", r->vin.index.address, r->vin.index.size,
            r->vin.index.version);
    app(buf, size, &n, "  3D states (Direct3D's names):");
    for (i = 0; i < (sizeof(names) / sizeof(names[0])); i++)
        app(buf, size, &n, " %s=%X", names[i].name, r->rs[names[i].id]);
    app(buf, size, &n, "\n");
    if (fixed8(r))
        app_states8(buf, size, &n, r, "  3D ");
    for (s = 0; s < N_STAGES; s++) {
        const uint32_t *t = r->tss[s];

        if (t[VGPU98_TSS_COLOROP] == VGPU98_TOP_DISABLE) {
            app(buf, size, &n, "  3D stage %u: disabled\n", s);
            break;
        }
        app(buf, size, &n, "  3D stage %u: ", s);
        app_stage(buf, size, &n, r, s, r->vin.program);
        app(buf, size, &n, "\n");
        app_mean(buf, size, &n, r, s, "  3D ");
        if (r->vin.program && (t[VGPU98_TSS_TEXTURETRANSFORMFLAGS] & VGPU98_TTFF_COUNT_MASK)) {
            app(buf, size, &n, "  3D stage %u matrix: ", s);
            app_matrix(buf, size, &n, r->xform[VGPU98_XFORM_TEXTURE + s]);
            app(buf, size, &n, "\n");
        }
    }
    app(buf, size, &n, "  3D host: %u images (%.1f of %u MB), %u pipelines, %u samplers%s\n", (unsigned) r->n_img,
        (double) r->img_bytes / (1u << 20), (unsigned) (r->img_budget >> 20), (unsigned) r->n_pipes, (unsigned) r->n_samps,
        r->broken ? "; 3D stopped until the next reset" : "");
    return n;
}

#else /* VGPU98_NO_VULKAN: built without the shader compiler; no 3D */

vgpu98_r3d_t *
vgpu98_r3d_init(vgpu98_gpu_t *g, const vgpu98_vram_t *vram)
{
    (void) vram;
    vgpu98_gpu_log(g, VGPU98_LOG_EVENTS, "VGPU98 3D: this 86Box was built without the GPU's shaders; no 3D\n");
    return NULL;
}

int
vgpu98_r3d_close(vgpu98_r3d_t *r)
{
    (void) r;
    return 1;
}

void
vgpu98_r3d_reset(vgpu98_r3d_t *r, uint64_t deadline_us, int forget)
{
    (void) r;
    (void) deadline_us;
    (void) forget;
}

void
vgpu98_r3d_uploaded(vgpu98_r3d_t *r, const vgpu98_surf_t *sf, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                    uint32_t start, uint32_t len)
{
    (void) r;
    (void) sf;
    (void) x0;
    (void) y0;
    (void) x1;
    (void) y1;
    (void) start;
    (void) len;
}

void
vgpu98_r3d_written(vgpu98_r3d_t *r, const vgpu98_surf_t *sf, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
{
    (void) r;
    (void) sf;
    (void) x0;
    (void) y0;
    (void) x1;
    (void) y1;
}

void
vgpu98_r3d_sync(vgpu98_r3d_t *r, uint32_t offset, uint32_t len, int cause)
{
    (void) r;
    (void) cause;
    (void) offset;
    (void) len;
}

void
vgpu98_r3d_discard(vgpu98_r3d_t *r, uint32_t offset, uint32_t len)
{
    (void) r;
    (void) offset;
    (void) len;
}

int
vgpu98_r3d_packet(vgpu98_r3d_t *r, uint32_t op, const uint32_t *payload, uint32_t len)
{
    (void) r;
    (void) op;
    (void) payload;
    (void) len;
    return VGPU98_ERR_PROFILE;
}

void
vgpu98_r3d_flush(vgpu98_r3d_t *r)
{
    (void) r;
}

void
vgpu98_r3d_publish(vgpu98_r3d_t *r, vgpu98_disp_t *d, uint32_t lo, uint32_t hi, uint32_t bpp)
{
    (void) r;
    (void) d;
    (void) lo;
    (void) hi;
    (void) bpp;
}

void
vgpu98_r3d_set_disp(vgpu98_r3d_t *r, vgpu98_disp_t *d)
{
    (void) r;
    (void) d;
}

void
vgpu98_r3d_closing(vgpu98_r3d_t *r)
{
    (void) r;
}

void
vgpu98_r3d_usage(const vgpu98_r3d_t *r, uint32_t *images, uint64_t *bytes, uint64_t *budget)
{
    (void) r;
    *images = 0;
    *bytes  = 0;
    *budget = 0;
}

void
vgpu98_r3d_vol_usage(const vgpu98_r3d_t *r, uint32_t *volumes, uint64_t *bytes)
{
    (void) r;
    *volumes = 0;
    *bytes   = 0;
}

void
vgpu98_r3d_wrange(const vgpu98_r3d_t *r, float *wnear, float *wfar)
{
    (void) r;
    *wnear = 1.0f;
    *wfar  = 1.0f;
}

void
vgpu98_r3d_tex_summary(vgpu98_r3d_t *r)
{
    (void) r;
}

void
vgpu98_r3d_frame(vgpu98_r3d_t *r, uint32_t start)
{
    (void) r;
    (void) start;
}

void
vgpu98_r3d_note(vgpu98_r3d_t *r, const uint32_t *p, uint32_t len, uint64_t at, uint64_t known)
{
    (void) r;
    (void) p;
    (void) len;
    (void) at;
    (void) known;
}

size_t
vgpu98_r3d_dump(vgpu98_r3d_t *r, uint32_t capture, uint32_t image, uint32_t sub, char *buf, size_t size)
{
    (void) r;
    (void) capture;
    (void) image;
    (void) sub;
    (void) buf;
    (void) size;
    return 0;
}

void
vgpu98_r3d_census_start(vgpu98_r3d_t *r)
{
    (void) r;
}

char *
vgpu98_r3d_census_end(vgpu98_r3d_t *r)
{
    (void) r;
    return NULL;
}

size_t
vgpu98_r3d_describe(const vgpu98_r3d_t *r, char *buf, size_t size)
{
    (void) r;
    if (size)
        buf[0] = 0;
    return 0;
}

#endif /* VGPU98_NO_VULKAN */
